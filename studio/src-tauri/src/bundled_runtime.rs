//! The Python runtime baked into the application bundle.
//!
//! Stage 2 of making the macOS app self-contained: `Unsloth.app` carries its own
//! CPython, its own site-packages and its own native helpers under
//! `Contents/Resources/runtime/`, so the first launch has nothing to install.
//!
//! ```text
//! Unsloth.app/Contents/Resources/runtime/
//!   python/                  relocatable CPython; python/bin/python3 is the interpreter
//!   site-packages/           all installed distributions (unsloth_cli/, studio/, deps)
//!   llama.cpp/               llama.cpp prebuilt (same layout as ~/.unsloth/llama.cpp)
//!   whisper.cpp/             whisper.cpp prebuilt binary
//!   node/                    trimmed Node (bin/node, npm)
//!   oxc-node-modules/        prefetched node_modules for the OXC validator
//!   BUNDLE_MANIFEST.json     component versions, source URLs, sha256s, python version
//! ```
//!
//! This module is the one seam the rest of the app asks: **is there a bundled
//! runtime?** Everything else — CLI resolution, preflight, repair, the installer
//! refusal — branches on [`bundled_runtime`] returning `Some` and never rebuilds
//! these paths itself.
//!
//! ## What moves and what does not
//!
//! Only *code* moves into the bundle. `~/.unsloth/studio` stays the writable data
//! root (DBs, logs, caches, models, the desktop auth secret), which is why nothing
//! here sets `UNSLOTH_STUDIO_HOME`: that variable is read as the data root by both
//! `unsloth_cli.commands.studio._resolve_studio_home` and
//! `studio/backend/utils/paths/storage_roots.studio_root`, so pointing it at the
//! bundle would move every database into a read-only, signed directory. The
//! existing scrub of `UNSLOTH_STUDIO_HOME` / `STUDIO_HOME` for managed children
//! therefore stays exactly as it was.
//!
//! ## Why no `__pycache__`
//!
//! The bundle is signed and, once dragged to `/Applications`, root-owned. A
//! `__pycache__` write there fails, and on a signed bundle a *successful* write
//! invalidates resource validation. `-B` on the interpreter, plus
//! `PYTHONDONTWRITEBYTECODE=1` for whatever it spawns, keeps the tree untouched.

use log::{info, warn};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{AppHandle, Manager};

/// The resource directory the payload lands in.
const RESOURCE_DIR_NAME: &str = "runtime";

/// Written by the payload build; read here only to prove the payload is whole.
const MANIFEST_NAME: &str = "BUNDLE_MANIFEST.json";

/// How the bootstrap below learns where the bundled distributions are.
///
/// An environment variable rather than a path baked into the `-c` string: paths
/// are `OsString`s and a lossy conversion into a Python string literal would
/// silently mangle a non-UTF-8 bundle path. It is always set explicitly on the
/// child (see [`BundledRuntime::child_env`]), so an inherited value can never
/// redirect an import.
pub(crate) const SITE_PACKAGES_ENV: &str = "UNSLOTH_BUNDLED_SITE_PACKAGES";

/// The interpreter bootstrap, mirroring [`crate::process::WINDOWS_CLI_ENTRYPOINT`]:
/// reach `unsloth_cli:app` through the interpreter instead of the generated
/// console script, which the bundle does not ship.
///
/// `-c` rather than `-m unsloth_cli`, and this is the one deliberate divergence
/// from the interface note that asked for `-I -P -m unsloth_cli`:
///
///   * `-I` implies `-E`, and `-E` discards `PYTHONPATH`. A bundled runtime keeps
///     its distributions OUTSIDE the interpreter's own prefix (`runtime/
///     site-packages`, not `runtime/python/lib/pythonX.Y/site-packages`), so
///     `PYTHONPATH` is the only env route to them and `-I` closes it. Reading the
///     directory from a private variable `-E` knows nothing about keeps full
///     isolation AND reaches the packages.
///   * `-P` is redundant: `-I` has dropped the working directory from `sys.path`
///     since 3.4, and the `sys.path[:1]` filter below is the same belt-and-braces
///     strip the Windows entry point does. It is also 3.11+, and while today's
///     payload pins CPython 3.13 (`studio/macos_runtime_pins.json`) this repo's
///     lock floor is 3.10 (`install_python_stack.py`,
///     `unsloth-lock-python-floor: 3.10`), so spelling the isolation with a flag
///     that predates both is one fewer thing a payload rebuild can break.
///
/// `append`, not `insert(0, ...)`: `site` appends site-packages after the stdlib,
/// and a distribution shadowing a stdlib module is not a behaviour to introduce
/// here. Under `-I` there is no other site-packages to outrank.
///
/// `sys.argv[0]` is assigned before the import because `unsloth_cli/__init__.py`
/// decides at import time whether it is the console script; that gate is what
/// applies the console-script stream setup and sets Typer's `prog_name`.
pub(crate) const BUNDLED_CLI_ENTRYPOINT: &str = "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.path.append(os.environ['UNSLOTH_BUNDLED_SITE_PACKAGES']); sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())";

/// Whether this build honours a bundled runtime at all.
///
/// macOS only for this stage. Linux ships an AppImage that scrubs `PYTHONHOME` /
/// `PYTHONPATH` for its managed children (`process::scrub_appimage_python_env`),
/// which is the opposite of what a bundled runtime needs, and Windows still
/// installs through `install.ps1`; neither payload nor CI exists for them. The
/// gate is one `cfg!` so widening it later is a one-line change with the whole of
/// this module already platform-neutral.
pub(crate) const fn bundled_runtime_supported() -> bool {
    cfg!(target_os = "macos")
}

/// A runtime found inside this app bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundledRuntime {
    root: PathBuf,
}

impl BundledRuntime {
    /// `Contents/Resources/runtime`.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// The interpreter every managed invocation runs.
    pub(crate) fn interpreter(&self) -> PathBuf {
        if cfg!(windows) {
            self.root.join("python").join("python.exe")
        } else {
            self.root.join("python").join("bin").join("python3")
        }
    }

    /// The interpreter's own prefix. Set as `PYTHONHOME` for descendants, which
    /// are ordinary (non-`-E`) interpreters and need to find this stdlib.
    pub(crate) fn python_home(&self) -> PathBuf {
        self.root.join("python")
    }

    /// Every installed distribution: `unsloth_cli/`, `studio/`, and their deps.
    pub(crate) fn site_packages(&self) -> PathBuf {
        self.root.join("site-packages")
    }

    pub(crate) fn llama_cpp_dir(&self) -> PathBuf {
        self.root.join("llama.cpp")
    }

    pub(crate) fn whisper_cpp_dir(&self) -> PathBuf {
        self.root.join("whisper.cpp")
    }

    /// The directory holding `node` (and `npm`), for the child `PATH`.
    pub(crate) fn node_bin_dir(&self) -> PathBuf {
        if cfg!(windows) {
            self.root.join("node")
        } else {
            self.root.join("node").join("bin")
        }
    }

    pub(crate) fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_NAME)
    }

    /// The stdlib landmark CPython itself looks for when it computes `sys.prefix`
    /// from the executable path. Under `-I` there is no `PYTHONHOME` to fall back
    /// on, so a payload missing this cannot start at all.
    fn stdlib_landmark(&self) -> Option<PathBuf> {
        let lib = self
            .python_home()
            .join(if cfg!(windows) { "Lib" } else { "lib" });
        if cfg!(windows) {
            let landmark = lib.join("os.py");
            return landmark.is_file().then_some(landmark);
        }
        let entries = std::fs::read_dir(&lib).ok()?;
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("python"))
            .map(|entry| entry.path().join("os.py"))
            .filter(|landmark| landmark.is_file())
            .collect();
        // read_dir order is unspecified and a payload could carry two stdlibs.
        found.sort();
        found.into_iter().next()
    }

    /// Whether this payload is whole enough to run, and if not, why.
    ///
    /// A filesystem check only, deliberately: it runs on the launch path. It is
    /// not a staleness check — a bundled runtime cannot be stale, since the
    /// desktop shell and the payload ship in the same `.app` from the same build,
    /// so "one bundle is one stack" is true by construction rather than something
    /// to measure. What it does catch is a payload that never arrived, was
    /// truncated, or was partly stripped: exactly the failures that would
    /// otherwise surface as an unexplained backend crash.
    pub(crate) fn health(&self) -> Result<(), String> {
        let interpreter = self.interpreter();
        if !interpreter.is_file() {
            return Err(format!(
                "the bundled Python interpreter is missing: {}",
                interpreter.display()
            ));
        }
        if self.stdlib_landmark().is_none() {
            return Err(format!(
                "the bundled Python standard library is missing under {}",
                self.python_home().display()
            ));
        }
        let site_packages = self.site_packages();
        for relative in [
            // What BUNDLED_CLI_ENTRYPOINT imports.
            Path::new("unsloth_cli").join("__init__.py"),
            // What `unsloth studio` loads to serve the backend
            // (unsloth_cli/commands/studio.py::_find_run_py).
            Path::new("studio").join("backend").join("run.py"),
        ] {
            let path = site_packages.join(&relative);
            if !path.is_file() {
                return Err(format!(
                    "the bundled runtime is incomplete: {} is missing",
                    path.display()
                ));
            }
        }
        // Parsed, not merely present: a truncated copy of the payload is the case
        // worth catching, and the manifest is written last. Its schema belongs to
        // the payload build, so only "it is a JSON object" is asserted here — the
        // component versions inside it are not a runtime input, since the shell
        // that reads them shipped in the same bundle.
        let manifest = self.manifest_path();
        let bytes = std::fs::read(&manifest)
            .map_err(|error| format!("could not read {}: {}", manifest.display(), error))?;
        match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(serde_json::Value::Object(_)) => Ok(()),
            Ok(_) => Err(format!("{} is not a JSON object", manifest.display())),
            Err(error) => Err(format!(
                "{} could not be parsed: {}",
                manifest.display(),
                error
            )),
        }
    }

    /// The argument vector that runs `unsloth <args>` on the bundled interpreter.
    ///
    /// `-X utf8` first, as on Windows today: `-I` implies `-E`, which would discard
    /// `PYTHONUTF8`, and the flag form survives it. `-B` for the same reason —
    /// `PYTHONDONTWRITEBYTECODE` is one of the variables `-E` drops, and not
    /// writing into a signed bundle is not optional.
    pub(crate) fn cli_args(&self, args: &[&str]) -> Vec<OsString> {
        let mut argv: Vec<OsString> = ["-X", "utf8", "-I", "-B", "-c", BUNDLED_CLI_ENTRYPOINT]
            .into_iter()
            .map(OsString::from)
            .collect();
        argv.extend(args.iter().copied().map(OsString::from));
        argv
    }

    /// Everything a managed child needs to find this runtime.
    ///
    /// Applied by [`crate::process::ManagedCliInvocation::to_command`] and its
    /// tokio twin, so no call site can build the invocation and forget the
    /// environment that makes it importable.
    pub(crate) fn child_env(&self) -> Vec<(OsString, OsString)> {
        self.child_env_from(&|name| std::env::var_os(name))
    }

    fn child_env_from(
        &self,
        lookup: &dyn Fn(&str) -> Option<OsString>,
    ) -> Vec<(OsString, OsString)> {
        let mut env: Vec<(OsString, OsString)> = Vec::new();
        let mut set = |name: &str, value: OsString| env.push((OsString::from(name), value));

        set(SITE_PACKAGES_ENV, self.site_packages().into_os_string());
        // Ignored by this interpreter (`-I` implies `-E`) and load-bearing for its
        // descendants: the backend spawns further copies of `sys.executable`
        // without `-I`, and those find neither their stdlib nor the bundled
        // distributions without these two.
        set("PYTHONHOME", self.python_home().into_os_string());
        set("PYTHONPATH", self.site_packages().into_os_string());
        set("PYTHONDONTWRITEBYTECODE", OsString::from("1"));

        // The two native runtimes that already have a documented directory
        // override, pointed at the bundled copies. A value the user set wins:
        // both are pre-existing user-facing settings, and process.rs deliberately
        // keeps them where it scrubs UNSLOTH_STUDIO_HOME. Only an unset (or blank)
        // value is filled in, so this is order-independent with the relative-path
        // pinning in `process::apply_managed_cli_context_*`, which rewrites a
        // user's relative value and must not be overruled here.
        for (name, dir) in [
            ("UNSLOTH_LLAMA_CPP_PATH", self.llama_cpp_dir()),
            ("UNSLOTH_WHISPER_CPP_PATH", self.whisper_cpp_dir()),
        ] {
            let user_set = lookup(name).is_some_and(|value| !value.is_empty());
            if !user_set && dir.is_dir() {
                set(name, dir.into_os_string());
            }
        }

        // Node has no directory override to set: `utils/node_runtime.py`
        // resolves it from `shutil.which("node")` first and only then from
        // `<STUDIO_HOME>/node`, so the bundled copy is reached by putting its bin
        // directory at the front of the child PATH. That is also what the setup
        // scripts do for their own process. Only `node`, `npm` and `npx` live
        // there, so nothing else is shadowed.
        let node_bin = self.node_bin_dir();
        if node_bin.is_dir() {
            let mut path = node_bin.into_os_string();
            if let Some(existing) = lookup("PATH").filter(|existing| !existing.is_empty()) {
                path.push(OsStr::new(if cfg!(windows) { ";" } else { ":" }));
                path.push(existing);
            }
            set("PATH", path);
        }

        env
    }
}

/// Whether `root` holds a bundled runtime.
///
/// Deliberately lenient — the directory existing is enough. A half-extracted
/// payload must still be *detected*, because the alternative is falling back to
/// `install.sh` and quietly building a second, mutable Python environment under
/// `~/.unsloth` that the app would then have to choose between. Whether the
/// payload is usable is [`BundledRuntime::health`]'s question, and preflight
/// reports it rather than repairing it.
pub(crate) fn locate(root: PathBuf) -> Option<BundledRuntime> {
    root.is_dir().then_some(BundledRuntime { root })
}

static BUNDLED_RUNTIME: OnceLock<Option<BundledRuntime>> = OnceLock::new();

/// Resolve the bundled runtime once, through Tauri's resource API.
///
/// Called from the setup hook before anything asks. `resolve` rather than
/// string-building from `current_exe()`: the layout of a resource directory
/// differs per bundle format and Tauri owns that knowledge.
pub fn init(app: &AppHandle) {
    let resolved = if !bundled_runtime_supported() {
        None
    } else {
        match app
            .path()
            .resolve(RESOURCE_DIR_NAME, tauri::path::BaseDirectory::Resource)
        {
            Ok(root) => locate(root),
            Err(error) => {
                warn!("Could not resolve the bundled runtime directory: {error}");
                None
            }
        }
    };

    match &resolved {
        Some(runtime) => match runtime.health() {
            Ok(()) => info!(
                "Using the bundled Python runtime at {}",
                runtime.root().display()
            ),
            Err(reason) => warn!(
                "The bundled Python runtime at {} is not usable: {}",
                runtime.root().display(),
                reason
            ),
        },
        None => info!("No bundled Python runtime; using the managed install under ~/.unsloth"),
    }

    // A second call is a programming error, not a state to reconcile.
    if BUNDLED_RUNTIME.set(resolved).is_err() {
        warn!("The bundled runtime was already resolved; ignoring the second attempt");
    }
}

/// The bundled runtime, or `None` when this build has none.
///
/// `None` before [`init`] runs, which is the honest answer: nothing on the launch
/// path asks before the setup hook, and a caller that did must behave as it did
/// before this file existed.
pub(crate) fn bundled_runtime() -> Option<&'static BundledRuntime> {
    BUNDLED_RUNTIME.get().and_then(Option::as_ref)
}

/// The writable data root, created if absent.
///
/// This is the whole of "first launch prepares state" for a bundled runtime: the
/// installer used to create this directory on the way to building a Python
/// environment in it, and only the directory part still has to happen. Everything
/// below it is created by whoever owns it — the desktop ownership id by
/// `desktop_backend_owner`, the databases and caches by the backend.
///
/// No permission change: the installer owns that policy for this directory, and a
/// user who already has one keeps whatever they have.
pub(crate) fn ensure_writable_state_root() -> Result<PathBuf, String> {
    let home =
        dirs::home_dir().ok_or_else(|| "could not resolve the home directory".to_string())?;
    let root = home.join(".unsloth").join("studio");
    std::fs::create_dir_all(&root)
        .map_err(|error| format!("could not create {}: {}", root.display(), error))?;
    Ok(root)
}

/// What "repair" means when the code is immutable.
///
/// The bundle cannot be rewritten — it is signed, read-only, and the way it
/// changes is an app update — so repair applies to the writable half only:
/// re-create the data root, re-create the desktop ownership id, and drop the
/// managed capability cache, which describes a `~/.unsloth` environment this app
/// no longer launches. A broken payload is reported instead of repaired, since no
/// amount of writing under `~/.unsloth` can fix it.
pub(crate) fn repair_writable_state(runtime: &BundledRuntime) -> Result<(), String> {
    runtime.health().map_err(|reason| {
        format!(
            "Unsloth's bundled runtime is damaged ({reason}). Reinstall the Unsloth app; \
             a repair cannot rewrite the runtime inside it."
        )
    })?;
    let root = ensure_writable_state_root()?;
    info!(
        "Bundled runtime repair: writable state root is {}",
        root.display()
    );
    crate::desktop_backend_owner::ensure_managed_studio_root_id()
        .map_err(|error| format!("could not prepare the desktop ownership id: {error}"))?;
    // Written by preflight for a `~/.unsloth` install only. It is never consulted
    // for a bundled runtime, so this only clears a stale file left by an earlier
    // managed install; a failure to remove it is not a repair failure.
    let cache = root.join("desktop_capability_cache.json");
    match std::fs::remove_file(&cache) {
        Ok(()) => info!("Bundled runtime repair: removed the stale capability cache"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => warn!("Could not remove {}: {}", cache.display(), error),
    }
    Ok(())
}

/// A whole payload on disk, for the tests in this crate that need one.
///
/// Shared rather than per-module so every test that asks "what happens with a
/// bundled runtime" asks it of the same layout this module documents; a private
/// copy in each module would drift from the real thing one field at a time.
#[cfg(test)]
pub(crate) struct Payload {
    root: PathBuf,
}

#[cfg(test)]
impl Drop for Payload {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[cfg(test)]
impl Payload {
    pub(crate) fn runtime(&self) -> BundledRuntime {
        locate(self.root.join(RESOURCE_DIR_NAME)).expect("a payload directory must be detected")
    }
}

#[cfg(test)]
pub(crate) fn payload(name: &str) -> Payload {
    use std::fs;
    let root = std::env::temp_dir().join(format!(
        "unsloth-bundled-runtime-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&root);
    let runtime = root.join(RESOURCE_DIR_NAME);
    if cfg!(windows) {
        fs::create_dir_all(runtime.join("python").join("Lib")).unwrap();
        fs::write(runtime.join("python").join("Lib").join("os.py"), "# stdlib").unwrap();
        fs::write(runtime.join("python").join("python.exe"), "").unwrap();
    } else {
        let stdlib = runtime.join("python").join("lib").join("python3.12");
        fs::create_dir_all(&stdlib).unwrap();
        fs::write(stdlib.join("os.py"), "# stdlib").unwrap();
        let bin = runtime.join("python").join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("python3"), "").unwrap();
    }
    let site_packages = runtime.join("site-packages");
    fs::create_dir_all(site_packages.join("unsloth_cli")).unwrap();
    fs::write(site_packages.join("unsloth_cli").join("__init__.py"), "").unwrap();
    fs::create_dir_all(site_packages.join("studio").join("backend")).unwrap();
    fs::write(
        site_packages.join("studio").join("backend").join("run.py"),
        "",
    )
    .unwrap();
    fs::write(
        runtime.join(MANIFEST_NAME),
        r#"{"python": {"version": "3.12.8"}}"#,
    )
    .unwrap();
    Payload { root }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn runtime_of(payload: &Payload) -> BundledRuntime {
        payload.runtime()
    }

    #[test]
    fn a_missing_runtime_directory_is_not_a_bundled_runtime() {
        // The whole fallback: no payload means today's ~/.unsloth behaviour.
        assert_eq!(
            locate(std::env::temp_dir().join("unsloth-no-such-runtime-dir")),
            None
        );
    }

    #[test]
    fn a_whole_payload_is_healthy() {
        let payload = payload("healthy");
        assert_eq!(runtime_of(&payload).health(), Ok(()));
    }

    #[test]
    fn a_half_extracted_payload_is_detected_but_unhealthy() {
        // Detected matters as much as unhealthy: an undetected payload sends the
        // app to install.sh, which builds a second Python environment.
        let payload = payload("half-extracted");
        let runtime = runtime_of(&payload);

        for missing in [
            runtime
                .site_packages()
                .join("unsloth_cli")
                .join("__init__.py"),
            runtime
                .site_packages()
                .join("studio")
                .join("backend")
                .join("run.py"),
            runtime.manifest_path(),
            runtime.interpreter(),
        ] {
            let saved = fs::read(&missing).unwrap();
            fs::remove_file(&missing).unwrap();
            assert!(
                locate(runtime.root().to_path_buf()).is_some(),
                "{} is still a bundled runtime",
                runtime.root().display()
            );
            assert!(
                runtime.health().is_err(),
                "removing {} must be reported",
                missing.display()
            );
            fs::write(&missing, saved).unwrap();
        }
        assert_eq!(runtime.health(), Ok(()));
    }

    #[test]
    fn a_payload_without_a_standard_library_cannot_start() {
        // Under -I there is no PYTHONHOME fallback, so the landmark search is the
        // only way the interpreter finds its own stdlib.
        let payload = payload("no-stdlib");
        let runtime = runtime_of(&payload);
        fs::remove_dir_all(
            runtime
                .python_home()
                .join(if cfg!(windows) { "Lib" } else { "lib" }),
        )
        .unwrap();
        assert!(runtime
            .health()
            .unwrap_err()
            .contains("standard library is missing"));
    }

    #[test]
    fn a_truncated_manifest_is_reported() {
        let payload = payload("truncated-manifest");
        let runtime = runtime_of(&payload);
        fs::write(runtime.manifest_path(), r#"{"python": {"vers"#).unwrap();
        assert!(runtime
            .health()
            .unwrap_err()
            .contains("could not be parsed"));
        // A JSON document that is not an object is not this manifest either.
        fs::write(runtime.manifest_path(), "[]").unwrap();
        assert!(runtime.health().unwrap_err().contains("not a JSON object"));
    }

    #[test]
    fn the_invocation_runs_the_bundled_interpreter_isolated_and_bytecode_free() {
        let payload = payload("invocation");
        let runtime = runtime_of(&payload);

        let args = runtime.cli_args(&["studio", "--api-only"]);
        let args: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            vec![
                "-X".to_string(),
                "utf8".to_string(),
                // -I, so no user site-packages and no working directory on sys.path.
                "-I".to_string(),
                // -B, because -I implies -E and PYTHONDONTWRITEBYTECODE would be
                // discarded; a __pycache__ write into a signed bundle either fails
                // or invalidates resource validation.
                "-B".to_string(),
                "-c".to_string(),
                BUNDLED_CLI_ENTRYPOINT.to_string(),
                "studio".to_string(),
                "--api-only".to_string(),
            ]
        );
        // -P is deliberately absent: it is 3.11+ and the lock floor is 3.10.
        assert!(!args.contains(&"-P".to_string()));
        // And no -m: the bundled distributions live outside the interpreter's own
        // prefix, so the bootstrap has to put them on sys.path itself.
        assert!(!args.contains(&"-m".to_string()));
        assert!(BUNDLED_CLI_ENTRYPOINT.contains(SITE_PACKAGES_ENV));
    }

    #[test]
    fn the_child_environment_points_at_the_bundle() {
        let payload = payload("child-env");
        let runtime = runtime_of(&payload);
        fs::create_dir_all(runtime.llama_cpp_dir()).unwrap();
        fs::create_dir_all(runtime.whisper_cpp_dir()).unwrap();
        fs::create_dir_all(runtime.node_bin_dir()).unwrap();

        let env = runtime.child_env_from(&|name| match name {
            "PATH" => Some(OsString::from(if cfg!(windows) {
                r"C:\bin"
            } else {
                "/usr/bin"
            })),
            _ => None,
        });
        let value = |name: &str| {
            env.iter()
                .find(|(key, _)| key == OsStr::new(name))
                .map(|(_, value)| value.to_string_lossy().into_owned())
        };

        assert_eq!(
            value(SITE_PACKAGES_ENV).as_deref(),
            Some(runtime.site_packages().to_string_lossy().as_ref())
        );
        assert_eq!(
            value("PYTHONHOME").as_deref(),
            Some(runtime.python_home().to_string_lossy().as_ref())
        );
        assert_eq!(
            value("PYTHONPATH").as_deref(),
            Some(runtime.site_packages().to_string_lossy().as_ref())
        );
        assert_eq!(value("PYTHONDONTWRITEBYTECODE").as_deref(), Some("1"));
        assert_eq!(
            value("UNSLOTH_LLAMA_CPP_PATH").as_deref(),
            Some(runtime.llama_cpp_dir().to_string_lossy().as_ref())
        );
        assert_eq!(
            value("UNSLOTH_WHISPER_CPP_PATH").as_deref(),
            Some(runtime.whisper_cpp_dir().to_string_lossy().as_ref())
        );
        // The bundled Node has to win shutil.which("node"), so it goes first.
        let path = value("PATH").unwrap();
        assert!(path.starts_with(runtime.node_bin_dir().to_string_lossy().as_ref()));
        assert!(path.ends_with(if cfg!(windows) { r"C:\bin" } else { "/usr/bin" }));
        // Never the data root: UNSLOTH_STUDIO_HOME is read as the place databases
        // live, and the bundle is read-only.
        assert!(!env
            .iter()
            .any(|(key, _)| key == OsStr::new("UNSLOTH_STUDIO_HOME")
                || key == OsStr::new("STUDIO_HOME")));
    }

    #[test]
    fn a_user_llama_cpp_directory_still_wins() {
        // Both overrides are pre-existing user settings; process.rs keeps them
        // where it scrubs the studio root, and so must this.
        let payload = payload("user-override");
        let runtime = runtime_of(&payload);
        fs::create_dir_all(runtime.llama_cpp_dir()).unwrap();
        fs::create_dir_all(runtime.whisper_cpp_dir()).unwrap();

        let env = runtime.child_env_from(&|name| match name {
            "UNSLOTH_LLAMA_CPP_PATH" => Some(OsString::from("/opt/mine/llama.cpp")),
            // A blank value is not a setting.
            "UNSLOTH_WHISPER_CPP_PATH" => Some(OsString::new()),
            _ => None,
        });

        assert!(!env
            .iter()
            .any(|(key, _)| key == OsStr::new("UNSLOTH_LLAMA_CPP_PATH")));
        assert!(env
            .iter()
            .any(|(key, value)| key == OsStr::new("UNSLOTH_WHISPER_CPP_PATH")
                && value == &runtime.whisper_cpp_dir().into_os_string()));
    }

    #[test]
    fn absent_native_runtimes_are_not_pointed_at() {
        // A payload that ships no whisper.cpp must not send the backend at a
        // directory that does not exist; the managed and PATH lookups behind it
        // still answer.
        let payload = payload("absent-natives");
        let runtime = runtime_of(&payload);
        let env = runtime.child_env_from(&|_| None);
        for name in ["UNSLOTH_LLAMA_CPP_PATH", "UNSLOTH_WHISPER_CPP_PATH", "PATH"] {
            assert!(
                !env.iter().any(|(key, _)| key == OsStr::new(name)),
                "{name} must be left alone when the payload has no copy"
            );
        }
    }

    #[test]
    fn no_bundled_runtime_is_resolved_in_this_test_binary() {
        // The unit tests never call init(), so every other module's tests see
        // exactly the pre-bundle behaviour.
        assert_eq!(bundled_runtime(), None);
    }
}
