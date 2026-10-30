//! Isolated, signed-out local execution: no credential, keychain, or listening server.
use std::{fs, path::Path, process::Command};

pub fn configure(command: &mut Command, root: &Path) {
    for (key, _) in std::env::vars_os().filter(|(key, _)| {
        let key = key.to_string_lossy();
        key.starts_with("RUNMAT_") || key.starts_with("OTEL_")
    }) {
        command.env_remove(key);
    }
    let config = root.join("isolated-user-config");
    let auth = root.join("signed-out-cli");
    fs::create_dir_all(&config).unwrap();
    fs::create_dir_all(&auth).unwrap();
    // Port zero cannot serve a collector or billing API. A regression that attempts
    // local authorization fails instead of being satisfied by a fixture server.
    command
        .env("XDG_CONFIG_HOME", config)
        .env(
            "RUNMAT_USER_SETTINGS",
            root.join("isolated-user-settings.json"),
        )
        .env("RUNMAT_CLI_CONFIG_DIR", auth)
        .env("RUNMAT_SERVER_URL", "http://127.0.0.1:0")
        .env("RUNMAT_TELEMETRY_ENABLED", "false")
        .env("RUNMAT_OTEL_ENABLED", "false")
        .env("RUNMAT_PACKAGE_CACHE_DIR", root.join("package-cache"))
        .env("RUNMAT_EXECUTION_STATE_DIR", root.join("execution-state"))
        .env("RUNMAT_FEA_ARTIFACT_ROOT", root.join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            root.join("study-artifacts"),
        );
}
