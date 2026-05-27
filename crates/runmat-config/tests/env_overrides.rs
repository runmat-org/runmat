mod support;

use runmat_config::runtime::ConfigLoader;
use tempfile::TempDir;

const ENV_VARS: &[&str] = &[
    "RUNMAT_CONFIG",
    "RUNMAT_CALLSTACK_LIMIT",
    "RUNMAT_JIT_THRESHOLD",
];

#[test]
fn runmat_config_env_selects_config_file() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("custom.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
callstack_limit = 123
jit = { enabled = true, threshold = 42, optimization_level = "speed" }
"#,
    )
    .unwrap();

    std::env::set_var("RUNMAT_CONFIG", &config_path);
    std::env::set_var("RUNMAT_CALLSTACK_LIMIT", "999");
    std::env::set_var("RUNMAT_JIT_THRESHOLD", "999");

    let config = ConfigLoader::load().unwrap();
    assert_eq!(config.runtime.callstack_limit, 123);
    assert_eq!(config.jit.threshold, 42);

    support::clear_env(ENV_VARS);
}

#[test]
fn runmat_config_env_missing_file_is_error() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);

    std::env::set_var(
        "RUNMAT_CONFIG",
        "/tmp/definitely-missing-runmat-config.toml",
    );
    let err = ConfigLoader::load().expect_err("missing RUNMAT_CONFIG path should fail");
    let message = err.to_string();
    assert!(message.contains("RUNMAT_CONFIG points to a missing file"));

    support::clear_env(ENV_VARS);
}
