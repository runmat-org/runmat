mod support;

use runmat_config::ConfigLoader;
use tempfile::TempDir;

const ENV_VARS: &[&str] = &["RUNMAT_CONFIG", "RUNMAT_TIMEOUT", "RUNMAT_JIT_THRESHOLD"];

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
timeout = 123
jit = { enabled = true, threshold = 42, optimization_level = "speed" }
"#,
    )
    .unwrap();

    std::env::set_var("RUNMAT_CONFIG", &config_path);
    std::env::set_var("RUNMAT_TIMEOUT", "999");
    std::env::set_var("RUNMAT_JIT_THRESHOLD", "999");

    let config = ConfigLoader::load().unwrap();
    assert_eq!(config.runtime.timeout, 123);
    assert_eq!(config.jit.threshold, 42);

    support::clear_env(ENV_VARS);
}
