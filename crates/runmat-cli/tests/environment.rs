use runmat_config::{ConfigLoader, JitOptLevel, RunMatConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn get_binary_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("runmat");
    path
}

fn run_runmat_with_env(args: &[&str], env_vars: HashMap<&str, &str>) -> std::process::Output {
    let mut cmd = Command::new(get_binary_path());
    cmd.args(args);
    cmd.env("NO_GUI", "1");

    for (key, value) in env_vars {
        cmd.env(key, value);
    }

    cmd.output().expect("Failed to execute runmat binary")
}

#[test]
fn runmat_config_env_loads_toml_runtime_settings() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
callstack_limit = 42
jit = { enabled = true, threshold = 25, optimization_level = "size" }
gc = { collect_stats = true }
"#,
    )
    .unwrap();

    let mut env = HashMap::new();
    env.insert("RUNMAT_CONFIG", config_path.to_str().unwrap());

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Threshold: 25"));
    assert!(stdout.contains("JIT Optimization: Size"));
    assert!(stdout.contains("GC Statistics: true"));
}

#[test]
fn runmat_config_env_loads_json_runtime_settings() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.json");
    std::fs::write(
        &config_path,
        r#"{
  "runtime": {
    "jit": { "enabled": false, "threshold": 11, "optimization_level": "none" },
    "gc": { "collect_stats": false }
  }
}"#,
    )
    .unwrap();

    let mut env = HashMap::new();
    env.insert("RUNMAT_CONFIG", config_path.to_str().unwrap());

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Compiler: disabled"));
    assert!(stdout.contains("JIT Threshold: 11"));
    assert!(stdout.contains("JIT Optimization: None"));
}

#[test]
fn cli_args_override_config_file() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
jit = { enabled = true, threshold = 5, optimization_level = "speed" }
"#,
    )
    .unwrap();

    let mut env = HashMap::new();
    env.insert("RUNMAT_CONFIG", config_path.to_str().unwrap());

    let output = run_runmat_with_env(&["--jit-threshold", "20", "info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Threshold: 20"));
}

#[test]
fn legacy_runtime_env_knobs_do_not_override_file_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
jit = { enabled = true, threshold = 25, optimization_level = "size" }
"#,
    )
    .unwrap();

    let mut env = HashMap::new();
    env.insert("RUNMAT_CONFIG", config_path.to_str().unwrap());
    env.insert("RUNMAT_JIT_THRESHOLD", "999");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Threshold: 25"));
}

#[test]
fn runmat_config_save_and_load_use_canonical_extensions() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");

    let mut config = RunMatConfig::default();
    config.jit.threshold = 33;
    config.jit.optimization_level = JitOptLevel::Aggressive;

    ConfigLoader::save_to_file(&config, &config_path).unwrap();
    let loaded = ConfigLoader::load_from_file(&config_path).unwrap();
    assert_eq!(loaded.jit.threshold, 33);
    assert_eq!(loaded.jit.optimization_level, JitOptLevel::Aggressive);
}

#[test]
fn info_output_shows_current_supported_env_keys() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(&config_path, "[runtime]\n").unwrap();

    let mut env = HashMap::new();
    env.insert("RUNMAT_CONFIG", config_path.to_str().unwrap());

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Environment:"));
    assert!(stdout.contains("RUNMAT_CONFIG"));
    assert!(stdout.contains("RUNMAT_SERVER_URL"));
    assert!(stdout.contains("RUNMAT_PROJECT_ID"));
}
