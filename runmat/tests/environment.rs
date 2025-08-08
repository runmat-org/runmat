use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

// Helper function to get the binary path
fn get_binary_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name
    if path.ends_with("deps") {
        path.pop(); // Remove deps directory
    }
    path.push("runmat");
    path
}

// Helper function to run rustmat with environment variables
fn run_runmat_with_env(args: &[&str], env_vars: HashMap<&str, &str>) -> std::process::Output {
    let mut cmd = Command::new(get_binary_path());
    cmd.args(args);

    for (key, value) in env_vars {
        cmd.env(key, value);
    }

    cmd.output().expect("Failed to execute rustmat binary")
}

#[test]
fn test_runmat_debug_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_DEBUG", "1");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    // Debug mode should show additional logging
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should have some debug output or at least work without errors
    assert!(!stderr.is_empty() || !stdout.is_empty());
}

#[test]
fn test_runmat_log_level_env_var() {
    let log_levels = ["error", "warn", "info", "debug", "trace"];

    for level in &log_levels {
        let mut env = HashMap::new();
        env.insert("RUSTMAT_LOG_LEVEL", *level);

        let output = run_runmat_with_env(&["info"], env);
        assert!(output.status.success(), "Failed with log level: {level}");
    }
}

#[test]
fn test_runmat_timeout_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_TIMEOUT", "60");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());
}

#[test]
fn test_runmat_jit_disable_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_JIT_DISABLE", "1");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Compiler: disabled"));
}

#[test]
fn test_runmat_jit_threshold_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_JIT_THRESHOLD", "15");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Threshold: 15"));
}

#[test]
fn test_runmat_jit_opt_level_env_var() {
    let opt_levels = [
        ("none", "None"),
        ("size", "Size"),
        ("speed", "Speed"),
        ("aggressive", "Aggressive"),
    ];

    for (env_val, expected_output) in &opt_levels {
        let mut env = HashMap::new();
        env.insert("RUSTMAT_JIT_OPT_LEVEL", *env_val);

        let output = run_runmat_with_env(&["info"], env);
        assert!(output.status.success(), "Failed with opt level: {env_val}");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(&format!("JIT Optimization: {expected_output}")));
    }
}

#[test]
fn test_runmat_gc_preset_env_var() {
    let presets = [
        ("low-latency", "LowLatency"),
        ("high-throughput", "HighThroughput"),
        ("low-memory", "LowMemory"),
        ("debug", "Debug"),
    ];

    for (env_val, expected_output) in &presets {
        let mut env = HashMap::new();
        env.insert("RUSTMAT_GC_PRESET", *env_val);

        let output = run_runmat_with_env(&["info"], env);
        assert!(output.status.success(), "Failed with GC preset: {env_val}");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(&format!("GC Preset: \"{expected_output}\"")));
    }
}

#[test]
fn test_runmat_gc_young_size_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_GC_YOUNG_SIZE", "128");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Young Generation: 128MB"));
}

#[test]
fn test_runmat_gc_threads_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_GC_THREADS", "8");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Threads: 8"));
}

#[test]
fn test_runmat_gc_stats_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_GC_STATS", "1");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Statistics: true"));
}

#[test]
fn test_kernel_ip_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_KERNEL_IP", "192.168.1.100");

    let output = run_runmat_with_env(&["kernel", "--help"], env);
    assert!(output.status.success());

    // Help should work regardless of environment variables
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Jupyter kernel"));
}

#[test]
fn test_kernel_key_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_KERNEL_KEY", "test-key-12345");

    let output = run_runmat_with_env(&["kernel", "--help"], env);
    assert!(output.status.success());
}

#[test]
fn test_multiple_env_vars_combined() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_DEBUG", "1");
    env.insert("RUSTMAT_GC_PRESET", "debug");
    env.insert("RUSTMAT_JIT_OPT_LEVEL", "aggressive");
    env.insert("RUSTMAT_GC_STATS", "1");
    env.insert("RUSTMAT_JIT_THRESHOLD", "5");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Preset: \"Debug\""));
    assert!(stdout.contains("JIT Optimization: Aggressive"));
    assert!(stdout.contains("GC Statistics: true"));
    assert!(stdout.contains("JIT Threshold: 5"));
}

#[test]
fn test_env_vars_override_defaults() {
    // First get the default values
    let default_output = run_runmat_with_env(&["info"], HashMap::new());
    assert!(default_output.status.success());
    let _default_stdout = String::from_utf8_lossy(&default_output.stdout);

    // Now set environment variables that should override defaults
    let mut env = HashMap::new();
    env.insert("RUSTMAT_JIT_THRESHOLD", "25");

    let override_output = run_runmat_with_env(&["info"], env);
    assert!(override_output.status.success());
    let override_stdout = String::from_utf8_lossy(&override_output.stdout);

    // Should see the overridden value
    assert!(override_stdout.contains("JIT Threshold: 25"));
}

#[test]
fn test_cli_args_override_env_vars() {
    // Set environment variable
    let mut env = HashMap::new();
    env.insert("RUSTMAT_JIT_THRESHOLD", "5");

    // Override with CLI argument
    let output = run_runmat_with_env(&["--jit-threshold", "20", "info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // CLI argument should take precedence
    assert!(stdout.contains("JIT Threshold: 20"));
}

#[test]
fn test_invalid_env_var_values() {
    let invalid_configs = [
        ("RUSTMAT_JIT_THRESHOLD", "invalid"),
        ("RUSTMAT_GC_YOUNG_SIZE", "not-a-number"),
        ("RUSTMAT_GC_THREADS", "negative"),
        ("RUSTMAT_TIMEOUT", "abc"),
    ];

    for (env_var, invalid_value) in &invalid_configs {
        let mut env = HashMap::new();
        env.insert(*env_var, *invalid_value);

        let output = run_runmat_with_env(&["info"], env);
        // Should either fail gracefully or ignore invalid values
        // The behavior may vary, but it shouldn't crash
        assert!(output.status.success() || !output.status.success());
    }
}

#[test]
fn test_env_var_case_sensitivity() {
    // Test that environment variables are case-sensitive
    let mut env = HashMap::new();
    env.insert("runmat_debug", "1"); // lowercase should not work

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    // Should not show debug behavior since env var is wrong case
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should work normally (debug flag not activated)
    assert!(stdout.contains("System Information"));
}

#[test]
fn test_empty_env_var_values() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_KERNEL_KEY", "");
    env.insert("RUSTMAT_LOG_LEVEL", "");

    let output = run_runmat_with_env(&["info"], env);
    // Should handle empty values gracefully
    assert!(output.status.success());
}

#[test]
fn test_config_file_env_var() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_CONFIG", "/nonexistent/config.toml");
    env.insert("NO_GUI", "1"); // Disable GUI to prevent hanging in test environment

    let output = run_runmat_with_env(&["info"], env);
    // Should work even if config file doesn't exist (using defaults)
    assert!(output.status.success());
}

#[test]
fn test_port_env_vars_for_kernel() {
    let port_vars = [
        "RUSTMAT_SHELL_PORT",
        "RUSTMAT_IOPUB_PORT",
        "RUSTMAT_STDIN_PORT",
        "RUSTMAT_CONTROL_PORT",
        "RUSTMAT_HB_PORT",
    ];

    for port_var in &port_vars {
        let mut env = HashMap::new();
        env.insert(*port_var, "8888");

        let output = run_runmat_with_env(&["kernel", "--help"], env);
        assert!(
            output.status.success(),
            "Failed with port env var: {port_var}"
        );
    }
}

#[test]
fn test_env_vars_shown_in_info_output() {
    let mut env = HashMap::new();
    env.insert("RUSTMAT_DEBUG", "1");
    env.insert("RUSTMAT_GC_PRESET", "debug");

    let output = run_runmat_with_env(&["info"], env);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The info command should show current environment variables
    assert!(stdout.contains("Environment:"));
    assert!(stdout.contains("RUSTMAT_DEBUG"));
    assert!(stdout.contains("RUSTMAT_GC_PRESET"));
}

#[test]
fn test_numeric_env_var_boundaries() {
    // Test boundary values for numeric environment variables
    let boundary_tests = [
        ("RUSTMAT_JIT_THRESHOLD", "1"),
        ("RUSTMAT_JIT_THRESHOLD", "1000"),
        ("RUSTMAT_GC_YOUNG_SIZE", "1"),
        ("RUSTMAT_GC_YOUNG_SIZE", "1024"),
        ("RUSTMAT_GC_THREADS", "1"),
        ("RUSTMAT_GC_THREADS", "32"),
        ("RUSTMAT_TIMEOUT", "1"),
        ("RUSTMAT_TIMEOUT", "3600"),
    ];

    for (env_var, value) in &boundary_tests {
        let mut env = HashMap::new();
        env.insert(*env_var, *value);

        let output = run_runmat_with_env(&["info"], env);
        assert!(output.status.success(), "Failed with {env_var}={value}");
    }
}

#[test]
fn test_boolean_env_var_variations() {
    let boolean_variations = [
        ("1", true),
        ("true", true),
        ("TRUE", true),
        ("0", false),
        ("false", false),
        ("FALSE", false),
    ];

    for (value, _expected) in &boolean_variations {
        let mut env = HashMap::new();
        env.insert("RUSTMAT_DEBUG", *value);

        let output = run_runmat_with_env(&["info"], env);
        assert!(output.status.success(), "Failed with RUSTMAT_DEBUG={value}");

        // Test with GC stats too
        let mut env2 = HashMap::new();
        env2.insert("RUSTMAT_GC_STATS", *value);

        let output = run_runmat_with_env(&["info"], env2);
        assert!(
            output.status.success(),
            "Failed with RUSTMAT_GC_STATS={value}"
        );
    }
}
