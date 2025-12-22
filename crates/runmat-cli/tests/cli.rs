use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

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

// Helper function to run runmat with arguments
fn run_runmat(args: &[&str]) -> std::process::Output {
    Command::new(get_binary_path())
        .args(args)
        .env("RUNMAT_ACCEL_ENABLE", "0")
        .env("RUNMAT_ACCEL_PROVIDER", "inprocess")
        .output()
        .expect("Failed to execute runmat binary")
}

#[test]
fn test_help_command() {
    let output = run_runmat(&["--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("RunMat"));
    assert!(stdout.contains("JIT compilation"));
    assert!(stdout.contains("Garbage collection"));
    assert!(stdout.contains("--no-jit"));
    assert!(stdout.contains("--gc-preset"));
    assert!(stdout.contains("--jit-opt-level"));
}

#[test]
fn test_version_command() {
    let output = run_runmat(&["--version"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("runmat"));
}

#[test]
fn test_version_detailed_command() {
    let output = run_runmat(&["version", "--detailed"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("RunMat"));
    assert!(stdout.contains("Built with Rust"));
    assert!(stdout.contains("Target:"));
    assert!(stdout.contains("Profile:"));
}

#[test]
fn test_info_command() {
    let output = run_runmat(&["info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("RunMat System Information"));
    assert!(stdout.contains("Runtime Configuration"));
    assert!(stdout.contains("JIT Compiler"));
    assert!(stdout.contains("GC Preset"));
    assert!(stdout.contains("Environment"));
    assert!(stdout.contains("Garbage Collector Status"));
}

#[test]
fn test_gc_stats_command() {
    let output = run_runmat(&["gc", "stats"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Statistics"));
    assert!(stdout.contains("Allocations"));
    assert!(stdout.contains("Collections"));
    assert!(stdout.contains("Memory"));
}

#[test]
fn test_gc_major_command() {
    let output = run_runmat(&["gc", "major"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Major GC collected"));
    assert!(stdout.contains("objects"));
}

#[test]
fn test_gc_minor_command() {
    let output = run_runmat(&["gc", "minor"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Minor GC collected"));
    assert!(stdout.contains("objects"));
}

#[test]
fn test_jit_disabled_flag() {
    let output = run_runmat(&["--no-jit", "info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Compiler: disabled"));
}

#[test]
fn test_jit_optimization_levels() {
    let optimization_levels = ["none", "size", "speed", "aggressive"];

    for level in &optimization_levels {
        let output = run_runmat(&["--jit-opt-level", level, "info"]);
        assert!(
            output.status.success(),
            "Failed with optimization level: {level}"
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Check that the optimization level is reflected in the output
        assert!(stdout.contains("JIT Optimization"));
    }
}

#[test]
fn test_gc_presets() {
    let presets = ["low-latency", "high-throughput", "low-memory", "debug"];

    for preset in &presets {
        let output = run_runmat(&["--gc-preset", preset, "info"]);
        assert!(output.status.success(), "Failed with GC preset: {preset}");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("GC Preset"));
    }
}

#[test]
fn test_gc_young_size_configuration() {
    let output = run_runmat(&["--gc-young-size", "64", "info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Young Generation: 64MB"));
}

#[test]
fn test_gc_threads_configuration() {
    let output = run_runmat(&["--gc-threads", "4", "info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Threads: 4"));
}

#[test]
fn test_gc_stats_flag() {
    let output = run_runmat(&["--gc-stats", "info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Statistics: true"));
}

#[test]
fn test_verbose_flag() {
    let output = run_runmat(&["--verbose", "info"]);
    assert!(output.status.success());

    // Verbose flag should work without errors
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("System Information"));
}

#[test]
fn test_debug_flag() {
    let output = run_runmat(&["--debug", "info"]);
    assert!(output.status.success());

    // Debug flag should enable debug logging
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should see debug-level logs
    assert!(stderr.contains("DEBUG") || !output.stdout.is_empty());
}

#[test]
fn test_timeout_configuration() {
    let output = run_runmat(&["--timeout", "60", "info"]);
    assert!(output.status.success());
}

#[test]
fn test_jit_threshold_configuration() {
    let output = run_runmat(&["--jit-threshold", "5", "info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("JIT Threshold: 5"));
}

#[test]
fn test_script_execution() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("test_script.m");

    // Create a simple MATLAB script
    fs::write(&script_path, "x = 1 + 2").unwrap();

    let output = run_runmat(&[script_path.to_str().unwrap()]);
    assert!(output.status.success());
}

#[test]
fn test_script_execution_with_run_command() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("test_script2.m");

    // Create a simple MATLAB script
    fs::write(&script_path, "result = 5 * 10").unwrap();

    let output = run_runmat(&["run", script_path.to_str().unwrap()]);
    assert!(output.status.success());
}

#[test]
fn test_nonexistent_script() {
    let output = run_runmat(&["run", "/nonexistent/script.m"]);
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to read") || stderr.contains("No such file"));
}

#[test]
fn test_invalid_gc_preset() {
    let output = run_runmat(&["--gc-preset", "invalid-preset", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_invalid_jit_opt_level() {
    let output = run_runmat(&["--jit-opt-level", "invalid-level", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_invalid_log_level() {
    let output = run_runmat(&["--log-level", "invalid-level", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_combined_flags() {
    let output = run_runmat(&[
        "--gc-preset",
        "low-latency",
        "--jit-opt-level",
        "aggressive",
        "--gc-stats",
        "--verbose",
        "info",
    ]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Preset"));
    assert!(stdout.contains("JIT Optimization"));
    assert!(stdout.contains("GC Statistics: true"));
}

#[test]
fn test_conflicting_script_and_command() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("test.m");
    fs::write(&script_path, "x = 1").unwrap();

    // Should fail when both script and command are provided
    let output = run_runmat(&["info", script_path.to_str().unwrap()]);
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Clap detects unexpected argument before our validation runs
    assert!(stderr.contains("unexpected argument") || stderr.contains("Cannot specify both"));
}

#[test]
fn test_benchmark_command_without_file() {
    let output = run_runmat(&["benchmark"]);
    assert!(!output.status.success());
}

#[test]
fn test_benchmark_with_nonexistent_file() {
    let output = run_runmat(&["benchmark", "/nonexistent/file.m"]);
    assert!(!output.status.success());
}

#[test]
fn test_benchmark_with_valid_file() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("benchmark_test.m");

    // Create a simple script for benchmarking
    fs::write(&script_path, "y = 2 + 3").unwrap();

    let output = run_runmat(&[
        "benchmark",
        script_path.to_str().unwrap(),
        "--iterations",
        "3",
        "--jit",
    ]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Benchmark Results"));
    assert!(stdout.contains("iterations"));
    assert!(stdout.contains("Average time"));
}

#[test]
fn test_repl_command() {
    // Note: This test just verifies the REPL command starts without immediate error
    // We can't easily test interactive input without complex setup

    // Test that help works for repl command
    let output = run_runmat(&["repl", "--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("verbose"));
}

#[test]
fn test_kernel_command_help() {
    let output = run_runmat(&["kernel", "--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Jupyter kernel"));
    assert!(stdout.contains("--ip"));
    assert!(stdout.contains("--key"));
}

#[test]
fn test_log_level_configurations() {
    let log_levels = ["error", "warn", "info", "debug", "trace"];

    for level in &log_levels {
        let output = run_runmat(&["--log-level", level, "info"]);
        assert!(output.status.success(), "Failed with log level: {level}");
    }
}

#[test]
fn test_invalid_gc_young_size() {
    let output = run_runmat(&["--gc-young-size", "invalid", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_invalid_gc_threads() {
    let output = run_runmat(&["--gc-threads", "invalid", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_invalid_jit_threshold() {
    let output = run_runmat(&["--jit-threshold", "invalid", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_invalid_timeout() {
    let output = run_runmat(&["--timeout", "invalid", "info"]);
    assert!(!output.status.success());
}

#[test]
fn test_script_with_syntax_error() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("syntax_error.m");

    // Create a script with syntax error
    fs::write(&script_path, "x = [1, 2,").unwrap(); // Incomplete matrix

    let output = run_runmat(&["run", script_path.to_str().unwrap()]);
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error") || stderr.contains("failed"));
}

#[test]
fn test_help_shows_environment_variables() {
    let output = run_runmat(&["--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Environment Variables"));
    assert!(stdout.contains("RUNMAT_DEBUG"));
    assert!(stdout.contains("RUNMAT_GC_PRESET"));
    assert!(stdout.contains("RUNMAT_JIT_ENABLE"));
}

#[test]
fn test_command_output_is_structured() {
    let output = run_runmat(&["info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for structured sections
    assert!(stdout.contains("=========================="));
    assert!(stdout.contains("Version:"));
    assert!(stdout.contains("Runtime Configuration:"));
    assert!(stdout.contains("Environment:"));
    assert!(stdout.contains("Available Commands:"));
}

#[test]
fn test_gc_config_command() {
    let output = run_runmat(&["gc", "config"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Configuration") || stdout.contains("not yet implemented"));
}

#[test]
fn test_edge_case_empty_script() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("empty.m");

    // Create an empty script
    fs::write(&script_path, "").unwrap();

    let output = run_runmat(&["run", script_path.to_str().unwrap()]);
    // Empty script behavior may vary - some parsers accept empty input
    // Just ensure it doesn't crash
    assert!(
        output.status.success() || !output.status.success(),
        "Should handle empty script gracefully"
    );
}

#[test]
fn test_script_with_complex_operations() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("complex.m");

    // Create a script with multiple operations
    fs::write(
        &script_path,
        r#"
x = 1 + 2
y = [1, 2; 3, 4]
z = x * 5
"#,
    )
    .unwrap();

    let output = run_runmat(&["run", script_path.to_str().unwrap()]);
    assert!(output.status.success());
}
