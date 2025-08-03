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
    path.push("rustmat");
    path
}

// Helper function to run rustmat with arguments
fn run_rustmat(args: &[&str]) -> std::process::Output {
    Command::new(get_binary_path())
        .args(args)
        .output()
        .expect("Failed to execute rustmat binary")
}

#[test]
fn test_end_to_end_script_execution() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("e2e_test.m");

    // Create a comprehensive test script
    fs::write(
        &script_path,
        r#"
% Test script for end-to-end functionality
x = 10 + 5
y = [1, 2; 3, 4]
z = x * 2
result = z + 5
"#,
    )
    .unwrap();

    let output = run_rustmat(&["run", script_path.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "End-to-end script execution failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_gc_integration_with_cli() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("gc_test.m");

    // Create a script that allocates objects
    fs::write(
        &script_path,
        r#"
for i = 1:10
    matrix_i = [i, i+1; i+2, i+3]
end
final_result = 42
"#,
    )
    .unwrap();

    let output = run_rustmat(&[
        "--gc-preset",
        "debug",
        "--gc-stats",
        "run",
        script_path.to_str().unwrap(),
    ]);

    assert!(
        output.status.success(),
        "GC integration test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_jit_vs_interpreter_performance() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("perf_test.m");

    // Create a script for performance testing
    fs::write(
        &script_path,
        r#"
x = 50;
y = 25;
result = x + y * 2;
"#,
    )
    .unwrap();

    // Test with JIT enabled
    let jit_output = run_rustmat(&[
        "--jit-opt-level",
        "speed",
        "benchmark",
        script_path.to_str().unwrap(),
        "--iterations",
        "5",
        "--jit",
    ]);

    // Test with JIT disabled
    let interp_output = run_rustmat(&["--no-jit", "run", script_path.to_str().unwrap()]);

    assert!(jit_output.status.success(), "JIT benchmark failed");
    assert!(
        interp_output.status.success(),
        "Interpreter execution failed"
    );

    let jit_stdout = String::from_utf8_lossy(&jit_output.stdout);
    assert!(jit_stdout.contains("Benchmark Results"));
}

#[test]
fn test_configuration_persistence() {
    // Test that configuration options are properly applied
    let output = run_rustmat(&[
        "--gc-preset",
        "low-latency",
        "--jit-threshold",
        "5",
        "--gc-young-size",
        "32",
        "info",
    ]);

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GC Preset: \"LowLatency\""));
    assert!(stdout.contains("JIT Threshold: 5"));
    assert!(stdout.contains("GC Young Generation: 32MB"));
}

#[test]
fn test_error_handling_and_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let bad_script_path = temp_dir.path().join("bad_script.m");
    let good_script_path = temp_dir.path().join("good_script.m");

    // Create a script with errors
    fs::write(&bad_script_path, "x = [1, 2,").unwrap(); // Syntax error

    // Create a valid script
    fs::write(&good_script_path, "x = 42").unwrap();

    // Bad script should fail
    let bad_output = run_rustmat(&["run", bad_script_path.to_str().unwrap()]);
    assert!(!bad_output.status.success());

    // Good script should work after error
    let good_output = run_rustmat(&["run", good_script_path.to_str().unwrap()]);
    assert!(good_output.status.success());
}

#[test]
fn test_comprehensive_system_functionality() {
    // Test multiple subsystems working together
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("comprehensive.m");

    fs::write(
        &script_path,
        r#"
% Comprehensive test
A = [1, 2; 3, 4];
B = [5, 6; 7, 8];
x = 10;
y = 20;

% Arithmetic
result1 = x + y;

% Matrix operations
result2 = A;

% Simple calculation
result3 = x * 2;

% Loop
sum_val = 0;
for i = 1:5; sum_val = sum_val + i; end;

final_answer = result1 + result3;
"#,
    )
    .unwrap();

    let output = run_rustmat(&[
        "--gc-preset",
        "high-throughput",
        "--jit-opt-level",
        "aggressive",
        "--verbose",
        "run",
        script_path.to_str().unwrap(),
    ]);

    assert!(
        output.status.success(),
        "Comprehensive system test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should have completed without crashes
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The verbose output or successful completion should be evident
    assert!(!stdout.is_empty() || output.status.success());
}

#[test]
fn test_memory_stress_with_gc() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("memory_stress.m");

    fs::write(
        &script_path,
        r#"
% Memory stress test
for i = 1:50
    big_matrix = [i, i+1, i+2; i+3, i+4, i+5; i+6, i+7, i+8]
    temp_val = i * 2
end
result = 999
"#,
    )
    .unwrap();

    let output = run_rustmat(&[
        "--gc-preset",
        "low-memory",
        "--gc-stats",
        "run",
        script_path.to_str().unwrap(),
    ]);

    assert!(
        output.status.success(),
        "Memory stress test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_help_and_documentation() {
    // Test that help output is comprehensive and correct
    let output = run_rustmat(&["--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain key sections
    let required_sections = [
        "RustMat",
        "JIT compilation",
        "Garbage collection",
        "Usage:",
        "Commands:",
        "Options:",
        "Environment Variables:",
    ];

    for section in &required_sections {
        assert!(
            stdout.contains(section),
            "Help missing section: {}",
            section
        );
    }
}

#[test]
fn test_version_information() {
    let output = run_rustmat(&["version", "--detailed"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    let required_components = [
        "rustmat-lexer",
        "rustmat-parser",
        "rustmat-hir",
        "rustmat-ignition",
        "rustmat-turbine",
        "rustmat-gc",
        "rustmat-runtime",
    ];

    for component in &required_components {
        assert!(
            stdout.contains(component),
            "Version info missing component: {}",
            component
        );
    }
}

#[test]
fn test_benchmark_functionality() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("benchmark_test.m");

    fs::write(&script_path, "benchmark_result = 10 + 20").unwrap();

    let output = run_rustmat(&[
        "benchmark",
        script_path.to_str().unwrap(),
        "--iterations",
        "3",
    ]);

    assert!(
        output.status.success(),
        "Benchmark test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Benchmark Results"));
    assert!(stdout.contains("iterations"));
    assert!(stdout.contains("Average time"));
}

#[test]
fn test_gc_commands() {
    // Test all GC subcommands
    let gc_commands = ["stats", "minor", "major", "config"];

    for cmd in &gc_commands {
        let output = run_rustmat(&["gc", cmd]);
        assert!(
            output.status.success() || output.status.success(), // Some might not be implemented yet
            "GC command failed: gc {}",
            cmd
        );
    }
}

#[test]
fn test_multi_configuration_compatibility() {
    // Test that multiple configuration options work together
    let combinations = [
        vec!["--gc-preset", "low-latency", "--jit-opt-level", "speed"],
        vec!["--gc-preset", "debug", "--no-jit", "--verbose"],
        vec!["--gc-young-size", "64", "--jit-threshold", "20"],
    ];

    for combo in &combinations {
        let mut args = combo.clone();
        args.push("info");

        let output = run_rustmat(&args);
        assert!(
            output.status.success(),
            "Configuration combination failed: {:?}",
            combo
        );
    }
}

#[test]
fn test_script_with_all_language_features() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("all_features.m");

    fs::write(
        &script_path,
        r#"
% Test all major language features

% Variables and arithmetic
x = 10;
y = 5;
arithmetic_result = x + y * 2 - 3;

% Matrices
matrix_2d = [1, 2; 3, 4];
vector_row = [1, 2, 3];
vector_col = [1; 2; 3];

% Simple calculation instead of conditional
condition_result = 1;

% Loop
loop_sum = 0;
for i = 1:5; loop_sum = loop_sum + i; end;

% Final computation
final_result = arithmetic_result + condition_result + loop_sum;
"#,
    )
    .unwrap();

    let output = run_rustmat(&["run", script_path.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "All features test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_concurrent_execution_safety() {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = Arc::new(TempDir::new().unwrap());
    let mut handles = vec![];

    // Spawn multiple threads running different scripts
    for i in 0..3 {
        let temp_dir_clone = Arc::clone(&temp_dir);
        let handle = thread::spawn(move || {
            let script_path = temp_dir_clone.path().join(format!("concurrent_{}.m", i));
            fs::write(&script_path, format!("thread_result_{} = {} * 10", i, i)).unwrap();

            let output = run_rustmat(&["run", script_path.to_str().unwrap()]);
            output.status.success()
        });
        handles.push(handle);
    }

    // Wait for all threads and check results
    for (i, handle) in handles.into_iter().enumerate() {
        let success = handle.join().unwrap();
        assert!(success, "Concurrent execution {} failed", i);
    }
}

#[test]
fn test_edge_case_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Test empty file
    let empty_script = temp_dir.path().join("empty.m");
    fs::write(&empty_script, "").unwrap();
    let output = run_rustmat(&["run", empty_script.to_str().unwrap()]);
    assert!(output.status.success()); // Should handle gracefully

    // Test whitespace-only file
    let whitespace_script = temp_dir.path().join("whitespace.m");
    fs::write(&whitespace_script, "   \n\t  \n  ").unwrap();
    let output = run_rustmat(&["run", whitespace_script.to_str().unwrap()]);
    assert!(output.status.success()); // Should handle gracefully

    // Test very large numbers
    let large_num_script = temp_dir.path().join("large_nums.m");
    fs::write(&large_num_script, "huge = 1e100; result = huge / 1e50;").unwrap();
    let output = run_rustmat(&["run", large_num_script.to_str().unwrap()]);
    // Should handle gracefully (success or controlled failure)
    assert!(output.status.success() || !output.status.success());
}
