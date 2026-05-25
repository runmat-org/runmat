// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use runmat_core::{RunError, RunMatSession};
use runmat_gc::{gc_test_context, GcConfig};

#[test]
fn test_repl_engine_creation() {
    gc_test_context(|| {
        let engine = RunMatSession::new();
        assert!(engine.is_ok());
    });
}

#[test]
fn test_repl_engine_with_jit_enabled() {
    gc_test_context(|| {
        let engine = RunMatSession::with_options(true, false);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        let stats = engine.stats();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.jit_compiled, 0);
        assert_eq!(stats.interpreter_fallback, 0);
    });
}

#[test]
fn test_repl_engine_with_jit_disabled() {
    gc_test_context(|| {
        let engine = RunMatSession::with_options(false, false);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        let stats = engine.stats();
        assert_eq!(stats.total_executions, 0);
    });
}

#[test]
fn test_simple_arithmetic_execution() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = 1 + 2");
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert!(execution_result.error.is_none());
        // Execution time is always valid (u64 type)

        let stats = engine.stats();
        assert_eq!(stats.total_executions, 1);
    });
}

#[test]
fn test_matrix_operations() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();

        // Test vector creation
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "y = [1, 2, 3]");
        assert!(result.is_ok());
        assert!(result.unwrap().error.is_none());

        // Test matrix creation
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "z = [1, 2; 3, 4]");
        assert!(result.is_ok());
        assert!(result.unwrap().error.is_none());

        let stats = engine.stats();
        assert_eq!(stats.total_executions, 2);
    });
}

#[test]
fn test_execution_statistics_tracking() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();

        // Execute multiple statements (testing individual execution)
        let inputs = ["x = 1", "y = 2", "z = 3"];
        for input in &inputs {
            let result = runmat_core::execute_text_request_for_testing(&mut engine, input);
            assert!(result.is_ok(), "Failed to execute: {input}");
        }

        let stats = engine.stats();
        assert_eq!(stats.total_executions, inputs.len());
        assert!(stats.average_execution_time_ms >= 0.0);
        // Total execution time is always valid (u64 type)
    });
}

#[test]
fn test_verbose_mode() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, true).unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = 42");
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert!(execution_result.error.is_none());
    });
}

#[test]
fn test_parse_error_handling() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = [1, 2,"); // Incomplete matrix
        match result {
            Err(RunError::Syntax(_)) => {}
            other => panic!("expected syntax error for incomplete matrix literal, got {other:?}"),
        }
    });
}

#[test]
fn test_invalid_syntax_handling() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = $invalid$");
        match result {
            Err(RunError::Syntax(_)) => {}
            other => panic!("expected syntax error for invalid tokens, got {other:?}"),
        }
    });
}

#[test]
fn test_execution_with_control_flow() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();

        let result =
            runmat_core::execute_text_request_for_testing(&mut engine, "if 1 > 0; x = 5; end")
                .expect("if-statement execution should succeed");
        assert!(
            result.error.is_none(),
            "if-statement should not report runtime errors"
        );

        let result =
            runmat_core::execute_text_request_for_testing(&mut engine, "for i = 1:3; y = i; end")
                .expect("for-loop execution should succeed");
        assert!(
            result.error.is_none(),
            "for-loop should not report runtime errors"
        );

        let x = runmat_core::execute_text_request_for_testing(&mut engine, "x")
            .expect("x readback should succeed");
        assert_eq!(x.value, Some(runmat_builtins::Value::Num(5.0)));
        let y = runmat_core::execute_text_request_for_testing(&mut engine, "y")
            .expect("y readback should succeed");
        assert_eq!(y.value, Some(runmat_builtins::Value::Num(3.0)));
    });
}

#[test]
fn test_gc_configuration() {
    gc_test_context(|| {
        let engine = RunMatSession::new().unwrap();
        let config = GcConfig::low_latency();
        let result = engine.configure_gc(config);
        assert!(result.is_ok());
    });
}

#[test]
fn test_gc_stats_retrieval() {
    gc_test_context(|| {
        let engine = RunMatSession::new().unwrap();
        let stats = engine.gc_stats();
        // Should be able to get stats without error
        let _allocations = stats
            .total_allocations
            .load(std::sync::atomic::Ordering::Relaxed);
    });
}

#[test]
fn test_system_info_display() {
    gc_test_context(|| {
        let engine = RunMatSession::new().unwrap();
        // This should not panic
        engine.show_system_info();
    });
}

#[test]
fn test_stats_reset() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();

        // Execute something to generate stats
        let _ = runmat_core::execute_text_request_for_testing(&mut engine, "x = 1");
        assert!(engine.stats().total_executions > 0);

        // Reset stats
        engine.reset_stats();
        assert_eq!(engine.stats().total_executions, 0);
        assert_eq!(engine.stats().total_execution_time_ms, 0);
        assert_eq!(engine.stats().average_execution_time_ms, 0.0);
    });
}

#[test]
fn test_multiple_executions_performance_tracking() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();

        // Execute the same operation multiple times to potentially trigger JIT
        for i in 1..=20 {
            let input = format!("x{i} = {i} + {i}");
            let result = runmat_core::execute_text_request_for_testing(&mut engine, &input);
            assert!(result.is_ok());
        }

        let stats = engine.stats();
        assert_eq!(stats.total_executions, 20);
        // At least some executions should have happened
        assert!(stats.jit_compiled + stats.interpreter_fallback == 20);
    });
}

#[test]
fn test_empty_input_handling() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "")
            .expect("empty input should execute");
        assert!(
            result.error.is_none(),
            "empty input should not produce runtime diagnostics"
        );
    });
}

#[test]
fn test_whitespace_only_input() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "   \t\n  ")
            .expect("whitespace-only input should execute");
        assert!(
            result.error.is_none(),
            "whitespace-only input should not produce runtime diagnostics"
        );
    });
}

#[test]
fn test_complex_expression_execution() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(
            &mut engine,
            "result = (1 + 2) * 3 - 4 / 2",
        );
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert!(execution_result.error.is_none());
        // Execution time can be 0 for very fast operations
    });
}

#[test]
fn test_concurrent_safety() {
    use std::sync::Arc;
    use std::thread;

    gc_test_context(|| {
        let engine = Arc::new(std::sync::Mutex::new(RunMatSession::new().unwrap()));
        let mut handles = vec![];

        // Spawn multiple threads executing different operations
        for i in 0..5 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let input = format!("x{i} = {i} * 2");
                let mut eng = engine_clone.lock().unwrap();
                runmat_core::execute_text_request_for_testing(&mut eng, &input)
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_execution_result_structure() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = 42").unwrap();

        // Check ExecutionResult structure
        // Execution time can be 0 for very fast operations
        assert!(result.error.is_none());
        // used_jit should be either true or false
        // Test just validates that the field exists and has a boolean value
        let _ = result.used_jit;
    });
}

#[test]
fn test_tokenize_detailed_compatibility() {
    let tokens = runmat_lexer::tokenize_detailed("x = 1 + 2");
    assert!(!tokens.is_empty());
    let labels = tokens
        .into_iter()
        .map(|token| format!("{:?}", token.token))
        .collect::<Vec<_>>();
    assert!(labels.contains(&"Ident".to_string()));
    assert!(labels.contains(&"Assign".to_string()));
    assert!(labels.contains(&"Integer".to_string()));
    assert!(labels.contains(&"Plus".to_string()));
}

#[test]
fn test_direct_execute_request_function() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "x = 1 + 2")
            .expect("execution should succeed");
        assert!(result.error.is_none());
    });
}

#[test]
fn test_direct_execute_request_error_handling() {
    gc_test_context(|| {
        let mut engine = RunMatSession::new().unwrap();
        let result = runmat_core::execute_text_request_for_testing(&mut engine, "invalid syntax $");
        assert!(result.is_err());
    });
}
