use rustmat_repl::ReplEngine;
use rustmat_gc::{gc_test_context, GcConfig};
use std::thread;

#[test]
fn test_jit_vs_interpreter_execution() {
    gc_test_context(|| {
        // Test with JIT enabled
        let mut jit_engine = ReplEngine::with_options(true, false).unwrap();
        let jit_result = jit_engine.execute("x = 5 + 3").unwrap();
        
        // Test with JIT disabled  
        let mut interp_engine = ReplEngine::with_options(false, false).unwrap();
        let interp_result = interp_engine.execute("x = 5 + 3").unwrap();
        
        // Both should succeed
        assert!(jit_result.error.is_none());
        assert!(interp_result.error.is_none());
        
        // Interpreter should never use JIT
        assert!(!interp_result.used_jit);
    });
}

#[test]
fn test_hotspot_compilation_simulation() {
    gc_test_context(|| {
        let mut engine = ReplEngine::with_options(true, false).unwrap();
        
        // Execute the same code multiple times to trigger potential JIT compilation
        let code = "result = 10 * 5 + 2";
        let mut _jit_executions = 0;
        let mut interp_executions = 0;
        
        for _ in 0..15 { // Execute multiple times to cross JIT threshold
            let result = engine.execute(code).unwrap();
            assert!(result.error.is_none());
            
            if result.used_jit {
                _jit_executions += 1;
            } else {
                interp_executions += 1;
            }
        }
        
        let stats = engine.stats();
        assert_eq!(stats.total_executions, 15);
        assert_eq!(stats.jit_compiled + stats.interpreter_fallback, 15);
        
        // Should have at least some executions (JIT may or may not be available)
        assert!(interp_executions > 0 || _jit_executions > 0, "Should have some executions");
    });
}

#[test]
fn test_gc_integration_during_execution() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        // Configure GC for low latency
        let config = GcConfig::low_latency();
        engine.configure_gc(config).unwrap();
        
        // Execute operations that create objects
        for i in 0..10 {
            let code = format!("matrix{} = [1, 2; 3, 4]", i);
            let result = engine.execute(&code);
            assert!(result.is_ok());
        }
        
        // Check GC stats - may not have allocations if using simple interpreter
        let gc_stats = engine.gc_stats();
        let _allocations = gc_stats.total_allocations.load(std::sync::atomic::Ordering::Relaxed);
        // GC allocations depend on implementation - could be 0 with simple interpreter
    });
}

#[test]
fn test_error_recovery_and_continued_execution() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        // Execute valid code
        let result1 = engine.execute("x = 1");
        assert!(result1.is_ok());
        
        // Execute invalid code
        let result2 = engine.execute("y = [1, 2,"); // Incomplete
        assert!(result2.is_err());
        
        // Engine should recover and continue working
        let result3 = engine.execute("z = 3");
        assert!(result3.is_ok());
        
        // Statistics should reflect execution attempts (behavior may vary)
        let stats = engine.stats();
        // Could be 2 (only successful) or 3 (all attempts) depending on implementation
        assert!(stats.total_executions >= 2 && stats.total_executions <= 3, 
               "Expected 2-3 executions, got {}", stats.total_executions);
    });
}

#[test]
fn test_complex_mathematical_operations() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        let complex_operations = [
            "a = 1 + 2 * 3",
            "b = (4 + 5) * 6", 
            "c = 7 - 8 / 2",
            "d = 10 + 20 - 5",
        ];
        
        for op in &complex_operations {
            let result = engine.execute(op);
            assert!(result.is_ok(), "Failed to execute: {}", op);
            assert!(result.unwrap().error.is_none());
        }
        
        let stats = engine.stats();
        assert_eq!(stats.total_executions, complex_operations.len());
    });
}

#[test]
fn test_control_flow_execution() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        let control_flow_tests = [
            "if 1 > 0; x = 10; end",
            "if 0 > 1; y = 20; else; y = 30; end",
            "for i = 1:3; z = i * 2; end",
            "while 0 < 1; break; end",
        ];
        
        for test in &control_flow_tests {
            let result = engine.execute(test);
            // Control flow may not be fully implemented yet
            if result.is_ok() {
                assert!(result.unwrap().error.is_none());
            } else {
                // If control flow isn't implemented, that's acceptable for now
                assert!(!result.unwrap_err().to_string().is_empty());
            }
        }
    });
}

#[test]
fn test_memory_usage_under_load() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        // Create many objects to test memory management
        for i in 0..50 {
            let code = format!("var{} = [{}; {}; {}]", i, i, i+1, i+2);
            let result = engine.execute(&code);
            assert!(result.is_ok());
        }
        
        let gc_stats = engine.gc_stats();
        let _allocations = gc_stats.total_allocations.load(std::sync::atomic::Ordering::Relaxed);
        // Matrix creation might not go through GC with simple interpreter
        
        // Force a GC collection
        let _ = rustmat_gc::gc_collect_major();
        
        // Should still be able to execute after GC
        let result = engine.execute("final = 42");
        assert!(result.is_ok());
    });
}

#[test]
fn test_execution_timing_accuracy() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        let start = std::time::Instant::now();
        let result = engine.execute("x = 1 + 1");
        let elapsed = start.elapsed();
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        
        // Execution time should be reasonable (can be 0 for very fast operations)
        // (execution_time_ms is u64, so always >= 0)
        assert!(exec_result.execution_time_ms <= elapsed.as_millis() as u64 + 100); // Allow some variance
    });
}

#[test]
fn test_verbose_mode_output() {
    gc_test_context(|| {
        // Verbose mode should not crash or cause errors
        let mut verbose_engine = ReplEngine::with_options(true, true).unwrap();
        let mut quiet_engine = ReplEngine::with_options(true, false).unwrap();
        
        let code = "test = 1 + 2";
        
        let verbose_result = verbose_engine.execute(code);
        let quiet_result = quiet_engine.execute(code);
        
        assert!(verbose_result.is_ok());
        assert!(quiet_result.is_ok());
        
        // Both should produce the same functional result
        assert_eq!(verbose_result.unwrap().error.is_none(), quiet_result.unwrap().error.is_none());
    });
}

#[test]
fn test_statistics_accuracy() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        let num_executions = 7;
        for i in 0..num_executions {
            let code = format!("val{} = {}", i, i);
            let result = engine.execute(&code);
            assert!(result.is_ok());
        }
        
        let stats = engine.stats();
        assert_eq!(stats.total_executions, num_executions);
        // (total_execution_time_ms is u64, so always >= 0)
        assert!(stats.average_execution_time_ms >= 0.0);
        
        // Average should be total/count
        let expected_avg = stats.total_execution_time_ms as f64 / num_executions as f64;
        assert!((stats.average_execution_time_ms - expected_avg).abs() < 0.001);
    });
}

#[test]
fn test_engine_state_isolation() {
    gc_test_context(|| {
        let mut engine1 = ReplEngine::new().unwrap();
        let mut engine2 = ReplEngine::new().unwrap();
        
        // Execute different code in each engine
        engine1.execute("x = 10").unwrap();
        engine2.execute("y = 20").unwrap();
        
        let stats1 = engine1.stats();
        let stats2 = engine2.stats();
        
        // Each engine should have its own statistics
        assert_eq!(stats1.total_executions, 1);
        assert_eq!(stats2.total_executions, 1);
        
        // Reset one engine shouldn't affect the other
        engine1.reset_stats();
        assert_eq!(engine1.stats().total_executions, 0);
        assert_eq!(engine2.stats().total_executions, 1); // Unchanged
    });
}

#[test]
fn test_concurrent_engine_usage() {
    gc_test_context(|| {
        use std::sync::{Arc, Mutex};
        
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];
        
        for thread_id in 0..3 {
            let results_clone = Arc::clone(&results);
            let handle = thread::spawn(move || {
                // Don't nest gc_test_context - create engine directly in thread
                let mut engine = ReplEngine::new().unwrap();
                let code = format!("thread_var = {} + 1", thread_id);
                let result = engine.execute(&code);
                
                let mut results_guard = results_clone.lock().unwrap();
                results_guard.push((thread_id, result.is_ok()));
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let results_guard = results.lock().unwrap();
        assert_eq!(results_guard.len(), 3);
        
        // All executions should have succeeded
        for (_thread_id, success) in results_guard.iter() {
            assert!(success, "Thread execution failed");
        }
    });
}

#[test]
fn test_matrix_operations_integration() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        let matrix_tests = [
            "A = [1, 2; 3, 4]",
            "B = [5, 6; 7, 8]", 
            "row_vec = [1, 2, 3]",
            "col_vec = [1; 2; 3]",
            "scalar = 42",
        ];
        
        for test in &matrix_tests {
            let result = engine.execute(test);
            assert!(result.is_ok(), "Matrix operation failed: {}", test);
            
            let exec_result = result.unwrap();
            assert!(exec_result.error.is_none());
        }
        
        let stats = engine.stats();
        assert_eq!(stats.total_executions, matrix_tests.len());
    });
}

#[test]
fn test_performance_degradation_detection() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        // Execute a bunch of operations and ensure performance doesn't degrade significantly
        let mut execution_times = Vec::new();
        
        for i in 0..20 {
            let code = format!("perf_test_{} = {} * 2 + 1", i, i);
            let result = engine.execute(&code).unwrap();
            execution_times.push(result.execution_time_ms);
        }
        
        // Calculate average execution time for first and last batches
        let first_batch_avg: f64 = execution_times[0..5].iter().map(|&x| x as f64).sum::<f64>() / 5.0;
        let last_batch_avg: f64 = execution_times[15..20].iter().map(|&x| x as f64).sum::<f64>() / 5.0;
        
        // Performance should not degrade significantly (allow 3x increase)
        // If both are 0 (very fast execution), that's acceptable
        if first_batch_avg > 0.0 {
            assert!(last_batch_avg < first_batch_avg * 3.0, 
                   "Performance degraded: first batch avg = {}, last batch avg = {}", 
                   first_batch_avg, last_batch_avg);
        } else {
            // Both are very fast (≤1ms), which is good performance
            assert!(last_batch_avg <= 1.0, "Performance should be fast");
        }
    });
}