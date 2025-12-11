use runmat_builtins::Value;
use runmat_gc::gc_test_context;
use runmat_hir::lower;
use runmat_ignition::compile;
use runmat_parser::parse;
use runmat_turbine::TurbineEngine;
use runmat_time::Instant;
use std::time::Duration;

#[test]
fn test_compilation_performance() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "perf_test = 42 + 58";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            let start = Instant::now();
            let result = engine.compile_bytecode(&bytecode);
            let compilation_time = start.elapsed();

            assert!(result.is_ok());

            // Compilation should be reasonably fast (less than 1 second for simple code)
            assert!(
                compilation_time < Duration::from_secs(1),
                "Compilation took too long: {compilation_time:?}"
            );
        }
    });
}

#[test]
fn test_execution_performance() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "execution_test = 100 * 200";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            let mut vars = vec![Value::Num(0.0); bytecode.var_count];

            // Measure execution time
            let start = Instant::now();
            let result = engine.execute_or_compile(&bytecode, &mut vars);
            let execution_time = start.elapsed();

            assert!(result.is_ok());

            // Execution should be fast (less than 100ms for simple arithmetic)
            assert!(
                execution_time < Duration::from_millis(100),
                "Execution took too long: {execution_time:?}"
            );
        }
    });
}

#[test]
fn test_repeated_execution_performance() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "repeated = 50 + 75";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            let mut execution_times = Vec::new();

            // Execute the same bytecode multiple times
            for _ in 0..10 {
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];

                let start = Instant::now();
                let result = engine.execute_or_compile(&bytecode, &mut vars);
                let execution_time = start.elapsed();

                assert!(result.is_ok());
                execution_times.push(execution_time);
            }

            // Later executions should not be significantly slower than early ones
            let first_execution = execution_times[0];
            let last_execution = execution_times[execution_times.len() - 1];

            // Allow up to 5x slowdown (generous threshold)
            assert!(
                last_execution < first_execution * 5,
                "Performance degraded: first={first_execution:?}, last={last_execution:?}"
            );
        }
    });
}

#[test]
fn test_compilation_cache_performance() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "cache_perf = 123 + 456";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            // First compilation
            let start1 = Instant::now();
            let result1 = engine.compile_bytecode(&bytecode);
            let time1 = start1.elapsed();
            assert!(result1.is_ok());

            // Second compilation (should use cache)
            let start2 = Instant::now();
            let result2 = engine.compile_bytecode(&bytecode);
            let time2 = start2.elapsed();
            assert!(result2.is_ok());

            // Cached compilation should be much faster
            assert!(
                time2 <= time1,
                "Cached compilation not faster: first={time1:?}, cached={time2:?}"
            );
        }
    });
}

#[test]
fn test_memory_usage_during_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let initial_stats = runmat_gc::gc_stats();
            let initial_memory = initial_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed);

            // Compile multiple functions
            for i in 0..10 {
                let source = format!("memory_test_{i} = {i} * 2");
                let ast = parse(&source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();

                let result = engine.compile_bytecode(&bytecode);
                assert!(result.is_ok());
            }

            let final_stats = runmat_gc::gc_stats();
            let final_memory = final_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed);

            // Memory usage should be reasonable (less than 10MB increase)
            let memory_increase = final_memory.saturating_sub(initial_memory);
            assert!(
                memory_increase < 10 * 1024 * 1024,
                "Excessive memory usage: increased by {memory_increase} bytes"
            );
        }
    });
}

#[test]
fn test_hotspot_compilation_timing() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "hotspot_timing = 789 + 321";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let mut times = Vec::new();

            // Execute multiple times to potentially trigger JIT compilation
            for _ in 0..15 {
                let start = Instant::now();
                let result = engine.execute_or_compile(&bytecode, &mut vars);
                let elapsed = start.elapsed();

                assert!(result.is_ok());
                times.push(elapsed);
            }

            // Check that execution times are reasonable
            for time in &times {
                assert!(
                    *time < Duration::from_millis(50),
                    "Individual execution too slow: {time:?}"
                );
            }

            // Average time should be reasonable
            let average_time: Duration = times.iter().sum::<Duration>() / times.len() as u32;
            assert!(
                average_time < Duration::from_millis(10),
                "Average execution time too slow: {average_time:?}"
            );
        }
    });
}

#[test]
fn test_compilation_scalability() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let mut compilation_times = Vec::new();

            // Compile increasingly complex expressions
            for i in 1..=10 {
                let mut expr = "1".to_string();
                for j in 2..=i {
                    expr = format!("({expr} + {j})");
                }
                let source = format!("scalability_test = {expr}");

                let ast = parse(&source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();

                let start = Instant::now();
                let result = engine.compile_bytecode(&bytecode);
                let elapsed = start.elapsed();

                if result.is_ok() {
                    compilation_times.push(elapsed);
                } else {
                    // Some complex expressions might not be JIT-compilable
                    break;
                }
            }

            // Compilation time shouldn't grow exponentially
            if compilation_times.len() >= 2 {
                let first_time = compilation_times[0];
                let last_time = compilation_times[compilation_times.len() - 1];

                // Allow up to 10x increase for 10x complexity
                assert!(
                    last_time < first_time * 20,
                    "Compilation time grew too much: first={first_time:?}, last={last_time:?}"
                );
            }
        }
    });
}

#[test]
fn test_concurrent_execution_performance() {
    gc_test_context(|| {
        if let Ok(engine) = TurbineEngine::new() {
            use std::sync::{Arc, Mutex};
            use std::thread;

            let engine = Arc::new(Mutex::new(engine));
            let mut handles = Vec::new();
            let start_time = Instant::now();

            // Spawn multiple threads executing in parallel
            for i in 0..4 {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    let source = format!("concurrent_{} = {} + {}", i, i, i + 1);
                    let ast = parse(&source).unwrap();
                    let hir = lower(&ast).unwrap();
                    let bytecode = compile(&hir).unwrap();

                    let mut vars = vec![Value::Num(0.0); bytecode.var_count];

                    let mut engine_guard = engine_clone.lock().unwrap();
                    engine_guard.execute_or_compile(&bytecode, &mut vars)
                });
                handles.push(handle);
            }

            // Wait for all threads
            for handle in handles {
                let result = handle.join().unwrap();
                assert!(result.is_ok());
            }

            let total_time = start_time.elapsed();

            // Concurrent execution should complete in reasonable time
            assert!(
                total_time < Duration::from_secs(5),
                "Concurrent execution took too long: {total_time:?}"
            );
        }
    });
}

#[test]
fn test_garbage_collection_impact() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "gc_impact = 555 + 444";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            // Perform some allocations to stress GC
            for i in 0..100 {
                let value = Value::Num(i as f64);
                let _ = runmat_gc::gc_allocate(value);
            }

            // Force a GC collection
            let _ = runmat_gc::gc_collect_major();

            // JIT should still work after GC
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let start = Instant::now();
            let result = engine.execute_or_compile(&bytecode, &mut vars);
            let elapsed = start.elapsed();

            assert!(result.is_ok());

            // Performance should not be severely impacted by GC
            assert!(
                elapsed < Duration::from_millis(100),
                "Execution after GC too slow: {elapsed:?}"
            );
        }
    });
}

#[test]
fn test_engine_statistics_overhead() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "stats_overhead = 999";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            let start = Instant::now();

            // Execute and collect stats multiple times
            for _ in 0..10 {
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let _ = engine.execute_or_compile(&bytecode, &mut vars);
                let _ = engine.stats(); // Get statistics
            }

            let elapsed = start.elapsed();

            // Statistics collection should not significantly impact performance
            assert!(
                elapsed < Duration::from_millis(500),
                "Statistics collection overhead too high: {elapsed:?}"
            );
        }
    });
}

#[test]
fn test_peak_memory_usage() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let initial_stats = runmat_gc::gc_stats();
            let initial_peak = initial_stats
                .peak_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed);

            // Compile and execute multiple functions
            for i in 0..20 {
                let source = format!("peak_test_{} = {} + {} * 2", i, i, i + 1);
                let ast = parse(&source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();

                let result = engine.compile_bytecode(&bytecode);
                if result.is_err() {
                    break; // Might hit resource limits
                }

                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let _ = engine.execute_or_compile(&bytecode, &mut vars);
            }

            let final_stats = runmat_gc::gc_stats();
            let final_peak = final_stats
                .peak_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed);

            // Peak memory usage should be reasonable (less than 50MB increase)
            let peak_increase = final_peak.saturating_sub(initial_peak);
            assert!(
                peak_increase < 50 * 1024 * 1024,
                "Peak memory usage too high: increased by {peak_increase} bytes"
            );
        }
    });
}

#[test]
fn test_cache_hit_ratio_performance() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "cache_hit_test = 111 + 222";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();

            // Multiple compilations to test cache behavior
            for _ in 0..10 {
                let result = engine.compile_bytecode(&bytecode);
                assert!(result.is_ok(), "Compilation should succeed");
            }

            let final_stats = engine.stats();

            // Cache hit rate should be a valid percentage (0.0 to 1.0)
            assert!(
                final_stats.cache_hit_rate >= 0.0 && final_stats.cache_hit_rate <= 1.0,
                "Cache hit rate should be between 0.0 and 1.0, got: {}",
                final_stats.cache_hit_rate
            );

            // Engine should maintain cache capacity information
            assert!(
                final_stats.cache_capacity > 0,
                "Cache should have a positive capacity"
            );
        }
    });
}
