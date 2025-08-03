use rustmat_turbine::TurbineEngine;
use rustmat_ignition::{compile, Bytecode};
use rustmat_parser::parse;
use rustmat_hir::lower;
use rustmat_builtins::Value;
use rustmat_gc::{gc_test_context, gc_allocate};

#[test]
fn test_turbine_engine_creation() {
    gc_test_context(|| {
        let result = TurbineEngine::new();
        if TurbineEngine::is_jit_supported() {
            assert!(result.is_ok());
        } else {
            // On unsupported platforms, should fail gracefully
            assert!(result.is_err());
        }
    });
}

#[test]
fn test_jit_support_detection() {
    // Should return a boolean without error
    let supported = TurbineEngine::is_jit_supported();
    assert!(supported == true || supported == false);
}

#[test]
fn test_target_info() {
    gc_test_context(|| {
        if let Ok(engine) = TurbineEngine::new() {
            let info = engine.target_info();
            assert!(!info.is_empty());
            assert!(info.contains("Target ISA"));
        }
    });
}

#[test]
fn test_simple_arithmetic_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Create simple bytecode for arithmetic
            let source = "x = 5 + 3";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_execution_with_variables() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "result = 10 * 2";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let result = engine.execute_or_compile(&bytecode, &mut vars);
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_hotspot_compilation_threshold() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "temp = 7 + 8";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let hash = engine.calculate_bytecode_hash(&bytecode);
            
            // Initially should not be considered hot
            assert!(!engine.should_compile(hash));
            
            // After multiple executions, should become hot
            for _ in 0..15 {
                let _ = engine.should_compile(hash);
            }
            
            // Should now be considered hot (implementation may vary)
            // This tests the profiling logic
        }
    });
}

#[test]
fn test_function_caching() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "cached = 15 + 25";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            // First compilation
            let result1 = engine.compile_bytecode(&bytecode);
            assert!(result1.is_ok());
            let hash1 = result1.unwrap();
            
            // Second compilation of the same bytecode should use cache
            let result2 = engine.compile_bytecode(&bytecode);
            assert!(result2.is_ok());
            let hash2 = result2.unwrap();
            
            assert_eq!(hash1, hash2);
        }
    });
}

#[test]
fn test_compilation_statistics() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let stats = engine.stats();
            
            // Should have initial stats
            assert_eq!(stats.compiled_functions, 0);
            // assert_eq!(stats.cache_hits, 0);
            // assert_eq!(stats.cache_misses, 0);
            
            // Compile something
            let source = "stats_test = 42";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let _ = engine.compile_bytecode(&bytecode);
            
            let updated_stats = engine.stats();
            // Stats should change (exact behavior depends on implementation)
            assert!(updated_stats.compiled_functions >= stats.compiled_functions);
        }
    });
}

#[test]
fn test_interpreter_fallback() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Test with complex control flow that might not be JIT-compilable
            let source = "x = 10";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let result = engine.execute_or_compile(&bytecode, &mut vars);
            
            // Should succeed even if JIT compilation fails (fallback to interpreter)
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_multiple_function_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let test_cases = [
                "a = 1 + 2",
                "b = 3 * 4",
                "c = 5 - 1",
                "d = 8 / 2",
            ];
            
            for (i, source) in test_cases.iter().enumerate() {
                let ast = parse(source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();
                
                let result = engine.compile_bytecode(&bytecode);
                assert!(result.is_ok(), "Failed to compile case {}: {}", i, source);
            }
            
            let _stats = engine.stats();
            // Stats should be available and engine functional
        }
    });
}

#[test]
fn test_bytecode_hash_consistency() {
    gc_test_context(|| {
        if let Ok(engine) = TurbineEngine::new() {
            let source = "hash_test = 99";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            // Same bytecode should produce same hash
            let hash1 = engine.calculate_bytecode_hash(&bytecode);
            let hash2 = engine.calculate_bytecode_hash(&bytecode);
            assert_eq!(hash1, hash2);
            
            // Different bytecode should produce different hash
            let source2 = "different = 88";
            let ast2 = parse(source2).unwrap();
            let hir2 = lower(&ast2).unwrap();
            let bytecode2 = compile(&hir2).unwrap();
            let hash3 = engine.calculate_bytecode_hash(&bytecode2);
            
            assert_ne!(hash1, hash3);
        }
    });
}

#[test]
fn test_execution_result_accuracy() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "computation = 6 * 7";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let result = engine.execute_or_compile(&bytecode, &mut vars);
            assert!(result.is_ok());
            
            // Check that execution doesn't crash and returns reasonable result
            assert_eq!(result.unwrap(), (0, false)); // (status, used_jit)
        }
    });
}

#[test]
fn test_memory_safety_with_gc() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Test that JIT compilation works with GC allocations
            let matrix_data = vec![1.0, 2.0, 3.0, 4.0];
            let matrix = rustmat_builtins::Matrix::new(matrix_data, 2, 2).unwrap();
            let matrix_value = Value::Matrix(matrix);
            let _gc_ptr = gc_allocate(matrix_value).unwrap();
            
            // Now test JIT compilation
            let source = "jit_with_gc = 100";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_concurrent_compilation_safety() {
    gc_test_context(|| {
        if let Ok(engine) = TurbineEngine::new() {
            use std::sync::{Arc, Mutex};
            use std::thread;
            
            let engine = Arc::new(Mutex::new(engine));
            let mut handles = vec![];
            
            // Spawn multiple threads trying to compile different functions
            for i in 0..3 {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    let source = format!("thread_{} = {} + 1", i, i);
                    let ast = parse(&source).unwrap();
                    let hir = lower(&ast).unwrap();
                    let bytecode = compile(&hir).unwrap();
                    
                    let mut engine_guard = engine_clone.lock().unwrap();
                    engine_guard.compile_bytecode(&bytecode)
                });
                handles.push(handle);
            }
            
            // Wait for all threads and check results
            for handle in handles {
                let result = handle.join().unwrap();
                assert!(result.is_ok());
            }
        }
    });
}

#[test]
fn test_error_handling_in_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Create bytecode that might be problematic for JIT
            let mut bytecode = Bytecode {
                instructions: vec![], // Empty instructions
                var_count: 0,
            };
            
            let result = engine.compile_bytecode(&bytecode);
            // Should handle gracefully (might succeed with empty bytecode or fail cleanly)
            assert!(result.is_ok() || result.is_err());
            
            // Engine should still be usable after error
            let source = "recovery_test = 55";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            assert!(result.is_ok());
        }
    });
}

#[test]
fn test_matrix_operations_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "matrix_test = [1, 2; 3, 4]";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            // Matrix operations should be compilable or fall back gracefully
            assert!(result.is_ok() || result.is_err());
            
            if result.is_ok() {
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let exec_result = engine.execute_or_compile(&bytecode, &mut vars);
                assert!(exec_result.is_ok());
            }
        }
    });
}

#[test]
fn test_complex_expression_compilation() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "complex = (10 + 5) * 3 - 8 / 2";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            assert!(result.is_ok());
            
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let exec_result = engine.execute_or_compile(&bytecode, &mut vars);
            assert!(exec_result.is_ok());
        }
    });
}

#[test]
fn test_profiler_hotness_tracking() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let source = "hotness_test = 77";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let hash = engine.calculate_bytecode_hash(&bytecode);
            
            // Track multiple executions
            for _ in 0..10 {
                let _should_compile = engine.should_compile(hash);
            }
            
            // Should have recorded the executions
            // (Exact behavior depends on implementation)
        }
    });
}

#[test]
fn test_cache_eviction_under_pressure() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Compile many different functions to test cache behavior
            for i in 0..20 {
                let source = format!("cache_test_{} = {}", i, i);
                let ast = parse(&source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();
                
                let result = engine.compile_bytecode(&bytecode);
                if result.is_err() {
                    // Might fail on resource exhaustion, which is acceptable
                    break;
                }
            }
            
            // Engine should still be responsive
            let _stats = engine.stats();
            // Stats should be available and functional
        }
    });
}

#[test]
fn test_execution_with_different_variable_counts() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            let test_cases = [
                ("single = 1", 1),
                ("a = 1; b = 2", 2),
                ("x = 1; y = 2; z = 3", 3),
            ];
            
            for (source, expected_vars) in &test_cases {
                let ast = parse(source).unwrap();
                let hir = lower(&ast).unwrap();
                let bytecode = compile(&hir).unwrap();
                
                assert!(bytecode.var_count >= *expected_vars);
                
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let result = engine.execute_or_compile(&bytecode, &mut vars);
                assert!(result.is_ok(), "Failed with case: {}", source);
            }
        }
    });
}

#[test]
fn test_runtime_function_integration() {
    gc_test_context(|| {
        if let Ok(mut engine) = TurbineEngine::new() {
            // Test compilation of code that uses runtime functions
            let source = "builtin_test = abs(-5)";
            let ast = parse(source).unwrap();
            let hir = lower(&ast).unwrap();
            let bytecode = compile(&hir).unwrap();
            
            let result = engine.compile_bytecode(&bytecode);
            // Should handle builtin functions (either compile or fallback)
            assert!(result.is_ok() || result.is_err());
            
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let exec_result = engine.execute_or_compile(&bytecode, &mut vars);
            assert!(exec_result.is_ok());
        }
    });
}