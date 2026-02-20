//! Stress tests for the garbage collector
//!
//! These tests verify that the GC can handle high-pressure scenarios
//! and integration with the interpreter and JIT.

use runmat_builtins::Value;
use runmat_gc::*;
use std::sync::atomic::Ordering;

#[test]
fn test_massive_allocation_cycle() {
    gc_test_context(|| {
        // Test allocating and releasing thousands of objects
        for cycle in 0..10 {
            let mut objects = Vec::new();

            // Allocate 1000 objects
            for i in 0..1000 {
                let value = match i % 4 {
                    0 => Value::Num(i as f64),
                    1 => Value::Int(runmat_builtins::IntValue::I32(i)),
                    2 => Value::Bool(i % 2 == 0),
                    _ => Value::String(format!("string_{i}")),
                };

                let ptr = gc_allocate(value).expect("allocation should succeed");
                objects.push(ptr);
            }

            // Force a collection
            let collected = gc_collect_major().expect("collection should succeed");
            println!("Cycle {cycle}: allocated 1000 objects, collected {collected}");

            // Clear references (making objects eligible for collection)
            objects.clear();

            // Force another collection
            let collected = gc_collect_major().expect("collection should succeed");
            println!("Cycle {cycle}: after clear, collected {collected}");
        }
    });
}

#[test]
fn test_large_matrix_stress() {
    gc_test_context(|| {
        // Test allocating large matrices that stress the GC
        let mut matrices = Vec::new();

        for i in 0..100 {
            let size = 100; // 100x100 matrix = 10,000 elements
            let data = vec![i as f64; size * size];
            let tensor = runmat_builtins::Tensor::new_2d(data, size, size)
                .expect("tensor creation should succeed");
            let ptr = gc_allocate(Value::Tensor(tensor)).expect("tensor allocation should succeed");
            matrices.push(ptr);

            // Trigger collection periodically
            if i % 10 == 9 {
                let collected = gc_collect_minor().expect("collection should succeed");
                println!(
                    "After {} large matrices, collected {} objects",
                    i + 1,
                    collected
                );
            }
        }

        // Final collection
        let collected = gc_collect_major().expect("final collection should succeed");
        println!("Final collection: {collected} objects");
    });
}

#[test]
fn test_nested_cell_stress() {
    gc_test_context(|| {
        // Test deeply nested cell structures
        fn create_nested_cell(depth: usize, width: usize) -> Value {
            if depth == 0 {
                Value::Num(42.0)
            } else {
                let mut cells = Vec::with_capacity(width);
                for _ in 0..width {
                    cells.push(create_nested_cell(depth - 1, width));
                }
                let ca = runmat_builtins::CellArray::new(cells, 1, width).expect("cell creation");
                Value::Cell(ca)
            }
        }

        let mut nested_objects = Vec::new();

        // Create nested structures of varying depths
        for depth in 1..=10 {
            let nested = create_nested_cell(depth, 3); // 3-way branching
            let ptr = gc_allocate(nested).expect("nested allocation should succeed");
            nested_objects.push(ptr);

            if depth % 3 == 0 {
                let collected = gc_collect_minor().expect("collection should succeed");
                println!("After depth {depth}, collected {collected} objects");
            }
        }

        // Major collection to clean up
        let collected = gc_collect_major().expect("major collection should succeed");
        println!("Final nested collection: {collected} objects");
    });
}

#[test]
fn test_gc_with_interpreter_integration() {
    gc_test_context(|| {
        // Test GC under interpreter load
        use futures::executor::block_on;
        use runmat_hir::{HirProgram, LoweringContext, SemanticError};
        use runmat_ignition::execute;
        use runmat_parser::parse;

        // Program that creates many temporary values
        let program = r#"
        result = 0;
        for i = 1:100
            temp1 = i * 2;
            temp2 = temp1 + 1;
            temp3 = temp2 * temp2;
            result = result + temp3;
        end
    "#;

        fn lower(ast: &runmat_parser::Program) -> std::result::Result<HirProgram, SemanticError> {
            runmat_hir::lower(ast, &LoweringContext::empty()).map(|result| result.hir)
        }

        // Run the program multiple times to stress the GC
        for run in 0..50 {
            let ast = parse(program).expect("parsing should succeed");
            let hir = lower(&ast).expect("lowering should succeed");
            let vars = block_on(execute(&hir)).expect("execution should succeed");

            // Verify the result is consistent
            let result: f64 = (&vars[0]).try_into().expect("result should be a number");
            assert!(result > 0.0);

            if run % 10 == 9 {
                let collected = gc_collect_minor().expect("collection should succeed");
                println!(
                    "After {} interpreter runs, collected {} objects",
                    run + 1,
                    collected
                );
            }
        }
    });
}

#[test]
fn test_gc_statistics_accuracy() {
    gc_test_context(|| {
        let initial_stats = gc_stats();
        let initial_allocations = initial_stats.total_allocations.load(Ordering::Relaxed);

        // Allocate a known number of objects
        let allocation_count = 500;
        let mut objects = Vec::new();

        for i in 0..allocation_count {
            let value = Value::Num(i as f64);
            let ptr = gc_allocate(value).expect("allocation should succeed");
            objects.push(ptr);
        }

        let after_alloc_stats = gc_stats();
        let final_allocations = after_alloc_stats.total_allocations.load(Ordering::Relaxed);

        // Check that statistics are accurate (at least the expected number of allocations)
        assert!(
            final_allocations - initial_allocations >= allocation_count,
            "Expected at least {} allocations, but got {}",
            allocation_count,
            final_allocations - initial_allocations
        );

        // Force collection and check collection stats
        let initial_collections = after_alloc_stats.minor_collections.load(Ordering::Relaxed);
        let collected = gc_collect_minor().expect("collection should succeed");
        let after_collect_stats = gc_stats();
        let final_collections = after_collect_stats
            .minor_collections
            .load(Ordering::Relaxed);

        let expected_increments = if collected > 0 { 1 } else { 0 };
        assert_eq!(
            final_collections - initial_collections,
            expected_increments,
            "minor collection stat mismatch: collected={collected}"
        );
        println!("Statistics test: allocated {allocation_count}, collected {collected}");
    });
}

#[test]
fn test_gc_under_memory_pressure() {
    gc_test_context(|| {
        // Configure GC for more aggressive collection
        let config = GcConfig {
            young_generation_size: 1024 * 1024, // 1MB
            minor_gc_threshold: 0.6,            // Collect at 60% full
            major_gc_threshold: 0.7,            // Major GC at 70% full
            ..GcConfig::default()
        };

        gc_configure(config).expect("configuration should succeed");

        let mut all_objects = Vec::new();
        let mut collection_count = 0;

        // Allocate until we trigger multiple collections
        for i in 0..10000 {
            // Create different types of objects to stress the allocator
            let value = match i % 5 {
                0 => Value::Num(i as f64),
                1 => Value::String(format!("pressure_test_{i}")),
                2 => Value::Tensor(
                    runmat_builtins::Tensor::new_2d(vec![i as f64; 100], 10, 10).unwrap(),
                ),
                3 => {
                    let ca = runmat_builtins::CellArray::new(
                        vec![
                            Value::Num(i as f64),
                            Value::Int(runmat_builtins::IntValue::I32(i)),
                        ],
                        1,
                        2,
                    )
                    .unwrap();
                    Value::Cell(ca)
                }
                _ => Value::Bool(i % 2 == 0),
            };

            let ptr = gc_allocate(value).expect("allocation should succeed");
            all_objects.push(ptr);

            // Check if collection was triggered
            let stats = gc_stats();
            let current_collections = stats.minor_collections.load(Ordering::Relaxed);
            if current_collections > collection_count {
                collection_count = current_collections;
                println!(
                    "Collection triggered after {} allocations (total collections: {})",
                    i + 1,
                    collection_count
                );
            }

            // Keep only recent objects to create pressure
            if all_objects.len() > 1000 {
                all_objects.drain(0..500); // Remove first 500 objects
            }
        }

        println!("Memory pressure test completed with {collection_count} collections");
        assert!(
            collection_count > 0,
            "Should have triggered at least one collection"
        );
    });
}

// NOTE: Concurrent allocation test removed as the current GC is not thread-safe
// This is a placeholder for future multi-threaded GC development

#[test]
fn test_gc_configuration_changes() {
    gc_test_context(|| {
        // Configure for frequent collections
        let aggressive_config = GcConfig {
            minor_gc_threshold: 0.3,          // Very aggressive
            young_generation_size: 64 * 1024, // Small generation
            ..GcConfig::default()
        };

        gc_configure(aggressive_config).expect("aggressive config should work");

        // Allocate under aggressive settings
        for i in 0..100 {
            let _ = gc_allocate(Value::String(format!("aggressive_{i}")))
                .expect("allocation should succeed");
        }

        let aggressive_stats = gc_stats();
        let aggressive_collections = aggressive_stats.minor_collections.load(Ordering::Relaxed);

        // Configure for less frequent collections
        let relaxed_config = GcConfig {
            minor_gc_threshold: 0.9,                // Very relaxed
            young_generation_size: 2 * 1024 * 1024, // Large generation
            ..GcConfig::default()
        };

        gc_configure(relaxed_config).expect("relaxed config should work");

        // Allocate under relaxed settings
        for i in 0..100 {
            let _ = gc_allocate(Value::String(format!("relaxed_{i}")))
                .expect("allocation should succeed");
        }

        let final_stats = gc_stats();
        let final_collections = final_stats.minor_collections.load(Ordering::Relaxed);

        println!(
            "Configuration test: aggressive collections: {aggressive_collections}, final collections: {final_collections}"
        );

        // Aggressive config should trigger more collections
        assert!(
            aggressive_collections >= 1,
            "Aggressive config should trigger collections"
        );
    });
}
