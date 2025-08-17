//! Basic allocation tests for the garbage collector

use runmat_builtins::Value;
use runmat_gc::*;

#[test]
fn test_basic_allocation() {
    let _ = gc_reset_for_test();
    let value = Value::Num(42.0);
    let ptr = gc_allocate(value).expect("allocation should succeed");

    assert_eq!(*ptr, Value::Num(42.0));
    assert!(!ptr.is_null());
}

#[test]
fn test_multiple_allocations() {
    let _ = gc_reset_for_test();
    let values = vec![
        Value::Num(1.0),
        Value::Int(2),
        Value::Bool(true),
        Value::String("test".to_string()),
    ];

    let mut ptrs = Vec::new();
    for value in values {
        let ptr = gc_allocate(value.clone()).expect("allocation should succeed");
        assert_eq!(*ptr, value);
        ptrs.push(ptr);
    }

    // Verify all pointers are still valid
    assert_eq!(*ptrs[0], Value::Num(1.0));
    assert_eq!(*ptrs[1], Value::Int(2));
    assert_eq!(*ptrs[2], Value::Bool(true));
    assert_eq!(*ptrs[3], Value::String("test".to_string()));
}

#[test]
fn test_matrix_allocation() {
    let _ = gc_reset_for_test();
    let tensor = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("tensor creation should succeed");
    let value = Value::Tensor(tensor);

    let ptr = gc_allocate(value).expect("allocation should succeed");

    if let Value::Tensor(ref m) = *ptr {
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("Expected Matrix value");
    }
}

#[test]
fn test_cell_allocation() {
    let _ = gc_reset_for_test();
    let cell_contents = vec![
        Value::Num(1.0),
        Value::String("nested".to_string()),
        Value::Bool(false),
    ];
    let cell = Value::Cell(runmat_builtins::CellArray::new(cell_contents.clone(), 1, 3).unwrap());

    let ptr = gc_allocate(cell).expect("allocation should succeed");

    if let Value::Cell(ref contents) = *ptr {
        assert_eq!(contents.data.len(), 3);
        assert_eq!(&*contents.data[0], &Value::Num(1.0));
        assert_eq!(&*contents.data[1], &Value::String("nested".to_string()));
        assert_eq!(&*contents.data[2], &Value::Bool(false));
    } else {
        panic!("Expected Cell value");
    }
}

#[test]
fn test_allocation_stats() {
    gc_test_context(|| {
        let initial_stats = gc_stats();
        let initial_allocations = initial_stats
            .total_allocations
            .load(std::sync::atomic::Ordering::Relaxed);

        // Allocate some values
        for i in 0..10 {
            let _ = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        }

        let final_stats = gc_stats();
        let final_allocations = final_stats
            .total_allocations
            .load(std::sync::atomic::Ordering::Relaxed);

        // Check that at least 10 allocations were made (allowing for test isolation issues)
        assert!(
            final_allocations - initial_allocations >= 10,
            "Expected at least 10 allocations, got {}",
            final_allocations - initial_allocations
        );
    });
}

#[test]
fn test_large_allocation() {
    // Ensure we have a reasonable configuration for this test
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");

    // Create a large matrix
    let size = 100;
    let data = vec![1.0; size * size];
    let tensor = runmat_builtins::Tensor::new_2d(data, size, size).expect("tensor creation should succeed");
    let value = Value::Tensor(tensor);

    let ptr = gc_allocate(value).expect("large allocation should succeed");

    if let Value::Tensor(ref m) = *ptr {
        assert_eq!(m.rows, size);
        assert_eq!(m.cols, size);
        assert_eq!(m.data.len(), size * size);
    } else {
        panic!("Expected Matrix value");
    }
}

#[test]
fn test_allocation_triggers_collection() {
    gc_test_context(|| {
        // Configure GC for more frequent collections
        let config = GcConfig {
            minor_gc_threshold: 0.05,   // Very low threshold (5% instead of 10%)
            young_generation_size: 512, // Even smaller generation
            ..GcConfig::default()
        };

        gc_configure(config).expect("configuration should succeed");

        let initial_stats = gc_stats();
        let initial_collections = initial_stats
            .minor_collections
            .load(std::sync::atomic::Ordering::Relaxed);

        // Allocate enough values to definitely trigger collection
        // Each Value is ~40 bytes, so 20 values = ~800 bytes, which should exceed 5% of 512 bytes
        for i in 0..20 {
            let _ = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        }

        let final_stats = gc_stats();
        let final_collections = final_stats
            .minor_collections
            .load(std::sync::atomic::Ordering::Relaxed);

        // Should have triggered at least one collection
        assert!(
            final_collections > initial_collections,
            "Expected collections to increase from {initial_collections} but got {final_collections}"
        );
    });
}

#[test]
fn test_nested_cell_allocation() {
    // Ensure we have a reasonable configuration for this test
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");

    // Create nested cell arrays
    let inner_cell = Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());

    let outer_cell = Value::Cell(runmat_builtins::CellArray::new(vec![inner_cell, Value::String("outer".to_string())], 1, 2).unwrap());

    let ptr = gc_allocate(outer_cell).expect("allocation should succeed");

    if let Value::Cell(ref outer_contents) = *ptr {
        assert_eq!(outer_contents.data.len(), 2);

        if let Value::Cell(ref inner_contents) = *outer_contents.data[0] {
            assert_eq!(inner_contents.data.len(), 2);
            assert_eq!(&*inner_contents.data[0], &Value::Num(1.0));
            assert_eq!(&*inner_contents.data[1], &Value::Num(2.0));
        } else {
            panic!("Expected inner Cell value");
        }

        assert_eq!(&*outer_contents.data[1], &Value::String("outer".to_string()));
    } else {
        panic!("Expected outer Cell value");
    }
}

#[test]
fn test_allocation_with_roots() {
    gc_test_context(|| {
        // Allocate some values and explicitly register them as roots
        let ptr1 = gc_allocate(Value::Num(1.0)).expect("allocation should succeed");
        let ptr2 = gc_allocate(Value::Num(2.0)).expect("allocation should succeed");

        // Register as roots to protect from collection
        gc_add_root(ptr1.clone()).expect("root registration should succeed");
        gc_add_root(ptr2.clone()).expect("root registration should succeed");

        assert_eq!(*ptr1, Value::Num(1.0));
        assert_eq!(*ptr2, Value::Num(2.0));

        // Force a collection - roots should keep their values alive
        let _collected = gc_collect_minor().expect("collection should succeed");

        // Values should still be accessible
        assert_eq!(*ptr1, Value::Num(1.0));
        assert_eq!(*ptr2, Value::Num(2.0));

        // Clean up roots
        gc_remove_root(ptr1).expect("root removal should succeed");
        gc_remove_root(ptr2).expect("root removal should succeed");
    });
}
