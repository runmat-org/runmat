//! Garbage collection behavior tests

use rustmat_gc::*;
use rustmat_builtins::Value;
use std::sync::atomic::Ordering;

#[test]
fn test_minor_collection() {
    let initial_stats = gc_stats();
    let initial_collections = initial_stats.minor_collections.load(Ordering::Relaxed);
    
    // Trigger a minor collection
    let _collected = gc_collect_minor().expect("minor collection should succeed");
    
    let final_stats = gc_stats();
    let final_collections = final_stats.minor_collections.load(Ordering::Relaxed);
    
    // Should have performed one additional collection
    assert_eq!(final_collections - initial_collections, 1);
    // collected is usize, always >= 0, just check it's a valid value
}

#[test]
fn test_major_collection() {
    gc_test_context(|| {
        let initial_stats = gc_stats();
        let initial_collections = initial_stats.major_collections.load(Ordering::Relaxed);
        
        // Trigger a major collection
        let _collected = gc_collect_major().expect("major collection should succeed");
        
        let final_stats = gc_stats();
        let final_collections = final_stats.major_collections.load(Ordering::Relaxed);
        
        // Should have performed one additional collection
        assert_eq!(final_collections - initial_collections, 1);
        // collected is always valid (usize)
    });
}

#[test]
fn test_collection_with_live_objects() {
    // Reset to default configuration
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");
    
    // Allocate some objects and keep references
    let mut live_objects = Vec::new();
    for i in 0..10 {
        let ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        gc_add_root(ptr).expect("root registration should succeed"); // Explicitly protect from collection
        live_objects.push(ptr);
    }
    
    // Allocate some objects without keeping references (potential garbage)
    for i in 0..10 {
        let _ptr = gc_allocate(Value::String(format!("temp_{}", i)))
            .expect("allocation should succeed");
        // These objects become garbage when _ptr goes out of scope
    }
    
    let before_collection = gc_stats();
    let before_allocations = before_collection.total_allocations.load(Ordering::Relaxed);
    
    // Force collection
    let _collected = gc_collect_minor().expect("collection should succeed");
    
    // Live objects should still be accessible
    for (i, ptr) in live_objects.iter().enumerate() {
        assert_eq!(**ptr, Value::Num(i as f64));
    }
    
    // Some garbage should have been collected
    // Note: In the current implementation, this is simulated
    // collected is always valid (usize)
    
    // Total allocations should remain the same (it's a cumulative counter)
    let after_collection = gc_stats();
    let after_allocations = after_collection.total_allocations.load(Ordering::Relaxed);
    assert_eq!(before_allocations, after_allocations);
    
    // Clean up roots
    for ptr in &live_objects {
        gc_remove_root(*ptr).expect("root removal should succeed");
    }
}

#[test]
fn test_collection_frequency() {
    // Configure for frequent collections but with reasonable size
    let config = GcConfig {
        minor_gc_threshold: 0.5, // Moderate threshold
        young_generation_size: 64 * 1024, // 64KB - reasonable size
        ..GcConfig::default()
    };
    
    gc_configure(config).expect("configuration should succeed");
    
    let initial_stats = gc_stats();
    let initial_collections = initial_stats.minor_collections.load(Ordering::Relaxed);
    
    // Allocate many objects to trigger multiple collections
    for i in 0..50 {
        let _ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        
        // Occasionally force collection
        if i % 10 == 0 {
            let _ = gc_collect_minor().expect("collection should succeed");
        }
    }
    
    let final_stats = gc_stats();
    let final_collections = final_stats.minor_collections.load(Ordering::Relaxed);
    
    // Should have triggered multiple collections
    assert!(final_collections > initial_collections);
    assert!(final_collections - initial_collections >= 5); // At least 5 forced collections
}

#[test]
fn test_collection_timing() {
    use std::time::Instant;
    
    // Reset to default configuration
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");
    
    // Allocate some objects
    for i in 0..20 {
        let _ptr = gc_allocate(Value::String(format!("object_{}", i)))
            .expect("allocation should succeed");
    }
    
    // Time a collection
    let start = Instant::now();
    let collected = gc_collect_minor().expect("collection should succeed");
    let duration = start.elapsed();
    
    // Collection should complete relatively quickly
    assert!(duration.as_millis() < 1000); // Less than 1 second
    
    // Should collect some objects (in simulation)
    // collected is always valid (usize)
    
    println!("Collected {} objects in {:?}", collected, duration);
}

#[test]
fn test_collection_with_different_generations() {
    // Reset to default configuration
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");
    
    // This test verifies the generational aspect of collection
    
    // First, allocate some objects (they go to young generation)
    let mut young_objects = Vec::new();
    for i in 0..5 {
        let ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        gc_add_root(ptr).expect("root registration should succeed"); // Protect from collection
        young_objects.push(ptr);
    }
    
    // Perform a minor collection (should promote survivors to next generation)
    let _collected_minor = gc_collect_minor().expect("minor collection should succeed");
    
    // Allocate more objects (these go to young generation)
    for i in 10..15 {
        let _ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
    }
    
    // Perform a major collection (collects all generations)
    let _collected_major = gc_collect_major().expect("major collection should succeed");
    
    // Young objects should still be accessible (they were promoted)
    for (i, ptr) in young_objects.iter().enumerate() {
        assert_eq!(**ptr, Value::Num(i as f64));
    }
    
    // Clean up roots
    for ptr in &young_objects {
        gc_remove_root(*ptr).expect("root removal should succeed");
    }
    
    // collected values are always valid (usize)
}

#[test]
fn test_collection_stats_accuracy() {
    gc_test_context(|| {
        let initial_stats = gc_stats();
        let initial_minor = initial_stats.minor_collections.load(Ordering::Relaxed);
        let initial_major = initial_stats.major_collections.load(Ordering::Relaxed);
        
        // Perform specific numbers of collections
        for _ in 0..3 {
            let _ = gc_collect_minor().expect("minor collection should succeed");
        }
        
        for _ in 0..2 {
            let _ = gc_collect_major().expect("major collection should succeed");
        }
        
        let final_stats = gc_stats();
        let final_minor = final_stats.minor_collections.load(Ordering::Relaxed);
        let final_major = final_stats.major_collections.load(Ordering::Relaxed);
        
        // Verify exact counts
        assert_eq!(final_minor - initial_minor, 3);
        assert_eq!(final_major - initial_major, 2);
    });
}

#[test]
fn test_collection_with_roots() {
    use rustmat_gc::{GlobalRoot, gc_register_root, gc_unregister_root};
    
    // Reset to default configuration
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");
    
    // Create a root with some values
    let root_values = vec![
        Value::Num(100.0),
        Value::String("persistent".to_string()),
    ];
    let root = Box::new(GlobalRoot::new(root_values, "test root".to_string()));
    let root_id = gc_register_root(root).expect("root registration should succeed");
    
    // Allocate some values that will become garbage
    for i in 0..10 {
        let _ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
    }
    
    // Collect garbage
    let _collected = gc_collect_major().expect("collection should succeed");
    
    // Root values should protect objects from collection
    // (In a real implementation, these would be traced from roots)
    // collected is always valid (usize)
    
    // Clean up
    gc_unregister_root(root_id).expect("root unregistration should succeed");
}

#[test]
fn test_allocation_after_collection() {
    // Allocate some objects
    for i in 0..10 {
        let _ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
    }
    
    // Collect garbage
    let _collected = gc_collect_minor().expect("collection should succeed");
    
    // Should still be able to allocate after collection
    let new_ptr = gc_allocate(Value::String("post-collection".to_string()))
        .expect("allocation after collection should succeed");
    
    assert_eq!(*new_ptr, Value::String("post-collection".to_string()));
}

#[cfg(feature = "debug-gc")]
#[test]
fn test_force_collection() {
    use rustmat_gc::gc_force_collect;
    
    // Allocate some objects
    for i in 0..5 {
        let _ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
    }
    
    // Force collection (debug feature)
    let collected = gc_force_collect().expect("forced collection should succeed");
    // collected is always valid (usize)
}

#[test]
fn test_collection_performance() {
    use std::time::Instant;
    
    // Reset to default configuration
    let config = GcConfig::default();
    gc_configure(config).expect("configuration should succeed");
    
    // Allocate a moderate number of objects
    let num_objects = 100;
    let mut objects = Vec::new();
    
    let alloc_start = Instant::now();
    for i in 0..num_objects {
        let ptr = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
        gc_add_root(ptr).expect("root registration should succeed"); // Protect from collection
        objects.push(ptr);
    }
    let alloc_duration = alloc_start.elapsed();
    
    let collect_start = Instant::now();
    let collected = gc_collect_minor().expect("collection should succeed");
    let collect_duration = collect_start.elapsed();
    
    println!("Allocated {} objects in {:?}", num_objects, alloc_duration);
    println!("Collected {} objects in {:?}", collected, collect_duration);
    
    // Allocation should be fast
    assert!(alloc_duration.as_millis() < 100);
    
    // Collection should be reasonably fast
    assert!(collect_duration.as_millis() < 50);
    
    // Objects should still be accessible (they're kept alive by the Vec)
    for (i, ptr) in objects.iter().enumerate() {
        assert_eq!(**ptr, Value::Num(i as f64));
    }
    
    // Clean up roots
    for ptr in &objects {
        gc_remove_root(*ptr).expect("root removal should succeed");
    }
}