use rustmat_turbine::{TurbineEngine, HotspotProfiler, FunctionCache};
use rustmat_ignition::{Bytecode, Instr};
use cranelift::prelude::isa::CallConv;

#[test]
fn test_turbine_engine_creation() {
    let engine = TurbineEngine::new();
    assert!(engine.is_ok());
    
    let engine = engine.unwrap();
    let stats = engine.stats();
    assert_eq!(stats.compiled_functions, 0);
    assert_eq!(stats.total_compilations, 0);
}

#[test]
fn test_hotspot_profiler() {
    let mut profiler = HotspotProfiler::new();
    
    // Function should not be hot initially
    assert!(!profiler.is_hot(123));
    assert_eq!(profiler.get_hotness(123), 0);
    
    // Record several executions
    for _ in 0..5 {
        profiler.record_execution(123);
    }
    
    assert_eq!(profiler.get_hotness(123), 5);
    assert!(!profiler.is_hot(123)); // Not hot yet (threshold is 10)
    
    // Make it hot
    for _ in 0..6 {
        profiler.record_execution(123);
    }
    
    assert!(profiler.is_hot(123));
    assert_eq!(profiler.get_hotness(123), 11);
    
    let stats = profiler.stats();
    assert_eq!(stats.total_functions, 1);
    assert_eq!(stats.hot_functions, 1);
    assert_eq!(stats.total_executions, 11);
}

#[test]
fn test_hotspot_profiler_multiple_functions() {
    let mut profiler = HotspotProfiler::new();
    
    // Create multiple functions with different hotness
    for _ in 0..15 {
        profiler.record_execution(100); // Very hot
    }
    
    for _ in 0..8 {
        profiler.record_execution(200); // Not quite hot
    }
    
    for _ in 0..12 {
        profiler.record_execution(300); // Hot
    }
    
    assert!(profiler.is_hot(100));
    assert!(!profiler.is_hot(200));
    assert!(profiler.is_hot(300));
    
    let hottest = profiler.get_hottest_functions(2);
    assert_eq!(hottest.len(), 2);
    assert_eq!(hottest[0], (100, 15)); // Hottest first
    assert_eq!(hottest[1], (300, 12)); // Second hottest
}

#[test]
fn test_function_cache() {
    let mut cache = FunctionCache::with_capacity(2);
    
    // Cache should be empty initially
    assert!(!cache.contains(123));
    assert_eq!(cache.stats().size, 0);
    
    // Insert a dummy compiled function
    let dummy_func = rustmat_turbine::CompiledFunction {
        ptr: std::ptr::null(),
        signature: cranelift::prelude::Signature::new(CallConv::SystemV),
        hotness: 10,
    };
    
    cache.insert(123, dummy_func);
    assert!(cache.contains(123));
    assert_eq!(cache.stats().size, 1);
    
    // Test cache retrieval
    let retrieved = cache.get(123);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().hotness, 10);
}

#[test]
fn test_function_cache_eviction() {
    let mut cache = FunctionCache::with_capacity(2);
    
    let dummy_func1 = rustmat_turbine::CompiledFunction {
        ptr: std::ptr::null(),
        signature: cranelift::prelude::Signature::new(CallConv::SystemV),
        hotness: 5,
    };
    
    let dummy_func2 = rustmat_turbine::CompiledFunction {
        ptr: std::ptr::null(),
        signature: cranelift::prelude::Signature::new(CallConv::SystemV),
        hotness: 8,
    };
    
    let dummy_func3 = rustmat_turbine::CompiledFunction {
        ptr: std::ptr::null(),
        signature: cranelift::prelude::Signature::new(CallConv::SystemV),
        hotness: 12,
    };
    
    // Fill cache to capacity
    cache.insert(100, dummy_func1);
    cache.insert(200, dummy_func2);
    assert_eq!(cache.stats().size, 2);
    
    // Access the first function to make it more recently used
    cache.get(100);
    
    // Insert third function - should evict least recently used (200)
    cache.insert(300, dummy_func3);
    assert_eq!(cache.stats().size, 2);
    assert!(cache.contains(100)); // Should still be there (recently accessed)
    assert!(!cache.contains(200)); // Should be evicted
    assert!(cache.contains(300)); // New function should be there
}

#[test]
fn test_bytecode_compilation() {
    let mut engine = TurbineEngine::new().unwrap();
    
    // Create simple bytecode
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(1.0),
            Instr::LoadConst(2.0),
            Instr::Add,
        ],
        var_count: 0,
    };
    
    // Initially should not compile (not hot)
    let hash = engine.calculate_bytecode_hash(&bytecode);
    assert!(!engine.should_compile(hash));
    
    // Make it hot by calling multiple times
    for _ in 0..10 {
        engine.should_compile(hash);
    }
    assert!(engine.should_compile(hash));
    
    // Compile the bytecode
    let result = engine.compile_bytecode(&bytecode);
    assert!(result.is_ok());
    
    let stats = engine.stats();
    assert_eq!(stats.compiled_functions, 1);
}

#[test]
fn test_profiler_reset() {
    let mut profiler = HotspotProfiler::new();
    
    // Add some data
    for _ in 0..15 {
        profiler.record_execution(123);
    }
    
    assert_eq!(profiler.total_executions(), 15);
    assert!(profiler.is_hot(123));
    
    // Reset should clear everything
    profiler.reset();
    assert_eq!(profiler.total_executions(), 0);
    assert!(!profiler.is_hot(123));
    assert_eq!(profiler.get_hotness(123), 0);
} 