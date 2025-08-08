use runmat_repl::ReplEngine;
use runmat_gc::gc_test_context;

fn main() {
    gc_test_context(|| {
        let mut engine = ReplEngine::new().unwrap();
        
        // Execute the same operation multiple times to potentially trigger JIT
        for i in 1..=5 {
            let input = format!("x{} = {} + {}", i, i, i);
            let result = engine.execute(&input);
            println!("Iteration {}: result = {:?}", i, result);
            let stats = engine.stats();
            println!("  Stats: total={}, jit={}, interpreter={}, sum={}", 
                stats.total_executions, stats.jit_compiled, stats.interpreter_fallback,
                stats.jit_compiled + stats.interpreter_fallback);
        }
        
        let final_stats = engine.stats();
        println!("Final stats: total={}, jit={}, interpreter={}, sum={}", 
            final_stats.total_executions, final_stats.jit_compiled, final_stats.interpreter_fallback,
            final_stats.jit_compiled + final_stats.interpreter_fallback);
    });
}
