use rustmat_gc::gc_test_context;
use rustmat_repl::ReplEngine;

fn main() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();
    
    println!("Testing: y = 42;");
    let result = engine.execute("y = 42;").unwrap();
    println!("Result: {:?}", result);
    println!("Has value: {}", result.value.is_some());
    if let Some(v) = result.value {
        println!("Value: {}", v);
    }
}
