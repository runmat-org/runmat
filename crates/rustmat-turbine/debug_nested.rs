use rustmat_turbine::TurbineEngine;
use rustmat_ignition::{Bytecode, Instr};
use rustmat_builtins::Value;

fn main() {
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(3.0),
            Instr::StoreVar(0),         // x = 3
            Instr::LoadVar(0),
            Instr::LoadConst(5.0),
            Instr::Less,                // x < 5? -> true
            Instr::JumpIfFalse(14),     // if false, jump to outer else (LoadConst(0.0))
            // Outer true branch
            Instr::LoadVar(0),
            Instr::LoadConst(2.0),
            Instr::Greater,             // x > 2? -> true
            Instr::JumpIfFalse(12),     // if false, jump to inner else (LoadConst(24.0))
            // Inner true branch
            Instr::LoadConst(42.0),     // result = 42
            Instr::Jump(15),            // jump to end
            // Inner false branch
            Instr::LoadConst(24.0),     // result = 24
            Instr::Jump(15),            // jump to end
            // Outer false branch
            Instr::LoadConst(0.0),      // result = 0
            // End
            Instr::StoreVar(1),
        ],
        var_count: 2,
    };
    
    println!("Testing with pure interpreter:");
    let result = rustmat_ignition::interpret(&bytecode);
    println!("Interpreter result: {:?}", result);
    
    if TurbineEngine::is_jit_supported() {
        let mut engine = TurbineEngine::new().unwrap();
        let mut vars = vec![Value::Num(0.0), Value::Num(0.0)];
        println!("Testing with JIT engine:");
        let result = engine.execute_or_compile(&bytecode, &mut vars);
        println!("JIT result: {:?}", result);
        println!("Vars: {:?}", vars);
    }
}
