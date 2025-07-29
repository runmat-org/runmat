use rustmat_ignition::execute;
use rustmat_hir::lower;
use rustmat_parser::parse;
use std::convert::TryInto;
use rustmat_builtins::Value;
use rustmat_ignition::Bytecode;
use rustmat_ignition::Instr;
use rustmat_ignition::interpret;

#[test]
fn arithmetic_and_assignment() {
    let input = "x = 1 + 2; y = x * x";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 3.0);
    assert_eq!(y, 9.0);
}

#[test]
fn while_loop_decrements() {
    let ast = parse("x=3; y=0; while x; y=y+1; x=x-1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 0.0); // x
    assert_eq!(y, 3.0); // y
}

#[test]
fn for_loop_sum() {
    let ast = parse("s=0; for i=1:4; s=s+i; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 10.0);
}

#[test]
fn continue_in_loop() {
    let ast = parse("x=0; for i=1:3; if i-2; x=x+i; else; continue; end; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 4.0);
}

#[test]
fn break_in_loop() {
    let ast = parse("x=0; while 1; x=x+1; break; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn continue_outside_loop_errors() {
    let ast = parse("continue").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "continue outside loop");
}

#[test]
fn break_outside_loop_errors() {
    let ast = parse("break").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "break outside loop");
}

#[test]
fn nested_loops_break_only_inner() {
    let ast = parse("x=0; for i=1:2; for j=1:3; x=x+1; if j-1; break; end; end; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 4.0);
}

#[test]
fn nested_loops_continue_only_inner() {
    let ast = parse("x=0; for i=1:2; for j=1:3; x=x+1; if j-2; continue; end; x=x+10; end; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 26.0);
}

#[test]
fn return_statement_halts_execution() {
    let ast = parse("x=1; return; x=2").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    // Return statement now properly halts execution
    assert_eq!(x, 1.0);
}

#[test]
fn elseif_executes_correct_branch() {
    let ast = parse("x=2; if x-2; y=1; elseif x-1; y=2; else; y=3; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 2.0);
}

#[test]
fn multiple_elseif_branches() {
    let ast = parse("x=3; if x-3; y=1; elseif x-3; y=2; elseif x-2; y=3; else; y=4; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 3.0);
}

#[test]
fn else_branch_when_all_false() {
    let ast = parse("x=0; if x; y=1; elseif x; y=2; else; y=3; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 3.0);
}

#[test]
fn for_loop_start_greater_than_end() {
    let ast = parse("x=0; for i=5:1; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn colon_operator_errors() {
    let ast = parse("x = 1:2:5").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_err());
}

#[test]
fn range_with_step_errors() {
    let ast = parse("x = 1:2:10").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_err());
}

#[test]
fn function_definition_errors() {
    let ast = parse("function y = f(x); y = x + 1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_err());
}

#[test]
fn function_call_with_builtin() {
    // Test that function calls are parsed and would call the dispatcher
    // This tests the infrastructure even though we don't have actual builtins registered in this test
    let ast = parse("x = add(1, 2)").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    // This should fail with "unknown builtin" since add isn't registered in this test context
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unknown builtin"));
}

#[test]
fn matrix_operations_basic() {
    // Test that basic matrix syntax works with actual values
    let ast = parse("x = [1, 2]").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
    
    // Matrix should contain the actual values, not just zeros
    let vars = result.unwrap();
    if let Value::Matrix(matrix) = &vars[0] {
        assert_eq!(matrix.rows, 1);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0, 2.0]);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn matrix_2d_compilation() {
    // Test 2D matrix with proper element ordering
    let ast = parse("x = [1, 2; 3, 4]").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
    
    let vars = result.unwrap();
    if let Value::Matrix(matrix) = &vars[0] {
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        // Should be in row-major order: [1, 2, 3, 4]
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
        // Verify individual element access
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 2.0);
        assert_eq!(matrix.get(1, 0).unwrap(), 3.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 4.0);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn test_turbine_control_flow_pattern() {
    // Test the exact same bytecode pattern that's failing in turbine
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(2.0),      // 0
            Instr::StoreVar(0),         // 1: x = 2
            Instr::LoadVar(0),          // 2
            Instr::LoadConst(5.0),      // 3
            Instr::Less,                // 4: x < 5? -> true for x=2
            Instr::JumpIfFalse(9),      // 5: if false, jump to else
            // True branch
            Instr::LoadConst(100.0),    // 6: result = 100
            Instr::StoreVar(1),         // 7
            Instr::Jump(11),            // 8: jump over else
            // False branch  
            Instr::LoadConst(200.0),    // 9: result = 200
            Instr::StoreVar(1),         // 10
            // End
            Instr::Return,              // 11
        ],
        var_count: 2,
    };
    
    let result = interpret(&bytecode);
    assert!(result.is_ok(), "Interpreter should handle this control flow: {:?}", result);
    
    let vars = result.unwrap();
    assert_eq!(vars.len(), 2);
    
    // x should be 2.0
    if let Value::Num(x) = &vars[0] {
        assert_eq!(*x, 2.0);
    } else {
        panic!("Variable 0 should be Num(2.0), got {:?}", vars[0]);
    }
    
    // result should be 100.0 (true branch executed since 2 < 5)
    if let Value::Num(result) = &vars[1] {
        assert_eq!(*result, 100.0);
    } else {
        panic!("Variable 1 should be Num(100.0), got {:?}", vars[1]);
    }
}
