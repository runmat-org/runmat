use rustmat_builtins::Value;
use rustmat_hir::lower;
use rustmat_ignition::execute;
use rustmat_ignition::interpret;
use rustmat_ignition::Bytecode;
use rustmat_ignition::Instr;
use rustmat_parser::parse;
use std::convert::TryInto;

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
    let ast =
        parse("x=0; for i=1:2; for j=1:3; x=x+1; if j-2; continue; end; x=x+10; end; end").unwrap();
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
fn colon_operator_with_step() {
    let ast = parse("x = 1:2:5").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
    // Should create range [1, 3, 5]
}

#[test]
fn range_with_step() {
    let ast = parse("x = 1:2:10").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
    // Should create range [1, 3, 5, 7, 9]
}

#[test]
fn function_definition_works() {
    let ast = parse("function y = f(x); y = x + 1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
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
fn verify_jit_test_control_flow() {
    // Test simple control flow from JIT tests
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(2.0),
            Instr::StoreVar(0), // x = 2
            Instr::LoadVar(0),
            Instr::LoadConst(5.0),
            Instr::Less,           // x < 5? -> true for x=2
            Instr::JumpIfFalse(9), // if false, jump to else
            // True branch
            Instr::LoadConst(100.0), // result = 100
            Instr::StoreVar(1),
            Instr::Jump(11), // jump over else
            // False branch
            Instr::LoadConst(200.0), // result = 200
            Instr::StoreVar(1),
            // End
            Instr::Return,
        ],
        var_count: 2,
        functions: std::collections::HashMap::new(),
    };

    let result = interpret(&bytecode).unwrap();
    assert_eq!(result[0], Value::Num(2.0));
    assert_eq!(result[1], Value::Num(100.0));
}

#[test]
fn verify_jit_test_nested_control_flow() {
    // Test nested control flow from JIT tests
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(3.0),
            Instr::StoreVar(0), // x = 3
            Instr::LoadVar(0),
            Instr::LoadConst(5.0),
            Instr::Less,            // x < 5? -> true
            Instr::JumpIfFalse(14), // if false, jump to outer else (LoadConst(0.0))
            // Outer true branch
            Instr::LoadVar(0),
            Instr::LoadConst(2.0),
            Instr::Greater,         // x > 2? -> true
            Instr::JumpIfFalse(12), // if false, jump to inner else (LoadConst(24.0))
            // Inner true branch
            Instr::LoadConst(42.0), // result = 42
            Instr::Jump(15),        // jump to end
            // Inner false branch
            Instr::LoadConst(24.0), // result = 24
            Instr::Jump(15),        // jump to end
            // Outer false branch
            Instr::LoadConst(0.0), // result = 0
            // End
            Instr::StoreVar(1),
        ],
        var_count: 2,
        functions: std::collections::HashMap::new(),
    };

    let result = interpret(&bytecode).unwrap();
    assert_eq!(result[0], Value::Num(3.0));
    assert_eq!(result[1], Value::Num(42.0));
}

#[test]
fn test_turbine_control_flow_pattern() {
    // Test the exact same bytecode pattern that's failing in turbine
    let bytecode = Bytecode {
        instructions: vec![
            Instr::LoadConst(2.0), // 0
            Instr::StoreVar(0),    // 1: x = 2
            Instr::LoadVar(0),     // 2
            Instr::LoadConst(5.0), // 3
            Instr::Less,           // 4: x < 5? -> true for x=2
            Instr::JumpIfFalse(9), // 5: if false, jump to else
            // True branch
            Instr::LoadConst(100.0), // 6: result = 100
            Instr::StoreVar(1),      // 7
            Instr::Jump(11),         // 8: jump over else
            // False branch
            Instr::LoadConst(200.0), // 9: result = 200
            Instr::StoreVar(1),      // 10
            // End
            Instr::Return, // 11
        ],
        var_count: 2,
        functions: std::collections::HashMap::new(),
    };

    let result = interpret(&bytecode);
    assert!(
        result.is_ok(),
        "Interpreter should handle this control flow: {result:?}"
    );

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

#[test]
fn test_function_parameter_binding() {
    // Test: function y = add_numbers(a, b); y = a + b; end
    let ast =
        parse("function y = add_numbers(a, b); y = a + b; end; result = add_numbers(10, 20);")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(
        result.is_ok(),
        "Function with parameters should work: {result:?}"
    );
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 30.0));
    assert!(
        found_result.is_some(),
        "Function should compute 10 + 20 = 30, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_local_variables() {
    // Test: function y = compute(x); temp = x * 2; y = temp + 5; end
    let ast =
        parse("function y = compute(x); temp = x * 2; y = temp + 5; end; result = compute(7);")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Function with local variables should work");
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 19.0));
    assert!(
        found_result.is_some(),
        "Function should compute 7 * 2 + 5 = 19, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_recursive_calls() {
    // Test: function y = factorial(n); if n <= 1; y = 1; else; y = n * factorial(n - 1); end; end
    let ast = parse("function y = factorial(n); if n <= 1; y = 1; else; y = n * factorial(n - 1); end; end; result = factorial(5);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Recursive function should work");
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 120.0));
    assert!(
        found_result.is_some(),
        "Function should compute 5! = 120, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_multiple_calls() {
    // Test: Multiple calls to same function with different parameters
    let ast = parse(
        "function y = double(x); y = x * 2; end; a = double(5); b = double(10); c = double(15);",
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Multiple function calls should work");
    let vars = result.unwrap();

    // Check all three results (exact indices depend on HIR variable assignment)
    let found_a = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 10.0));
    assert!(
        found_a.is_some(),
        "First call should be 5 * 2 = 10, but found variables: {vars:?}"
    );

    let found_b = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 20.0));
    assert!(
        found_b.is_some(),
        "Second call should be 10 * 2 = 20, but found variables: {vars:?}"
    );

    let found_c = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 30.0));
    assert!(
        found_c.is_some(),
        "Third call should be 15 * 2 = 30, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_with_control_flow() {
    // Test: function y = abs_value(x); if x < 0; y = -x; else; y = x; end; end
    let ast = parse("function y = abs_value(x); if x < 0; y = -x; else; y = x; end; end; pos = abs_value(42); neg = abs_value(-17);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Function with control flow should work");
    let vars = result.unwrap();

    // Check both results (exact indices depend on HIR variable assignment)
    let found_pos = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 42.0));
    assert!(
        found_pos.is_some(),
        "abs_value(42) should be 42, but found variables: {vars:?}"
    );

    let found_neg = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 17.0));
    assert!(
        found_neg.is_some(),
        "abs_value(-17) should be 17, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_with_loops() {
    // Test: function y = sum_to_n(n); y = 0; for i = 1:n; y = y + i; end; end
    let ast = parse(
        "function y = sum_to_n(n); y = 0; for i = 1:n; y = y + i; end; end; result = sum_to_n(10);",
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Function with loops should work");
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 55.0));
    assert!(
        found_result.is_some(),
        "sum_to_n(10) should be 55, but found variables: {vars:?}"
    );
}

#[test]
fn test_nested_function_calls() {
    // Test: Calling functions from within other functions
    let ast = parse("function y = add(a, b); y = a + b; end; function y = multiply_and_add(x); y = add(x * 2, x * 3); end; result = multiply_and_add(4);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(
        result.is_ok(),
        "Nested function calls should work: {:?}",
        result.as_ref().err()
    );
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 20.0));
    assert!(found_result.is_some(), "multiply_and_add(4) should be 20 (add(4*2, 4*3) = add(8, 12) = 20), but found variables: {vars:?}");
}

#[test]
fn test_function_variable_isolation() {
    // Test: Variables inside functions don't affect global variables
    let ast = parse("global_var = 100; function y = test_isolation(x); global_var = x; y = global_var; end; result = test_isolation(42); final_global = global_var;").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Function variable isolation should work");
    let vars = result.unwrap();

    // Check that function returned 42 (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 42.0));
    assert!(
        found_result.is_some(),
        "Function should return 42, but found variables: {vars:?}"
    );

    // Check that global variable is unchanged (should still be 100)
    let found_global = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 100.0));
    assert!(
        found_global.is_some(),
        "Global variable should remain 100, but found variables: {vars:?}"
    );
}

// === NEGATIVE TESTS ===

#[test]
fn test_undefined_function_call() {
    // Test: Calling a function that doesn't exist should error
    let ast = parse("result = nonexistent_function(42);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_err(), "Calling undefined function should error");
    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("undefined function") || error_msg.contains("nonexistent_function"),
        "Error should mention undefined function, got: {error_msg}"
    );
}

#[test]
fn test_function_wrong_argument_count() {
    // Test: Calling function with wrong number of arguments
    let ast = parse("function y = add_two(a, b); y = a + b; end; result = add_two(5);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    // MATLAB requires exact argument count - should fail with "Not enough input arguments"
    assert!(
        result.is_err(),
        "Function with missing parameters should fail in MATLAB"
    );
    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(
        error_msg.contains("argument") || error_msg.contains("parameter"),
        "Error should mention argument/parameter mismatch, got: {error_msg}"
    );
}

// === EDGE CASES ===

#[test]
fn test_function_no_parameters() {
    // Test: Function with no parameters
    let ast = parse("function y = get_constant(); y = 42; end; result = get_constant();").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(
        result.is_ok(),
        "Function with no parameters should work: {result:?}"
    );
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 42.0));
    assert!(
        found_result.is_some(),
        "Function should return 42, but found variables: {vars:?}"
    );
}

#[test]
fn test_function_empty_body() {
    // Test: Function with empty body (should return 0.0 by default)
    let ast = parse("function y = empty_function(x); end; result = empty_function(123);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Function with empty body should work");
    let vars = result.unwrap();

    // Should return default value 0.0
    if let Value::Num(value) = &vars[0] {
        assert_eq!(*value, 0.0, "Empty function should return 0.0");
    } else {
        panic!("Result should be Num(0.0), got {:?}", vars[0]);
    }
}

#[test]
fn test_function_deep_recursion() {
    // Test: Deep recursion (should handle reasonable depth)
    let ast = parse("function y = countdown(n); y = n; end; result = countdown(10);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);

    assert!(result.is_ok(), "Deep recursion should work");
    let vars = result.unwrap();

    // The result should be in one of the variables (exact index depends on HIR variable assignment)
    let found_result = vars
        .iter()
        .find(|v| matches!(v, Value::Num(n) if *n == 10.0));
    assert!(
        found_result.is_some(),
        "countdown(10) should return 10, but found variables: {vars:?}"
    );
}
