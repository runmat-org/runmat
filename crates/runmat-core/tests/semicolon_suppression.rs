use runmat_gc::gc_test_context;
use runmat_core::RunMatSession as ReplEngine;

/// Test basic semicolon suppression behavior
#[test]
fn test_semicolon_suppresses_output() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Expression without semicolon should return a value
    let result = engine.execute("2 + 3").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "5");

    // Expression with semicolon should suppress output
    let result = engine.execute("2 + 3;").unwrap();
    assert!(result.value.is_none()); // No value returned due to semicolon suppression
}

/// Test semicolon suppression with assignments
#[test]
fn test_assignment_with_semicolon() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Assignment without semicolon should return the assigned value
    let result = engine.execute("x = 42").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "42");

    // Assignment with semicolon should suppress output
    let result = engine.execute("y = 42;").unwrap();
    assert!(result.value.is_none()); // No value returned due to semicolon suppression

    // But the variable should still be assigned
    let result = engine.execute("y").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "42");
}

/// Test mixed expressions with and without semicolons
#[test]
fn test_mixed_semicolon_behavior() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Set up some variables
    engine.execute("a = 10;").unwrap(); // No output
    engine.execute("b = 20;").unwrap(); // No output

    let result = engine.execute("a").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "10");
    let result = engine.execute("b").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "20");

    // Expression without semicolon should show result
    let result = engine.execute("a + b").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "30");

    // Same expression with semicolon should suppress output
    let result = engine.execute("a + b;").unwrap();
    assert!(result.value.is_none());
}

/// Test semicolon suppression with function calls
#[test]
fn test_function_call_semicolon_suppression() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Function call without semicolon should return result
    let result = engine.execute("sin(0)").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "0");

    // Function call with semicolon should suppress output
    let result = engine.execute("sin(0);").unwrap();
    assert!(result.value.is_none());
}

/// Test semicolon suppression with matrix operations
#[test]
fn test_matrix_semicolon_suppression() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Matrix creation without semicolon should show result
    let result = engine.execute("[1, 2, 3]").unwrap();
    assert!(result.value.is_some());
    assert!(result.value.unwrap().to_string().contains("1"));

    // Matrix creation with semicolon should suppress output
    let result = engine.execute("[1, 2, 3];").unwrap();
    assert!(result.value.is_none());
}

/// Test semicolon suppression with complex expressions
#[test]
fn test_complex_expression_semicolon_suppression() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Complex expression without semicolon
    let result = engine.execute("(2 + 3) * (4 - 1)").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "15");

    // Same complex expression with semicolon
    let result = engine.execute("(2 + 3) * (4 - 1);").unwrap();
    assert!(result.value.is_none());
}

/// Test that errors are always shown regardless of semicolons
#[test]
fn test_errors_always_shown() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Error without semicolon
    let result = engine.execute("undefined_var");
    assert!(result.is_err() || result.unwrap().error.is_some());

    // Error with semicolon should still be shown
    let result = engine.execute("undefined_var;");
    assert!(result.is_err() || result.unwrap().error.is_some());
}

/// Test that type information is shown for semicolon-suppressed assignments
#[test]
fn test_type_info_display() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Test scalar assignment with semicolon
    let result = engine.execute("x = 42;").unwrap();
    assert!(result.value.is_none());
    assert_eq!(result.type_info, Some("scalar".to_string()));

    // Test matrix assignment with semicolon
    let result = engine.execute("y = [1, 2; 3, 4];").unwrap();
    assert!(result.value.is_none());
    assert_eq!(result.type_info, Some("2x2 matrix".to_string()));

    // Test vector assignment with semicolon
    let result = engine.execute("z = [1, 2, 3];").unwrap();
    assert!(result.value.is_none());
    assert_eq!(result.type_info, Some("1x3 vector".to_string()));

    // Test char array assignment with semicolon
    let result = engine.execute("w = 'hello';").unwrap();
    assert!(result.value.is_none());
    assert_eq!(result.type_info, Some("1x5 char array".to_string()));

    // Test that assignments without semicolon still show values, not type info
    let result = engine.execute("a = 100").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "100");
    assert_eq!(result.type_info, None);
}

/// Test that statement-only constructs (like if/while) are not affected by semicolons
#[test]
fn test_control_flow_not_affected() {
    let mut engine = gc_test_context(ReplEngine::new).unwrap();

    // Set up a variable
    engine.execute("x = 5;").unwrap();

    // Control flow statements shouldn't return values regardless of semicolons
    let result = engine.execute("if x > 0; y = 1; end").unwrap();
    assert!(result.value.is_none()); // Control flow doesn't return values

    let result = engine.execute("if x > 0; y = 2; end;").unwrap();
    assert!(result.value.is_none()); // Still no values

    // But the assignment inside should work
    let result = engine.execute("y").unwrap();
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap().to_string(), "2");
}
