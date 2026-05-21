#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_semantic_source;

#[test]
fn closure_simple_no_capture() {
    let vars = execute_semantic_source("f = @(x) x + 1; y = feval(f, 2);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn closure_captures_free_variables() {
    let vars = execute_semantic_source("a=1; b=2; f=@(x) x + a + b; y = feval(f, 3);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
}

#[test]
fn nested_closures_capture_outer() {
    let vars =
        execute_semantic_source("a=10; f=@(x) @(y) (x + y + a); g = feval(f, 2); r = feval(g, 3);")
            .unwrap();
    // Expect 2 + 3 + 10 = 15
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[test]
fn feval_with_string_handle() {
    let vars = execute_semantic_source("r = feval('@max', 2, 5);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn feval_with_string_handle_resolves_local_semantic_function() {
    let vars =
        execute_semantic_source("function y = inc(x); y = x + 1; end; r = feval('@inc', 2);")
            .unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn feval_with_unresolved_string_handle_errors() {
    let err = execute_semantic_source("r = feval('@definitely_missing_callback', 1);")
        .expect_err("unresolved @string handle should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn feval_string_without_at_errors_with_identifier_contract() {
    let err = execute_semantic_source("r = feval('sin', 0);")
        .expect_err("string handle without @ should fail");
    assert_eq!(err.identifier(), Some("RunMat:FevalHandleStringInvalid"));
}

#[test]
fn feval_nonrow_char_handle_errors_with_identifier_contract() {
    let err = execute_semantic_source("r = feval(['@'; 's'], 0);")
        .expect_err("non-row char handle should fail");
    assert_eq!(err.identifier(), Some("RunMat:FevalHandleShapeInvalid"));
}

#[test]
fn str2func_and_func2str_round_trip_for_builtin_handle() {
    let vars = execute_semantic_source("f = str2func('sin'); name = func2str(f); y = feval(f, 0);")
        .unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "sin")));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if n.abs() < 1e-9)));
}

#[test]
fn func2str_non_handle_errors_with_identifier_contract() {
    let err = execute_semantic_source("name = func2str(1);")
        .expect_err("func2str non-handle input should fail");
    assert_eq!(err.identifier(), Some("RunMat:Func2StrHandleTypeInvalid"));
}

#[test]
fn str2func_resolves_local_semantic_function_handle() {
    let vars = execute_semantic_source(
        "function y = inc(x); y = x + 1; end; f = str2func('inc'); r = feval(f, 2);",
    )
    .unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn str2func_unresolved_external_callback_errors_without_legacy_fallback() {
    let err =
        execute_semantic_source("f = str2func('definitely_missing_callback'); y = feval(f, 1);")
            .expect_err("unresolved str2func callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_unresolved_external_callback_zero_output_errors_without_legacy_fallback() {
    let err = execute_semantic_source("f = str2func('definitely_missing_callback'); feval(f, 1);")
        .expect_err("unresolved str2func callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_qualified_external_callback_errors_without_legacy_fallback() {
    let err = execute_semantic_source("f = str2func('pkg.remote_inc'); y = feval(f, 1);")
        .expect_err("unresolved qualified str2func callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_qualified_external_callback_zero_output_errors_without_legacy_fallback() {
    let err = execute_semantic_source("f = str2func('pkg.remote_inc'); feval(f, 1);")
        .expect_err("unresolved qualified str2func callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_qualified_external_expand_callback_errors_without_legacy_fallback() {
    let err = execute_semantic_source(
        "f = str2func('pkg.remote_inc'); C = deal(1,2); y = feval(f, C{:});",
    )
    .expect_err("unresolved qualified str2func expanded callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_qualified_external_direct_call_errors_without_legacy_fallback() {
    let err = execute_semantic_source("f = str2func('pkg.remote_inc'); y = f(1);")
        .expect_err("unresolved qualified str2func direct call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn str2func_empty_name_errors_with_identifier_contract() {
    let err = execute_semantic_source("f = str2func('');")
        .expect_err("empty str2func function name should fail");
    assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
}

#[test]
fn str2func_nontext_name_type_errors_with_identifier_contract() {
    let err = execute_semantic_source("f = str2func(1);")
        .expect_err("non-text str2func function name should fail");
    assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
}

#[test]
fn str2func_nonrow_char_name_errors_with_identifier_contract() {
    let err = execute_semantic_source("f = str2func(['a'; 'b']);")
        .expect_err("non-row char-array str2func function name should fail");
    assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
}

#[test]
fn fzero_accepts_anonymous_function() {
    let vars = execute_semantic_source("f = @(x) cos(x) - x; r = fzero(f, 0.5);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 0.7390851332).abs() < 1e-6)));
}

#[test]
fn fzero_accepts_optimset_options() {
    let vars = execute_semantic_source(
        "opts = optimset('TolX', 1e-10, 'Display', 'off'); r = fzero(@sin, [3 4], opts);",
    )
    .unwrap();
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Num(n) if (*n - std::f64::consts::PI).abs() < 1e-8)
    ));
}

#[test]
fn fsolve_accepts_anonymous_vector_function() {
    let vars = execute_semantic_source(
        "F = @(x) [x(1)^2 + x(2)^2 - 4; x(1)*x(2) - 1]; x = fsolve(F, [1; 1]);",
    )
    .unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(t) = v {
            t.data.len() == 2
                && (t.data[0] * t.data[0] + t.data[1] * t.data[1] - 4.0).abs() < 1e-5
                && (t.data[0] * t.data[1] - 1.0).abs() < 1e-5
        } else {
            false
        }
    }));
}

#[test]
fn ode45_accepts_anonymous_rhs_function() {
    let vars = execute_semantic_source("y = ode45(@(t, y) -y, [0 1], 1);").unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(t) = v {
            t.cols() == 1 && (t.data[t.rows() - 1] - (-1.0_f64).exp()).abs() < 1e-2
        } else {
            false
        }
    }));
}

#[test]
fn ode23_accepts_two_output_assignment() {
    let vars =
        execute_semantic_source("[t, y] = ode23(@(t, y) -2*y, [0 0.25 0.5 1.0], 1);").unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(tensor) = v {
            tensor.cols() == 1
                && tensor.rows() == 4
                && (tensor.data[tensor.rows() - 1] - (-2.0_f64).exp()).abs() < 2e-2
        } else {
            false
        }
    }));
}
