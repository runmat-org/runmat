#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

#[test]
fn fminunc_solves_quadratic_from_source() {
    let vars = execute_source(
        "fun = @(x) sum((x - [1; 2; 3]).^2); [x, fval, exitflag] = fminunc(fun, [0; 0; 0]); y = x(2);",
    )
    .unwrap();
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(t) if t.shape == vec![3, 1] && (t.materialize_f64()[0] - 1.0).abs() < 1.0e-4 && (t.materialize_f64()[1] - 2.0).abs() < 1.0e-4 && (t.materialize_f64()[2] - 3.0).abs() < 1.0e-4)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if n.abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 2.0).abs() < 1.0e-4)));
}

#[test]
fn optimization_integer_boundaries_execute_from_compiled_source() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "b = fminbnd(@(x) (x-2).^2, int16(0), uint16(5), struct('MaxIter',uint8(100))); if abs(b-2) > 1e-4; error('compiled fminbnd integer boundary failed'); end; u = fminunc(@(x) (x-3).^2, int8(0), struct('MaxIter',uint16(100))); if abs(u-3) > 1e-4; error('compiled fminunc integer boundary failed'); end; [s,residual,flag,output,jacobian] = fsolve(@(x) x-4, uint8(0), struct('MaxIter',uint16(100))); if abs(s-4) > 1e-5 || abs(residual) > 1e-5 || flag <= 0 || output.iterations < 0 || abs(jacobian-1) > 1e-4; error('compiled fsolve integer boundary failed'); end; [z,zval,zflag] = fzero(@(x) x-5, single([0])); if ~strcmp(class(z),'single'); error('compiled fzero root class failed'); end; if ~strcmp(class(zval),'single'); error('compiled fzero value class failed'); end; if ~strcmp(class(zflag),'single'); error('compiled fzero flag class failed'); end; if abs(double(z)-5) > 1e-4; error('compiled fzero value failed'); end;",
    )
    .expect("compiled optimization integer semantics");
}

#[test]
fn linprog_solves_bounded_program_from_source() {
    let vars = execute_source(
        "f = [-1; -2]; A = [1 1]; b = 4; [x, fval, exitflag] = linprog(f, A, b, [], [], [0; 0], []); y = x(2);",
    )
    .unwrap();
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(t) if t.shape == vec![2, 1] && (t.materialize_f64()[0] - 0.0).abs() < 1.0e-7 && (t.materialize_f64()[1] - 4.0).abs() < 1.0e-7)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n + 8.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 4.0).abs() < 1.0e-7)));
}

#[test]
fn linprog_solves_equality_and_bounds_from_source() {
    let vars = execute_source(
        "f = [1; 2]; A = []; b = []; Aeq = [1 1]; beq = 3; [x, fval] = linprog(f, A, b, Aeq, beq, [1; 0], []);",
    )
    .unwrap();
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(t) if t.shape == vec![2, 1] && (t.materialize_f64()[0] - 3.0).abs() < 1.0e-7 && t.materialize_f64()[1].abs() < 1.0e-7)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1.0e-7)));
}

#[test]
fn linprog_solves_sparse_one_sided_bound_from_source() {
    let vars = execute_source(
        "f = [1; 0]; [x, fval, exitflag] = linprog(f, [], [], [], [], [2; -Inf], []); y = x(1);",
    )
    .unwrap();
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(t) if t.shape == vec![2, 1] && (t.materialize_f64()[0] - 2.0).abs() < 1.0e-7 && t.materialize_f64()[1].abs() < 1.0e-7)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 2.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-7)));
}

#[test]
fn linprog_optimizes_along_equality_face_from_source() {
    let vars = execute_source(
        "f = [-1; 0; 0]; A = [1 0 0]; b = 1; Aeq = [0 0 1]; beq = 0; [x, fval, exitflag] = linprog(f, A, b, Aeq, beq, [], []); y = x(1);",
    )
    .unwrap();
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(t) if t.shape == vec![3, 1] && (t.materialize_f64()[0] - 1.0).abs() < 1.0e-7 && t.materialize_f64()[1].abs() < 1.0e-7 && t.materialize_f64()[2].abs() < 1.0e-7)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n + 1.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-7)));
}

#[test]
fn linprog_reports_infeasible_status_from_source() {
    let vars =
        execute_source("[x, fval, exitflag, output] = linprog(1, [], [], [], [], 2, 1);").unwrap();
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Tensor(t) if t.shape == vec![0, 0])));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n + 2.0).abs() < 1.0e-7)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Struct(s) if s.fields.contains_key("message"))));
}
