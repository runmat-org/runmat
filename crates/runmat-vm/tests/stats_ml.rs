use runmat_builtins::{Tensor, Value};

#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

fn has_num(vars: &[Value], expected: f64) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - expected).abs() < 1.0e-8))
}

fn has_tensor_shape(vars: &[Value], shape: &[usize]) -> bool {
    vars.iter().any(|value| match value {
        Value::Tensor(Tensor {
            shape: tensor_shape,
            ..
        }) => tensor_shape == shape,
        _ => false,
    })
}

#[test]
fn lasso_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0 1; 1 1; 2 1; 3 1; 4 1]; y = [1; 3; 5; 7; 9]; [B,FitInfo] = lasso(X, y, 'Lambda', 0, 'Standardize', false); slope = B(1); intercept = FitInfo.Intercept(1);",
    )
    .expect("lasso script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 1.0));
}
