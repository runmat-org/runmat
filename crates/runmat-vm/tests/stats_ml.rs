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

fn has_bool(vars: &[Value], expected: bool) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Bool(value) if *value == expected))
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

#[test]
fn student_t_distribution_surface_executes_from_scripts() {
    let vars = execute_source(
        "p = tcdf(0, 10); q = tcdf(10, 99, 'upper'); d = tpdf(0, 1); x = tinv(0.95, 50); r = trnd(5, 2, 3);",
    )
    .expect("student t distribution script");
    assert!(has_num(&vars, 0.5));
    assert!(has_num(&vars, std::f64::consts::FRAC_1_PI));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - 1.675905).abs() < 1.0e-5)));
    assert!(has_tensor_shape(&vars, &[2, 3]));
}

#[test]
fn grpstats_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [1 10; 3 30; 2 20; 4 40]; g = [2; 1; 2; 1]; [m,s,names] = grpstats(X, g, [\"mean\", \"std\", \"gname\"]); firstMean = m(1,1); secondMean = m(2,2);",
    )
    .expect("grpstats script");
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, 3.5));
    assert!(has_num(&vars, 15.0));
}

#[test]
fn ttest2_surface_executes_from_scripts() {
    let vars = execute_source(
        "x = [1;2;3;4]; y = [2;4;6;8]; [h,p,ci,stats] = ttest2(x, y); tval = stats.tstat; df = stats.df; sd = stats.sd; lo = ci(1); hi = ci(2);",
    )
    .expect("ttest2 script");
    assert!(has_bool(&vars, false));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 0.133_974_596));
    assert!(has_num(&vars, -1.732_050_808));
    assert!(has_num(&vars, 6.0));
    assert!(has_num(&vars, 2.041_241_452));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value + 6.031_813).abs() < 1.0e-6)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - 1.031_813).abs() < 1.0e-6)));
}

#[test]
fn bootstrp_surface_executes_from_scripts() {
    let vars =
        execute_source("rng('default'); x = [1;2;3;4]; [b,s] = bootstrp(4, @mean, x); n = s(1,1);")
            .expect("bootstrp script");
    assert!(has_tensor_shape(&vars, &[4, 1]));
    assert!(has_tensor_shape(&vars, &[4, 4]));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (1.0..=4.0).contains(value))));
}
