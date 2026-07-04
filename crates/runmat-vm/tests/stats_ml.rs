use runmat_builtins::{LogicalArray, Tensor, Value};

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

fn has_logical_shape(vars: &[Value], shape: &[usize]) -> bool {
    vars.iter().any(|value| match value {
        Value::LogicalArray(LogicalArray {
            shape: logical_shape,
            ..
        }) => logical_shape == shape,
        _ => false,
    })
}

fn has_logical_true_at(vars: &[Value], index: usize) -> bool {
    vars.iter().any(|value| match value {
        Value::LogicalArray(LogicalArray { data, .. }) => {
            data.get(index).copied().unwrap_or(0) != 0
        }
        _ => false,
    })
}

fn disable_interactive_plots_for_test() -> runmat_runtime::builtins::plotting::PlotTestLockGuard {
    let guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    runmat_runtime::builtins::plotting::reset_plot_state();
    guard
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
fn ridge_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3]; y = [1; 3; 5; 7]; B = ridge(y, X, [0 1], 0); b0 = B(1,1); b1 = B(2,1); shrink = B(2,2);",
    )
    .expect("ridge script");
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 1.5));
}

#[test]
fn fitlm_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3]; y = [1; 3; 5; 7]; mdl = fitlm(X, y); coefs = mdl.Coefficients.Estimate; b0 = coefs(1); b1 = coefs(2); yhat = predict(mdl, [4; 5]); p1 = yhat(1); p2 = yhat(2); r2 = mdl.Rsquared.Ordinary;",
    )
    .expect("fitlm script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 9.0));
    assert!(has_num(&vars, 11.0));
}

#[test]
fn fitctree_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3]; y = [0; 0; 1; 1]; mdl = fitctree(X, y, 'MaxNumSplits', 1, 'MinParentSize', 2); [label,score,node,cnum] = predict(mdl, [0.5; 2.5]); a = label(1); b = label(2); s11 = score(1,1); n1 = node(1); c2 = cnum(2);",
    )
    .expect("fitctree script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, 0.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
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

#[test]
fn ecdf_and_cdfplot_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "y = [3;1;2;2]; [f,x] = ecdf(y); f2 = f(2); x3 = x(3); [h,stats] = cdfplot([1;2;3]); mn = stats.mean; md = stats.median;",
    )
    .expect("ecdf and cdfplot script");
    assert!(has_tensor_shape(&vars, &[4, 1]));
    assert!(has_num(&vars, 0.25));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 2.0));
}

#[test]
fn outlier_cleanup_surface_executes_from_scripts() {
    let vars = execute_source(
        "A = [1;2;100;4;5]; [tf,L,U,C] = isoutlier(A); B = filloutliers(A, 'linear'); C2 = filloutliers(A, -1); b3 = B(3); c3 = C2(3);",
    )
    .expect("outlier cleanup script");
    assert!(has_logical_shape(&vars, &[5, 1]));
    assert!(has_logical_true_at(&vars, 2));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, -1.0));
}

#[test]
fn lhsdesign_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng('default'); X = lhsdesign(6, 3, 'Smooth', 'off', 'Criterion', 'correlation', 'Iterations', 3); s = size(X); first = X(1,1);",
    )
    .expect("lhsdesign script");
    assert!(has_tensor_shape(&vars, &[6, 3]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Num(value) if *value > 0.0 && *value < 1.0) }));
}

#[test]
fn distance_helper_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0 0; 3 4; 4 0; 0 2]; D = pdist(X); M = squareform(D); D2 = squareform(M); Y = [1 0; 3 0]; P = pdist2(X, Y, 'squaredeuclidean'); [N,I] = pdist2(X, Y, 'euclidean', 'Smallest', 1); d12 = D(1); d14 = D(3); m14 = M(1,4); p22 = P(2,2); n2 = N(2); i2 = I(2);",
    )
    .expect("distance helper script");
    assert!(has_tensor_shape(&vars, &[1, 6]));
    assert!(has_tensor_shape(&vars, &[4, 4]));
    assert!(has_tensor_shape(&vars, &[4, 2]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 5.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 16.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 3.0));
}
