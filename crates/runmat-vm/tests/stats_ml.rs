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
fn fitclinear_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3]; y = [0; 0; 1; 1]; [mdl,FitInfo] = fitclinear(X, y, 'Learner', 'logistic', 'Lambda', 0, 'Solver', 'sgd'); [label,score] = predict(mdl, [0.2; 2.8]); a = label(1); b = label(2); beta = mdl.Beta(1); bias = mdl.Bias(1); obj = FitInfo.Objective(1);",
    )
    .expect("fitclinear script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, 0.0));
    assert!(has_num(&vars, 1.0));
}

#[test]
fn tsne_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng('default'); X = [0 0; 0.2 0.1; 9.8 9.9; NaN 1]; [Y,loss] = tsne(X, 'Algorithm', 'exact', 'Perplexity', 2, 'NumDimensions', 2, 'Options', struct('MaxIter', 5)); s = size(Y); rows = s(1); cols = s(2);",
    )
    .expect("tsne script");
    assert!(has_tensor_shape(&vars, &[3, 2]));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 2.0));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if value.is_finite())));
}

#[test]
fn statset_and_statget_surface_executes_from_scripts() {
    let vars = execute_source(
        "opts = statset('nbinfit'); opts = statset(opts, 'TolX', 1e-8, 'UseParallel', 'off'); tx = statget(opts, 'TolX'); mi = statget(opts, 'MaxIter'); tf = statget(opts, 'TolFun', 7); up = statget(opts, 'UseParallel'); missing = statget(opts, 'OutputFcn', 42);",
    )
    .expect("statset/statget script");
    assert!(has_num(&vars, 1.0e-8));
    assert!(has_num(&vars, 200.0));
    assert!(has_num(&vars, 1.0e-6));
    assert!(has_num(&vars, 42.0));
    assert!(has_bool(&vars, false));
}

#[test]
fn statset_options_feed_named_stats_builtins() {
    let vars = execute_source(
        "X = [0 0; 0.2 0.1; 9.8 9.9; 10 10.1]; kopts = statset('kmeans'); [idx,C] = kmeans(X, 2, 'Start', [0 0; 10 10], 'Options', kopts); rng('default'); topts = statset('tsne'); Y = tsne(X, 'Algorithm', 'exact', 'Perplexity', 2, 'Options', topts); ks = size(C); ts = size(Y); krows = ks(1); kcols = ks(2); trows = ts(1); tcols = ts(2);",
    )
    .expect("statset options should feed supported stats builtins");
    assert!(has_tensor_shape(&vars, &[4, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[4, 2]));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 4.0));
}

#[test]
fn classify_surface_executes_from_scripts() {
    let vars = execute_source(
        "training = [0; 0.5; 2; 2.5]; group = [1; 1; 2; 2]; [class,err,posterior,logp,coeff] = classify([0.2; 2.2], training, group, 'linear', 'empirical'); c1 = class(1); c2 = class(2); p11 = posterior(1,1); k = coeff(1,2).const;",
    )
    .expect("classify script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 0.0));
}

#[test]
fn kmeans_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0 0; 0.2 0.1; 9.8 9.9; 10 10.1]; [idx,C,sumd,D] = kmeans(X, 2, 'Start', [0 0; 10 10], 'MaxIter', 20); a = idx(1); b = idx(4); c11 = C(1,1); c22 = C(2,2); dshape = size(D);",
    )
    .expect("kmeans script");
    assert!(has_tensor_shape(&vars, &[4, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[4, 2]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 0.1));
    assert!(has_num(&vars, 10.0));
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
fn dividerand_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng('default'); [tr,val,te] = dividerand(10, 0.6, 0.2, 0.2); ntr = numel(tr); nv = numel(val); nt = numel(te); total = ntr + nv + nt; first = tr(1);",
    )
    .expect("dividerand script");
    assert!(has_tensor_shape(&vars, &[1, 6]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 6.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 10.0));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (1.0..=10.0).contains(value))));
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
fn boxplot_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "h = boxplot([1;2;10;20], {'A';'A';'B';'B'}, 'Labels', {'First','Second'}, 'GroupOrder', {'A','B'}, 'Orientation', 'horizontal', 'Whisker', 1, 'Symbol', '+r'); n = numel(h); first = h(1);",
    )
    .expect("boxplot script");
    assert!(has_tensor_shape(&vars, &[1, 12]));
    assert!(has_num(&vars, 12.0));
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
        "X = [0 0; 3 4; 4 0; 0 2]; D = pdist(X); M = squareform(D); D2 = squareform(M); Y = [1 0; 3 0]; P = pdist2(X, Y, 'squaredeuclidean'); [N,I] = pdist2(X, Y, 'euclidean', 'Smallest', 1); [Idx,KD] = knnsearch(X, Y, 'K', 2, 'Distance', 'euclidean'); T = knnsearch(X, [0 0], 'K', 1, 'IncludeTies', true); d12 = D(1); d14 = D(3); m14 = M(1,4); p22 = P(2,2); n2 = N(2); i2 = I(2); idx21 = Idx(2,1); kd21 = KD(2,1); tie1 = T{1}(1);",
    )
    .expect("distance helper script");
    assert!(has_tensor_shape(&vars, &[1, 6]));
    assert!(has_tensor_shape(&vars, &[4, 4]));
    assert!(has_tensor_shape(&vars, &[4, 2]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 5.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 16.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 1.0));
}

#[test]
fn linkage_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0 0; 3 4; 4 0; 0 2]; Z = linkage(X, 'single'); D = pdist(X); Zc = linkage(D, 'complete'); Za = linkage(X, 'average', {'minkowski', 3}, 'savememory', 'off'); zcheck = 100*Z(1,1) + 10*Z(1,2) + Z(1,3); zccheck = 100*Zc(3,1) + 10*Zc(3,2) + Zc(3,3); za13 = Za(1,3);",
    )
    .expect("linkage script");
    assert!(has_tensor_shape(&vars, &[3, 3]));
    assert!(has_num(&vars, 142.0));
    assert!(has_num(&vars, 565.0));
    assert!(has_num(&vars, 2.0));
}

#[test]
fn covariance_conversion_surface_executes_from_scripts() {
    let vars = execute_source(
        "C = [4 2; 2 9]; [R,s] = corrcov(C); Q = cov2corr(C); r12 = R(1,2); s2 = s(2); q21 = Q(2,1);",
    )
    .expect("corrcov script");
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - (1.0 / 3.0)).abs() < 1.0e-8)));
    assert!(has_num(&vars, 3.0));
}

#[test]
fn distribution_compatibility_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng('default'); b = binocdf(55,100,0.5); bu = binocdf(55,100,0.5,'upper'); c = chi2cdf(3,5); w = wblinv(0.5,3,4); gr = gamrnd(2,3,2,2); br = binornd(10,0.5,2,3); wr = wblrnd(4,3,[2 2]);",
    )
    .expect("distribution compatibility script");
    assert!(has_num(&vars, 0.864_373_487_963_083));
    assert!(has_num(&vars, 0.135_626_512_036_917));
    assert!(has_num(&vars, 0.300_014_164_121_372));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - 3.0 * std::f64::consts::LN_2.powf(0.25)).abs() < 1.0e-8)));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 3]));
}

#[test]
fn mvnrnd_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng('default'); R = mvnrnd([1 2], [4 1; 1 9], 5); S = mvnrnd([0 0; 10 20], cat(3, eye(2), [4 0; 0 9])); T = mvnrnd([0 0], [1 4], 3); U = mvnrnd([0; 10; 20], 4); rcheck = 100*size(R,1)+size(R,2); scheck = 100*size(S,1)+size(S,2); tcheck = 100*size(T,1)+size(T,2); ucheck = 100*size(U,1)+size(U,2); r11 = R(1,1); s22 = S(2,2);",
    )
    .expect("mvnrnd script");
    assert!(has_tensor_shape(&vars, &[5, 2]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[3, 2]));
    assert!(has_tensor_shape(&vars, &[3, 1]));
    assert!(has_num(&vars, 502.0));
    assert!(has_num(&vars, 202.0));
    assert!(has_num(&vars, 302.0));
    assert!(has_num(&vars, 301.0));
}

#[test]
fn cvpartition_surface_executes_from_scripts() {
    let vars = execute_source(
        "c = cvpartition(6, 'KFold', 3); te = test(c, 2); tr = training(c, 'all'); nts = c.NumTestSets; ts = c.TestSize(1); h = cvpartition([1;1;2;2], 'Holdout', 0.5); hm = test(h); hs = h.TestSize;",
    )
    .expect("cvpartition script");
    assert!(has_logical_shape(&vars, &[6, 1]));
    assert!(has_logical_shape(&vars, &[6, 3]));
    assert!(has_logical_shape(&vars, &[4, 1]));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 2.0));
}
