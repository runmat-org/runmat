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

fn has_tensor_data(vars: &[Value], shape: &[usize], expected: &[f64]) -> bool {
    vars.iter().any(|value| match value {
        Value::Tensor(tensor) => {
            let data = tensor.materialize_f64();
            tensor.shape == shape
                && data.len() == expected.len()
                && data
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (*actual - *expected).abs() < 1.0e-8)
        }
        _ => false,
    })
}

fn has_bool(vars: &[Value], expected: bool) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Bool(value) if *value == expected))
}

fn has_string(vars: &[Value], expected: &str) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::String(value) if value == expected))
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
fn lassoglm_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3; 4; 5]; y = [0; 0; 0; 1; 1; 1]; opts = statset('lassoglm'); [B,FitInfo] = lassoglm(X, y, 'binomial', 'Lambda', [0 0.1], 'CV', 3, 'Options', opts); sz = size(B); rows = sz(1); cols = sz(2); d = FitInfo.Deviance(1); i = FitInfo.Intercept(1); idx = FitInfo.IndexMinDeviance; lp = FitInfo.LambdaMinDeviance; score = (d >= 0) + (idx >= 1) + (lp >= 0);",
    )
    .expect("lassoglm script");
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 3.0));
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
fn fitdist_surface_executes_from_scripts() {
    let vars = execute_source(
        "pd = fitdist([1; 2; 3], 'Normal'); vals = pd.ParameterValues; mu = vals(1); sigma = vals(2); y = pdf(pd, 2); p = cdf(pd, 2); x = icdf(pd, 0.5); r = random(pd, [2 3]); yn = pdf('Normal', 0, 0, 1); pn = cdf('Poisson', 2, 3); rn = random('Weibull', 2, 3, 2, 2); sz = size(r); rows = sz(1); cols = sz(2);",
    )
    .expect("fitdist script");
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, (2.0_f64 / 3.0).sqrt()));
    assert!(has_num(&vars, 0.5));
    assert!(has_tensor_shape(&vars, &[2, 3]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, 3.0));
}

#[test]
fn mnrfit_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [0; 1; 2; 3; 4]; y = [1; 2; 1; 2; 2]; [B,dev,stats] = mnrfit(X, y, 'IterationLimit', 200, 'Tolerance', 1e-7); sz = size(B); rows = sz(1); cols = sz(2); se = stats.se; sesz = size(se); serows = sesz(1); secols = sesz(2); dfe = stats.dfe; cn = stats.classNames; c1 = cn(1); c2 = cn(2);",
    )
    .expect("mnrfit script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 3.0));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if value.is_finite())));
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
fn kstest_surface_executes_from_scripts() {
    let vars = execute_source(
        "h = kstest([-1 -0.25 0 0.25 1]); [h2,p2,ks,cv] = kstest([2;2.5;3;3.5;4], 'Tail', 'smaller');",
    )
    .expect("kstest script");
    assert!(has_bool(&vars, false));
    assert!(has_bool(&vars, true));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if *value > 0.0 && *value < 1.0)));
}

#[test]
fn refline_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "plot([2 4], [10 20]); h = refline(2, 1); xd = get(h, 'XData'); yd = get(h, 'YData'); x1 = xd(1); x2 = xd(2); y1 = yd(1); y2 = yd(2);",
    )
    .expect("refline script");
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 4.0));
    assert!(has_num(&vars, 5.0));
    assert!(has_num(&vars, 9.0));
}

#[test]
fn lsline_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "scatter([1 2 3], [2 4 6]); h = lsline; xd = get(h, 'XData'); yd = get(h, 'YData'); x1 = xd(1); x2 = xd(2); y1 = yd(1); y2 = yd(2);",
    )
    .expect("lsline script");
    assert!(has_tensor_shape(&vars, &[1, 3]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 6.0));
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
fn confusionmat_surface_executes_from_scripts() {
    let vars = execute_source(
        "truth = [1; 1; 2; 2; NaN]; pred = [1; 2; 2; 1; 1]; [C,ord] = confusionmat(truth, pred, 'Order', [2; 1]); a = C(1,1); b = C(1,2); c = C(2,1); d = C(2,2); o1 = ord(1); labels = [\"cat\";\"dog\";\"dog\"]; guesses = [\"dog\";\"dog\";\"cat\"]; [S,sord] = confusionmat(labels, guesses); s11 = S(1,1); s12 = S(1,2);",
    )
    .expect("confusionmat script");
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
}

#[test]
fn perfcurve_surface_executes_from_scripts() {
    let vars = execute_source(
        "labels = [1; 0; 1; 0]; scores = [0.9; 0.8; 0.4; 0.1]; [X,Y,T,AUC,opt,suby,names] = perfcurve(labels, scores, 1); n = length(T); xl = X(end); yl = Y(end); [R,P] = perfcurve(labels, scores, 1, 'XCrit', 'reca', 'YCrit', 'prec', 'TVals', [0.8 0.4]); r2 = R(2); p1 = P(1);",
    )
    .expect("perfcurve script");
    assert!(has_tensor_shape(&vars, &[5, 1]));
    assert!(has_tensor_shape(&vars, &[1, 2]));
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_num(&vars, 0.75));
    assert!(has_num(&vars, 5.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 0.5));
}

#[test]
fn qqplot_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "h = qqplot([1;2;4;8]); xd = get(h(1), 'XData'); yd = get(h(1), 'YData'); mk = get(h(1), 'Marker'); qstyle = get(h(2), 'LineStyle'); rstyle = get(h(3), 'LineStyle'); n = length(h); first = yd(1); last = yd(end); h2 = qqplot([0;10;20], [], [25 50 75]); yd2 = get(h2(1), 'YData'); mid = yd2(2); h3 = qqplot([0.1;0.2], [0.3;0.4]); n3 = length(h3);",
    )
    .expect("qqplot script");
    assert!(has_tensor_shape(&vars, &[3, 1]));
    assert!(has_tensor_shape(&vars, &[1, 4]));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 8.0));
    assert!(has_num(&vars, 10.0));
    assert!(has_string(&vars, "+"));
    assert!(has_string(&vars, "-"));
    assert!(has_string(&vars, "--"));
}

#[test]
fn binscatter_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "h = binscatter([0;0.2;0.8;1], [0;0.1;0.9;1], [2 2]); vals = get(h, 'Values'); nb = get(h, 'NumBins'); typ = get(h, 'Type'); set(h, 'NumBins', [1 1]); vals_after = get(h, 'Values'); nb_after = get(h, 'NumBins'); h2 = binscatter([0;1;2], [0;1;2], 'NumBins', [2 2], 'XLimits', [0 2], 'YLimits', [0 2], 'ShowEmptyBins', 'on', 'FaceAlpha', 0.5, 'DisplayName', 'density'); alpha = get(h2, 'FaceAlpha'); show = get(h2, 'ShowEmptyBins'); edges = get(h2, 'XBinEdges'); name = get(h2, 'DisplayName');",
    )
    .expect("binscatter script");
    assert!(has_tensor_data(&vars, &[2, 2], &[2.0, 0.0, 0.0, 2.0]));
    assert!(has_tensor_data(&vars, &[1, 2], &[2.0, 2.0]));
    assert!(has_tensor_data(&vars, &[1, 1], &[4.0]));
    assert!(has_tensor_data(&vars, &[1, 2], &[1.0, 1.0]));
    assert!(has_tensor_data(&vars, &[1, 3], &[0.0, 1.0, 2.0]));
    assert!(has_num(&vars, 0.5));
    assert!(has_bool(&vars, true));
    assert!(has_string(&vars, "binscatter"));
    assert!(has_string(&vars, "density"));
}

#[test]
fn binscatter_integer_classes_execute_from_compiled_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "h1 = binscatter(int8([0;1]),int8([0;1]),uint8([2 2])); c1 = class(get(h1,'XData')); h2 = binscatter(int16([0;1]),int16([0;1]),uint16([2 2])); c2 = class(get(h2,'XData')); h3 = binscatter(int32([0;1]),int32([0;1]),uint32([2 2])); c3 = class(get(h3,'XData')); h4 = binscatter(int64([0;1]),int64([0;1]),uint64([2 2])); c4 = class(get(h4,'XData')); h5 = binscatter(uint8([0;1]),uint8([0;1]),int8([2 2])); c5 = class(get(h5,'XData')); h6 = binscatter(uint16([0;1]),uint16([0;1]),int16([2 2])); c6 = class(get(h6,'XData')); h7 = binscatter(uint32([0;1]),uint32([0;1]),int32([2 2])); c7 = class(get(h7,'XData')); h8 = binscatter(uint64([0;1]),uint64([0;1]),int64([2 2])); c8 = class(get(h8,'XData')); vals = get(h8,'Values');",
    )
    .expect("compiled integer binscatter script");
    for class in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        assert!(has_string(&vars, class), "missing class {class}");
    }
    assert!(has_tensor_data(&vars, &[2, 2], &[1.0, 0.0, 0.0, 1.0]));
}

#[test]
fn scatterhist_surface_executes_from_scripts() {
    let _plot_guard = disable_interactive_plots_for_test();
    let vars = execute_source(
        "h = scatterhist([0;0.2;0.8;1], [0;0.1;0.9;1], 'NBins', [2 2]); n = length(h); g = [\"a\";\"b\";\"a\";\"b\"]; h2 = scatterhist([0;0.2;0.8;1], [0;0.1;0.9;1], 'Group', g, 'Kernel', 'on', 'Legend', 'on', 'Location', 'southwest', 'Direction', 'in'); n2 = length(h2);",
    )
    .expect("scatterhist script");
    assert!(has_tensor_shape(&vars, &[3, 1]));
    assert!(has_num(&vars, 3.0));
}

#[test]
fn encoding_surface_executes_from_scripts() {
    let vars = execute_source(
        "D = dummyvar([1;2;1]); labels = [\"red\";\"blue\";\"red\"]; H = onehotencode(labels, 2, 'ClassNames', [\"blue\";\"red\"], 'OutputType', 'logical'); dec = onehotdecode(H, [\"blue\";\"red\"], 2, 'string'); d11 = D(1,1); d22 = D(2,2);",
    )
    .expect("encoding script");
    assert!(has_tensor_shape(&vars, &[3, 2]));
    assert!(has_logical_shape(&vars, &[3, 2]));
    assert!(has_num(&vars, 1.0));
}

#[test]
fn regress_surface_executes_from_scripts() {
    let vars = execute_source(
        "X = [1 0; 1 1; 1 2; 1 3]; y = [1; 3; 5; 7]; [b,bint,r,rint,stats] = regress(y, X); intercept = b(1); slope = b(2); r2 = stats(1);",
    )
    .expect("regress script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[4, 1]));
    assert!(has_tensor_shape(&vars, &[4, 2]));
    assert!(has_tensor_shape(&vars, &[1, 4]));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 2.0));
}

#[test]
fn lscov_surface_executes_from_scripts() {
    let vars = execute_source(
        "A = [1 0; 1 1; 1 2]; B = [1; 2; 10]; V = [1; 1; 100]; [x,stdx,mse,S] = lscov(A, B, V, 'orth'); intercept = x(1); slope = x(2);",
    )
    .expect("lscov script");
    assert!(has_tensor_shape(&vars, &[2, 1]));
    assert!(has_tensor_shape(&vars, &[1, 1]));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_num(&vars, -0.3972055888223553));
    assert!(has_num(&vars, 5.191616766467066));
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
        "rng('default'); b = binocdf(55,100,0.5); bu = binocdf(55,100,0.5,'upper'); c = chi2cdf(3,5); w = wblinv(0.5,3,4); icn = icdf('Normal',0.5,10,2); ict = icdf('t',0.95,50); icw = icdf('Weibull',0.5,3,4); icc = icdf('Chi-square',0.5,2); icb = icdf('Binomial',0.75,10,0.5); ice = icdf('Exponential',0.5,4); icu = icdf('Uniform',0.25,10,20); icl = icdf('Lognormal',0.5,1,2); icp = icdf('Poisson',0.8,3); icg = icdf('Gamma',0.5,2,3); icbeta = icdf('Beta',0.5,2,2); icf = icdf('F',0.5,5,10); gr = gamrnd(2,3,2,2); br = binornd(10,0.5,2,3); wr = wblrnd(4,3,[2 2]);",
    )
    .expect("distribution compatibility script");
    assert!(has_num(&vars, 0.864_373_487_963_083));
    assert!(has_num(&vars, 0.135_626_512_036_917));
    assert!(has_num(&vars, 0.300_014_164_121_372));
    assert!(has_num(&vars, 10.0));
    assert!(has_num(&vars, 6.0));
    assert!(has_num(&vars, 4.0 * std::f64::consts::LN_2));
    assert!(has_num(&vars, 12.5));
    assert!(has_num(&vars, std::f64::consts::E));
    assert!(has_num(&vars, 4.0));
    assert!(has_num(&vars, 0.5));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - 3.0 * std::f64::consts::LN_2.powf(0.25)).abs() < 1.0e-8)));
    assert!(vars.iter().any(
        |value| matches!(value, Value::Num(value) if (*value - 2.0 * std::f64::consts::LN_2).abs() < 1.0e-10)
    ));
    assert!(has_tensor_shape(&vars, &[2, 2]));
    assert!(has_tensor_shape(&vars, &[2, 3]));
}

#[test]
fn binocdf_integer_extensions_cover_every_class_and_preserve_single_output() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "x = [binocdf(int8(1),2,0.5) binocdf(int16(1),2,0.5) binocdf(int32(1),2,0.5) binocdf(int64(1),2,0.5) binocdf(uint8(1),2,0.5) binocdf(uint16(1),2,0.5) binocdf(uint32(1),2,0.5) binocdf(uint64(1),2,0.5)]; n = [binocdf(0,int8(1),0.5) binocdf(0,int16(1),0.5) binocdf(0,int32(1),0.5) binocdf(0,int64(1),0.5) binocdf(0,uint8(1),0.5) binocdf(0,uint16(1),0.5) binocdf(0,uint32(1),0.5) binocdf(0,uint64(1),0.5)]; p = [binocdf(0,1,int8(1)) binocdf(0,1,int16(1)) binocdf(0,1,int32(1)) binocdf(0,1,int64(1)) binocdf(0,1,uint8(1)) binocdf(0,1,uint16(1)) binocdf(0,1,uint32(1)) binocdf(0,1,uint64(1))]; score = sum(x) + sum(n) + sum(p); integer_class = class(x); single_class = class(binocdf(single([1 1]),2,0.5));",
    )
    .expect("binocdf integer extension script");
    assert!(has_num(&vars, 10.0));
    assert!(has_string(&vars, "double"));
    assert!(has_string(&vars, "single"));
}

#[test]
fn binornd_integer_extensions_cover_every_class_input_position_and_single_output() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "rng('default'); n = [binornd(int8(1),1) binornd(int16(1),1) binornd(int32(1),1) binornd(int64(1),1) binornd(uint8(1),1) binornd(uint16(1),1) binornd(uint32(1),1) binornd(uint64(1),1)]; p = [binornd(1,int8(1)) binornd(1,int16(1)) binornd(1,int32(1)) binornd(1,int64(1)) binornd(1,uint8(1)) binornd(1,uint16(1)) binornd(1,uint32(1)) binornd(1,uint64(1))]; sizes = numel(binornd(1,1,int8(2))) + numel(binornd(1,1,int16(2))) + numel(binornd(1,1,int32(2))) + numel(binornd(1,1,int64(2))) + numel(binornd(1,1,uint8(2))) + numel(binornd(1,1,uint16(2))) + numel(binornd(1,1,uint32(2))) + numel(binornd(1,1,uint64(2))); score = sum(n) + sum(p); integer_class = class(n); single_class = class(binornd(single([1 1]),1));",
    )
    .expect("binornd integer extension script");
    assert!(has_num(&vars, 16.0));
    assert!(has_num(&vars, 32.0));
    assert!(has_string(&vars, "double"));
    assert!(has_string(&vars, "single"));
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

#[test]
fn crossvalind_surface_executes_from_scripts() {
    let vars = execute_source(
        "rng(2026); idx = crossvalind('HoldOut', 10, 0.3); ntrain = sum(idx); ntest = sum(~idx); rng(7); fold = crossvalind('KFold', 9, 3); fmin = min(fold); fmax = max(fold); [tr,te] = crossvalind('LeaveMOut', 6, 2); ltrain = sum(tr); ltest = sum(te);",
    )
    .expect("crossvalind script");
    assert!(has_logical_shape(&vars, &[10, 1]));
    assert!(has_tensor_shape(&vars, &[9, 1]));
    assert!(has_logical_shape(&vars, &[6, 1]));
    assert!(has_num(&vars, 7.0));
    assert!(has_num(&vars, 3.0));
    assert!(has_num(&vars, 1.0));
    assert!(has_num(&vars, 4.0));
    assert!(has_num(&vars, 2.0));
}
