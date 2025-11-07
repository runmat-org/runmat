use runmat_accelerate::fusion::{FusionKind, FusionPattern, FusionPlan};
use runmat_accelerate::graph::{AccelGraph, AccelNodeLabel, PrimitiveOp, ValueOrigin};
use runmat_hir::lower;
use runmat_ignition::compile;
use runmat_parser::parse;

fn compile_graph(source: &str) -> AccelGraph {
    let ast = parse(source).expect("parse");
    let hir = lower(&ast).expect("lower");
    let bytecode = compile(&hir).expect("compile");
    bytecode.accel_graph.clone().expect("bytecode accel graph")
}

fn has_builtin(graph: &AccelGraph, name: &str) -> bool {
    graph.nodes.iter().any(|node| match &node.label {
        AccelNodeLabel::Builtin { name: node_name } => node_name.eq_ignore_ascii_case(name),
        _ => false,
    })
}

fn count_primitives(graph: &AccelGraph, op: PrimitiveOp) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| matches!(node.label, AccelNodeLabel::Primitive(p) if p == op))
        .count()
}

#[test]
fn stats_centered_gram_pattern() {
    let source = r#"
    n = 11;
    d = 5;
    A = rand(n, d);
    mu = mean(A, 1);
    centered = A - mu;
    cov = (centered.' * centered) / (n - 1);
    lambda = diag(cov);
    "#;
    let graph = compile_graph(source);

    assert!(has_builtin(&graph, "mean"));
    assert!(count_primitives(&graph, PrimitiveOp::Sub) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Transpose) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Mul) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Div) >= 1);
    assert!(has_builtin(&graph, "diag"));
}

#[test]
fn signal_power_spectrum_pattern() {
    let source = r#"
    t = 32;
    channels = 6;
    X = rand(t, channels);
    drift = mean(X, 2);
    Xc = X - drift;
    spectrum = Xc.' * Xc;
    scales = sqrt(sum(spectrum.^2, 1) + 1e-4);
    normalized = spectrum ./ scales;
    band_power = diag(normalized);
    "#;
    let graph = compile_graph(source);

    assert!(has_builtin(&graph, "mean"));
    assert!(count_primitives(&graph, PrimitiveOp::Sub) >= 1);
    assert!(has_builtin(&graph, "sum"));
    assert!(has_builtin(&graph, "sqrt"));
    assert!(has_builtin(&graph, "diag"));
}

#[test]
fn random_projection_power_step_pattern() {
    let source = r#"
    rows = 9;
    cols = 3;
    G = rand(rows, rows);
    Q = rand(rows, cols);
    Q = mtimes(G, Q);
    norms = sqrt(sum(Q.^2, 1) + 1e-6);
    Qn = Q ./ norms;
    cov = Qn.' * Qn;
    explained = diag(cov);
    "#;
    let graph = compile_graph(source);

    assert!(has_builtin(&graph, "sum"));
    assert!(has_builtin(&graph, "sqrt"));
    assert!(count_primitives(&graph, PrimitiveOp::ElemDiv) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Transpose) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Mul) >= 1);
    assert!(has_builtin(&graph, "diag"));
}

#[test]
fn monte_carlo_factor_risk_pattern() {
    let source = r#"
    steps = 17;
    factors = 8;
    paths = randn(steps, factors);
    offset = mean(paths, 1);
    centered = paths - offset;
    cov = (centered.' * centered) / (steps - 1);
    loadings = rand(factors, 4);
    loadings = loadings ./ sqrt(sum(loadings.^2, 1) + 1e-6);
    risk = diag(loadings.' * cov * loadings);
    "#;
    let graph = compile_graph(source);

    assert!(has_builtin(&graph, "randn"));
    assert!(has_builtin(&graph, "mean"));
    assert!(count_primitives(&graph, PrimitiveOp::Sub) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Transpose) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::Mul) >= 2);
    assert!(count_primitives(&graph, PrimitiveOp::Div) >= 1);
    assert!(has_builtin(&graph, "diag"));
}

#[test]
fn detects_centered_gram_group() {
    let source = r#"
    n = 7;
    d = 4;
    A = rand(n, d);
    mu = mean(A, 1);
    centered = A - mu;
    cov = (centered.' * centered) / (n - 1);
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::CenteredGram)));
}

#[test]
fn detects_power_step_group() {
    let source = r#"
    G = [
        2, -1, 0;
        0, 1, 3;
        4, 2, -2
    ];
    Q = [
        1, 2;
        3, 4;
        5, 6
    ];
    Q = mtimes(G, Q);
    norms = sqrt(sum(Q.^2, 1) + 1e-6);
    Q = Q ./ norms;
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::PowerStepNormalize)));
}

#[test]
fn detects_explained_variance_group() {
    let source = r#"
    G = [
        1, 0, 2;
        -1, 3, 0;
        0.5, -0.25, 1
    ];
    Q = [
        1, 2;
        3, 4;
        5, 6
    ];
    tmp = mtimes(Q.', G);
    cov = mtimes(tmp, Q);
    eval = diag(cov);
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::ExplainedVariance)));
}

#[test]
fn detects_image_normalize_group() {
    let source = r#"
    seed = 0;
    B = 4; H = 8; W = 12;
    gain = single(1.0123);
    bias = single(-0.02);
    gamma = single(1.8);
    eps0 = single(1e-6);
    imgs = rand(B, H, W, 'single');
    mu = mean(mean(imgs, 2), 3);
    sigma = sqrt(mean(mean((imgs - mu).^2, 2), 3) + eps0);
    out = ((imgs - mu) ./ sigma) * gain + bias;
    out = max(out, single(0));
    out = out .^ gamma;
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    let plan = FusionPlan::from_graph(&graph, &groups);
    let image_group = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ImageNormalize))
        .expect("image normalize group not found");
    match image_group.pattern.as_ref() {
        Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) => {
            assert!(matches!(pattern.epsilon, runmat_accelerate::fusion::ImageScalar::Constant(_)));
            assert!(pattern
                .gain
                .as_ref()
                .map(|s| matches!(s, runmat_accelerate::fusion::ImageScalar::Constant(_)))
                .unwrap_or(true));
            assert!(pattern
                .bias
                .as_ref()
                .map(|s| matches!(s, runmat_accelerate::fusion::ImageScalar::Constant(_)))
                .unwrap_or(true));
        }
        _ => panic!("missing image normalize pattern"),
    }
}

#[test]
fn explained_variance_plan_inputs() {
    let source = r#"
    G = [
        1, 0, 2;
        -1, 3, 0;
        0.5, -0.25, 1
    ];
    Q = [
        1, 2;
        3, 4;
        5, 6
    ];
    tmp = mtimes(Q.', G);
    cov = mtimes(tmp, Q);
    eval = diag(cov);
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    let plan = FusionPlan::from_graph(&graph, &groups);
    let explained = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ExplainedVariance))
        .expect("explained variance group not found");
    dbg!(&explained.group.nodes);
    dbg!(&explained.group.nodes);
    dbg!(&explained.inputs);
    assert_eq!(explained.inputs.len(), 2);
    dbg!(&explained.stack_pattern);
    let input_origins: Vec<_> = explained
        .inputs
        .iter()
        .map(|vid| graph.value(*vid).map(|info| info.origin.clone()))
        .collect();
    dbg!(input_origins);
}

#[test]
fn explained_variance_plan_inputs_with_rand() {
    let source = r#"
    rows = 4;
    cols = 2;
    G = rand(rows, rows);
    Q = rand(rows, cols);
    tmp = mtimes(Q.', G);
    cov = mtimes(tmp, Q);
    eval = diag(cov);
    "#;
    let graph = compile_graph(source);
    let groups = graph.detect_fusion_groups();
    let plan = FusionPlan::from_graph(&graph, &groups);
    let explained = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ExplainedVariance))
        .expect("explained variance group not found");
    dbg!(&explained.inputs);
    dbg!(&explained.stack_pattern);
    let origins: Vec<_> = explained
        .inputs
        .iter()
        .map(|vid| graph.value(*vid).map(|info| info.origin.clone()))
        .collect();
    dbg!(origins);
    dbg!(&explained.const_values);
    for node_id in &explained.group.nodes {
        if let Some(node) = graph.node(*node_id) {
            dbg!(node_id, &node.label, &node.inputs, &node.outputs);
        }
    }
    let variable_values: Vec<_> = graph
        .values
        .iter()
        .enumerate()
        .filter(|(_, info)| matches!(info.origin, ValueOrigin::Variable { .. }))
        .collect();
    dbg!(variable_values);
}