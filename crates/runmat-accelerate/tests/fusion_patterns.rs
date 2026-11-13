use runmat_accelerate::fusion::{detect_fusion_groups, FusionKind, FusionPlan};
use runmat_accelerate::graph::{AccelGraph, AccelNodeLabel, PrimitiveOp, ShapeInfo, ValueOrigin};
use runmat_hir::LoweringContext;
use runmat_parser::parse;
use std::sync::Once;

fn init_logger() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });
}
fn compile_bytecode(source: &str) -> runmat_vm::Bytecode {
    let trimmed = source.trim_start_matches(|c: char| c.is_whitespace());
    let ast = parse(trimmed).expect("parse");
    let hir = runmat_hir::lower(&ast, &LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let entrypoint = hir.assembly.entrypoints[0].id;
    runmat_vm::compile(&hir.assembly, &mir, entrypoint).expect("compile semantic")
}
fn compile_graph(source: &str) -> AccelGraph {
    compile_bytecode(source)
        .accel_graph
        .expect("bytecode accel graph")
}

fn count_matrix_products(graph: &AccelGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| node.is_matrix_product())
        .count()
}

fn has_builtin(graph: &AccelGraph, name: &str) -> bool {
    graph.nodes.iter().any(|node| match &node.label {
        AccelNodeLabel::Builtin {
            name: node_name, ..
        } => node_name.eq_ignore_ascii_case(name),
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
    assert!(count_matrix_products(&graph) >= 1);
    assert!(count_primitives(&graph, PrimitiveOp::ElemDiv) >= 1);
    assert!(has_builtin(&graph, "diag"));
}

#[test]
#[ignore]
fn debug_qr_power_iter_graph() {
    let source = r#"
    rows = 16;
    cols = 4;
    G = rand(rows, rows);
    Q = rand(rows, cols);
    [Q, R_unused] = qr(G * Q, 'econ');
    "#;
    let graph = compile_graph(source);
    println!("nodes: {}", graph.nodes.len());
    for node in &graph.nodes {
        println!(
            "node {} {:?} inputs={:?} outputs={:?}",
            node.id, node.label, node.inputs, node.outputs
        );
    }
    for value in &graph.values {
        println!(
            "value {} origin={:?} shape={:?} const={:?}",
            value.id, value.origin, value.shape, value.constant
        );
    }
    panic!("debug");
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
#[ignore]
fn accel_graph_records_node_bindings() {
    let source = r#"
        pts = 8;
        x = rand(pts, 1);
        y = tanh(x);
        z = y + single(0.1) .* y;
    "#;
    let graph = compile_graph(source);
    assert!(
        !graph.node_bindings.is_empty(),
        "expected node_bindings to capture StoreVar writes"
    );
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
    assert!(count_matrix_products(&graph) >= 1);
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
    assert!(count_matrix_products(&graph) >= 2);
    assert!(count_primitives(&graph, PrimitiveOp::ElemDiv) >= 1);
    assert!(has_builtin(&graph, "diag"));
}

#[test]
fn detects_centered_gram_group() {
    let source = r#"
    n = 7;
    d = 4;
    A = rand(7, 4);
    mu = mean(A, 1);
    centered = A - mu;
    cov = (centered.' * centered) / 6;
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::CenteredGram)));
    for expression in [
        "(centered.' * centered) / (n - 1)",
        "(centered.' * A) / 6",
        "(centered * centered.') / 6",
        "(centered.' .* centered) / 6",
        "(centered.' * centered) / 8",
    ] {
        let source =
            format!("n=7; d=4; A=rand(7,4); mu=mean(A,1); centered=A-mu; cov={expression};");
        assert!(
            !detect_fusion_groups(&compile_graph(&source))
                .iter()
                .any(|group| matches!(group.kind, FusionKind::CenteredGram)),
            "{expression}"
        );
    }
    let wrong_axis =
        compile_graph("A=rand(7,4); mu=mean(A,2); centered=A-mu; cov=(centered.'*centered)/6;");
    assert!(!detect_fusion_groups(&wrong_axis)
        .iter()
        .any(|group| matches!(group.kind, FusionKind::CenteredGram)));
    let entry = runmat_builtins::MTIMES_CATALOG_ENTRY;
    let identity = entry.binding_identity(&entry.bindings[0]);
    for origin in [
        runmat_accelerate::graph::FusionCallOrigin::NamedUnresolved,
        runmat_accelerate::graph::FusionCallOrigin::Operator(runmat_types::OperatorKind::Add),
        runmat_accelerate::graph::FusionCallOrigin::Named {
            binding: runmat_builtins::BuiltinBindingIdentity {
                variant: "not-canonical",
                ..identity
            },
        },
    ] {
        let mut forged = graph.clone();
        let product = forged
            .nodes
            .iter_mut()
            .find(|node| node.is_matrix_product())
            .unwrap();
        product.label = AccelNodeLabel::Builtin {
            name: entry.identity.name.into(),
            origin,
        };
        assert!(!product.is_matrix_product());
        assert!(!detect_fusion_groups(&forged)
            .iter()
            .any(|group| matches!(group.kind, FusionKind::CenteredGram)));
    }
}

#[test]
fn detects_power_step_group() {
    let source = r#"
    seed = 0;
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
    let groups = detect_fusion_groups(&graph);
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::PowerStepNormalize)));
}

#[test]
fn detects_explained_variance_group() {
    let source = r#"
    seed = 0;
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
    let groups = detect_fusion_groups(&graph);
    assert!(groups
        .iter()
        .any(|group| matches!(group.kind, FusionKind::ExplainedVariance)));
}

#[test]
fn explained_variance_does_not_reorder_matrix_products() {
    for expression in ["diag((G * Q.') * Q)", "diag(Q * (Q.' * G))"] {
        let source = format!("Q = [1, 2; 3, 4]; G = [5, 6; 7, 8]; result = {expression};");
        let graph = compile_graph(&source);
        assert!(
            !detect_fusion_groups(&graph)
                .iter()
                .any(|group| matches!(group.kind, FusionKind::ExplainedVariance)),
            "must retain matrix operand order: {expression}",
        );
    }
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
    let bytecode = compile_bytecode(source);
    let graph = bytecode.accel_graph.as_ref().expect("compiled graph");
    let groups = detect_fusion_groups(graph);
    let plan = FusionPlan::from_graph(graph, &groups);
    let image_group = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ImageNormalize))
        .expect("image normalize group not found");
    match image_group.pattern.as_ref() {
        Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) => {
            let assert_live = |scalar: &runmat_accelerate::fusion::ImageScalar, name: &str| {
                let runmat_accelerate::fusion::ImageScalar::Value(id) = scalar else {
                    panic!("{name}: published scalar must not be frozen into a literal");
                };
                let binding = graph
                    .var_binding(*id)
                    .or_else(|| match graph.value(*id)?.origin {
                        ValueOrigin::NodeOutput { node, .. } => graph.node_binding(node),
                        _ => None,
                    })
                    .expect("exact live scalar binding");
                let expected = bytecode
                    .var_names
                    .iter()
                    .find(|(_, candidate)| candidate.as_str() == name)
                    .map(|(slot, _)| *slot)
                    .expect("compiled control variable");
                assert_eq!(
                    binding.index, expected,
                    "{name}: wrong live scalar identity"
                );
            };
            assert_live(&pattern.epsilon, "eps0");
            assert_live(pattern.gain.as_ref().expect("source gain"), "gain");
            assert_live(pattern.bias.as_ref().expect("source bias"), "bias");
            assert_live(pattern.gamma.as_ref().expect("source exponent"), "gamma");
            assert!(
                pattern.clamp_zero,
                "clamped image normalize script should set clamp_zero"
            );
        }
        _ => panic!("missing image normalize pattern"),
    }
}

#[test]
fn detects_image_normalize_group_with_vecdim_and_clamp() {
    let source = r#"
    rng(0); B=2; H=4; W=5;
    gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

    imgs = rand(B, H, W, 'single');
    mu = single(mean(imgs, [2 3], 'native'));
    sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
    out = single(((imgs - mu) ./ sigma) * gain + bias);
    out = max(out, single(0));
    out = single(out .^ gamma);
    mse = mean((out - imgs).^2, 'all');

    fprintf('RESULT_ok MSE=%.6e\n', double(mse));
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
    let plan = FusionPlan::from_graph(&graph, &groups);
    let image_group = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ImageNormalize))
        .expect("image normalize group not found for vecdim clamped form");
    match image_group.pattern.as_ref() {
        Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) => {
            assert!(
                pattern.clamp_zero,
                "clamped vecdim image normalize script should set clamp_zero"
            );
            assert!(pattern.requires_single_input);
        }
        _ => panic!("missing image normalize pattern"),
    }

    let mut known_scalar_cast = graph.clone();
    let clamp = known_scalar_cast
        .nodes
        .iter()
        .find(|node| matches!(&node.label, AccelNodeLabel::Builtin { name, .. } if name == "max"))
        .expect("compiled clamp");
    let cast_output = clamp.inputs[0];
    let cast_value = known_scalar_cast
        .values
        .iter_mut()
        .find(|value| value.id == cast_output)
        .expect("compiled cast output");
    assert!(matches!(cast_value.shape, ShapeInfo::Unknown));
    cast_value.shape = ShapeInfo::Scalar;
    assert!(known_scalar_cast
        .detect_fusion_groups()
        .iter()
        .all(|group| group.kind != FusionKind::ImageNormalize));
}

#[test]
fn detects_image_normalize_group_with_cast_wrapped_clamp() {
    let source = r#"
    rng(0); B=2; H=4; W=5;
    gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

    imgs = rand(B, H, W, 'single');
    mu = single(mean(imgs, [2 3], 'native'));
    sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
    out = single(((imgs - mu) ./ sigma) * gain + bias);
    out = single(max(out, single(0)));
    out = out .^ gamma;
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
    let plan = FusionPlan::from_graph(&graph, &groups);
    let image_group = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ImageNormalize))
        .expect("image normalize group not found for cast-wrapped clamp");
    match image_group.pattern.as_ref() {
        Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) => {
            assert!(
                pattern.clamp_zero,
                "cast-wrapped max clamp should set clamp_zero"
            );
        }
        _ => panic!("missing image normalize pattern"),
    }
}

#[test]
fn detects_image_normalize_group_with_vecdim_without_clamp() {
    let source = r#"
    rng(0); B=2; H=4; W=5;
    gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

    imgs = rand(B, H, W, 'single');
    mu = mean(imgs, [2 3]);
    sigma = sqrt(mean((imgs - mu).^2, [2 3]) + eps0);
    out = ((imgs - mu) ./ sigma) * gain + bias;
    out = out .^ gamma;
    mse = mean((out - imgs).^2, 'all');

    fprintf('RESULT_ok MSE=%.6e\n', double(mse));
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
    let plan = FusionPlan::from_graph(&graph, &groups);
    let image_group = plan
        .groups
        .iter()
        .find(|g| matches!(g.group.kind, FusionKind::ImageNormalize))
        .expect("image normalize group not found for vecdim exact form");
    match image_group.pattern.as_ref() {
        Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) => {
            assert!(
                !pattern.clamp_zero,
                "unclamped image normalize script should preserve unclamped semantics"
            );
        }
        _ => panic!("missing image normalize pattern"),
    }
}

#[test]
fn live_gpu_zero_without_a_validated_value_proof_retains_ordinary_execution() {
    init_logger();
    let source = r#"
    seed = 0;
    B = 4; H = 8; W = 12;
    gain_default = single(1.0123);
    bias_default = single(-0.02);
    gamma_default = single(1.8);
    eps0_default = single(1e-6);
    gain = gpuArray(gain_default);
    bias = gpuArray(bias_default);
    gamma = gpuArray(gamma_default);
    eps0 = gpuArray(eps0_default);
    zero_proto = gpuArray(single(0));
    imgs = rand(B, H, W, 'like', zero_proto);
    mu = single(mean(imgs, [2 3], 'native'));
    sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
    out = single(((imgs - mu) ./ sigma) * gain + bias);
    zero_scalar = gpuArray(single(0));
    out = max(out, zero_scalar);
    out = single(out .^ gamma);
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
    assert!(
        !groups
            .iter()
            .any(|group| matches!(group.kind, FusionKind::ImageNormalize)),
        "a live gpuArray zero is not a certified immutable zero control"
    );
    assert!(has_builtin(&graph, "max"));
    assert!(has_builtin(&graph, "gpuArray"));
    // This is a source-plan fixture, not a numerical execution witness. The
    // original bytecode and explicit GPU intent remain the ordinary route;
    // no specialized kernel may invent a live-control proof.
}

#[test]
fn explained_variance_plan_inputs() {
    let source = r#"
    seed = 0;
    G = reshape([1, -1, 0.5, 0, 3, -0.25, 2, 0, 1], 3, 3);
    Q = reshape([1, 3, 5, 2, 4, 6], 3, 2);
    tmp = mtimes(Q.', G);
    cov = mtimes(tmp, Q);
    eval = diag(cov);
    "#;
    let graph = compile_graph(source);
    let groups = detect_fusion_groups(&graph);
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
    let groups = detect_fusion_groups(&graph);
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
