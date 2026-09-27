#![cfg(feature = "native-accel")]

#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn actual_compiler_windows_include_canonical_sin_and_single() {
    let bytecode =
        test_helpers::compile_source("x = 1:4096; y = single(sin(x) .* x + 2);").unwrap();
    let groups = bytecode.runtime_fusion_groups();
    assert!(!groups.is_empty());
    for name in ["sin", "single"] {
        let sites = bytecode.instructions.iter().enumerate().filter(|(_, instruction)| {
            matches!(instruction, runmat_vm::Instr::CallBuiltinMulti(actual, 1, 1) if actual == name)
        }).map(|(pc, _)| pc).collect::<Vec<_>>();
        assert!(!sites.is_empty(), "actual {name} call must be emitted");
        for pc in sites {
            assert!(
                groups
                    .iter()
                    .any(|group| group.span.start <= pc && pc <= group.span.end),
                "canonical {name} at {pc} was excluded from compiler fusion windows: {groups:?}"
            );
        }
    }
}

#[test]
fn canonical_only_chain_produces_mir_candidates_without_arithmetic_signal() {
    let bytecode = test_helpers::compile_source("x = 1:4096; y = single(sin(x));").unwrap();
    assert!(bytecode.fusion_metadata.mir_fusion_candidate_group_count > 0);
    let groups = bytecode.runtime_fusion_groups();
    for (pc, instruction) in bytecode.instructions.iter().enumerate() {
        if matches!(instruction, runmat_vm::Instr::CallBuiltinMulti(name, 1, 1) if name == "sin" || name == "single")
        {
            assert!(groups
                .iter()
                .any(|group| group.span.start <= pc && pc <= group.span.end));
        }
    }
}

#[test]
fn effectful_calls_remain_window_boundaries() {
    let bytecode =
        test_helpers::compile_source("x = 1:4096; y = sin(x) + 2; disp(y); z = single(y .* x);")
            .unwrap();
    let pc = bytecode.instructions.iter().position(|instruction| {
        matches!(instruction, runmat_vm::Instr::CallBuiltinMulti(name, _, _) if name == "disp")
    }).expect("actual display call");
    assert!(bytecode
        .runtime_fusion_groups()
        .iter()
        .all(|group| pc < group.span.start || pc > group.span.end));
}

#[test]
fn valid_like_forms_do_not_enter_plain_unary_windows() {
    for name in ["single", "double", "sin", "cos", "tan", "atan"] {
        let bytecode = test_helpers::compile_source(&format!(
            "x = 1:4096; y = {name}(x .* x + 2, 'like', single(0));"
        ))
        .unwrap();
        let pc = bytecode.instructions.iter().position(|instruction| {
            matches!(instruction, runmat_vm::Instr::CallBuiltinMulti(actual, 3, 1) if actual == name)
        }).expect("actual three-argument like form");
        assert!(
            bytecode
                .runtime_fusion_groups()
                .iter()
                .all(|group| pc < group.span.start || pc > group.span.end),
            "{name}: prototype must not be discarded by plain unary fusion"
        );
    }
}

#[test]
fn named_and_operator_products_keep_real_matmul_epilogue_graphs() {
    use runmat_accelerate::{graph::AccelOpCategory, FusionKind};
    for expression in ["mtimes(A, B)", "A * B"] {
        let bytecode = test_helpers::compile_source(&format!(
            "A = rand(4, 3); B = rand(3, 2); y = {expression} .* 2;"
        ))
        .unwrap();
        let windows = bytecode.runtime_fusion_groups();
        assert!(!windows.is_empty(), "{expression}: compiler windows");
        let graph = bytecode
            .runtime_accel_graph_for_fusion(&windows)
            .expect("compiler-authorized product graph");
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.category == AccelOpCategory::MatMul));
        assert!(
            graph
                .detect_fusion_groups()
                .iter()
                .any(|group| matches!(group.kind, FusionKind::MatmulEpilogue)),
            "{expression}: real matmul epilogue detector"
        );
    }
}

#[test]
fn named_products_restore_explained_variance_idiom_graph() {
    use runmat_accelerate::FusionKind;
    let bytecode = test_helpers::compile_source(
        "G = rand(4, 4); Q = rand(4, 2); tmp = mtimes(Q.', G); prod = mtimes(tmp, Q); ev = diag(prod);"
    ).unwrap();
    let windows = bytecode.runtime_fusion_groups();
    let graph = bytecode
        .runtime_accel_graph_for_fusion(&windows)
        .expect("named-product explained variance graph");
    assert!(graph
        .detect_fusion_groups()
        .iter()
        .any(|group| matches!(group.kind, FusionKind::ExplainedVariance)));
}
