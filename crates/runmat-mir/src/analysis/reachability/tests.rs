use runmat_hir::{lower, LoweringContext};

use super::{
    analyze_reachability, analyze_reachability_with_functions, function_dependencies,
    selected_function_closure, ReachabilityCertainty, ReachabilityNames, ReachabilityNodeKind,
    ReachabilityReason,
};

fn lower_mir(source: &str) -> crate::MirAssembly {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
    crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR")
}

#[test]
fn reports_direct_functions_and_builtins_with_reasons() {
    let assembly = lower_mir(
        r#"
result = helper(4);
result = abs(result);
function out = helper(value)
  out = value + 1;
end
"#,
    );
    let report = analyze_reachability(&assembly, &ReachabilityNames::default());

    let helper = report
        .nodes
        .iter()
        .find(|node| node.kind == ReachabilityNodeKind::Function && node.symbol == "helper")
        .expect("helper node");
    assert_eq!(helper.certainty, ReachabilityCertainty::Definite);
    assert!(report.edges.iter().any(|edge| {
        edge.to == helper.id
            && edge.reason == ReachabilityReason::DirectCall
            && edge.certainty == ReachabilityCertainty::Definite
    }));
    assert!(report.nodes.iter().any(|node| {
        node.id == "builtin:abs"
            && node.module == "runmat-builtins"
            && node.symbol == "abs"
            && node.builtin_id() == Some(&runmat_types::BuiltinId("abs".into()))
    }));
    assert!(report.edges.iter().any(|edge| {
        edge.to == "builtin:plus"
            && edge.reason == ReachabilityReason::OperatorDispatch
            && edge.certainty == ReachabilityCertainty::FiniteDynamic
    }));
    assert!(!report.has_unknown_edges);
}

#[test]
fn dotted_semantic_fallback_is_a_reachable_alternative_not_a_definite_call() {
    let assembly = lower_mir(
        "obj = 2; y = obj.bump(3);\nfunction out = bump(receiver, value)\n out = receiver + value;\nend",
    );
    let report = analyze_reachability(&assembly, &ReachabilityNames::default());
    let bump = report
        .nodes
        .iter()
        .find(|node| node.kind == ReachabilityNodeKind::Function && node.symbol == "bump")
        .expect("method-style fallback function is retained");
    assert!(report.edges.iter().any(|edge| {
        edge.to == bump.id
            && edge.reason == ReachabilityReason::MethodDispatch
            && edge.certainty == ReachabilityCertainty::FiniteDynamic
    }));
}

#[test]
fn function_dependencies_keep_typed_dotted_fallback_edges() {
    let assembly = lower_mir(
        "obj = 2; y = obj.bump(3);\nfunction out = bump(receiver, value)\n out = receiver + value;\nend",
    );
    let caller = assembly.entrypoints[0];
    let callee = *assembly
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "bump")
        .map(|(id, _)| id)
        .expect("bound fallback");
    assert!(
        function_dependencies(&assembly, &std::iter::once(caller).collect())
            .get(&caller)
            .is_some_and(|children| children.contains(&callee))
    );
}

#[test]
fn selected_static_closure_does_not_seed_unrelated_public_functions() {
    let assembly = lower_mir(
        "out = helper(4);\nfunction y = helper(x)\n y = x + 1;\nend\nfunction y = unused(x)\n y = x + 2;\nend",
    );
    let root = assembly.entrypoints[0];
    let function = |name: &str| {
        *assembly
            .functions
            .iter()
            .find(|(_, metadata)| metadata.name.0 == name)
            .map(|(id, _)| id)
            .expect("named function")
    };
    let selected = selected_function_closure(&assembly, &std::iter::once(root).collect());
    assert!(selected.contains(&root));
    assert!(selected.contains(&function("helper")));
    assert!(!selected.contains(&function("unused")));
    let (_, public_report_retention) =
        analyze_reachability_with_functions(&assembly, &ReachabilityNames::default());
    assert!(public_report_retention.contains(&function("unused")));
}

#[test]
fn typed_retention_does_not_depend_on_report_node_spelling() {
    let assembly =
        lower_mir("result = helper(4);\nfunction out = helper(value)\n out = value + 1;\nend");
    let (mut report, retained) =
        analyze_reachability_with_functions(&assembly, &ReachabilityNames::default());
    let helper = *assembly
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "helper")
        .map(|(function, _)| function)
        .expect("helper function");
    assert!(retained.contains(&helper));
    for node in &mut report.nodes {
        node.id = "presentation-only".into();
    }
    assert!(retained.contains(&helper));
}

#[test]
fn reports_unbounded_dynamic_calls_explicitly() {
    let mut assembly = lower_mir("out = abs(1);");
    let call = assembly
        .bodies
        .values_mut()
        .flat_map(|body| &mut body.blocks)
        .flat_map(|block| &mut block.statements)
        .find_map(|statement| match &mut statement.kind {
            crate::MirStmtKind::Assign {
                value: crate::MirRvalue::Call(call),
                ..
            }
            | crate::MirStmtKind::MultiAssign {
                value: crate::MirRvalue::Call(call),
                ..
            }
            | crate::MirStmtKind::Expr(crate::MirRvalue::Call(call)) => Some(call),
            _ => None,
        })
        .expect("call in fixture");
    call.callee = crate::MirCallee::Dynamic(crate::MirOperand::Local(crate::MirLocalId(0)));
    let report = analyze_reachability(&assembly, &ReachabilityNames::default());

    assert!(report.has_unknown_edges);
    assert!(report.edges.iter().any(|edge| {
        edge.to == "runtime_catalog:unknown"
            && edge.reason == ReachabilityReason::DynamicCall
            && edge.certainty == ReachabilityCertainty::Unknown
    }));
}

#[test]
fn report_json_is_deterministic() {
    let assembly = lower_mir("value = abs(-2);");
    let first = analyze_reachability(&assembly, &ReachabilityNames::default());
    let second = analyze_reachability(&assembly, &ReachabilityNames::default());
    assert_eq!(
        serde_json::to_vec(&first).expect("serialize first"),
        serde_json::to_vec(&second).expect("serialize second")
    );
}

#[test]
fn resolves_constant_feval_targets_to_real_candidates() {
    let assembly = lower_mir(
        r#"
result = feval('helper', 2);
function out = helper(value)
  out = value + 1;
end
"#,
    );
    let report = analyze_reachability(&assembly, &ReachabilityNames::default());
    let helper = report
        .nodes
        .iter()
        .find(|node| node.kind == ReachabilityNodeKind::Function && node.symbol == "helper")
        .expect("helper candidate");
    assert!(report.edges.iter().any(|edge| {
        edge.to == helper.id
            && edge.reason == ReachabilityReason::DynamicNamedCall
            && edge.certainty == ReachabilityCertainty::FiniteDynamic
            && edge.detail.as_deref() == Some("constant string dynamic call")
    }));
    assert!(!report.has_unknown_edges);
}
