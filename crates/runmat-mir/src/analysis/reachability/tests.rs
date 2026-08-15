use runmat_hir::{lower, LoweringContext};

use super::{
    analyze_reachability, ReachabilityCertainty, ReachabilityNames, ReachabilityNodeKind,
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
        node.id == "builtin:abs" && node.module == "runmat-builtins" && node.symbol == "abs"
    }));
    assert!(report.edges.iter().any(|edge| {
        edge.to == "builtin:plus"
            && edge.reason == ReachabilityReason::OperatorDispatch
            && edge.certainty == ReachabilityCertainty::FiniteDynamic
    }));
    assert!(!report.has_unknown_edges);
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
