use runmat_hir::{
    lower, CallKind, CallSyntax, HirCallableRef, HirExprKind, HirStmtKind, HirSubscriptStep,
    LoweringContext, RequestedOutputCount,
};
use runmat_static_analysis::frontend::{
    analyze_source, AnalysisCompleteness, ResolutionState, DIAGNOSTIC_RUNTIME_METHOD_DISPATCH,
    DIAGNOSTIC_UNRESOLVED_FUNCTION,
};

use runmat_runtime as _;

#[test]
fn dotted_chain_dispatch_remains_runtime_dependent_for_methods_and_callable_fields() {
    for source in [
        "receiver = struct(); value = receiver.compute(1);",
        "receiver = struct(); value = receiver.compute(1).finish(2);",
        "receiver = struct('compute', @(x) x); value = receiver.compute(1); function y = compute(obj, x); y = x; end",
    ] {
        let analysis = analyze_source(
            source,
            runmat_parser::CompatMode::default(),
            &LoweringContext::empty(),
        );
        assert!(analysis.lowering_failure.is_none());
        assert_eq!(
            analysis.domains.name_resolution,
            AnalysisCompleteness::RuntimeDependent
        );
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_METHOD_DISPATCH));
        assert!(!analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));
        let calls: Vec<_> = analysis
            .resolution
            .iter()
            .filter(|evidence| evidence.name == "compute" || evidence.name == "finish")
            .collect();
        let expected = if source.contains(".finish") { 2 } else { 1 };
        assert_eq!(calls.len(), expected);
        assert!(calls.iter().all(|evidence| {
            evidence.state == ResolutionState::RuntimeDependent && evidence.definition.is_none()
        }));
    }
}

#[test]
fn dotted_chain_index_keeps_fallback_ambiguity_requested_outputs_and_step_spans() {
    let source = "receiver = struct(); value = receiver.compute(1).finish(2); [a,b] = receiver.compute(3); function y = compute(obj, x); y = x; end";
    let ast = runmat_parser::parse(source).unwrap();
    let source_id = runmat_types::ProgramSourceId(37);
    let result = lower(
        &ast,
        &LoweringContext::empty().with_primary_source(source_id),
    )
    .unwrap();
    let calls: Vec<_> = result
        .hir_index
        .calls
        .iter()
        .filter(|call| matches!(call.syntax, CallSyntax::DottedInvoke))
        .collect();
    assert_eq!(calls.len(), 3);
    for (call, spelling, outputs) in [
        (calls[0], "receiver.compute(1)", RequestedOutputCount::One),
        (
            calls[1],
            "receiver.compute(1).finish(2)",
            RequestedOutputCount::One,
        ),
        (
            calls[2],
            "receiver.compute(3)",
            RequestedOutputCount::Exactly(2),
        ),
    ] {
        assert!(matches!(call.kind, CallKind::Dynamic));
        assert!(matches!(call.callee, HirCallableRef::Unresolved(_)));
        assert_eq!(call.requested_outputs, outputs);
        assert_eq!(call.source, Some(source_id));
        assert_eq!(&source[call.span.start..call.span.end], spelling);
    }
    assert!(result.assembly.functions.iter().any(|function| {
        function.body.statements.iter().any(|statement| {
            matches!(&statement.kind, HirStmtKind::Assign(_, value, _)
            if matches!(&value.kind, HirExprKind::SubscriptChain(chain)
                if matches!(chain.steps.first(), Some(HirSubscriptStep::DottedInvoke {
                    semantic_fallback: Some(runmat_types::CallableIdentity::BoundFunction(_)),
                    ..
                }))))
        })
    }));
}

#[test]
fn members_indices_and_static_namespaces_do_not_invent_dotted_dispatch() {
    for source in [
        "receiver = struct('values', {[1 2]}); value = receiver.values{1};",
        "value = uint8.empty(1,0);",
    ] {
        let analysis = analyze_source(
            source,
            runmat_parser::CompatMode::default(),
            &LoweringContext::empty(),
        );
        assert!(analysis.lowering_failure.is_none());
        assert!(!analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_METHOD_DISPATCH));
        assert!(!analysis
            .lowering
            .unwrap()
            .hir_index
            .calls
            .iter()
            .any(|call| matches!(call.syntax, CallSyntax::DottedInvoke)));
    }
}
