use super::*;
use crate::MirStmtKind;

#[test]
fn native_metadata_includes_hidden_region_constructs_and_requirements() {
    use runmat_hir::{CallSyntax, CallableFallbackPolicy, CallableIdentity, RequestedOutputCount};
    use runmat_types::{BuiltinId, EffectKind};

    let call_effects = runmat_builtins::BuiltinEffects::none().with_host_callback();

    let region = MirExpressionRegion {
        steps: vec![
            let_step(1, MirRvalue::End),
            let_step(
                2,
                MirRvalue::Call(crate::MirCall {
                    callee: crate::MirCallee::Static(CallableIdentity::Builtin(BuiltinId(
                        "contextual_fixture".into(),
                    ))),
                    args: vec![crate::MirCallArg::Single(MirOperand::Local(MirLocalId(1)))],
                    arg_spans: vec![Span::default()],
                    syntax: CallSyntax::Command,
                    requested_outputs: RequestedOutputCount::Exactly(1),
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: crate::AsyncBehaviorFact::MaySuspend,
                    effects: call_effects,
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Impure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            ),
        ],
        result: MirOperand::Local(MirLocalId(2)),
    };
    let value = MirRvalue::Index {
        base: MirOperand::Constant(crate::MirConstant::EmptyArray),
        indexing: crate::MirIndexing {
            argument_spans: None,
            kind: runmat_hir::IndexKind::Paren,
            plan: crate::MirIndexPlan::Slice,
            components: vec![crate::MirIndexComponent::ContextualExpr(region)],
            result_context: runmat_hir::IndexResultContext::ReadSingle,
            cell_expand_all: false,
        },
    };
    let (effects, capabilities) = crate::rvalue_declared_requirements(&value);
    assert!(effects.0.contains(&EffectKind::HostCallback));
    assert!(effects.0.contains(&EffectKind::MaySuspend));
    assert!(capabilities.0.is_empty());
    assert_eq!(
        crate::rvalue_construct_inventory(&value),
        vec![
            crate::MirConstructKind::Index,
            crate::MirConstructKind::End,
            crate::MirConstructKind::Call,
        ]
    );

    let MirRvalue::Index { indexing, .. } = value else {
        unreachable!()
    };
    let capture = MirStmtKind::CaptureSequence {
        destination: MirSequenceLocalId(9),
        source: crate::MirExpansionSource::CellContents {
            base: MirOperand::Constant(crate::MirConstant::EmptyArray),
            indexing,
        },
    };
    let (effects, capabilities) = crate::statement_declared_requirements(&capture);
    assert!(effects.0.contains(&EffectKind::Unknown));
    assert!(effects.0.contains(&EffectKind::HostCallback));
    assert!(effects.0.contains(&EffectKind::MaySuspend));
    assert!(capabilities.0.is_empty());
    assert_eq!(
        crate::statement_construct_inventory(&capture),
        vec![
            crate::MirConstructKind::CaptureSequence,
            crate::MirConstructKind::End,
            crate::MirConstructKind::Call,
        ]
    );
}
