use super::*;
use crate::{
    MirAggregateElement, MirAggregateKind, MirLocalId, MirOperand, MirRvalue, MirSequenceLocalId,
    MirStmt, MirStmtKind,
};
use runmat_hir::Span;

fn let_step(local: usize, value: MirRvalue) -> MirExpressionStep {
    MirExpressionStep::Let {
        local: MirLocalId(local),
        value,
        span: Span::default(),
    }
}

#[test]
fn validates_ordered_local_and_sequence_definitions() {
    let forward_local = MirExpressionRegion {
        steps: vec![
            let_step(1, MirRvalue::Use(MirOperand::Local(MirLocalId(2)))),
            let_step(
                2,
                MirRvalue::Use(MirOperand::Constant(crate::MirConstant::Bool(true))),
            ),
        ],
        result: MirOperand::Local(MirLocalId(1)),
    };
    assert!(forward_local
        .validate()
        .unwrap_err()
        .contains("before its definition"));

    let sequence = MirSequenceLocalId(3);
    let forward_sequence = MirExpressionRegion {
        steps: vec![
            let_step(
                1,
                MirRvalue::Aggregate {
                    kind: MirAggregateKind::Tensor,
                    row_lengths: vec![1],
                    elements: vec![MirAggregateElement::CapturedSequence(sequence)],
                },
            ),
            MirExpressionStep::CaptureSequence {
                destination: sequence,
                source: crate::MirExpansionSource::ReturnedOutputs(MirOperand::Constant(
                    crate::MirConstant::EmptyArray,
                )),
                span: Span::default(),
            },
        ],
        result: MirOperand::Local(MirLocalId(1)),
    };
    assert!(forward_sequence
        .validate()
        .unwrap_err()
        .contains("before its capture"));
}

#[test]
fn operand_traversal_includes_steps_sources_and_result() {
    let mut region = MirExpressionRegion {
        steps: vec![
            let_step(2, MirRvalue::Use(MirOperand::Local(MirLocalId(1)))),
            MirExpressionStep::CaptureSequence {
                destination: MirSequenceLocalId(0),
                source: crate::MirExpansionSource::ReturnedOutputs(MirOperand::Local(MirLocalId(
                    2,
                ))),
                span: Span::default(),
            },
        ],
        result: MirOperand::Local(MirLocalId(2)),
    };
    let mut seen = Vec::new();
    region.visit_operands(|operand| {
        if let MirOperand::Local(local) = operand {
            seen.push(local.0);
        }
    });
    assert_eq!(seen, vec![1, 2, 2]);
    let mut external = Vec::new();
    region.visit_external_operands(|operand| {
        if let MirOperand::Local(local) = operand {
            external.push(local.0);
        }
    });
    assert_eq!(external, vec![1]);
    region.visit_operands_mut(|operand| {
        if let MirOperand::Local(local) = operand {
            local.0 += 10;
        }
    });
    let mut remapped = Vec::new();
    region.visit_operands(|operand| {
        if let MirOperand::Local(local) = operand {
            remapped.push(local.0);
        }
    });
    assert_eq!(remapped, vec![11, 12, 12]);
}

fn body_with_region(region: MirExpressionRegion) -> crate::MirBody {
    use crate::{
        BasicBlock, BasicBlockId, MirIndexPlan, MirIndexing, MirLocal, MirLocalKind, MirPlace,
        MirTerminator, MirTerminatorKind,
    };
    use runmat_hir::{FunctionAbi, FunctionId, IndexKind, IndexResultContext};
    let span = Span::default();
    crate::MirBody {
        function: FunctionId(0),
        abi: FunctionAbi {
            fixed_inputs: Vec::new(),
            varargin: None,
            fixed_outputs: Vec::new(),
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        locals: (0..2)
            .map(|id| MirLocal {
                id: MirLocalId(id),
                binding: None,
                kind: MirLocalKind::Temporary,
                span,
            })
            .collect(),
        blocks: vec![BasicBlock {
            id: BasicBlockId(0),
            statements: vec![MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Index {
                        base: MirOperand::Constant(crate::MirConstant::EmptyArray),
                        indexing: MirIndexing {
                            kind: IndexKind::Paren,
                            plan: MirIndexPlan::Slice,
                            components: vec![crate::MirIndexComponent::ContextualExpr(region)],
                            result_context: IndexResultContext::ReadSingle,
                            cell_expand_all: false,
                        },
                    },
                },
                span,
            }],
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                span,
            },
        }],
    }
}

#[test]
fn body_admission_confines_contextual_temporaries() {
    let region = MirExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let body = body_with_region(region.clone());
    body.validate_expression_regions().unwrap();

    let mut wrong_kind = body.clone();
    wrong_kind.locals[1].kind = crate::MirLocalKind::Binding;
    assert!(wrong_kind
        .validate_expression_regions()
        .unwrap_err()
        .contains("must be a temporary"));

    let mut escaped = body;
    escaped.blocks[0].terminator.kind =
        crate::MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]);
    assert!(escaped
        .validate_expression_regions()
        .unwrap_err()
        .contains("used outside its region"));

    let context_free = MirExpressionRegion {
        steps: vec![let_step(
            1,
            MirRvalue::Use(MirOperand::Constant(crate::MirConstant::Bool(true))),
        )],
        result: MirOperand::Local(MirLocalId(1)),
    };
    assert!(context_free
        .validate()
        .unwrap_err()
        .contains("does not depend on the indexing context"));
}

#[test]
fn mutation_annotation_does_not_duplicate_the_store_region() {
    let region = MirExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let mut body = body_with_region(region);
    let place = match &body.blocks[0].statements[0].kind {
        MirStmtKind::Assign { place, .. } => place.clone(),
        other => panic!("expected assignment, got {other:?}"),
    };
    body.blocks[0].statements.insert(
        0,
        MirStmt {
            kind: MirStmtKind::PlaceMutation(crate::MirPlaceMutation {
                place,
                kind: runmat_hir::PlaceMutationKind::IndexedAssign,
                creation_policy: runmat_hir::AssignmentCreationPolicy::ExistingOnly,
                shape_policy: runmat_hir::AssignmentShapePolicy::MatlabCompatible,
            }),
            span: Span::default(),
        },
    );
    body.validate_expression_regions().unwrap();

    let mut missing_store = body.clone();
    missing_store.blocks[0].statements.remove(1);
    assert!(missing_store
        .validate_expression_regions()
        .unwrap_err()
        .contains("no following assignment"));

    let mut mismatched_store = body;
    let MirStmtKind::Assign { place, .. } = &mut mismatched_store.blocks[0].statements[1].kind
    else {
        unreachable!()
    };
    *place = crate::MirPlace::Local(MirLocalId(9));
    assert!(mismatched_store
        .validate_expression_regions()
        .unwrap_err()
        .contains("exact assignment"));
}

#[test]
fn body_admission_confines_contextual_sequences() {
    let sequence = MirSequenceLocalId(7);
    let region = MirExpressionRegion {
        steps: vec![
            let_step(1, MirRvalue::End),
            MirExpressionStep::CaptureSequence {
                destination: sequence,
                source: crate::MirExpansionSource::ReturnedOutputs(MirOperand::Local(MirLocalId(
                    1,
                ))),
                span: Span::default(),
            },
            let_step(
                2,
                MirRvalue::Aggregate {
                    kind: MirAggregateKind::Tensor,
                    row_lengths: vec![1],
                    elements: vec![MirAggregateElement::CapturedSequence(sequence)],
                },
            ),
        ],
        result: MirOperand::Local(MirLocalId(2)),
    };
    let mut body = body_with_region(region);
    body.locals.push(crate::MirLocal {
        id: MirLocalId(2),
        binding: None,
        kind: crate::MirLocalKind::Temporary,
        span: Span::default(),
    });
    body.validate_expression_regions().unwrap();

    body.blocks[0].statements.push(MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Aggregate {
            kind: MirAggregateKind::Cell,
            row_lengths: vec![1],
            elements: vec![MirAggregateElement::CapturedSequence(sequence)],
        }),
        span: Span::default(),
    });
    assert!(body
        .validate_expression_regions()
        .unwrap_err()
        .contains("expected its single use inside the defining region"));
}

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
