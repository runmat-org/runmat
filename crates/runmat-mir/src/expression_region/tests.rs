use super::*;
use crate::{
    MirAggregateElement, MirAggregateKind, MirLocalId, MirOperand, MirRvalue, MirSequenceLocalId,
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
fn subscript_chain_rejects_invalid_contextual_dotted_argument() {
    let invalid = MirExpressionRegion {
        steps: vec![
            let_step(1, MirRvalue::Use(MirOperand::Local(MirLocalId(2)))),
            let_step(
                2,
                MirRvalue::Use(MirOperand::Constant(crate::MirConstant::Bool(true))),
            ),
        ],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let chain = crate::MirSubscriptChain {
        root: MirOperand::Local(MirLocalId(0)),
        steps: vec![crate::MirSubscriptStep::DottedInvoke {
            semantic_fallback: None,
            member: runmat_hir::MemberName("method".to_owned()),
            arguments: vec![crate::MirDottedInvokeArgument::Contextual(invalid)],
        }],
        sequence_use: runmat_types::SequenceUse::RequireSingle,
        context: runmat_types::ObjectIndexingContext::Expression,
    };
    assert_eq!(
        chain.validate(),
        Err(crate::MirSubscriptChainError::InvalidDottedArgumentRegion)
    );
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
fn prepared_region_keeps_contextual_end_validation_distinct() {
    let steps = vec![let_step(
        1,
        MirRvalue::Use(MirOperand::Constant(crate::MirConstant::Bool(true))),
    )];
    let result = MirOperand::Local(MirLocalId(1));
    assert!(MirPreparedExpressionRegion {
        steps: steps.clone(),
        result: result.clone(),
    }
    .validate()
    .is_ok());
    assert!(MirExpressionRegion { steps, result }
        .validate()
        .unwrap_err()
        .contains("does not depend on the indexing context"));

    assert!(MirPreparedExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    }
    .validate()
    .unwrap_err()
    .contains("without its own indexing receiver"));

    let nested_contextual = MirExpressionRegion {
        steps: vec![let_step(3, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(3)),
    };
    let owned_end = MirPreparedExpressionRegion {
        steps: vec![let_step(
            4,
            MirRvalue::Index {
                base: MirOperand::Constant(crate::MirConstant::EmptyArray),
                indexing: crate::MirIndexing {
                    argument_spans: None,
                    kind: runmat_hir::IndexKind::Paren,
                    plan: crate::MirIndexPlan::Scalar,
                    components: vec![crate::MirIndexComponent::ContextualExpr(nested_contextual)],
                    result_context: runmat_hir::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
        )],
        result: MirOperand::Local(MirLocalId(4)),
    };
    assert!(owned_end.validate().is_ok());
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

#[path = "tests/body.rs"]
mod body;
#[path = "tests/requirements.rs"]
mod requirements;
