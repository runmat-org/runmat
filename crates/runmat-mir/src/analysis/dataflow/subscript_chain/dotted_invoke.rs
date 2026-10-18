use crate::{MirCallArg, MirCallArgumentRegion, MirDottedInvokeArgument, MirExpressionStep};
use runmat_hir::{IndexKind, IndexResultContext};
use runmat_types::{DynamicReason, SequenceFactInference, ValueFact, ValueSequenceFact};

use super::super::{index_selector_for_operand, simple_rvalue_fact};
use super::callable;
use super::simple_operand_fact;

/// Compatibility adapter for value-only inference callers. New sequence-aware
/// MIR analysis must use [`infer_sequence`] and retain the sequence fact.
pub(super) fn infer_legacy_value(
    member: &ValueFact,
    arguments: &[MirDottedInvokeArgument],
    sequence_use: runmat_types::SequenceUse,
    facts: &[Option<ValueFact>],
) -> (
    runmat_types::FactInference,
    runmat_types::EffectSet,
    runmat_types::CapabilitySet,
) {
    let (inferred, effects, capabilities) = infer_sequence(member, arguments, sequence_use, facts);
    let fact = callable::selected_fact(&inferred.sequence, sequence_use);
    (
        runmat_types::FactInference {
            fact,
            diagnostics: inferred.diagnostics,
        },
        effects,
        capabilities,
    )
}

pub(super) fn infer_sequence(
    member: &ValueFact,
    arguments: &[MirDottedInvokeArgument],
    sequence_use: runmat_types::SequenceUse,
    facts: &[Option<ValueFact>],
) -> (
    SequenceFactInference,
    runmat_types::EffectSet,
    runmat_types::CapabilitySet,
) {
    if arguments.iter().any(|argument| {
        matches!(
            argument,
            MirDottedInvokeArgument::Call(region)
                if !matches!(region.argument, MirCallArg::Single(_))
        )
    }) {
        let mut effects = runmat_types::EffectSet::default();
        effects.0.insert(runmat_types::EffectKind::Unknown);
        effects.0.insert(runmat_types::EffectKind::MayThrow);
        return (
            SequenceFactInference {
                sequence: ValueSequenceFact::dynamic(),
                diagnostics: Vec::new(),
            },
            effects,
            Default::default(),
        );
    }
    if !matches!(&member.kind, runmat_types::ValueKindFact::Callable(_)) {
        // A non-callable member follows ordinary parentheses indexing. The
        // expression preparation remains call-shaped; unknown expanded widths
        // cannot be turned into a false scalar selector fact here.
        let inference = runmat_types::infer_index_sequence(
            member,
            IndexKind::Paren,
            &arguments
                .iter()
                .map(|argument| match argument {
                    MirDottedInvokeArgument::Colon => runmat_types::IndexSelectorFact::Colon,
                    MirDottedInvokeArgument::Call(region) => match &region.argument {
                        MirCallArg::Single(_) => prepared_selector_fact(region, facts),
                        MirCallArg::Expansion(_) | MirCallArg::CapturedSequence(_) => {
                            runmat_types::IndexSelectorFact::Numeric(ValueFact::unknown(
                                DynamicReason::DynamicDispatch,
                            ))
                        }
                    },
                    MirDottedInvokeArgument::Contextual(_) => {
                        runmat_types::IndexSelectorFact::Numeric(ValueFact::unknown(
                            DynamicReason::Unspecified,
                        ))
                    }
                })
                .collect::<Vec<_>>(),
            IndexResultContext::ReadSingle,
        );
        return (inference, Default::default(), Default::default());
    }
    callable::infer(
        member,
        arguments
            .iter()
            .map(|component| match component {
                MirDottedInvokeArgument::Call(region) => match &region.argument {
                    MirCallArg::Single(operand) => simple_operand_fact(operand, facts),
                    MirCallArg::Expansion(_) | MirCallArg::CapturedSequence(_) => {
                        ValueFact::unknown(DynamicReason::DynamicDispatch)
                    }
                },
                MirDottedInvokeArgument::Colon | MirDottedInvokeArgument::Contextual(_) => {
                    ValueFact::unknown(DynamicReason::DynamicDispatch)
                }
            })
            .collect(),
        vec![runmat_types::LiteralValue::Unknown; arguments.len()],
        sequence_use,
    )
    .expect("callable kind was established")
}

fn prepared_selector_fact(
    region: &MirCallArgumentRegion,
    facts: &[Option<ValueFact>],
) -> runmat_types::IndexSelectorFact {
    let mut prepared = facts.to_vec();
    for step in &region.steps {
        if let MirExpressionStep::Let { local, value, .. } = step {
            if local.0 >= prepared.len() {
                prepared.resize(local.0 + 1, None);
            }
            prepared[local.0] = Some(simple_rvalue_fact(value, &prepared));
        }
    }
    match &region.argument {
        MirCallArg::Single(operand) => index_selector_for_operand(operand, &prepared),
        MirCallArg::Expansion(_) | MirCallArg::CapturedSequence(_) => {
            runmat_types::IndexSelectorFact::Numeric(ValueFact::unknown(
                DynamicReason::DynamicDispatch,
            ))
        }
    }
}
