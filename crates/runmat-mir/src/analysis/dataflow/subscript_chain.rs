use crate::{MirOperand, MirSubscriptChain, MirSubscriptStep};
use runmat_types::{DynamicReason, ValueFact, ValueKindFact};

use super::{index_selectors, simple_operand_fact};

mod dotted_invoke;

pub(crate) struct SubscriptChainInference {
    pub inference: runmat_types::FactInference,
    pub disposition: SubscriptPathDisposition,
    pub effects: runmat_types::EffectSet,
    pub capabilities: runmat_types::CapabilitySet,
}

pub(crate) enum SubscriptPathDisposition {
    DefaultOnly,
    UnresolvedDynamic { step: usize },
}

pub(crate) fn infer(
    chain: &MirSubscriptChain,
    facts: &[Option<ValueFact>],
) -> SubscriptChainInference {
    let mut current = simple_operand_fact(&chain.root, facts);
    let mut diagnostics = Vec::new();
    let mut effects = runmat_types::EffectSet::default();
    let mut capabilities = runmat_types::CapabilitySet::default();
    for (position, step) in chain.steps.iter().enumerate() {
        if matches!(
            current.kind,
            ValueKindFact::Object(_) | ValueKindFact::Unknown
        ) {
            return dynamic_dispatch(position, diagnostics, effects, capabilities);
        }
        let terminal_use = if position + 1 == chain.steps.len() {
            chain.sequence_use
        } else {
            runmat_types::SequenceUse::RequireSingle
        };
        let inference = match step {
            MirSubscriptStep::Index(indexing) => runmat_types::infer_index(
                &current,
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            ),
            MirSubscriptStep::Member(member) => {
                runmat_types::infer_member_read(&current, member, terminal_use)
            }
            MirSubscriptStep::DynamicMember(member) => {
                let Some(name) = constant_member_name(member) else {
                    return dynamic_dispatch(position, diagnostics, effects, capabilities);
                };
                runmat_types::infer_member_read(
                    &current,
                    &runmat_types::MemberName(name),
                    terminal_use,
                )
            }
            MirSubscriptStep::DottedInvoke { member, indexing } => {
                let member_inference = runmat_types::infer_member_read(
                    &current,
                    member,
                    runmat_types::SequenceUse::RequireSingle,
                );
                diagnostics.extend(member_inference.diagnostics);
                let (inference, call_effects, call_capabilities) =
                    dotted_invoke::infer_legacy_value(
                        &member_inference.fact,
                        indexing,
                        terminal_use,
                        facts,
                    );
                effects.0.extend(call_effects.0);
                capabilities.0.extend(call_capabilities.0);
                inference
            }
        };
        diagnostics.extend(inference.diagnostics);
        current = inference.fact;
    }
    SubscriptChainInference {
        inference: runmat_types::FactInference {
            fact: current,
            diagnostics,
        },
        disposition: SubscriptPathDisposition::DefaultOnly,
        effects,
        capabilities,
    }
}

pub(crate) fn infer_sequence(
    chain: &MirSubscriptChain,
    facts: &[Option<ValueFact>],
) -> runmat_types::SequenceFactInference {
    let mut current = simple_operand_fact(&chain.root, facts);
    let mut diagnostics = Vec::new();
    for (position, step) in chain.steps.iter().enumerate() {
        if matches!(
            current.kind,
            ValueKindFact::Object(_) | ValueKindFact::Unknown
        ) {
            return runmat_types::SequenceFactInference {
                sequence: runmat_types::ValueSequenceFact::dynamic(),
                diagnostics,
            };
        }
        let terminal = position + 1 == chain.steps.len();
        if terminal {
            let mut inferred = match step {
                MirSubscriptStep::Index(indexing) => runmat_types::infer_index_sequence(
                    &current,
                    indexing.kind,
                    &index_selectors(indexing, facts),
                    indexing.result_context,
                ),
                MirSubscriptStep::Member(member) => {
                    runmat_types::infer_member_sequence(&current, member)
                }
                MirSubscriptStep::DynamicMember(member) => {
                    let Some(name) = constant_member_name(member) else {
                        return runmat_types::SequenceFactInference {
                            sequence: runmat_types::ValueSequenceFact::dynamic(),
                            diagnostics,
                        };
                    };
                    runmat_types::infer_member_sequence(&current, &runmat_types::MemberName(name))
                }
                MirSubscriptStep::DottedInvoke { member, indexing } => {
                    let member_inference = runmat_types::infer_member_read(
                        &current,
                        member,
                        runmat_types::SequenceUse::RequireSingle,
                    );
                    diagnostics.extend(member_inference.diagnostics);
                    let (inferred, _, _) = dotted_invoke::infer_sequence(
                        &member_inference.fact,
                        indexing,
                        chain.sequence_use,
                        facts,
                    );
                    inferred
                }
            };
            diagnostics.append(&mut inferred.diagnostics);
            inferred.diagnostics = diagnostics;
            return inferred;
        }

        let inference = match step {
            MirSubscriptStep::Index(indexing) => runmat_types::infer_index(
                &current,
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            ),
            MirSubscriptStep::Member(member) => runmat_types::infer_member_read(
                &current,
                member,
                runmat_types::SequenceUse::RequireSingle,
            ),
            MirSubscriptStep::DynamicMember(member) => {
                let Some(name) = constant_member_name(member) else {
                    return runmat_types::SequenceFactInference {
                        sequence: runmat_types::ValueSequenceFact::dynamic(),
                        diagnostics,
                    };
                };
                runmat_types::infer_member_read(
                    &current,
                    &runmat_types::MemberName(name),
                    runmat_types::SequenceUse::RequireSingle,
                )
            }
            MirSubscriptStep::DottedInvoke { member, indexing } => {
                let member_inference = runmat_types::infer_member_read(
                    &current,
                    member,
                    runmat_types::SequenceUse::RequireSingle,
                );
                diagnostics.extend(member_inference.diagnostics);
                let (inference, _, _) = dotted_invoke::infer_legacy_value(
                    &member_inference.fact,
                    indexing,
                    runmat_types::SequenceUse::RequireSingle,
                    facts,
                );
                inference
            }
        };
        diagnostics.extend(inference.diagnostics);
        current = inference.fact;
    }
    runmat_types::SequenceFactInference {
        sequence: runmat_types::ValueSequenceFact::single(current),
        diagnostics,
    }
}

fn dynamic_dispatch(
    step: usize,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
    effects: runmat_types::EffectSet,
    capabilities: runmat_types::CapabilitySet,
) -> SubscriptChainInference {
    SubscriptChainInference {
        inference: runmat_types::FactInference {
            fact: ValueFact::unknown(DynamicReason::DynamicDispatch),
            diagnostics,
        },
        disposition: SubscriptPathDisposition::UnresolvedDynamic { step },
        effects,
        capabilities,
    }
}

fn constant_member_name(operand: &MirOperand) -> Option<String> {
    let MirOperand::Constant(crate::MirConstant::String(value)) = operand else {
        return None;
    };
    Some(value.runtime_text())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn constant_dynamic_member_uses_canonical_string_without_retrimming() {
        let name = "'quoted'".to_owned();
        let field = ValueFact::scalar(ValueKindFact::Logical);
        let root = ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact::scalar(
            BTreeMap::from([(name.clone(), field.clone())]),
            true,
        )));
        let chain = MirSubscriptChain {
            root: MirOperand::Local(crate::MirLocalId(0)),
            steps: vec![MirSubscriptStep::DynamicMember(MirOperand::Constant(
                crate::MirConstant::String(runmat_hir::StringLiteral("'''quoted'''".to_owned())),
            ))],
            sequence_use: runmat_types::SequenceUse::RequireSingle,
            context: runmat_types::ObjectIndexingContext::Expression,
        };
        let inferred = infer(&chain, &[Some(root)]);
        assert_eq!(inferred.inference.fact, field);
        assert!(matches!(
            inferred.disposition,
            SubscriptPathDisposition::DefaultOnly
        ));
    }
}
