use std::collections::BTreeMap;

use runmat_types::{
    CapabilitySet, DistributedValueId, EffectSet, FactJoin, FactWiden, LiteralValue, ValueFact,
    ValueSequenceFact,
};

use crate::{MirBody, MirSequenceLocalId};

use super::super::InitFact;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalState {
    pub fact: Option<ValueFact>,
    pub literal: LiteralValue,
    pub assignment: InitFact,
}

impl LocalState {
    fn unassigned() -> Self {
        Self {
            fact: None,
            literal: LiteralValue::Unknown,
            assignment: InitFact::Unassigned,
        }
    }

    pub fn assigned(fact: ValueFact, literal: LiteralValue) -> Self {
        Self {
            fact: Some(fact),
            literal,
            assignment: InitFact::DefinitelyAssigned,
        }
    }

    pub fn set(&mut self, fact: ValueFact, literal: LiteralValue) {
        self.fact = Some(fact);
        self.literal = literal;
        self.assignment = InitFact::DefinitelyAssigned;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlowState {
    pub locals: Vec<LocalState>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub distributed: BTreeMap<DistributedValueId, ValueFact>,
    sequences: BTreeMap<MirSequenceLocalId, ValueSequenceFact>,
}

impl FlowState {
    pub fn entry(body: &MirBody, parameters: &[ValueFact], captures: &[ValueFact]) -> Self {
        let mut state = Self {
            locals: vec![LocalState::unassigned(); body.locals.len()],
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            distributed: BTreeMap::new(),
            sequences: BTreeMap::new(),
        };
        let mut parameter = 0;
        let mut capture = 0;
        for local in &body.locals {
            let fact = match local.kind {
                crate::MirLocalKind::Parameter => {
                    let value = parameters.get(parameter).cloned();
                    parameter += 1;
                    value
                }
                crate::MirLocalKind::Capture => {
                    let value = captures.get(capture).cloned();
                    capture += 1;
                    value
                }
                _ => None,
            };
            if matches!(
                local.kind,
                crate::MirLocalKind::Parameter | crate::MirLocalKind::Capture
            ) {
                state.locals[local.id.0] = LocalState::assigned(
                    fact.unwrap_or_else(|| {
                        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue)
                    }),
                    LiteralValue::Unknown,
                );
            }
        }
        state
    }

    pub fn join_from(&mut self, incoming: &Self, widen: bool) -> bool {
        let mut changed = false;
        for (slot, next) in self.locals.iter_mut().zip(&incoming.locals) {
            let assignment = join_assignment(slot.assignment, next.assignment);
            let fact = join_optional(slot.fact.as_ref(), next.fact.as_ref(), widen);
            let literal = if slot.literal == next.literal {
                slot.literal.clone()
            } else {
                LiteralValue::Unknown
            };
            if slot.assignment != assignment || slot.fact != fact || slot.literal != literal {
                slot.assignment = assignment;
                slot.fact = fact;
                slot.literal = literal;
                changed = true;
            }
        }
        let effects = EffectSet(self.effects.0.union(&incoming.effects.0).copied().collect());
        let capabilities = CapabilitySet(
            self.capabilities
                .0
                .union(&incoming.capabilities.0)
                .copied()
                .collect(),
        );
        if self.effects != effects {
            self.effects = effects;
            changed = true;
        }
        if self.capabilities != capabilities {
            self.capabilities = capabilities;
            changed = true;
        }
        for (id, next) in &incoming.distributed {
            match self.distributed.get(id) {
                Some(current) => {
                    let joined = if widen {
                        current.widen(next)
                    } else {
                        current.join(next)
                    };
                    if &joined != current {
                        self.distributed.insert(*id, joined);
                        changed = true;
                    }
                }
                None => {
                    self.distributed.insert(*id, next.clone());
                    changed = true;
                }
            }
        }
        let sequence_ids = self
            .sequences
            .keys()
            .chain(incoming.sequences.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        for id in sequence_ids {
            match (self.sequences.get(&id), incoming.sequences.get(&id)) {
                (Some(current), Some(next)) => {
                    let joined = if widen {
                        current.widen(next)
                    } else {
                        current.join(next)
                    };
                    if &joined != current {
                        self.sequences.insert(id, joined);
                        changed = true;
                    }
                }
                (Some(current), None) if current != &ValueSequenceFact::dynamic() => {
                    self.sequences.insert(id, ValueSequenceFact::dynamic());
                    changed = true;
                }
                (None, Some(_)) => {
                    self.sequences.insert(id, ValueSequenceFact::dynamic());
                    changed = true;
                }
                (Some(_), None) | (None, None) => {}
            }
        }
        changed
    }

    pub fn final_facts(&self) -> impl Iterator<Item = (usize, &LocalState)> {
        self.locals.iter().enumerate()
    }

    pub fn value_facts(&self) -> Vec<Option<ValueFact>> {
        self.locals.iter().map(|local| local.fact.clone()).collect()
    }

    pub fn replace_value_facts(&mut self, facts: Vec<Option<ValueFact>>) {
        for (local, fact) in self.locals.iter_mut().zip(facts) {
            if fact != local.fact {
                local.fact = fact;
                local.literal = LiteralValue::Unknown;
                local.assignment = InitFact::DefinitelyAssigned;
            }
        }
    }

    pub fn set_sequence(&mut self, id: MirSequenceLocalId, fact: ValueSequenceFact) {
        self.sequences.insert(id, fact);
    }

    pub fn sequence_facts(&self) -> &BTreeMap<MirSequenceLocalId, ValueSequenceFact> {
        &self.sequences
    }
}

fn join_optional(
    current: Option<&ValueFact>,
    next: Option<&ValueFact>,
    widen: bool,
) -> Option<ValueFact> {
    match (current, next) {
        (Some(current), Some(next)) if widen => Some(current.widen(next)),
        (Some(current), Some(next)) => Some(current.join(next)),
        (Some(current), None) => Some(current.clone()),
        (None, Some(next)) => Some(next.clone()),
        (None, None) => None,
    }
}

fn join_assignment(left: InitFact, right: InitFact) -> InitFact {
    match (left, right) {
        (InitFact::Unassigned, InitFact::Unassigned) => InitFact::Unassigned,
        (InitFact::DefinitelyAssigned, InitFact::DefinitelyAssigned) => {
            InitFact::DefinitelyAssigned
        }
        _ => InitFact::MaybeAssigned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::{FunctionAbi, FunctionId};
    use runmat_types::{DynamicReason, NumericClass, NumericDomain, NumericFact, ValueKindFact};

    fn empty_body() -> MirBody {
        MirBody {
            function: FunctionId(0),
            abi: FunctionAbi {
                fixed_inputs: Vec::new(),
                varargin: None,
                fixed_outputs: Vec::new(),
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            locals: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn number() -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    }

    #[test]
    fn sequence_facts_join_exactly_only_when_all_predecessors_define_them() {
        let body = empty_body();
        let id = MirSequenceLocalId(3);
        let mut left = FlowState::entry(&body, &[], &[]);
        left.set_sequence(id, ValueSequenceFact::single(number()));
        let right = FlowState::entry(&body, &[], &[]);
        assert!(left.join_from(&right, false));
        assert_eq!(
            left.sequence_facts().get(&id),
            Some(&ValueSequenceFact::dynamic())
        );

        let mut first = FlowState::entry(&body, &[], &[]);
        first.set_sequence(id, ValueSequenceFact::single(number()));
        let mut second = FlowState::entry(&body, &[], &[]);
        second.set_sequence(
            id,
            ValueSequenceFact::single(ValueFact::unknown(DynamicReason::RuntimeValue)),
        );
        assert!(first.join_from(&second, false));
        assert_eq!(first.sequence_facts()[&id].outputs.len(), 1);
        assert!(!first.sequence_facts()[&id].variadic);
    }
}
