use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_types::{
    CapabilitySet, DistributedValueId, EffectSet, FactJoin, FactWiden, LiteralValue, ValueFact,
    ValueSequenceFact,
};

use crate::{MirBody, MirLocalId, MirPlace, MirSequenceLocalId, SequenceRootCreation};

use super::super::InitFact;
use crate::analysis::workspace_effect::current_scope_bindings;

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

    fn runtime_workspace_lookup() -> Self {
        Self {
            fact: None,
            literal: LiteralValue::Unknown,
            assignment: InitFact::RuntimeWorkspaceLookup,
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

    fn invalidate_for_current_workspace_effect(&mut self) {
        *self = Self::runtime_workspace_lookup();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlowState {
    pub locals: Vec<LocalState>,
    current_workspace_bindings: Arc<[MirLocalId]>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub distributed: BTreeMap<DistributedValueId, ValueFact>,
    sequences: BTreeMap<MirSequenceLocalId, ValueSequenceFact>,
    /// Abstract counterpart of the single invocation-owned fixed destination.
    /// RHS workspace changes cannot replace this captured receiver fact.
    pub fixed_receiver: Option<(runmat_hir::StmtId, MirLocalId, LocalState)>,
}

impl FlowState {
    /// Only the local lifecycle proves an absent binding. An unknown value,
    /// parameter, capture, or declared workspace slot cannot be used as a
    /// creation receipt for a new cell root.
    pub(crate) fn absent_cell_root_for_place(
        &self,
        place: &MirPlace,
        creation: SequenceRootCreation,
    ) -> Option<(MirLocalId, ValueFact)> {
        if creation != SequenceRootCreation::EmptyCell {
            return None;
        }
        let local = place.root_local()?;
        (self.locals.get(local.0)?.assignment == InitFact::Unassigned)
            .then(|| (local, runmat_types::infer_cell_aggregate(&[]).fact))
    }

    pub fn entry(
        body: &MirBody,
        parameters: &[ValueFact],
        captures: &[ValueFact],
        constructor_instance: Option<MirLocalId>,
    ) -> Self {
        let mut state = Self {
            locals: vec![LocalState::unassigned(); body.locals.len()],
            current_workspace_bindings: current_scope_bindings(body).into(),
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            distributed: BTreeMap::new(),
            sequences: BTreeMap::new(),
            fixed_receiver: None,
        };
        let mut parameter = 0;
        let mut capture = 0;
        for local in &body.locals {
            if local.binding == body.abi.varargout && local.binding.is_some() {
                state.locals[local.id.0] = LocalState::assigned(
                    ValueFact::proven(
                        runmat_types::ValueKindFact::Cell(runmat_types::CellFact {
                            element: Box::new(ValueFact::unknown(
                                runmat_types::DynamicReason::RuntimeValue,
                            )),
                            elements: Vec::new(),
                            elements_complete: true,
                        }),
                        runmat_types::ShapeFact::from(vec![Some(1), Some(0)]),
                        runmat_types::StorageFact::Dense,
                    ),
                    LiteralValue::Unknown,
                );
                continue;
            }
            if local.binding.is_some_and(|binding| {
                body.abi.implicit_nargin == Some(binding)
                    || body.abi.implicit_nargout == Some(binding)
            }) {
                state.locals[local.id.0] = LocalState::assigned(
                    ValueFact::scalar(runmat_types::ValueKindFact::Numeric(
                        runmat_types::NumericFact {
                            class: runmat_types::NumericClass::Double,
                            domain: runmat_types::NumericDomain::Real,
                        },
                    )),
                    LiteralValue::Unknown,
                );
                continue;
            }
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
                crate::MirLocalKind::IncomingWorkspace => None,
                _ => None,
            };
            if matches!(
                local.kind,
                crate::MirLocalKind::Parameter
                    | crate::MirLocalKind::Capture
                    | crate::MirLocalKind::IncomingWorkspace
            ) {
                state.locals[local.id.0] = LocalState::assigned(
                    fact.unwrap_or_else(|| {
                        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue)
                    }),
                    LiteralValue::Unknown,
                );
            } else if matches!(local.kind, crate::MirLocalKind::ExternalWorkspace) {
                state.locals[local.id.0] = LocalState::runtime_workspace_lookup();
            }
        }
        if let Some(local) = constructor_instance {
            state.locals[local.0] = LocalState::assigned(
                ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
                LiteralValue::Unknown,
            );
        }
        state
    }

    pub(crate) fn invalidate_current_workspace_bindings(&mut self) {
        for binding in self.current_workspace_bindings.iter() {
            self.locals[binding.0].invalidate_for_current_workspace_effect();
        }
    }

    pub fn join_from(&mut self, incoming: &Self, widen: bool) -> bool {
        let mut changed = false;
        let receiver = match (&self.fixed_receiver, &incoming.fixed_receiver) {
            (Some((id, root, current)), Some((next_id, next_root, next)))
                if id == next_id && root == next_root =>
            {
                let assignment = join_assignment(current.assignment, next.assignment);
                Some((
                    *id,
                    *root,
                    LocalState {
                        fact: if assignment == InitFact::RuntimeWorkspaceLookup {
                            None
                        } else {
                            join_optional(current.fact.as_ref(), next.fact.as_ref(), widen)
                        },
                        literal: if current.literal == next.literal {
                            current.literal.clone()
                        } else {
                            LiteralValue::Unknown
                        },
                        assignment,
                    },
                ))
            }
            _ => None,
        };
        if self.fixed_receiver != receiver {
            self.fixed_receiver = receiver;
            changed = true;
        }
        for (slot, next) in self.locals.iter_mut().zip(&incoming.locals) {
            let assignment = join_assignment(slot.assignment, next.assignment);
            let fact = if assignment == InitFact::RuntimeWorkspaceLookup {
                None
            } else {
                join_optional(slot.fact.as_ref(), next.fact.as_ref(), widen)
            };
            let literal = if assignment == InitFact::RuntimeWorkspaceLookup {
                LiteralValue::Unknown
            } else if slot.literal == next.literal {
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
        (InitFact::RuntimeWorkspaceLookup, _) | (_, InitFact::RuntimeWorkspaceLookup) => {
            InitFact::RuntimeWorkspaceLookup
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
            catch_scopes: Default::default(),
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
        let mut left = FlowState::entry(&body, &[], &[], None);
        left.set_sequence(id, ValueSequenceFact::single(number()));
        let right = FlowState::entry(&body, &[], &[], None);
        assert!(left.join_from(&right, false));
        assert_eq!(
            left.sequence_facts().get(&id),
            Some(&ValueSequenceFact::dynamic())
        );

        let mut first = FlowState::entry(&body, &[], &[], None);
        first.set_sequence(id, ValueSequenceFact::single(number()));
        let mut second = FlowState::entry(&body, &[], &[], None);
        second.set_sequence(
            id,
            ValueSequenceFact::single(ValueFact::unknown(DynamicReason::RuntimeValue)),
        );
        assert!(first.join_from(&second, false));
        assert_eq!(first.sequence_facts()[&id].outputs.len(), 1);
        assert!(!first.sequence_facts()[&id].variadic);
    }
}
