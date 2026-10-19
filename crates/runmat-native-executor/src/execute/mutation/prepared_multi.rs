use std::collections::BTreeMap;

use runmat_mir::{MirOutputTargetList, MirPlace};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::sequence::{DestinationLayout, PreparedSequenceDestination};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::continuation;
use crate::execute::state::HostState;

mod roots;
mod target;
mod preparation;
pub(in crate::execute) use preparation::PendingMultiPreparation;
use roots::local_value;

#[derive(Clone)]
pub(super) enum PreparedTarget {
    Place(MirPlace),
    Sequence {
        root: runmat_mir::MirLocalId,
        destination: PreparedSequenceDestination,
        absent_root: Option<NativeValueRef>,
        root_creation: runmat_mir::SequenceRootCreation,
    },
    Discard,
}

pub(in crate::execute) struct PreparedMultiAssignment {
    targets: Vec<PreparedTarget>,
    layout: DestinationLayout,
    values: Vec<Vec<NativeValueRef>>,
    retained: Vec<NativeValueRef>,
    next_target: usize,
    roots: BTreeMap<runmat_mir::MirLocalId, NativeValueRef>,
}

impl PreparedMultiAssignment {
    pub(in crate::execute) fn prepare(
        state: &mut HostState,
        targets: &MirOutputTargetList,
    ) -> NativeExecutorResult<Self> {
        preparation::prepare(state, targets)
    }

    pub(in crate::execute) const fn cardinality(&self) -> usize {
        self.layout.total()
    }

    pub(in crate::execute) fn skip_preparation_regions(
        &self,
        state: &mut HostState,
    ) -> NativeExecutorResult<()> {
        for target in &self.targets {
            if matches!(target, PreparedTarget::Sequence { .. }) {
                continuation::skip_completed_region(state)?;
            }
        }
        Ok(())
    }

    pub(in crate::execute) fn bind_values(
        mut self,
        state: &mut HostState,
        values: Vec<NativeValueRef>,
    ) -> NativeExecutorResult<Self> {
        let values = values
            .into_iter()
            .map(|value| state.arena.get(value).cloned())
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        self.values = self
            .layout
            .clone()
            .distribute(values)?
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| state.arena.insert(value))
                    .collect()
            })
            .collect();
        Ok(self)
    }

    pub(in crate::execute) fn root_references(
        &self,
    ) -> impl Iterator<Item = (u32, NativeValueRef)> + '_ {
        (0_u32..).zip(
            self.values
                .iter()
                .flatten()
                .copied()
                .chain(self.retained.iter().copied())
                .chain(self.targets.iter().filter_map(|target| match target {
                    PreparedTarget::Sequence { absent_root, .. } => *absent_root,
                    _ => None,
                }))
                .chain(self.roots.values().copied()),
        )
    }

    pub(in crate::execute) fn commit(&mut self, state: &mut HostState) -> NativeExecutorResult<()> {
        for index in 0..self.targets.len() {
            let target = self.targets[index].clone();
            if index < self.next_target {
                match &target {
                    PreparedTarget::Place(place) => continuation::skip_completed(state, place)?,
                    PreparedTarget::Sequence { .. } => continuation::skip_completed_region(state)?,
                    PreparedTarget::Discard => {}
                }
                continue;
            }
            let values = self.values[index].clone();
            let values = values
                .into_iter()
                .map(|value| state.arena.get(value).cloned())
                .collect::<NativeExecutorResult<Vec<_>>>()?;
            match target {
                PreparedTarget::Discard => {}
                PreparedTarget::Place(place) => {
                    let rhs = values.into_iter().next().ok_or_else(|| {
                        NativeExecutorError::Host(
                            "prepared fixed output target has no assigned value".into(),
                        )
                    })?;
                    let (local, reference) = match &place {
                        // A plain output initializes its local; unlike a
                        // nested mutation, it has no previous receiver to read.
                        MirPlace::Local(local) => (*local, state.arena.insert(rhs)),
                        _ => {
                            let local = target::root_local(&place)?;
                            let creation = place.assignment_creation_policy(false);
                            let root = match self.roots.get(&local) {
                                Some(reference) => state.arena.get(*reference)?.clone(),
                                None => continuation::assignment_root_for_place(
                                    state, &place, creation,
                                )?,
                            };
                            let (local, updated) = continuation::assign_to_root(
                                state, &place, root, rhs, false, creation,
                            )?;
                            (local, state.arena.insert(updated))
                        }
                    };
                    self.roots.insert(local, reference);
                }
                PreparedTarget::Sequence {
                    root,
                    destination,
                    absent_root,
                    root_creation,
                } => {
                    let current = match self.roots.get(&root) {
                        Some(reference) => state.arena.get(*reference)?.clone(),
                        None => {
                            if root_creation == runmat_mir::SequenceRootCreation::EmptyCell
                                && state.sequence_root_still_absent(root.0)?
                            {
                                match absent_root {
                                    Some(reference) => state.arena.get(reference)?.clone(),
                                    None => HostState::empty_cell_root(),
                                }
                            } else {
                                local_value(state, root)?
                            }
                        }
                    };
                    let updated = continuation::assign_prepared_sequence(
                        state,
                        destination,
                        current,
                        values,
                    )?;
                    self.roots.insert(root, state.arena.insert(updated));
                }
            }
            self.next_target = index + 1;
        }
        for (local, reference) in std::mem::take(&mut self.roots) {
            state.set_local(local.0, reference)?;
        }
        Ok(())
    }
}
