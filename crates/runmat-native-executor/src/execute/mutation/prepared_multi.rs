use std::collections::BTreeMap;

use runmat_mir::{MirOutputTargetList, MirPlace};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::sequence::{DestinationLayout, PreparedSequenceDestination};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::continuation;
use crate::execute::state::HostState;

mod fixed_source;
mod preparation;
mod roots;
mod target;
use fixed_source::FixedSourceOwner;
pub(in crate::execute) use preparation::PendingMultiPreparation;
pub(in crate::execute) use preparation::{has_fixed_end_context, resolve_fixed_end};
use roots::local_value;

#[expect(
    clippy::large_enum_variant,
    reason = "Sequence retains its move-once destination inline; boxing adds an assignment allocation, while hoisting it creates invalid Place/Discard destination states"
)]
pub(super) enum PreparedTarget {
    Place(MirPlace),
    Sequence {
        root: runmat_mir::MirLocalId,
        destination: Option<PreparedSequenceDestination>,
        absent_root: Option<NativeValueRef>,
        root_creation: runmat_mir::SequenceRootCreation,
        original_root: Option<NativeValueRef>,
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
    fixed: Option<FixedPreparedAssignment>,
}

struct FixedPreparedAssignment {
    source: FixedSourceOwner,
    root: runmat_mir::MirLocalId,
    original: NativeValueRef,
    destination: Option<PreparedSequenceDestination>,
    rhs: Option<NativeValueRef>,
    updated: Option<NativeValueRef>,
    original_released: bool,
}

impl PreparedMultiAssignment {
    pub(in crate::execute) fn release_fixed_original(
        &mut self,
        arena: &mut crate::memory::ValueArena,
    ) -> NativeExecutorResult<()> {
        if let Some(fixed) = &mut self.fixed {
            if !fixed.original_released {
                arena.release(fixed.original)?;
                fixed.original_released = true;
            }
        }
        Ok(())
    }
    pub(in crate::execute) fn prepare_fixed(
        state: &mut HostState,
        phase: &runmat_mir::MirFixedPreparation,
    ) -> NativeExecutorResult<Option<Self>> {
        preparation::prepare_fixed(state, phase)
    }

    pub(in crate::execute) fn matches_fixed(
        &self,
        owner: &runmat_mir::MirPreparedAssignment,
    ) -> bool {
        self.fixed
            .as_ref()
            .is_some_and(|fixed| fixed.source.matches(owner))
    }

    pub(in crate::execute) fn fixed_values_bound(&self) -> bool {
        self.fixed.as_ref().is_some_and(|fixed| fixed.rhs.is_some())
    }

    pub(in crate::execute) fn bind_fixed_value(
        &mut self,
        value: NativeValueRef,
    ) -> NativeExecutorResult<()> {
        let fixed = self.fixed.as_mut().ok_or_else(|| {
            NativeExecutorError::Host("fixed RHS bound to a sequence destination".into())
        })?;
        if fixed.rhs.is_some() {
            return Err(NativeExecutorError::Host(
                "fixed RHS was already bound".into(),
            ));
        }
        fixed.rhs = Some(value);
        Ok(())
    }
    pub(in crate::execute) fn prepare(
        state: &mut HostState,
        targets: &MirOutputTargetList,
    ) -> NativeExecutorResult<Self> {
        preparation::prepare(state, targets)
    }

    pub(in crate::execute) const fn cardinality(&self) -> usize {
        if self.fixed.is_some() {
            1
        } else {
            self.layout.total()
        }
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
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>()
            })
            .collect::<NativeExecutorResult<_>>()?;
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
                .chain(self.roots.values().copied())
                .chain(self.fixed.iter().flat_map(|fixed| {
                    std::iter::once(fixed.original)
                        .chain(fixed.rhs)
                        .chain(fixed.updated)
                })),
        )
    }

    pub(in crate::execute) fn commit(&mut self, state: &mut HostState) -> NativeExecutorResult<()> {
        if let Some(fixed) = &mut self.fixed {
            if fixed.updated.is_none() {
                let rhs = fixed.rhs.ok_or_else(|| {
                    NativeExecutorError::Host("fixed assignment has no bound RHS".into())
                })?;
                let value = continuation::assign_prepared_fixed(
                    state,
                    &mut fixed.destination,
                    fixed.original,
                    rhs,
                )?;
                fixed.updated = Some(value);
            }
            return state.set_local(
                fixed.root.0,
                fixed.updated.expect("completed fixed publication"),
            );
        }
        for index in 0..self.targets.len() {
            let target = &mut self.targets[index];
            if index < self.next_target {
                match target {
                    PreparedTarget::Place(place) => continuation::skip_completed(state, place)?,
                    PreparedTarget::Sequence { .. } => continuation::skip_completed_region(state)?,
                    PreparedTarget::Discard => {}
                }
                continue;
            }
            let values = self.values[index]
                .iter()
                .map(|value| state.arena.get(*value).cloned())
                .collect::<NativeExecutorResult<Vec<_>>>()?;
            match target {
                PreparedTarget::Discard => {}
                PreparedTarget::Place(place) => {
                    let rhs = values.into_iter().next().ok_or_else(|| {
                        NativeExecutorError::Host(
                            "prepared fixed output target has no assigned value".into(),
                        )
                    })?;
                    let (local, reference) = match &*place {
                        // A plain output initializes its local; unlike a
                        // nested mutation, it has no previous receiver to read.
                        MirPlace::Local(local) => {
                            (*local, state.arena.insert(rhs, Some(&state.runtime))?)
                        }
                        _ => {
                            let local = target::root_local(place)?;
                            let creation = place.assignment_creation_policy(false);
                            let root = match self.roots.get(&local) {
                                Some(reference) => state.arena.get(*reference)?.clone(),
                                None => {
                                    continuation::assignment_root_for_place(state, place, creation)?
                                }
                            };
                            let (local, updated) = continuation::assign_to_root(
                                state, place, root, rhs, false, creation,
                            )?;
                            (local, state.arena.insert(updated, Some(&state.runtime))?)
                        }
                    };
                    self.roots.insert(local, reference);
                }
                PreparedTarget::Sequence {
                    root,
                    destination,
                    absent_root,
                    root_creation,
                    original_root,
                } => {
                    let current = match *original_root {
                        Some(original) => state.arena.get(original)?.clone(),
                        None => match self.roots.get(root) {
                            Some(reference) => state.arena.get(*reference)?.clone(),
                            None => {
                                if *root_creation == runmat_mir::SequenceRootCreation::EmptyCell
                                    && state.sequence_root_still_absent(root.0)?
                                {
                                    match *absent_root {
                                        Some(reference) => state.arena.get(reference)?.clone(),
                                        None => HostState::empty_cell_root(),
                                    }
                                } else {
                                    local_value(state, *root)?
                                }
                            }
                        },
                    };
                    let updated = continuation::assign_prepared_sequence(
                        state,
                        destination,
                        current,
                        values,
                    )?;
                    self.roots
                        .insert(*root, state.arena.insert(updated, Some(&state.runtime))?);
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
