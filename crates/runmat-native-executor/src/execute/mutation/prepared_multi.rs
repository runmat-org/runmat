use std::collections::BTreeMap;

use runmat_mir::{MirOutputTargetList, MirPlace};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::sequence::{DestinationLayout, PreparedSequenceDestination};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{continuation, root_local_value};
use crate::execute::state::HostState;

mod roots;
mod target;
mod preparation;
pub(super) use preparation::PendingMultiPreparation;
use roots::local_value;

#[derive(Clone)]
pub(super) enum PreparedTarget {
    Place(MirPlace),
    Sequence {
        root: runmat_mir::MirLocalId,
        destination: PreparedSequenceDestination,
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
                    let local = target::root_local(&place)?;
                    let root = match self.roots.get(&local) {
                        Some(reference) => state.arena.get(*reference)?.clone(),
                        None => root_local_value(state, &place)?,
                    };
                    let (local, updated) =
                        continuation::assign_to_root(state, &place, root, rhs, false, true)?;
                    self.roots.insert(local, state.arena.insert(updated));
                }
                PreparedTarget::Sequence { root, destination } => {
                    let current = match self.roots.get(&root) {
                        Some(reference) => state.arena.get(*reference)?.clone(),
                        None => local_value(state, root)?,
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
