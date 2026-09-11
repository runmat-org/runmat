use std::collections::BTreeMap;

use runmat_mir::{MirOutputTarget, MirOutputTargetList, MirPlace};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::sequence::{
    DestinationCardinality, DestinationLayout, PreparedSequenceDestination,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::assign_place_to_root;
use crate::execute::state::HostState;

mod roots;
mod target;
use roots::{local_value, root_for_update};
use target::prepare_sequence_target;

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
}

impl PreparedMultiAssignment {
    pub(in crate::execute) fn prepare(
        state: &mut HostState,
        targets: &MirOutputTargetList,
    ) -> NativeExecutorResult<Self> {
        let mut prepared = Vec::with_capacity(targets.targets.len());
        let mut cardinalities = Vec::with_capacity(targets.targets.len());
        for target in &targets.targets {
            match target {
                MirOutputTarget::Place(place) => {
                    prepared.push(PreparedTarget::Place(place.clone()));
                    cardinalities.push(DestinationCardinality::Fixed);
                }
                MirOutputTarget::Discard => {
                    prepared.push(PreparedTarget::Discard);
                    cardinalities.push(DestinationCardinality::Discard);
                }
                MirOutputTarget::Sequence(target) => {
                    let (root, destination) = prepare_sequence_target(state, target)?;
                    cardinalities.push(DestinationCardinality::Sequence(destination.cardinality()));
                    prepared.push(PreparedTarget::Sequence { root, destination });
                }
            }
        }
        Ok(Self {
            targets: prepared,
            layout: DestinationLayout::new(cardinalities)?,
            values: Vec::new(),
        })
    }

    pub(in crate::execute) const fn cardinality(&self) -> usize {
        self.layout.total()
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
        (0_u32..).zip(self.values.iter().flatten().copied())
    }

    pub(in crate::execute) fn commit(self, state: &mut HostState) -> NativeExecutorResult<()> {
        let mut roots = BTreeMap::<runmat_mir::MirLocalId, Value>::new();
        for (target, values) in self.targets.into_iter().zip(self.values) {
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
                    let root = root_for_update(state, &mut roots, &place)?;
                    let (local, updated) =
                        assign_place_to_root(state, &place, root, rhs, false, true)?;
                    roots.insert(local, updated);
                }
                PreparedTarget::Sequence { root, destination } => {
                    let current = match roots.remove(&root) {
                        Some(value) => value,
                        None => local_value(state, root)?,
                    };
                    let updated = super::super::sync::complete(
                        &state.runtime,
                        destination.assign(current, values, Some(&state.function.name)),
                        "prepared native output assignment",
                    )?;
                    roots.insert(root, updated);
                }
            }
        }
        for (local, value) in roots {
            let reference = state.arena.insert(value);
            state.set_local(local.0, reference)?;
        }
        Ok(())
    }
}
