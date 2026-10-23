//! The one existing frame capture table: admitted index/entries, canonical
//! rooted output backing, move-only publication between typed phase slots.
use runmat_mir::MirSequenceLocalId;
use runmat_runtime::context::host_allocation::{
    FrameCollectionGate, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::execution::{RootedValueLease, RootedValueSequence, RootedValuesIntoIter};

use super::{continuation_admission_error, ContinuationMetadata, HostState};
use crate::{NativeExecutorError, NativeExecutorResult};

/// Physical ownership persists after logical sequence consumption, until a
/// selector's successor root plan is published. The existing OrderedCache
/// allocator quotes this actual slot layout, including its retained lease.
pub(super) enum CapturedSequenceSlot {
    Vacant,
    Live(RootedValueSequence),
    Consumed { _roots: RootedValueLease },
}

impl CapturedSequenceSlot {
    fn live(&self) -> Option<&RootedValueSequence> {
        match self {
            Self::Live(sequence) => Some(sequence),
            _ => None,
        }
    }
    fn occupied(&self) -> bool {
        !matches!(self, Self::Vacant)
    }
    fn take_live(&mut self) -> Option<RootedValueSequence> {
        if !matches!(self, Self::Live(_)) {
            return None;
        }
        match std::mem::replace(self, Self::Vacant) {
            Self::Live(sequence) => Some(sequence),
            _ => unreachable!("checked live slot"),
        }
    }
}

impl HostState {
    pub(in crate::execute) fn captured_sequence_width(
        &self,
        id: MirSequenceLocalId,
    ) -> NativeExecutorResult<usize> {
        self.contextual_regions
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.captured_sequences.get(&id))
            .and_then(CapturedSequenceSlot::live)
            .map(|sequence| sequence.len())
            .ok_or_else(|| {
                NativeExecutorError::Host("fixed selector sequence receipt is unavailable".into())
            })
    }
    pub(in crate::execute) fn captured_sequence_live(&self, id: MirSequenceLocalId) -> bool {
        self.contextual_regions
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.captured_sequences.get(&id))
            .is_some_and(CapturedSequenceSlot::occupied)
    }

    pub(in crate::execute) fn captured_sequences_live(&self) -> bool {
        self.contextual_regions
            .metadata
            .as_ref()
            .is_some_and(|metadata| {
                metadata
                    .captured_sequences
                    .values()
                    .any(CapturedSequenceSlot::occupied)
            })
    }

    /// Fund entry/index growth before source publication. Pure functions may
    /// capture without expression cursors: admit the SAME metadata gate lazily,
    /// not a phantom expression-region capacity or a second registry.
    pub(in crate::execute) fn prepare_captured_sequence_slot(
        &mut self,
        id: MirSequenceLocalId,
    ) -> NativeExecutorResult<()> {
        if self.captured_sequence_live(id) {
            return Err(NativeExecutorError::Host(
                "native sequence slot overwritten before consumption".into(),
            ));
        }
        if self.contextual_regions.metadata.is_none() {
            let retained = FrameCollectionGate::retained_bytes()
                .map_err(|cause| continuation_admission_error(&self.runtime, cause))?;
            let plan = HostAllocationPlan::new(0, retained)
                .map_err(|cause| continuation_admission_error(&self.runtime, cause))?;
            let receipt = HostAllocationReceipt::admit(Some(&self.runtime), plan)
                .map_err(|cause| continuation_admission_error(&self.runtime, cause))?;
            let gate = match receipt {
                Some(receipt) => FrameCollectionGate::prepare(&self.runtime, &receipt)
                    .map_err(|cause| continuation_admission_error(&self.runtime, cause))?,
                None => FrameCollectionGate::unbounded(),
            };
            self.contextual_regions.metadata = Some(ContinuationMetadata::new(gate));
        }
        let metadata = self.contextual_regions.metadata_mut()?;
        if metadata.captured_sequences.get(&id).is_none() {
            metadata
                .captured_sequences
                .prepare_one_more(&self.runtime, &metadata.gate)?;
            metadata
                .captured_sequences
                .insert_ready(id, CapturedSequenceSlot::Vacant);
        }
        Ok(())
    }

    pub(in crate::execute) fn publish_captured_sequence(
        &mut self,
        id: MirSequenceLocalId,
        value: RootedValueSequence,
    ) -> NativeExecutorResult<()> {
        let slot = self
            .contextual_regions
            .metadata_mut()?
            .captured_sequences
            .get_mut(&id)
            .ok_or_else(|| {
                NativeExecutorError::Host("sequence publication has no admitted frame slot".into())
            })?;
        if slot.occupied() {
            return Err(NativeExecutorError::Host(
                "sequence publication replaced a live receipt".into(),
            ));
        }
        *slot = CapturedSequenceSlot::Live(value);
        Ok(())
    }

    pub(in crate::execute) fn take_captured_sequence_owner(
        &mut self,
        id: MirSequenceLocalId,
    ) -> NativeExecutorResult<RootedValueSequence> {
        self.contextual_regions
            .metadata_mut()?
            .captured_sequences
            .get_mut(&id)
            .and_then(CapturedSequenceSlot::take_live)
            .ok_or_else(|| {
                NativeExecutorError::Host(format!(
                    "native sequence local {} was read before capture or after consumption",
                    id.0
                ))
            })
    }

    /// Consume once while leaving the original registration in the SAME
    /// admitted slot through all subsequent selector construction/admission.
    pub(in crate::execute) fn take_selector_sequence(
        &mut self,
        id: MirSequenceLocalId,
    ) -> NativeExecutorResult<RootedValuesIntoIter> {
        let slot = self
            .contextual_regions
            .metadata_mut()?
            .captured_sequences
            .get_mut(&id)
            .ok_or_else(|| {
                NativeExecutorError::Host("selector sequence has no admitted capture slot".into())
            })?;
        let sequence = slot.take_live().ok_or_else(|| {
            NativeExecutorError::Host(
                "selector sequence was unavailable or already consumed".into(),
            )
        })?;
        *slot = CapturedSequenceSlot::Consumed {
            _roots: sequence.root_lease(),
        };
        sequence.try_into_values().map_err(Into::into)
    }

    /// Called only after the assembled selector has a published successor
    /// root plan. No scan, new backing, or root re-registration is required.
    pub(in crate::execute) fn retire_selector_sequence_lease(&mut self, id: MirSequenceLocalId) {
        if let Some(metadata) = self.contextual_regions.metadata.as_mut() {
            if let Some(slot) = metadata.captured_sequences.get_mut(&id) {
                if matches!(slot, CapturedSequenceSlot::Consumed { .. }) {
                    *slot = CapturedSequenceSlot::Vacant;
                }
            }
        }
    }

    pub(in crate::execute) fn move_captured_sequence(
        &mut self,
        from: MirSequenceLocalId,
        to: MirSequenceLocalId,
    ) -> NativeExecutorResult<()> {
        if from == to {
            return Err(NativeExecutorError::Host(
                "sequence move aliases its source".into(),
            ));
        }
        self.prepare_captured_sequence_slot(to)?;
        let value = self.take_captured_sequence_owner(from)?;
        self.publish_captured_sequence(to, value)
    }

    pub(in crate::execute) fn discard_captured_sequence(&mut self, id: MirSequenceLocalId) {
        if let Some(metadata) = self.contextual_regions.metadata.as_mut() {
            if let Some(slot) = metadata.captured_sequences.get_mut(&id) {
                *slot = CapturedSequenceSlot::Vacant;
            }
        }
    }

    pub(in crate::execute) fn abandon_captured_sequences(&mut self) {
        if let Some(metadata) = self.contextual_regions.metadata.as_mut() {
            // Reuse admitted static IDs across loops; owners retire in source
            // insertion order without copying their physical output backing.
            metadata
                .captured_sequences
                .values_mut()
                .for_each(|slot| *slot = CapturedSequenceSlot::Vacant);
        }
    }
}
