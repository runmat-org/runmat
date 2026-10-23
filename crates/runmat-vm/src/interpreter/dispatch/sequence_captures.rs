//! Mutable metadata for the existing frame CSL windows. Moving a receipt
//! changes only its slot identity, never the underlying operand-stack Values.
use super::{SequenceRegister, SequenceState};
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, host_allocation_runtime_error, FrameCollectionGate,
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::{context::RuntimeContext, RuntimeError};
use std::collections::hash_map::RandomState;

impl Default for SequenceState {
    /// Ordinary/test ingress; actual interpreter invocations use new_in_context.
    fn default() -> Self {
        Self::with_capture_gate(FrameCollectionGate::unbounded(), None)
    }
}

impl SequenceState {
    pub(super) fn capture_scalar_top(
        &mut self,
        stack: &crate::interpreter::stack::InterpreterStack,
    ) -> Result<(), RuntimeError> {
        let start = stack.len().checked_sub(1).ok_or_else(|| {
            crate::interpreter::errors::mex("StackUnderflow", "scalar sequence has no operand")
        })?;
        if self.assignment.is_some() {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "scalar sequence overwrites a live assignment window",
            ));
        }
        self.assignment = Some(SequenceRegister { start, len: 1 });
        Ok(())
    }

    pub(super) fn capture_rooted(
        &mut self,
        slot: usize,
        stack: &mut crate::interpreter::stack::InterpreterStack,
        sequence: runmat_runtime::execution::RootedValueSequence,
    ) -> Result<(), RuntimeError> {
        if self.captures.contains_key(&slot) {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "captured sequence slot was overwritten before consumption",
            ));
        }
        self.reserve_capture_entry()?;
        let values = sequence.try_into_values()?;
        let window = SequenceRegister {
            start: stack.len(),
            len: values.len(),
        };
        stack.try_extend(values)?;
        self.captures.insert(slot, window);
        Ok(())
    }
    pub(crate) fn validate_capture_context(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        match &self.capture_context {
            Some(original) if original.local_identity() != runtime.local_identity() => Err(
                capture_allocation(Some(runtime), HostAllocationError::StaleAuthority),
            ),
            None if HostAllocationReceipt::policy(Some(runtime))
                .map_err(|source| capture_allocation(Some(runtime), source))?
                != runmat_runtime::context::host_allocation::HostAllocationPolicy::Unbounded =>
            {
                Err(capture_allocation(
                    Some(runtime),
                    HostAllocationError::MissingManagedOwner,
                ))
            }
            _ => Ok(()),
        }
    }
    pub(crate) fn capture_rooted_values(
        &mut self,
        slot: usize,
        stack: &mut crate::interpreter::stack::InterpreterStack,
        values: runmat_runtime::execution::RootedValues,
    ) -> Result<(), RuntimeError> {
        if self.captures.contains_key(&slot) {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "captured sequence slot was overwritten before consumption",
            ));
        }
        self.reserve_capture_entry()?;
        let values = values.try_into_values()?;
        let window = SequenceRegister {
            start: stack.len(),
            len: values.len(),
        };
        stack.try_extend(values)?;
        self.captures.insert(slot, window);
        Ok(())
    }
    pub(crate) fn frame_vector<T>(
        &self,
        count: usize,
    ) -> Result<crate::runtime::workspace::FrameVec<T>, RuntimeError> {
        let layout = std::alloc::Layout::array::<T>(count).map_err(|_| {
            capture_allocation(self.capture_context.as_ref(), HostAllocationError::Overflow)
        })?;
        let bytes = if count == 0 {
            0
        } else {
            runmat_runtime::context::host_allocation::collection_block_retained_bytes(layout)
                .map_err(|source| capture_allocation(self.capture_context.as_ref(), source))?
        };
        self.with_metadata_allocation(bytes, |allocator| {
            let mut values = crate::runtime::workspace::FrameVec::new_in(allocator);
            values.try_reserve_exact(count).map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?;
            Ok(values)
        })
    }

    pub(crate) fn frame_map<T>(
        &self,
        count: usize,
    ) -> Result<crate::runtime::workspace::FrameMap<usize, T>, RuntimeError> {
        let bytes = frame_hash_table_retained_bytes::<usize, T>(count)
            .map_err(|source| capture_allocation(self.capture_context.as_ref(), source))?;
        self.with_metadata_allocation(bytes, |allocator| {
            let mut values =
                crate::runtime::workspace::FrameMap::with_hasher_in(RandomState::new(), allocator);
            values.try_reserve(count).map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?;
            Ok(values)
        })
    }

    fn with_metadata_allocation<T>(
        &self,
        bytes: u64,
        initialize: impl FnOnce(
            runmat_runtime::context::host_allocation::OperationCollectionAllocator,
        ) -> Result<T, HostAllocationError>,
    ) -> Result<T, RuntimeError> {
        let allocation = |source| capture_allocation(self.capture_context.as_ref(), source);
        let receipt = HostAllocationReceipt::admit(
            self.capture_context.as_ref(),
            HostAllocationPlan::new(0, bytes).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let _installed = match (&self.capture_context, &receipt) {
            (Some(context), Some(receipt)) => self
                .capture_gate
                .install(context, receipt)
                .map_err(allocation)?,
            _ => self.capture_gate.install_unbounded().map_err(allocation)?,
        };
        let value = initialize(self.capture_gate.allocator()).map_err(allocation)?;
        if let (Some(context), Some(receipt)) = (&self.capture_context, &receipt) {
            receipt.checkpoint(context).map_err(allocation)?;
        }
        Ok(value)
    }

    fn with_capture_gate(gate: FrameCollectionGate, context: Option<RuntimeContext>) -> Self {
        Self {
            assignment: None,
            call_outputs: None,
            captures: crate::runtime::workspace::FrameMap::with_hasher_in(
                RandomState::new(),
                gate.allocator(),
            ),
            capture_context: context,
            output_targets: Vec::new(),
            fixed_target: None,
            expected_output_targets: None,
            fixed_assignment_owner: None,
            destination_builder: None,
            prepared_index_component_count: None,
            prepared_index_receiver: None,
            contextual_indices: crate::runtime::workspace::FrameVec::new_in(gate.allocator()),
            subscript_receivers: crate::runtime::workspace::FrameVec::new_in(gate.allocator()),
            cleanup_windows: crate::runtime::workspace::FrameVec::new_in(gate.allocator()),
            capture_gate: gate,
        }
    }

    pub(crate) fn new_in_context(context: &RuntimeContext) -> Result<Self, RuntimeError> {
        let allocation = |source| capture_allocation(Some(context), source);
        let plan = HostAllocationPlan::new(
            0,
            FrameCollectionGate::retained_bytes().map_err(allocation)?,
        )
        .map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan).map_err(allocation)?;
        let gate = match &receipt {
            Some(receipt) => FrameCollectionGate::prepare(context, receipt).map_err(allocation)?,
            None => FrameCollectionGate::unbounded(),
        };
        Ok(Self::with_capture_gate(gate, Some(context.clone())))
    }

    /// Reserve one genuinely insertable slot before moving either the source
    /// window or payload. A replacement retains old and new physical funding
    /// simultaneously; no assumed tombstone reuse or amortized-layout quote.
    pub(super) fn reserve_capture_entry(&mut self) -> Result<(), RuntimeError> {
        self.reserve_cleanup_window()?;
        let allocation = |source| capture_allocation(self.capture_context.as_ref(), source);
        let target = self
            .captures
            .len()
            .checked_add(1)
            .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
        let bytes = if target > self.captures.capacity() {
            frame_hash_table_retained_bytes::<usize, SequenceRegister>(target)
                .map_err(allocation)?
        } else {
            0
        };
        let receipt = HostAllocationReceipt::admit(
            self.capture_context.as_ref(),
            HostAllocationPlan::new(0, bytes).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let _installed = match (&self.capture_context, &receipt) {
            (Some(context), Some(receipt)) => self
                .capture_gate
                .install(context, receipt)
                .map_err(allocation)?,
            _ => self.capture_gate.install_unbounded().map_err(allocation)?,
        };
        if target > self.captures.capacity() {
            let mut replacement = crate::runtime::workspace::FrameMap::with_hasher_in(
                RandomState::new(),
                self.capture_gate.allocator(),
            );
            replacement.try_reserve(target).map_err(|_| {
                allocation(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ))
            })?;
            if let (Some(context), Some(receipt)) = (&self.capture_context, &receipt) {
                receipt.checkpoint(context).map_err(allocation)?;
            }
            replacement.extend(self.captures.drain());
            self.captures = replacement;
        }
        Ok(())
    }

    fn reserve_cleanup_window(&mut self) -> Result<(), RuntimeError> {
        let required = self
            .captures
            .len()
            .checked_add(self.subscript_receivers.len())
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| {
                capture_allocation(self.capture_context.as_ref(), HostAllocationError::Overflow)
            })?;
        if required > self.cleanup_windows.capacity() {
            let capacity = required.checked_mul(2).ok_or_else(|| {
                capture_allocation(self.capture_context.as_ref(), HostAllocationError::Overflow)
            })?;
            self.cleanup_windows = self.frame_vector(capacity)?;
        }
        Ok(())
    }

    /// Reserve all retirement metadata before a receiver protocol can run.
    pub(super) fn prepare_subscript_receiver(&mut self) -> Result<(), RuntimeError> {
        self.reserve_cleanup_window()?;
        if self.subscript_receivers.len() == self.subscript_receivers.capacity() {
            let capacity = self
                .subscript_receivers
                .len()
                .checked_add(1)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| {
                    capture_allocation(self.capture_context.as_ref(), HostAllocationError::Overflow)
                })?;
            let mut replacement = self.frame_vector(capacity)?;
            replacement.extend(self.subscript_receivers.drain(..));
            self.subscript_receivers = replacement;
        }
        Ok(())
    }

    pub(crate) fn move_capture(
        &mut self,
        source: usize,
        destination: usize,
    ) -> Result<(), RuntimeError> {
        if source == destination
            || self.captures.contains_key(&destination)
            || !self.captures.contains_key(&source)
        {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "captured sequence move has no unique live source and empty destination",
            ));
        }
        self.reserve_capture_entry()?;
        let window = self
            .captures
            .remove(&source)
            .expect("validated live capture");
        self.captures.insert(destination, window);
        Ok(())
    }
}

fn capture_allocation(
    context: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:CommaSeparatedListState",
        "frame captured-sequence metadata admission failed",
        source,
    )
}
