//! Incremental, context-bound ownership of retained aggregate root metadata.

use std::{alloc::Layout, fmt};

use allocator_api2::vec::Vec as AllocatedVec;
use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};

use crate::context::host_allocation::{
    collection_block_retained_bytes, FrameCollectionGate, HostAllocationCheckpoint,
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    OperationCollectionAllocator,
};
use crate::context::RuntimeContext;

use super::RetainedValueRoots;

const ROOT_CHUNK_CAPACITY: usize = 64;

#[derive(Debug)]
enum RootSlots {
    Empty,
    Unbounded(Vec<RetainedValueRoots>),
    Managed(AllocatedVec<RetainedValueRoots, OperationCollectionAllocator>),
}

impl RootSlots {
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Unbounded(values) => values.len(),
            Self::Managed(values) => values.len(),
        }
    }
}

/// Only the construction owner borrows Runtime. Finished chunks retain their
/// own physical claims and can outlive the operation without retaining a whole
/// RuntimeContext or registering any managed handle again.
pub(crate) struct RootedValueChildren<'a> {
    current: RootSlots,
    sealed: Option<ExactShared<RootChunk>>,
    gate: Option<FrameCollectionGate>,
    runtime: Option<&'a RuntimeContext>,
    policy: HostAllocationPolicy,
    checkpoint: Option<HostAllocationCheckpoint>,
}

impl fmt::Debug for RootedValueChildren<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RootedValueChildren")
            .field("current", &self.current.len())
            .field("has_sealed_chunks", &self.sealed.is_some())
            .field("policy", &self.policy)
            .finish()
    }
}

impl Default for RootedValueChildren<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RootedValueChildren<'a> {
    /// Compatibility builders explicitly retain their ordinary unbounded policy.
    pub(crate) fn new() -> Self {
        Self {
            current: RootSlots::Unbounded(Vec::new()),
            sealed: None,
            gate: None,
            runtime: None,
            policy: HostAllocationPolicy::Unbounded,
            checkpoint: None,
        }
    }

    /// Select policy without allocating. Empty/scalar children require no root
    /// chunk, table, gate, or speculative worst-case metadata reservation.
    pub(crate) fn in_context(runtime: &'a RuntimeContext) -> Result<Self, HostAllocationError> {
        let checkpoint = HostAllocationReceipt::capture_owner(Some(runtime))?;
        let policy = if checkpoint.is_some() {
            HostAllocationPolicy::Managed
        } else {
            HostAllocationPolicy::Unbounded
        };
        Ok(Self {
            current: if policy == HostAllocationPolicy::Managed {
                RootSlots::Empty
            } else {
                RootSlots::Unbounded(Vec::new())
            },
            sealed: None,
            gate: None,
            runtime: Some(runtime),
            policy,
            checkpoint,
        })
    }

    fn checkpoint(&self) -> Result<(), HostAllocationError> {
        if let Some(checkpoint) = &self.checkpoint {
            return checkpoint.check(self.runtime.expect("managed root owner has a runtime"));
        }
        if HostAllocationReceipt::policy(self.runtime)? != self.policy {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }

    /// Infallible ordinary-builder path; never selected by a context-bound
    /// finite owner. It preserves the existing public builder's contract.
    pub(in crate::execution::completion) fn push(&mut self, roots: RetainedValueRoots) {
        assert!(
            self.runtime.is_none(),
            "context-bound roots require fallible admission"
        );
        self.push_unbounded(roots);
    }

    fn push_unbounded(&mut self, roots: RetainedValueRoots) {
        if roots.is_empty() {
            return;
        }
        if self.current.len() == ROOT_CHUNK_CAPACITY {
            let node = RootChunk {
                _roots: std::mem::replace(
                    &mut self.current,
                    RootSlots::Unbounded(Vec::with_capacity(ROOT_CHUNK_CAPACITY)),
                ),
                previous: self.sealed.take(),
                _claim: None,
            };
            self.sealed = Some(ExactShared::new(node));
        }
        let RootSlots::Unbounded(values) = &mut self.current else {
            unreachable!("unbounded root owner retains ordinary slots");
        };
        values.push(roots);
    }

    /// Failure returns the incoming root owner intact. Existing chunks are
    /// never retired by a failed metadata allocation while Values still live.
    pub(super) fn try_push(
        &mut self,
        roots: RetainedValueRoots,
    ) -> Result<(), (RetainedValueRoots, HostAllocationError)> {
        if let Err(error) = self.checkpoint() {
            return Err((roots, error));
        }
        if roots.is_empty() {
            return Ok(());
        }
        if self.policy == HostAllocationPolicy::Unbounded {
            self.push_unbounded(roots);
            return Ok(());
        }
        if let Err(error) = self.prepare_slot() {
            return Err((roots, error));
        }
        if let Err(error) = self.checkpoint() {
            return Err((roots, error));
        }
        let RootSlots::Managed(values) = &mut self.current else {
            unreachable!("managed root slots were admitted");
        };
        values.push(roots); // prepare_slot admitted capacity before this move.
        Ok(())
    }

    fn prepare_slot(&mut self) -> Result<(), HostAllocationError> {
        self.checkpoint()?;
        let runtime = self.runtime.expect("managed root owner has a runtime");
        if self.gate.is_none() {
            let plan = HostAllocationPlan::new(0, FrameCollectionGate::retained_bytes()?)?;
            let receipt = HostAllocationReceipt::admit(Some(runtime), plan)?
                .ok_or(HostAllocationError::StaleAuthority)?;
            self.gate = Some(FrameCollectionGate::prepare(runtime, &receipt)?);
        }
        let gate = self.gate.as_ref().expect("prepared root collection gate");
        if self.current.len() == ROOT_CHUNK_CAPACITY {
            // Allocate the empty node BEFORE moving any retained roots or
            // sibling links, so even failure leaves the old forest protected.
            let mut node = self.prepare_node()?;
            node.get_mut_unique()._roots = std::mem::replace(
                &mut self.current,
                RootSlots::Managed(AllocatedVec::new_in(gate.allocator())),
            );
            node.get_mut_unique().previous = self.sealed.take();
            self.sealed = Some(node);
        }
        if matches!(self.current, RootSlots::Empty) {
            self.current = RootSlots::Managed(AllocatedVec::new_in(gate.allocator()));
        }
        let RootSlots::Managed(values) = &mut self.current else {
            return Err(HostAllocationError::StaleAuthority);
        };
        if values.len() < values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .max(2)
            .checked_mul(2)
            .ok_or(HostAllocationError::Overflow)?
            .min(ROOT_CHUNK_CAPACITY);
        let layout = Layout::array::<RetainedValueRoots>(capacity)
            .map_err(|_| HostAllocationError::Overflow)?;
        let plan = HostAllocationPlan::new(0, collection_block_retained_bytes(layout)?)?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)?
            .ok_or(HostAllocationError::StaleAuthority)?;
        let _permit = gate.install(runtime, &receipt)?;
        let result = values.try_reserve_exact(capacity - values.len());
        receipt.checkpoint(runtime)?;
        self.checkpoint()?;
        result.map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })
    }

    fn prepare_node(&self) -> Result<ExactShared<RootChunk>, HostAllocationError> {
        self.checkpoint()?;
        let runtime = self.runtime.expect("managed root owner has a runtime");
        let raw = ExactShared::<RootChunk>::requested_bytes();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?;
        let receipt =
            HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, retained)?)?
                .ok_or(HostAllocationError::StaleAuthority)?;
        let permit = receipt.output_permit(runtime)?;
        let claim = permit
            .reserve_exact_claim(raw)
            .map_err(HostAllocationError::Numeric)?;
        let result = ExactShared::try_new(RootChunk {
            _roots: RootSlots::Empty,
            previous: None,
            _claim: Some(claim),
        });
        receipt.checkpoint(runtime)?;
        self.checkpoint()?;
        result.map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })
    }

    pub(in crate::execution::completion) fn finish(mut self) -> RetainedValueRoots {
        assert!(
            self.runtime.is_none(),
            "context-bound roots require fallible publication"
        );
        self.finish_unbounded()
    }

    fn finish_unbounded(&mut self) -> RetainedValueRoots {
        if self.current.len() != 0 {
            self.sealed = Some(ExactShared::new(RootChunk {
                _roots: std::mem::replace(&mut self.current, RootSlots::Empty),
                previous: self.sealed.take(),
                _claim: None,
            }));
        }
        self.sealed
            .take()
            .map_or(RetainedValueRoots::Empty, |root| {
                RetainedValueRoots::Aggregate { _root: root }
            })
    }

    pub(super) fn try_finish(&mut self) -> Result<RetainedValueRoots, HostAllocationError> {
        self.checkpoint()?;
        if self.runtime.is_none() {
            return Ok(self.finish_unbounded());
        }
        if self.current.len() != 0 {
            let mut node = if self.policy == HostAllocationPolicy::Managed {
                self.prepare_node()?
            } else {
                ExactShared::new(RootChunk {
                    _roots: RootSlots::Empty,
                    previous: None,
                    _claim: None,
                })
            };
            node.get_mut_unique()._roots = std::mem::replace(&mut self.current, RootSlots::Empty);
            node.get_mut_unique().previous = self.sealed.take();
            self.sealed = Some(node);
        }
        Ok(self
            .sealed
            .take()
            .map_or(RetainedValueRoots::Empty, |root| {
                RetainedValueRoots::Aggregate { _root: root }
            }))
    }
}

#[derive(Debug)]
pub(in crate::execution::completion) struct RootChunk {
    _roots: RootSlots,
    previous: Option<ExactShared<RootChunk>>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Drop for RootChunk {
    fn drop(&mut self) {
        let mut previous = self.previous.take();
        while let Some(node) = previous {
            let Ok(mut node) = node.try_unwrap() else {
                break;
            };
            previous = node.previous.take();
        }
    }
}
