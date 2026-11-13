//! Allocation-aware storage for mutable poll-local projection overlays.
//!
//! This is the physical owner shared by call, output, and console projection
//! domains. Variable-size console text retains its own separate claim.

use allocator_api2::vec::Vec as AllocatedVec;
use runmat_execution::cancellation::CancellationToken;
use runmat_value::exact_capacity_vec;
use std::alloc::Layout;

use super::host_allocation::{
    collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
    HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt, HostOutputPermit,
};
use super::{RuntimeContext, RuntimeContextLocalId};

type ManagedOverlay<T> = AllocatedVec<T, AdmittedCollectionAllocator>;

/// Consuming iterator that preserves the physical owner of either backing.
/// In the managed case allocator claims remain live until the iterator and
/// every unconsumed element have been physically dropped.
pub(crate) enum AdmittedOverlayIntoIter<T> {
    Unbounded(std::vec::IntoIter<T>),
    Managed(allocator_api2::vec::IntoIter<T, AdmittedCollectionAllocator>),
}

/// Ordered overlay storage whose managed backing retains its exact physical
/// allocation claim.
pub(crate) enum AdmittedOverlayVec<T> {
    Unbounded {
        values: Vec<T>,
        generation: u64,
    },
    Managed {
        values: ManagedOverlay<T>,
        generation: u64,
        authority: RuntimeContextLocalId,
        cancellation: CancellationToken,
    },
}

#[derive(Clone)]
enum OverlayExpectation {
    Unbounded {
        generation: u64,
        len: usize,
        capacity: usize,
    },
    Managed {
        generation: u64,
        len: usize,
        capacity: usize,
        authority: RuntimeContextLocalId,
        cancellation: CancellationToken,
    },
}

/// Off-owner preparation for one push.  A replacement allocation, when
/// present, is complete and fenced before the live overlay is borrowed.
pub(crate) struct PreparedOverlayGrowth<T> {
    expected: OverlayExpectation,
    next_generation: u64,
    receipt: Option<HostAllocationReceipt>,
    replacement: Option<ManagedOverlay<T>>,
    ordinary_replacement: Option<Vec<T>>,
}

/// Allocation-free observation of the live overlay used to prepare growth
/// after releasing the projection borrow.
pub(crate) struct OverlayPushSnapshot {
    expected: OverlayExpectation,
    next_generation: u64,
}

impl OverlayPushSnapshot {
    pub(crate) fn len(&self) -> usize {
        match &self.expected {
            OverlayExpectation::Unbounded { len, .. } | OverlayExpectation::Managed { len, .. } => {
                *len
            }
        }
    }
}

/// A prepared replacement whose captured authority was checked immediately
/// before the caller acquires the live projection borrow.
pub(crate) struct CheckedOverlayGrowth<T>(PreparedOverlayGrowth<T>);

/// Backing displaced by a successful managed growth.  Callers keep this
/// carrier outside projection borrows so allocator and element destruction
/// cannot reenter a live `RefCell` guard.
pub(crate) struct OverlayRetirement<T> {
    _backing: Option<ManagedOverlay<T>>,
    _ordinary_backing: Option<Vec<T>>,
    _receipt: Option<HostAllocationReceipt>,
    _source: Option<AdmittedOverlayIntoIter<T>>,
}

/// A stale installation returns every callback-bearing owner to the caller.
/// It must be dropped only after the live projection borrow is released.
pub(crate) struct OverlayInstallError<T> {
    pub(crate) cause: HostAllocationError,
    pub(crate) value: T,
    pub(crate) prepared: CheckedOverlayGrowth<T>,
}

/// Empty managed backing prepared without borrowing its live projection.
/// The receipt stays with the token until installation has been validated.
pub(crate) struct PreparedManagedOverlay<T> {
    replacement: AdmittedOverlayVec<T>,
    receipt: HostAllocationReceipt,
}

/// Former empty backing retired only after the projection borrow is released.
pub(crate) struct ManagedOverlayRetirement<T> {
    _backing: AdmittedOverlayVec<T>,
    _receipt: HostAllocationReceipt,
}

pub(crate) struct ManagedOverlayInstallError<T> {
    pub(crate) cause: HostAllocationError,
    _prepared: PreparedManagedOverlay<T>,
}

impl<T> AdmittedOverlayVec<T> {
    pub(crate) const fn unbounded() -> Self {
        Self::Unbounded {
            values: Vec::new(),
            generation: 0,
        }
    }

    /// Construct an empty managed owner from an already admitted operation.
    /// The allocator is allocation-free here; the first backing block is
    /// quoted and claimed by `prepare_push`.
    pub(crate) fn managed(
        context: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> Result<Self, HostAllocationError> {
        let permit = receipt.output_permit(context)?;
        Self::managed_with_permit(context, permit)
    }

    /// Construct an empty managed overlay inside a larger admitted projection
    /// operation. No collection block exists yet; the retained permit is used
    /// only as the allocator identity until the first prepared replacement is
    /// installed.
    pub(crate) fn managed_with_permit(
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
            return Err(HostAllocationError::StaleAuthority);
        }
        if context.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let allocator = AdmittedCollectionAllocator::new(permit);
        Ok(Self::Managed {
            values: AllocatedVec::new_in(allocator),
            generation: 0,
            authority: context.local_identity(),
            cancellation: context.cancellation(),
        })
    }

    /// Whether a direct synchronous Managed scope must first replace this
    /// empty legacy backing. Live legacy overlays cannot be migrated because
    /// their guards belong to the original generation.
    pub(crate) fn requires_managed_owner(&self) -> Result<bool, HostAllocationError> {
        match self {
            Self::Unbounded { values, .. } if values.is_empty() => Ok(true),
            Self::Unbounded { .. } => Err(HostAllocationError::MissingManagedOwner),
            Self::Managed { .. } => Ok(false),
        }
    }

    /// Prepare and fence an empty managed owner before borrowing live state.
    pub(crate) fn prepare_managed_owner(
        context: &RuntimeContext,
    ) -> Result<PreparedManagedOverlay<T>, HostAllocationError> {
        if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
            return Err(HostAllocationError::StaleAuthority);
        }
        let receipt = HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, 0)?)?
            .ok_or(HostAllocationError::MissingManagedOwner)?;
        let replacement = Self::managed(context, &receipt)?;
        receipt.checkpoint(context)?;
        Ok(PreparedManagedOverlay {
            replacement,
            receipt,
        })
    }

    /// Allocation-free terminal policy installation. A stale token returns
    /// its prepared owner intact for off-borrow retirement.
    pub(crate) fn install_managed_owner(
        &mut self,
        prepared: PreparedManagedOverlay<T>,
    ) -> Result<ManagedOverlayRetirement<T>, ManagedOverlayInstallError<T>> {
        if !matches!(self, Self::Unbounded { values, .. } if values.is_empty()) {
            return Err(ManagedOverlayInstallError {
                cause: HostAllocationError::StaleAuthority,
                _prepared: prepared,
            });
        }
        let PreparedManagedOverlay {
            replacement,
            receipt,
        } = prepared;
        let backing = std::mem::replace(self, replacement);
        Ok(ManagedOverlayRetirement {
            _backing: backing,
            _receipt: receipt,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Unbounded { values, .. } => values,
            Self::Managed { values, .. } => values,
        }
    }

    /// Borrow elements without exposing or detaching the admitted backing.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Unbounded { values, .. } => values.as_mut_slice(),
            Self::Managed { values, .. } => values.as_mut_slice(),
        }
    }

    /// Preserve the ordinary `Vec` fast path when no RuntimeContext is
    /// active. A managed owner must always use prepare/checkpoint/install.
    pub(crate) fn push_unbounded(&mut self, value: T) -> Result<(), T> {
        match self {
            Self::Unbounded { values, generation } => {
                let Some(next_generation) = generation.checked_add(1) else {
                    return Err(value);
                };
                values.push(value);
                *generation = next_generation;
                Ok(())
            }
            Self::Managed { .. } => Err(value),
        }
    }

    /// Capture the complete allocation-free state needed to prepare one push
    /// after the caller releases its projection borrow.
    pub(crate) fn snapshot_push(&self) -> Result<OverlayPushSnapshot, HostAllocationError> {
        let next_generation = self
            .generation()
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let expected = match self {
            Self::Unbounded { values, generation } => OverlayExpectation::Unbounded {
                generation: *generation,
                len: values.len(),
                capacity: values.capacity(),
            },
            Self::Managed {
                values,
                generation,
                authority,
                cancellation,
            } => OverlayExpectation::Managed {
                generation: *generation,
                len: values.len(),
                capacity: values.capacity(),
                authority: *authority,
                cancellation: cancellation.clone(),
            },
        };
        Ok(OverlayPushSnapshot {
            expected,
            next_generation,
        })
    }

    pub(crate) fn prepare_push(
        context: &RuntimeContext,
        snapshot: OverlayPushSnapshot,
    ) -> Result<PreparedOverlayGrowth<T>, HostAllocationError> {
        Self::prepare_append(context, snapshot, 1)
    }

    /// Quote one whole owned source before any element moves into the live
    /// overlay. The source's physical owner is retained through installation.
    pub(crate) fn prepare_append(
        context: &RuntimeContext,
        snapshot: OverlayPushSnapshot,
        additional: usize,
    ) -> Result<PreparedOverlayGrowth<T>, HostAllocationError> {
        let OverlayPushSnapshot {
            expected,
            next_generation,
        } = snapshot;
        match &expected {
            OverlayExpectation::Unbounded { len, capacity, .. } => {
                if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Unbounded
                {
                    return Err(HostAllocationError::MissingManagedOwner);
                }
                let required = len
                    .checked_add(additional)
                    .ok_or(HostAllocationError::Overflow)?;
                let ordinary_replacement = growth_capacity(*capacity, required)?
                    .map(|capacity| {
                        exact_capacity_vec(capacity).map_err(HostAllocationError::Numeric)
                    })
                    .transpose()?;
                Ok(PreparedOverlayGrowth {
                    expected,
                    next_generation,
                    receipt: None,
                    replacement: None,
                    ordinary_replacement,
                })
            }
            OverlayExpectation::Managed {
                len,
                capacity,
                authority,
                cancellation,
                ..
            } => {
                validate_managed_authority(context, *authority, cancellation)?;
                let required = len
                    .checked_add(additional)
                    .ok_or(HostAllocationError::Overflow)?;
                let Some(capacity) = growth_capacity(*capacity, required)? else {
                    return Ok(PreparedOverlayGrowth {
                        expected,
                        next_generation,
                        receipt: None,
                        replacement: None,
                        ordinary_replacement: None,
                    });
                };
                let retained = collection_bytes::<T>(capacity)?;
                let receipt = HostAllocationReceipt::admit(
                    Some(context),
                    HostAllocationPlan::new(0, retained)?,
                )?
                .ok_or(HostAllocationError::MissingManagedOwner)?;
                let allocator = receipt.collection_allocator(context)?;
                let mut replacement = AllocatedVec::new_in(allocator);
                replacement
                    .try_reserve_exact(capacity)
                    .map_err(|_| HostAllocationError::Overflow)?;
                receipt.checkpoint(context)?;
                Ok(PreparedOverlayGrowth {
                    expected,
                    next_generation,
                    receipt: Some(receipt),
                    replacement: Some(replacement),
                    ordinary_replacement: None,
                })
            }
        }
    }

    /// Revalidate the captured operation immediately before borrowing the
    /// live projection for installation.
    /// Allocation-free terminal installation.  On error, the supplied value
    /// and prepared replacement are returned intact for off-borrow retirement.
    pub(crate) fn install_push(
        &mut self,
        prepared: CheckedOverlayGrowth<T>,
        value: T,
    ) -> Result<OverlayRetirement<T>, OverlayInstallError<T>> {
        if let Some(cause) = prepared.0.install_error(self) {
            return Err(OverlayInstallError {
                cause,
                value,
                prepared,
            });
        }
        let mut prepared = prepared.0;
        let mut ordinary_retired = None;
        let retired = match self {
            Self::Unbounded { values, generation } => {
                ordinary_retired = prepared.ordinary_replacement.take().map(|mut replacement| {
                    std::mem::swap(values, &mut replacement);
                    values.append(&mut replacement);
                    replacement
                });
                values.push(value);
                *generation = prepared.next_generation;
                None
            }
            Self::Managed {
                values, generation, ..
            } => {
                let retired = prepared.replacement.take().map(|mut replacement| {
                    std::mem::swap(values, &mut replacement);
                    values.extend(replacement.drain(..));
                    replacement
                });
                values.push(value);
                *generation = prepared.next_generation;
                retired
            }
        };
        Ok(OverlayRetirement {
            _backing: retired,
            _ordinary_backing: ordinary_retired,
            _receipt: prepared.receipt.take(),
            _source: None,
        })
    }

    /// Move a preowned exact-count batch into a prevalidated replacement.
    /// The source iterator (and its former claim) retires after the caller
    /// releases its projection borrow.
    pub(crate) fn install_append(
        &mut self,
        prepared: CheckedOverlayGrowth<T>,
        source: AdmittedOverlayVec<T>,
    ) -> Result<
        OverlayRetirement<T>,
        (
            HostAllocationError,
            CheckedOverlayGrowth<T>,
            AdmittedOverlayVec<T>,
        ),
    > {
        if let Some(cause) = prepared.0.install_error(self) {
            return Err((cause, prepared, source));
        }
        let additional = source.len();
        let enough = self
            .len()
            .checked_add(additional)
            .is_some_and(|required| match self {
                Self::Unbounded { values, .. } => prepared
                    .0
                    .ordinary_replacement
                    .as_ref()
                    .map_or(required <= values.capacity(), |replacement| {
                        required <= replacement.capacity()
                    }),
                Self::Managed { values, .. } => prepared
                    .0
                    .replacement
                    .as_ref()
                    .map_or(required <= values.capacity(), |replacement| {
                        required <= replacement.capacity()
                    }),
            });
        if !enough {
            return Err((HostAllocationError::StaleAuthority, prepared, source));
        }
        let mut prepared = prepared.0;
        let mut source = source.into_iter();
        let mut ordinary_retired = None;
        let retired = match self {
            Self::Unbounded { values, generation } => {
                ordinary_retired = prepared.ordinary_replacement.take().map(|mut replacement| {
                    std::mem::swap(values, &mut replacement);
                    values.append(&mut replacement);
                    replacement
                });
                values.extend(source.by_ref());
                *generation = prepared.next_generation;
                None
            }
            Self::Managed {
                values, generation, ..
            } => {
                let retired = prepared.replacement.take().map(|mut replacement| {
                    std::mem::swap(values, &mut replacement);
                    values.extend(replacement.drain(..));
                    replacement
                });
                values.extend(source.by_ref());
                *generation = prepared.next_generation;
                retired
            }
        };
        Ok(OverlayRetirement {
            _backing: retired,
            _ordinary_backing: ordinary_retired,
            _receipt: prepared.receipt.take(),
            _source: Some(source),
        })
    }

    /// Remove one overlay without allocating.  The displaced element is
    /// returned so its destructor can run after the projection borrow ends.
    pub(crate) fn retire(&mut self, index: usize) -> Option<T> {
        let next_generation = self.generation().saturating_add(1);
        let value = match self {
            Self::Unbounded { values, .. } => {
                (index < values.len()).then(|| values.remove(index))?
            }
            Self::Managed { values, .. } => (index < values.len()).then(|| values.remove(index))?,
        };
        match self {
            Self::Unbounded { generation, .. } | Self::Managed { generation, .. } => {
                *generation = next_generation;
            }
        }
        Some(value)
    }

    fn generation(&self) -> u64 {
        match self {
            Self::Unbounded { generation, .. } | Self::Managed { generation, .. } => *generation,
        }
    }
}

impl<T> IntoIterator for AdmittedOverlayVec<T> {
    type Item = T;
    type IntoIter = AdmittedOverlayIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Unbounded { values, .. } => {
                AdmittedOverlayIntoIter::Unbounded(values.into_iter())
            }
            Self::Managed { values, .. } => AdmittedOverlayIntoIter::Managed(values.into_iter()),
        }
    }
}

impl<T> Iterator for AdmittedOverlayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Unbounded(values) => values.next(),
            Self::Managed(values) => values.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Unbounded(values) => values.size_hint(),
            Self::Managed(values) => values.size_hint(),
        }
    }
}

impl<T> DoubleEndedIterator for AdmittedOverlayIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Unbounded(values) => values.next_back(),
            Self::Managed(values) => values.next_back(),
        }
    }
}

impl<T> ExactSizeIterator for AdmittedOverlayIntoIter<T> {
    fn len(&self) -> usize {
        match self {
            Self::Unbounded(values) => values.len(),
            Self::Managed(values) => values.len(),
        }
    }
}

impl<T> PreparedOverlayGrowth<T> {
    /// Capture the final operation fence before the live projection is
    /// borrowed.  Failure returns the prepared backing for off-borrow drop.
    pub(crate) fn checkpoint(
        self,
        context: &RuntimeContext,
    ) -> Result<CheckedOverlayGrowth<T>, (HostAllocationError, Self)> {
        if let Err(cause) = self.check_authority(context) {
            return Err((cause, self));
        }
        Ok(CheckedOverlayGrowth(self))
    }

    fn check_authority(&self, context: &RuntimeContext) -> Result<(), HostAllocationError> {
        match &self.expected {
            OverlayExpectation::Unbounded { .. } => {
                if HostAllocationReceipt::policy(Some(context))? == HostAllocationPolicy::Unbounded
                {
                    Ok(())
                } else {
                    Err(HostAllocationError::MissingManagedOwner)
                }
            }
            OverlayExpectation::Managed {
                authority,
                cancellation,
                ..
            } => {
                validate_managed_authority(context, *authority, cancellation)?;
                if let Some(receipt) = &self.receipt {
                    receipt.checkpoint(context)?;
                }
                Ok(())
            }
        }
    }

    fn install_error(&self, owner: &AdmittedOverlayVec<T>) -> Option<HostAllocationError> {
        if let OverlayExpectation::Managed { cancellation, .. } = &self.expected {
            if cancellation.is_cancelled() {
                return Some(HostAllocationError::Cancelled);
            }
        }
        (!self.matches(owner)).then_some(HostAllocationError::StaleAuthority)
    }

    fn matches(&self, owner: &AdmittedOverlayVec<T>) -> bool {
        match (&self.expected, owner) {
            (
                OverlayExpectation::Unbounded {
                    generation,
                    len,
                    capacity,
                },
                AdmittedOverlayVec::Unbounded {
                    values,
                    generation: actual,
                },
            ) => generation == actual && *len == values.len() && *capacity == values.capacity(),
            (
                OverlayExpectation::Managed {
                    generation,
                    len,
                    capacity,
                    authority,
                    cancellation,
                },
                AdmittedOverlayVec::Managed {
                    values,
                    generation: actual,
                    authority: actual_authority,
                    cancellation: actual_cancellation,
                },
            ) => {
                generation == actual
                    && *len == values.len()
                    && *capacity == values.capacity()
                    && authority == actual_authority
                    && cancellation.same_generation(actual_cancellation)
            }
            _ => false,
        }
    }
}

fn validate_managed_authority(
    context: &RuntimeContext,
    authority: RuntimeContextLocalId,
    cancellation: &CancellationToken,
) -> Result<(), HostAllocationError> {
    if context.local_identity() != authority
        || !cancellation.same_generation(&context.cancellation())
    {
        return Err(HostAllocationError::StaleAuthority);
    }
    if context.is_cancelled() {
        return Err(HostAllocationError::Cancelled);
    }
    if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
        return Err(HostAllocationError::StaleAuthority);
    }
    Ok(())
}

fn growth_capacity(current: usize, required: usize) -> Result<Option<usize>, HostAllocationError> {
    if required <= current {
        return Ok(None);
    }
    let doubled = current
        .max(1)
        .checked_mul(2)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(Some(doubled.max(required)))
}

fn collection_bytes<T>(capacity: usize) -> Result<u64, HostAllocationError> {
    let layout = Layout::array::<T>(capacity).map_err(|_| HostAllocationError::Overflow)?;
    collection_block_retained_bytes(layout)
}

#[cfg(test)]
mod tests {
    use super::AdmittedOverlayVec;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn stale_preparation_preserves_the_supplied_value() {
        let mut overlays = AdmittedOverlayVec::unbounded();
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let first_snapshot = overlays.snapshot_push().expect("first snapshot");
        let first = AdmittedOverlayVec::<String>::prepare_push(&context, first_snapshot)
            .expect("first preparation")
            .checkpoint(&context)
            .unwrap_or_else(|(cause, _)| panic!("first checkpoint: {cause}"));
        let second_snapshot = overlays.snapshot_push().expect("second snapshot");
        let second = AdmittedOverlayVec::<String>::prepare_push(&context, second_snapshot)
            .expect("second preparation")
            .checkpoint(&context)
            .unwrap_or_else(|(cause, _)| panic!("second checkpoint: {cause}"));
        assert!(overlays.install_push(first, String::from("first")).is_ok());
        let failure = match overlays.install_push(second, String::from("second")) {
            Ok(_) => panic!("stale install succeeded"),
            Err(failure) => failure,
        };
        assert_eq!(failure.value, "second");
        assert_eq!(overlays.as_slice(), ["first"]);
    }

    #[test]
    fn managed_backing_retains_its_claim_until_the_owner_drops() {
        let ledger = ManagedHostEffectLedger::new_with_epoch(65_536, 7);
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
                );
        let mut overlays = AdmittedOverlayVec::unbounded();
        assert!(overlays
            .requires_managed_owner()
            .expect("empty legacy owner"));
        let prepared =
            AdmittedOverlayVec::prepare_managed_owner(&context).expect("managed owner preparation");
        let retirement = overlays
            .install_managed_owner(prepared)
            .unwrap_or_else(|failure| panic!("managed owner install: {}", failure.cause));
        drop(retirement);

        let snapshot = overlays.snapshot_push().expect("growth snapshot");
        let prepared = AdmittedOverlayVec::<u64>::prepare_push(&context, snapshot)
            .expect("growth preparation")
            .checkpoint(&context)
            .unwrap_or_else(|(cause, _)| panic!("growth checkpoint: {cause}"));
        let retirement = overlays
            .install_push(prepared, 17_u64)
            .unwrap_or_else(|failure| panic!("growth install: {}", failure.cause));
        drop(retirement);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        let mut values = overlays.into_iter();
        assert_eq!(values.len(), 1);
        assert_eq!(values.next(), Some(17));
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(values);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
