//! One owner for intermediate resident allocations and output admission.
use std::error::Error;
use std::fmt;

use runmat_accelerate_api::{
    allocation::GpuHandleReleaseError, handle_integer_type, handle_is_logical, handle_precision,
    handle_storage, provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle,
    GpuTensorLayout, GpuTensorStorage, ProviderExecutionError, ProviderExecutionResult,
};

const MAX_STAGES: usize = 24;

pub(super) enum StageClass {
    Real,
    Logical,
    /// `reduce_all` is a logical scalar on the host provider and a 0/1
    /// floating scalar on WGPU. Both are admitted numeric predicates.
    Predicate,
}

#[derive(Debug)]
struct StageRejected {
    reason: &'static str,
    cleanup: Option<GpuHandleReleaseError>,
}

impl fmt::Display for StageRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.reason)
    }
}

impl Error for StageRejected {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.cleanup
            .as_ref()
            .map(|cleanup| cleanup as &(dyn Error + 'static))
    }
}

#[derive(Debug)]
struct StageCompletionFailed {
    primary: Option<ProviderExecutionError>,
    cleanup: Vec<GpuHandleReleaseError>,
}

impl fmt::Display for StageCompletionFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "iir_zero_phase: {} stage retirement failure(s)",
            self.cleanup.len()
        )?;
        if let Some(primary) = &self.primary {
            write!(f, "; original execution failure: {primary}")?;
        }
        for cleanup in &self.cleanup {
            write!(f, "; {cleanup}")?;
        }
        Ok(())
    }
}

impl Error for StageCompletionFailed {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.primary
            .as_ref()
            .map(|primary| primary as &(dyn Error + 'static))
            .or_else(|| {
                self.cleanup
                    .first()
                    .map(|cleanup| cleanup as &(dyn Error + 'static))
            })
    }
}

pub(super) struct Stages<'a> {
    provider: &'a dyn AccelProvider,
    protected: [&'a GpuTensorHandle; 3],
    handles: Vec<GpuTensorHandle>,
    cleanup: Vec<GpuHandleReleaseError>,
}

impl<'a> Stages<'a> {
    pub fn new(
        provider: &'a dyn AccelProvider,
        protected: [&'a GpuTensorHandle; 3],
    ) -> ProviderExecutionResult<Self> {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(MAX_STAGES)
            .map_err(ProviderExecutionError::from_source)?;
        let mut cleanup = Vec::new();
        cleanup
            .try_reserve_exact(MAX_STAGES)
            .map_err(ProviderExecutionError::from_source)?;
        Ok(Self {
            provider,
            protected,
            handles,
            cleanup,
        })
    }

    pub fn get(&self, index: usize) -> &GpuTensorHandle {
        &self.handles[index]
    }

    pub fn admit(
        &mut self,
        handle: GpuTensorHandle,
        shape: &[usize],
        class: StageClass,
    ) -> ProviderExecutionResult<usize> {
        let exact_owner = provider_for_handle(&handle)
            .is_some_and(|owner| same_provider_instance(owner, self.provider));
        let valid_class = match class {
            StageClass::Real => {
                handle_precision(&handle) == Some(self.provider.precision())
                    && handle_storage(&handle) == GpuTensorStorage::Real
                    && handle_integer_type(&handle).is_none()
                    && !handle_is_logical(&handle)
            }
            StageClass::Logical => {
                handle_storage(&handle) == GpuTensorStorage::Real && handle_is_logical(&handle)
            }
            StageClass::Predicate => {
                handle_storage(&handle) == GpuTensorStorage::Real
                    && (handle_is_logical(&handle)
                        || (handle_precision(&handle) == Some(self.provider.precision())
                            && handle_integer_type(&handle).is_none()))
            }
        };
        let distinct = self
            .protected
            .iter()
            .copied()
            .chain(self.handles.iter())
            .all(|other| !handle.shares_allocation_with(other));
        let physical_extent = shape
            .iter()
            .try_fold(1u64, |count, &extent| {
                count.checked_mul(u64::try_from(extent).ok()?)
            })
            .is_some_and(|count| {
                handle.physical().element_capacity() >= count
                    && count
                        .checked_mul(handle.physical().element_type().element_size() as u64)
                        .is_some_and(|bytes| handle.physical().payload_bytes() >= bytes)
            });
        let valid = exact_owner
            && handle.device_id() == self.provider.device_id()
            && handle.allocation_lease().validate_live().is_ok()
            && handle.shape() == shape
            && handle.view().layout() == GpuTensorLayout::Contiguous
            && valid_class
            && physical_extent
            && distinct
            && self.handles.len() < MAX_STAGES;
        if !valid {
            // A provider may hand back a wrapper for a caller operand or an
            // earlier stage. Retiring that alias would invalidate the owner
            // that remains live in the invocation.
            let cleanup = if distinct {
                handle.release().err()
            } else {
                None
            };
            return Err(ProviderExecutionError::from_source(StageRejected {
                reason: "iir_zero_phase: malformed or aliased provider stage output",
                cleanup,
            }));
        }
        let index = self.handles.len();
        self.handles.push(handle);
        Ok(index)
    }

    fn retire(&mut self) {
        for handle in self.handles.drain(..) {
            if let Err(error) = handle.release() {
                self.cleanup.push(error);
            }
        }
    }

    pub fn finish(
        mut self,
        result: ProviderExecutionResult<usize>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let output = result
            .as_ref()
            .ok()
            .map(|&index| self.handles.swap_remove(index));
        self.retire();
        if self.cleanup.is_empty() {
            return match (result, output) {
                (Ok(_), Some(output)) => Ok(output),
                (Err(primary), _) => Err(primary),
                _ => unreachable!("result index always identifies an admitted handle"),
            };
        }
        if let Some(output) = output {
            if let Err(error) = output.release() {
                self.cleanup.push(error);
            }
        }
        Err(ProviderExecutionError::from_source(StageCompletionFailed {
            primary: result.err(),
            cleanup: self.cleanup,
        })
        .after_effect())
    }
}
