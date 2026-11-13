//! One owner for intermediate resident allocations and output admission.
use std::error::Error;
use std::fmt;

use runmat_accelerate_api::{
    allocation::GpuHandleReleaseError, handle_integer_type, handle_is_logical, handle_precision,
    handle_storage, provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle,
    GpuTensorLayout, GpuTensorStorage, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionResult, ProviderIirFilterResult,
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

pub(super) struct Stages<'a> {
    provider: &'a dyn AccelProvider,
    protected: Vec<&'a GpuTensorHandle>,
    handles: Vec<GpuTensorHandle>,
    cleanup: Vec<GpuHandleReleaseError>,
}

impl<'a> Stages<'a> {
    pub fn new(
        provider: &'a dyn AccelProvider,
        protected: [&'a GpuTensorHandle; 3],
        borrowed: &'a [&'a GpuTensorHandle],
    ) -> ProviderExecutionResult<Self> {
        let protected_len = borrowed.len().checked_add(protected.len()).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "iir_zero_phase: borrowed operand count overflow"
            ))
        })?;
        let mut protected_handles = Vec::new();
        protected_handles
            .try_reserve_exact(protected_len)
            .map_err(ProviderExecutionError::from_source)?;
        protected_handles.extend(protected);
        protected_handles.extend_from_slice(borrowed);
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
            protected: protected_handles,
            handles,
            cleanup,
        })
    }

    pub fn get(&self, index: usize) -> &GpuTensorHandle {
        &self.handles[index]
    }

    pub fn protected(&self) -> &[&GpuTensorHandle] {
        &self.protected
    }

    pub fn admit_flip(
        &mut self,
        output: Option<GpuTensorHandle>,
        source: usize,
        axis: usize,
        shape: &[usize],
    ) -> ProviderExecutionResult<usize> {
        let output = output.ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "iir_zero_phase: required flip stage unavailable"
            ))
        })?;
        let input = self.get(source);
        if !runmat_types::geometry::FlipGeometry::new(input.shape(), &[axis]).has_effect()
            && output.shares_allocation_with(input)
            && output.view() == input.view()
            && output.shape() == shape
        {
            output.allocation_lease().validate_live()?;
            return Ok(source);
        }
        self.admit(output, shape, StageClass::Real)
    }

    pub fn admit(
        &mut self,
        handle: GpuTensorHandle,
        shape: &[usize],
        class: StageClass,
    ) -> ProviderExecutionResult<usize> {
        let distinct = self.is_distinct(&handle, None);
        let valid =
            self.valid_stage(&handle, shape, class, None) && self.handles.len() < MAX_STAGES;
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

    fn is_distinct(&self, handle: &GpuTensorHandle, sibling: Option<&GpuTensorHandle>) -> bool {
        self.protected
            .iter()
            .copied()
            .chain(self.handles.iter())
            .chain(sibling)
            .all(|other| !handle.shares_allocation_with(other))
    }

    fn valid_stage(
        &self,
        handle: &GpuTensorHandle,
        shape: &[usize],
        class: StageClass,
        sibling: Option<&GpuTensorHandle>,
    ) -> bool {
        let exact_owner = provider_for_handle(handle)
            .is_some_and(|owner| same_provider_instance(owner, self.provider));
        let valid_class = match class {
            StageClass::Real => {
                handle_precision(handle) == Some(self.provider.precision())
                    && handle_storage(handle) == GpuTensorStorage::Real
                    && handle_integer_type(handle).is_none()
                    && !handle_is_logical(handle)
            }
            StageClass::Logical => {
                handle_storage(handle) == GpuTensorStorage::Real && handle_is_logical(handle)
            }
            StageClass::Predicate => {
                handle_storage(handle) == GpuTensorStorage::Real
                    && (handle_is_logical(handle)
                        || (handle_precision(handle) == Some(self.provider.precision())
                            && handle_integer_type(handle).is_none()))
            }
        };
        let distinct = self.is_distinct(handle, sibling);
        exact_owner
            && handle.device_id() == self.provider.device_id()
            && handle.allocation_lease().validate_live().is_ok()
            && handle.shape() == shape
            && handle.view().layout() == GpuTensorLayout::Contiguous
            && valid_class
            && handle.view().validate(handle.physical()).is_ok()
            && distinct
    }

    /// Admit both outputs as one callback result. Validate their ownership and
    /// mutual distinctness before either handle becomes a stage; on rejection,
    /// retain every fresh-output cleanup failure alongside the primary fault.
    pub fn admit_filter_result(
        &mut self,
        result: ProviderIirFilterResult,
        signal_shape: &[usize],
        state_shape: &[usize],
        state_required: bool,
    ) -> ProviderExecutionResult<usize> {
        let ProviderIirFilterResult {
            output,
            final_state,
        } = result;
        let capacity = self
            .handles
            .len()
            .checked_add(1 + usize::from(final_state.is_some()));
        let valid = capacity.is_some_and(|len| len <= MAX_STAGES)
            && (!state_required || final_state.is_some())
            && self.valid_stage(
                &output,
                signal_shape,
                StageClass::Real,
                final_state.as_ref(),
            )
            && final_state.as_ref().is_none_or(|state| {
                self.valid_stage(state, state_shape, StageClass::Real, Some(&output))
            });
        if !valid {
            let output_fresh = self.is_distinct(&output, None);
            let state_fresh = final_state
                .as_ref()
                .is_some_and(|state| self.is_distinct(state, None));
            if output_fresh {
                if let Err(error) = output.release() {
                    self.cleanup.push(error);
                }
            }
            if let Some(state) = final_state {
                if state_fresh {
                    if let Err(error) = state.release() {
                        self.cleanup.push(error);
                    }
                }
            }
            return Err(ProviderExecutionError::from_source(StageRejected {
                reason: "iir_zero_phase: malformed provider filter output pair",
                cleanup: None,
            }));
        }
        let index = self.handles.len();
        self.handles.push(output);
        if let Some(state) = final_state {
            self.handles.push(state);
        }
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
        Err(
            ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                result.err(),
                self.cleanup,
            ))
            .after_effect(),
        )
    }
}
