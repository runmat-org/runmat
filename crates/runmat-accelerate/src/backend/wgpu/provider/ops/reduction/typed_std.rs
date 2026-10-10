//! Scoped physical owner for public standard-deviation reductions.
use super::super::validated_fusion::ValidatedFusionPipeline;
use super::*;
use runmat_accelerate_api::allocation::{AllocationIdentity, GpuHandleReleaseError};
use runmat_accelerate_api::{
    ProviderExecutionCompletionError, ProviderExecutionError, ProviderExecutionResult,
};
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

#[path = "typed_std/execute.rs"]
mod execute;
#[path = "typed_std/frame.rs"]
mod frame;
#[cfg(test)]
#[path = "typed_std/tests.rs"]
mod tests;

/// Every cold numerical constructor is validated before the first std effect.
pub(super) struct StdSelected {
    sum: Option<Arc<ValidatedFusionPipeline>>,
    binary: Option<Arc<ValidatedFusionPipeline>>,
    fill: Arc<ValidatedFusionPipeline>,
    abs: Option<Arc<ValidatedFusionPipeline>>,
    sqrt: Option<Arc<ValidatedFusionPipeline>>,
    temporaries: RefCell<StdCleanupOwner>,
}

struct StdCleanupOwner {
    borrowed: AllocationIdentity,
    live: Vec<GpuTensorHandle>,
    cleanup_errors: Vec<GpuHandleReleaseError>,
}

impl StdCleanupOwner {
    fn new(borrowed: AllocationIdentity) -> Self {
        Self {
            borrowed,
            live: Vec::new(),
            cleanup_errors: Vec::new(),
        }
    }

    fn track(&mut self, handle: &GpuTensorHandle) {
        let identity = handle.allocation_identity();
        if identity != self.borrowed
            && !self
                .live
                .iter()
                .any(|existing| existing.allocation_identity() == identity)
        {
            self.live.push(handle.clone());
        }
    }

    fn retire(&mut self, identity: AllocationIdentity) {
        if let Some(index) = self
            .live
            .iter()
            .position(|handle| handle.allocation_identity() == identity)
        {
            let handle = self.live.remove(index);
            self.cleanup_errors.extend(handle.release().err());
        }
    }

    fn live_count(&self) -> usize {
        self.live.len()
    }

    fn live_requested_bytes(&self) -> u64 {
        self.live.iter().fold(0u64, |total, handle| {
            total.saturating_add(handle.physical().requested_bytes())
        })
    }

    fn complete(
        mut self,
        result: ProviderExecutionResult<GpuTensorHandle>,
        effect: bool,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        for handle in self.live.drain(..) {
            self.cleanup_errors.extend(handle.release().err());
        }
        let cleanup = self.cleanup_errors;
        match (result, cleanup.is_empty()) {
            (Ok(output), true) => Ok(output),
            (Err(primary), true) => Err(if effect {
                primary.after_effect()
            } else {
                primary.with_unproven_effects()
            }),
            (Err(primary), false) => {
                let error = ProviderExecutionError::from_source(
                    ProviderExecutionCompletionError::new(Some(primary), cleanup),
                );
                Err(if effect {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                })
            }
            (Ok(output), false) => {
                let mut cleanup = cleanup;
                cleanup.extend(output.release().err());
                let error = ProviderExecutionError::from_source(
                    ProviderExecutionCompletionError::new(None, cleanup),
                );
                Err(if effect {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                })
            }
        }
    }
}

impl StdSelected {
    pub(super) fn track(&self, handle: &GpuTensorHandle) {
        self.temporaries.borrow_mut().track(handle);
    }

    pub(super) fn retire_last_use(&self, handle: GpuTensorHandle) {
        let identity = handle.allocation_identity();
        drop(handle);
        self.temporaries.borrow_mut().retire(identity);
    }

    fn retained_temporary_evidence(&self) -> (usize, u64) {
        let owner = self.temporaries.borrow();
        (owner.live_count(), owner.live_requested_bytes())
    }

    fn complete(
        self,
        result: ProviderExecutionResult<GpuTensorHandle>,
        effect: bool,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.temporaries.into_inner().complete(result, effect)
    }
}

pub(super) fn finish_std_temporary(handle: GpuTensorHandle, selected: Option<&StdSelected>) {
    if let Some(selected) = selected {
        selected.retire_last_use(handle);
    } else {
        drop(handle);
    }
}

impl WgpuProvider {
    pub(super) fn std_binary_primitive_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        selected: Option<&StdSelected>,
        effect: Option<&Cell<bool>>,
    ) -> Result<GpuTensorHandle> {
        match (selected, effect) {
            (Some(selected), Some(effect)) => {
                let output = self.binary_std_selected_exec(
                    op,
                    a,
                    b,
                    selected
                        .binary
                        .as_ref()
                        .ok_or_else(|| anyhow!("std binary pair absent"))?,
                    effect,
                )?;
                selected.track(&output);
                Ok(output)
            }
            (None, None) => self.binary_op_exec(op, a, b),
            _ => Err(anyhow!("std selected binary phase carrier mismatch")),
        }
    }

    pub(super) fn std_fill_primitive_exec(
        &self,
        shape: &[usize],
        value: f64,
        selected: Option<&StdSelected>,
        effect: Option<&Cell<bool>>,
    ) -> Result<GpuTensorHandle> {
        match (selected, effect) {
            (Some(selected), Some(effect)) => {
                let output =
                    self.fill_std_selected_exec(shape, value, &selected.fill, effect, None)?;
                selected.track(&output);
                Ok(output)
            }
            (None, None) => self.fill_exec(shape, value),
            _ => Err(anyhow!("std selected fill phase carrier mismatch")),
        }
    }

    pub(super) fn std_unary_primitive_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
        selected: Option<&StdSelected>,
        effect: Option<&Cell<bool>>,
    ) -> Result<GpuTensorHandle> {
        match (selected, effect) {
            (Some(selected), Some(effect)) => {
                let pair = match op {
                    crate::backend::wgpu::types::UnaryOpCode::Abs => selected.abs.as_ref(),
                    crate::backend::wgpu::types::UnaryOpCode::Sqrt => selected.sqrt.as_ref(),
                    _ => None,
                }
                .ok_or_else(|| anyhow!("std unary pair absent or operation unadmitted"))?;
                self.unary_std_selected_exec(op, a, pair, effect)
            }
            (None, None) => self.unary_op_exec(op, a),
            _ => Err(anyhow!("std selected unary phase carrier mismatch")),
        }
    }
}

fn terminal(error: anyhow::Error, effect: bool) -> ProviderExecutionError {
    let typed = ProviderExecutionError::from(error);
    if effect {
        typed.after_effect()
    } else {
        typed.with_unproven_effects()
    }
}
