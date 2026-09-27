//! Scoped ownership of a validated numeric result during output adaptation.

use crate::builtins::common::gpu_helpers;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

/// Owns a top-level resident numeric result until it is published or replaced.
///
/// The caller validates the result's class, shape and storage before capture,
/// and supplies every borrowed operand/prototype. Container results require the
/// separate recursive alias protocol and must not use this numeric-only guard.
pub(crate) struct ResidentResultGuard<'a> {
    resident: Option<OwnedResident>,
    protected: &'a [&'a GpuTensorHandle],
}

struct OwnedResident {
    handle: GpuTensorHandle,
    owner: &'static dyn AccelProvider,
}

impl<'a> ResidentResultGuard<'a> {
    pub(crate) fn new(value: &Value, protected: &'a [&'a GpuTensorHandle]) -> Result<Self, String> {
        let resident = match value {
            Value::GpuTensor(handle) => {
                let owner = gpu_helpers::exact_provider_for_handle(handle)
                    .ok_or_else(|| "no provider owns the resident result".to_string())?;
                Some(OwnedResident {
                    handle: handle.clone(),
                    owner,
                })
            }
            _ => None,
        };
        Ok(Self {
            resident,
            protected,
        })
    }

    /// Transfer the original allocation only when the final result retains it.
    /// A different result leaves this guard responsible for the intermediate.
    pub(crate) fn publish(mut self, output: &Value) {
        if let (Some(intermediate), Value::GpuTensor(output)) = (&self.resident, output) {
            if gpu_helpers::same_gpu_handle(&intermediate.handle, output) {
                self.resident = None;
            }
        }
    }
}

impl Drop for ResidentResultGuard<'_> {
    fn drop(&mut self) {
        if let Some(resident) = &self.resident {
            if self
                .protected
                .iter()
                .any(|input| gpu_helpers::same_gpu_handle(&resident.handle, input))
            {
                return;
            }
            // Cleanup uses the authority admitted at capture, even after the
            // execution lease/thread selection or mutable metadata disappear.
            if resident.owner.free(&resident.handle).is_ok() {
                runmat_accelerate_api::clear_handle_metadata(&resident.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests;
