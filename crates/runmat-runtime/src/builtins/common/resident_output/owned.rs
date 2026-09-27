//! Scoped ownership of a validated numeric result during output adaptation.

use crate::builtins::common::gpu_helpers;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

/// Owns a top-level resident numeric result until it is published or replaced.
///
/// The caller validates the result's class, shape and storage before capture,
/// and supplies every borrowed operand/prototype. Container results require the
/// separate recursive alias protocol and must not use this numeric-only guard.
pub(crate) struct ResidentResultGuard<'a> {
    resident: Option<GpuTensorHandle>,
    protected: &'a [&'a GpuTensorHandle],
}

impl<'a> ResidentResultGuard<'a> {
    pub(crate) fn new(value: &Value, protected: &'a [&'a GpuTensorHandle]) -> Result<Self, String> {
        let resident = match value {
            Value::GpuTensor(handle) => {
                gpu_helpers::exact_provider_for_handle(handle)
                    .ok_or_else(|| "no provider owns the resident result".to_string())?;
                Some(handle.clone())
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
            if gpu_helpers::same_gpu_handle(intermediate, output) {
                self.resident = None;
            }
        }
    }
}

impl Drop for ResidentResultGuard<'_> {
    fn drop(&mut self) {
        if let Some(handle) = &self.resident {
            gpu_helpers::free_unprotected_exact_owner(handle, self.protected);
        }
    }
}

#[cfg(test)]
mod tests;
