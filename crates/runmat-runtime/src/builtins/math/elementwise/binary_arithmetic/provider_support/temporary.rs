use crate::builtins::common::gpu_helpers;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

/// A fresh intermediate allocation, never a borrowed operand or public result.
pub(in crate::builtins::math::elementwise::binary_arithmetic) struct Temporary<'a> {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) handle: GpuTensorHandle,
    provider: &'a dyn AccelProvider,
    protected: Vec<&'a GpuTensorHandle>,
}

impl<'a> Temporary<'a> {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn new(
        handle: GpuTensorHandle,
        provider: &'a dyn AccelProvider,
        protected: &[&'a GpuTensorHandle],
    ) -> Self {
        Self {
            handle,
            provider,
            protected: protected.to_vec(),
        }
    }
}

impl Drop for Temporary<'_> {
    fn drop(&mut self) {
        gpu_helpers::free_rejected_provider_output(&self.handle, &self.protected, self.provider);
    }
}
