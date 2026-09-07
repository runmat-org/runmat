use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

use crate::builtins::common::gpu_helpers;

pub(in crate::builtins::math::elementwise::binary_arithmetic) struct ExpandedPair<'a> {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) provider: &'a dyn AccelProvider,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) original_left:
        &'a GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) original_right:
        &'a GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) left: GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) right: GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) owns_left: bool,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) owns_right: bool,
}

impl ExpandedPair<'_> {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn release(
        &self,
        additional_protected: &[&GpuTensorHandle],
    ) {
        if self.owns_left {
            let mut protected = vec![self.original_left];
            protected.extend_from_slice(additional_protected);
            gpu_helpers::free_rejected_provider_output(&self.left, &protected, self.provider);
        }
        if self.owns_right {
            let mut protected = vec![self.original_right];
            protected.extend_from_slice(additional_protected);
            gpu_helpers::free_rejected_provider_output(&self.right, &protected, self.provider);
        }
    }
}
