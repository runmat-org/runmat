use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

pub(in crate::builtins::math::elementwise::binary_arithmetic) struct ExpandedPair<'a> {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) provider: &'a dyn AccelProvider,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) original_left:
        &'a GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) original_right:
        &'a GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) left: GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) right: GpuTensorHandle,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) expanded_left: bool,
    pub(in crate::builtins::math::elementwise::binary_arithmetic) expanded_right: bool,
}
