use super::{ConvParams, Geometry};
use crate::backend::wgpu::provider::WgpuProvider;
use runmat_accelerate_api::DeepLearningConv1DRequest;

impl WgpuProvider {
    /// The same physical geometry and buffer accounting used by the direct
    /// convolution kernel, for a whole-network route certificate.
    pub(crate) fn deep_learning_conv1d_forward_buffer_demand(
        &self,
        request: &DeepLearningConv1DRequest,
    ) -> Option<u64> {
        Geometry::new(request, self.precision)
            .ok()??
            .forward_owned_bytes(self.element_size as u64)
    }
}

impl Geometry {
    /// Explicit buffers retained by the forward dispatch. Borrowed input and
    /// parameter allocations, view materialization, and driver resources are
    /// outside this operation-owned count and must be added by route admission.
    pub fn forward_owned_bytes(&self, element_size: u64) -> Option<u64> {
        self.buffer_bytes(self.output_elements, element_size)?
            .checked_add(self.uniform_bytes(self.output_elements)?)
    }

    /// All three gradient outputs coexist. Each nonempty dispatch chunk has
    /// its own immutable uniform because command submission can outlive the
    /// caller's stack frame. No atomic or reduction scratch is allocated.
    pub fn adjoint_owned_bytes(&self, element_size: u64) -> Option<u64> {
        [
            self.input_elements,
            self.weight_elements,
            self.bias_elements,
        ]
        .into_iter()
        .try_fold(0u64, |total, elements| {
            total
                .checked_add(self.buffer_bytes(elements, element_size)?)?
                .checked_add(self.uniform_bytes(elements)?)
        })
    }

    fn buffer_bytes(&self, elements: usize, element_size: u64) -> Option<u64> {
        u64::try_from(elements)
            .ok()?
            .max(1)
            .checked_mul(element_size)
    }

    fn uniform_bytes(&self, elements: usize) -> Option<u64> {
        let lanes = u64::from(
            crate::backend::wgpu::shaders::deep_learning_convolution::CONV_WORKGROUP_SIZE,
        );
        let capacity =
            u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS).checked_mul(lanes)?;
        u64::try_from(elements)
            .ok()?
            .div_ceil(capacity)
            .checked_mul(std::mem::size_of::<ConvParams>() as u64)
    }
}
