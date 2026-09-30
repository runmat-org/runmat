use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_scratch_bytes(
        &self,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<u64> {
        let (groups, _) = geometry(request)?;
        u64::try_from(groups)?
            .checked_mul(3)
            .and_then(|count| count.checked_mul(self.element_size as u64))
            .ok_or_else(|| anyhow!("normalization scratch byte count overflows"))
    }

    pub(crate) fn deep_learning_layer_norm_capability_exec(
        &self,
        input: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        ensure!(
            input.shape() == request.shape,
            "normalization input shape differs from admitted geometry"
        );
        self.deep_learning_layer_norm_planned_capability_exec(input, scale, offset, request)
    }

    pub(crate) fn deep_learning_layer_norm_planned_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        geometry(request)?;
        if let Some(scale) = scale {
            ensure!(
                scale.shape() == request.scale_shape,
                "normalization Scale descriptor shape mismatch"
            );
        }
        if let Some(offset) = offset {
            ensure!(
                offset.shape() == request.offset_shape,
                "normalization Offset descriptor shape mismatch"
            );
        }
        for handle in std::iter::once(anchor).chain(scale).chain(offset) {
            let entry = self.get_entry_raw(handle)?;
            ensure!(
                entry.element_type == handle.physical().element_type()
                    && entry.storage == handle.physical().storage()
                    && entry.len == shape_count(handle.shape())?
                    && entry.logical == runmat_accelerate_api::handle_is_logical(handle)
                    && entry.integer_type() == runmat_accelerate_api::handle_integer_type(handle),
                "normalization handle differs from provider-owned descriptor"
            );
            if entry.storage != GpuTensorStorage::Real
                || entry.logical
                || entry.integer_type().is_some()
                || entry.element_type != anchor.physical().element_type()
            {
                return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
            }
        }
        self.deep_learning_layer_norm_capability_for_type(anchor.physical().element_type(), request)
    }

    pub(crate) fn deep_learning_layer_norm_capability_for_type(
        &self,
        element_type: runmat_accelerate_api::NumericElementType,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        let (groups, group_size) = geometry(request)?;
        let total = shape_count(&request.shape)?;
        if !self.supports_native_real_element_type(element_type) {
            return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
        }
        if self.precision == crate::backend::wgpu::types::NumericPrecision::F32
            && (!(request.epsilon as f32).is_finite() || request.epsilon as f32 == 0.0)
        {
            return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
        }
        if total > u32::MAX as usize
            || groups > u32::MAX as usize
            || group_size > u32::MAX as usize
            || groups.checked_mul(3).is_none_or(|n| n > u32::MAX as usize)
            || request
                .shape
                .iter()
                .any(|&extent| extent > u32::MAX as usize)
        {
            return Ok(DeepLearningLayerNormCapability::ResourceLimit);
        }
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        let element_size = u64::try_from(self.element_size)?;
        if [total, groups * 3, request.shape[request.channel_axis]]
            .iter()
            .any(|&count| {
                u64::try_from(count)
                    .ok()
                    .and_then(|n| n.checked_mul(element_size))
                    .is_none_or(|bytes| bytes > binding_limit)
            })
        {
            return Ok(DeepLearningLayerNormCapability::ResourceLimit);
        }
        Ok(DeepLearningLayerNormCapability::Ready)
    }
}
