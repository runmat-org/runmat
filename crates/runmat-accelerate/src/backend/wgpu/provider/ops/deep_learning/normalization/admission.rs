use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_adjoint_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        statistics: Option<&GpuTensorHandle>,
        upstream: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        let capability =
            self.deep_learning_layer_norm_planned_capability_exec(anchor, scale, offset, request)?;
        if capability != DeepLearningLayerNormCapability::Ready {
            return Ok(capability);
        }
        let (groups, _) = geometry(request)?;
        let expected_statistics = [3, groups];
        for (handle, expected) in [
            (statistics, expected_statistics.as_slice()),
            (upstream, request.shape.as_slice()),
        ] {
            if let Some(handle) = handle {
                let owned = self.get_entry_raw(handle)?;
                ensure!(
                    handle.shape() == expected
                        && owned.shape.as_slice() == expected
                        && owned.len == shape_count(expected)?
                        && owned.element_type == handle.physical().element_type()
                        && owned.storage == handle.physical().storage()
                        && owned.logical == runmat_accelerate_api::handle_is_logical(handle)
                        && owned.integer_type()
                            == runmat_accelerate_api::handle_integer_type(handle),
                    "normalization adjoint descriptor differs from provider-owned geometry"
                );
                if owned.element_type != anchor.physical().element_type()
                    || owned.storage != GpuTensorStorage::Real
                    || owned.logical
                    || owned.integer_type().is_some()
                {
                    return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
                }
            }
        }
        let total = shape_count(&request.shape)?;
        let channels = request.shape[request.channel_axis];
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        let element_size = u64::try_from(self.element_size)?;
        if [total, groups * 2, channels].iter().any(|&count| {
            u64::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(element_size))
                .is_none_or(|bytes| bytes > binding_limit)
        }) {
            return Ok(DeepLearningLayerNormCapability::ResourceLimit);
        }
        Ok(DeepLearningLayerNormCapability::Ready)
    }

    /// Exact explicit statistics/output/uniform demand for one layer-norm
    /// dispatch. Input/parameter view materializations are accounted for by
    /// the whole-route caller before this kernel is invoked.
    pub(crate) fn layer_norm_buffer_demand(
        &self,
        request: &DeepLearningLayerNormRequest,
    ) -> Option<u64> {
        let (groups, _) = geometry(request).ok()?;
        let total = shape_count(&request.shape).ok()? as u64;
        let output = total.max(1).checked_mul(self.element_size as u64)?;
        if total == 0 {
            return Some(output);
        }
        let statistics = self.deep_learning_layer_norm_scratch_bytes(request).ok()?;
        let capacity = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
            .checked_mul(u64::from(
                crate::backend::wgpu::shaders::deep_learning_normalization::NORMALIZATION_WORKGROUP_SIZE,
            ))?;
        let chunks = (groups as u64)
            .div_ceil(capacity)
            .checked_add(total.div_ceil(capacity))?;
        output
            .checked_add(statistics)?
            .checked_add(chunks.checked_mul(std::mem::size_of::<NormParams>() as u64)?)
    }

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
        if total != 0 && !self.ieee_words_available(self.precision) {
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
