use super::*;
use runmat_accelerate_api::{GpuTensorLayout, NumericElementType};

impl WgpuProvider {
    /// Pure complete-kernel admission. An anchor may describe a prior stage,
    /// so its shape is not required to equal the future convolution input.
    /// Every present source is still authenticated against the allocation
    /// owner before any planned upload or dispatch is allowed.
    pub(crate) fn deep_learning_conv1d_capability_exec(
        &self,
        operands: DeepLearningConv1DOperands<'_>,
        request: &DeepLearningConv1DRequest,
    ) -> Result<DeepLearningConv1DCapability> {
        let anchor_type = self.authenticate_conv_source(operands.anchor, None)?;
        // Authenticate every supplied lease before classifying any otherwise
        // unsupported class/layout or resource demand. A foreign later operand
        // must not be hidden by an earlier unsupported source.
        let mut supplied_types = [None; 4];
        for (slot, (handle, shape)) in [
            (operands.input, request.input_shape.as_slice()),
            (operands.weights, request.weights_shape.as_slice()),
            (operands.bias, request.bias_shape.as_slice()),
            (operands.upstream, request.output_shape.as_slice()),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(handle) = handle {
                supplied_types[slot] = Some(self.authenticate_conv_source(handle, Some(shape))?);
            }
        }
        let Some(anchor_type) = anchor_type else {
            return Ok(DeepLearningConv1DCapability::UnsupportedOperation);
        };
        if supplied_types
            .into_iter()
            .flatten()
            .any(|element_type| element_type != Some(anchor_type))
        {
            return Ok(DeepLearningConv1DCapability::UnsupportedOperation);
        }
        self.deep_learning_conv1d_capability_for_type(anchor_type, request)
    }

    /// Pure physical capability for a proposed host-origin source. Live
    /// resident operands are authenticated separately before this call.
    pub(crate) fn deep_learning_conv1d_capability_for_type(
        &self,
        element_type: NumericElementType,
        request: &DeepLearningConv1DRequest,
    ) -> Result<DeepLearningConv1DCapability> {
        let geometry = Geometry::new(request, self.precision)?;
        if !self.supports_native_real_element_type(element_type) {
            return Ok(DeepLearningConv1DCapability::UnsupportedOperation);
        }
        // Authentication of every supplied lease precedes a legal resource
        // rejection. An oversized request cannot mask a foreign/stale owner.
        let Some(geometry) = geometry else {
            return Ok(DeepLearningConv1DCapability::ResourceLimit);
        };
        if [
            geometry.input_elements,
            geometry.output_elements,
            geometry.weight_elements,
            geometry.bias_elements,
        ]
        .iter()
        .any(|&elements| !self.within_binding_limit(elements.max(1)))
        {
            return Ok(DeepLearningConv1DCapability::ResourceLimit);
        }
        // The complete route must separately include borrowed inputs,
        // uploads, optional materializations, output presentation, and any
        // concurrent layers in its provider-owned resource certificate.
        if geometry
            .forward_owned_bytes(self.element_size as u64)
            .is_none()
            || geometry
                .adjoint_owned_bytes(self.element_size as u64)
                .is_none()
        {
            return Ok(DeepLearningConv1DCapability::ResourceLimit);
        }
        Ok(DeepLearningConv1DCapability::Ready)
    }

    fn authenticate_conv_source(
        &self,
        handle: &GpuTensorHandle,
        expected_shape: Option<&[usize]>,
    ) -> Result<Option<NumericElementType>> {
        let entry = self.get_entry_raw(handle)?;
        ensure!(
            entry.element_type == handle.physical().element_type()
                && entry.storage == handle.physical().storage()
                && entry.logical == runmat_accelerate_api::handle_is_logical(handle)
                && entry.integer_type() == runmat_accelerate_api::handle_integer_type(handle)
                && entry.len
                    == handle
                        .shape()
                        .iter()
                        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                        .ok_or_else(|| anyhow!("convolution source shape overflows"))?,
            "convolution source differs from its provider-owned allocation"
        );
        if let Some(shape) = expected_shape {
            ensure!(
                handle.shape() == shape,
                "convolution source shape differs from admitted geometry"
            );
        }
        if (expected_shape.is_some()
            && !matches!(handle.view().layout(), GpuTensorLayout::Contiguous))
            || entry.storage != GpuTensorStorage::Real
            || entry.logical
            || entry.integer_type().is_some()
        {
            return Ok(None);
        }
        Ok(Some(entry.element_type))
    }
}
