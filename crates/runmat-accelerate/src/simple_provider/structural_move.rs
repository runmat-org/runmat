//! Bit-preserving in-process structural producers over authoritative payloads.
use super::*;

impl InProcessProvider {
    pub(super) fn flip_structural(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        let result = (|| -> Result<Option<GpuTensorHandle>> {
            let data = {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("flip: allocation owner is poisoned"))?;
                let data = guard
                    .get(handle)
                    .ok_or_else(|| anyhow!("flip: input is not a live owned allocation"))?;
                if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                    || handle.physical().storage() == GpuTensorStorage::ComplexInterleaved
                    || matches!(data, InProcessNumericData::F32(_))
                    || (handle.shape().contains(&0) && handle.physical().element_capacity() != 0)
                {
                    return Ok(None);
                }
                if !runmat_types::geometry::FlipGeometry::new(handle.shape(), axes).has_effect() {
                    return Ok(Some(handle.clone()));
                }
                data.clone()
            };
            let data = match data {
                InProcessNumericData::F64(values) => {
                    InProcessNumericData::F64(flip_data(&values, handle.shape(), axes)?)
                }
                InProcessNumericData::Integer(values) => {
                    InProcessNumericData::Integer(flip_integer_data(&values, handle.shape(), axes)?)
                }
                InProcessNumericData::F32(_) => {
                    unreachable!("native Single declined before cloning")
                }
            };
            self.allocate_produced_observed(
                data,
                handle.shape().to_vec(),
                handle.physical().storage(),
                handle.view().is_logical(),
                Some(&allocated),
            )
            .map(Some)
        })();
        result.map_err(|error| {
            let error = runmat_accelerate_api::ProviderExecutionError::from(error);
            if allocated.get() {
                error.after_effect()
            } else {
                error.with_unproven_effects()
            }
        })
    }

    pub(super) fn permute_structural(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        let result = (|| -> Result<Option<GpuTensorHandle>> {
            let data = {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("permute: allocation owner is poisoned"))?;
                let data = guard
                    .get(handle)
                    .ok_or_else(|| anyhow!("permute: input is not a live owned allocation"))?;
                // Capability absence is decided while borrowing authoritative
                // storage, before cloning or allocating a producer payload.
                if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                    || (matches!(data, InProcessNumericData::Integer(_))
                        && handle.physical().storage() == GpuTensorStorage::ComplexInterleaved)
                    || (handle.shape().contains(&0) && handle.physical().element_capacity() != 0)
                {
                    return Ok(None);
                }
                data.clone()
            };
            let storage = handle.physical().storage();
            let lanes = if storage == GpuTensorStorage::ComplexInterleaved {
                2
            } else {
                1
            };
            let (data, shape) = match data {
                InProcessNumericData::F32(values) => {
                    let (values, shape) = permute_data(&values, handle.shape(), order, lanes)?;
                    (InProcessNumericData::F32(values), shape)
                }
                InProcessNumericData::F64(values) => {
                    let (values, shape) = permute_data(&values, handle.shape(), order, lanes)?;
                    (InProcessNumericData::F64(values), shape)
                }
                InProcessNumericData::Integer(values) => {
                    let (values, shape) = permute_integer_data(&values, handle.shape(), order)?;
                    (InProcessNumericData::Integer(values), shape)
                }
            };
            self.allocate_produced_observed(
                data,
                shape,
                storage,
                handle.view().is_logical(),
                Some(&allocated),
            )
            .map(Some)
        })();
        result.map_err(|error| {
            let error = runmat_accelerate_api::ProviderExecutionError::from(error);
            if allocated.get() {
                error.after_effect()
            } else {
                error.with_unproven_effects()
            }
        })
    }

    pub(super) fn replicate_structural(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        let result = (|| -> Result<Option<GpuTensorHandle>> {
            let data = {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("repmat: allocation owner is poisoned"))?;
                let data = guard
                    .get(handle)
                    .ok_or_else(|| anyhow!("repmat: input is not a live owned allocation"))?;
                // Capability absence is decided while borrowing authoritative
                // storage, before cloning or allocating a producer payload.
                if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                    || (matches!(data, InProcessNumericData::Integer(_))
                        && handle.physical().storage() == GpuTensorStorage::ComplexInterleaved)
                    || (handle.shape().contains(&0) && handle.physical().element_capacity() != 0)
                {
                    return Ok(None);
                }
                data.clone()
            };
            let storage = handle.physical().storage();
            let (data, shape) = match data {
                InProcessNumericData::F32(values) => {
                    let (values, shape) = if storage == GpuTensorStorage::ComplexInterleaved {
                        repmat_complex_interleaved(&values, handle.shape(), reps)?
                    } else {
                        repmat_numeric(&values, handle.shape(), reps)?
                    };
                    (InProcessNumericData::F32(values), shape)
                }
                InProcessNumericData::F64(values) => {
                    let (values, shape) = if storage == GpuTensorStorage::ComplexInterleaved {
                        repmat_complex_interleaved(&values, handle.shape(), reps)?
                    } else {
                        repmat_numeric(&values, handle.shape(), reps)?
                    };
                    (InProcessNumericData::F64(values), shape)
                }
                InProcessNumericData::Integer(values) => {
                    let (values, shape) = repmat_integer_data(&values, handle.shape(), reps)?;
                    (InProcessNumericData::Integer(values), shape)
                }
            };
            self.allocate_produced_observed(
                data,
                shape,
                storage,
                handle.view().is_logical(),
                Some(&allocated),
            )
            .map(Some)
        })();
        result.map_err(|error| {
            let error = runmat_accelerate_api::ProviderExecutionError::from(error);
            if allocated.get() {
                error.after_effect()
            } else {
                error.with_unproven_effects()
            }
        })
    }
}
