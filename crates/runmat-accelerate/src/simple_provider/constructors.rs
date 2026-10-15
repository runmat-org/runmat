//! Concrete in-process constant constructors; absence is decided before effects.
use super::*;
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use std::cell::Cell;

fn execute_constructor(
    action: impl FnOnce(&Cell<bool>) -> Result<Option<GpuTensorHandle>>,
) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
    let allocated = Cell::new(false);
    action(&allocated).map_err(|source| {
        let source = ProviderExecutionError::from(source);
        if allocated.get() {
            source.after_effect()
        } else {
            source.with_unproven_effects()
        }
    })
}

impl InProcessProvider {
    pub(super) fn fill_constructor(
        &self,
        shape: &[usize],
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            let len = logical_len_for_shape("fill", shape)?;
            // Preserve the admitted provider's existing zeros + scalar-add
            // numerical route, but never retry any raw error from either stage.
            let base = self.allocate_produced_observed(
                InProcessNumericData::F64(vec![0.0; len]),
                shape.to_vec(),
                GpuTensorStorage::Real,
                false,
                Some(effect),
            )?;
            if value == 0.0 {
                return Ok(Some(base));
            }
            self.scalar_add(&base, value).map(Some)
        })
    }

    pub(super) fn zeros_like_constructor(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            let logical = {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
                guard
                    .get(prototype)
                    .ok_or_else(|| anyhow!("zeros_like requires a live owned prototype"))?;
                guard
                    .producer_is_logical(prototype)
                    .ok_or_else(|| anyhow!("zeros_like producer facts unavailable"))?
            };
            let len = logical_len_for_shape("zeros_like", shape)?;
            let storage = prototype.physical().storage();
            let lanes = len
                .checked_mul(if storage == GpuTensorStorage::ComplexInterleaved {
                    2
                } else {
                    1
                })
                .ok_or_else(|| anyhow!("zeros_like: tensor size overflow"))?;
            let data = match prototype.physical().element_type() {
                NumericElementType::F64 => InProcessNumericData::F64(vec![0.0; lanes]),
                NumericElementType::F32 => InProcessNumericData::F32(vec![0.0; lanes]),
                element => {
                    let data = match element
                        .integer_type()
                        .ok_or_else(|| anyhow!("zeros_like: unsupported element type"))?
                    {
                        IntegerElementType::I8 => HostIntegerDataOwned::I8(vec![0; lanes]),
                        IntegerElementType::I16 => HostIntegerDataOwned::I16(vec![0; lanes]),
                        IntegerElementType::I32 => HostIntegerDataOwned::I32(vec![0; lanes]),
                        IntegerElementType::I64 => HostIntegerDataOwned::I64(vec![0; lanes]),
                        IntegerElementType::U8 => HostIntegerDataOwned::U8(vec![0; lanes]),
                        IntegerElementType::U16 => HostIntegerDataOwned::U16(vec![0; lanes]),
                        IntegerElementType::U32 => HostIntegerDataOwned::U32(vec![0; lanes]),
                        IntegerElementType::U64 => HostIntegerDataOwned::U64(vec![0; lanes]),
                    };
                    InProcessNumericData::Integer(data)
                }
            };
            self.allocate_produced_observed(
                data,
                shape.to_vec(),
                storage,
                logical && prototype.view().is_logical(),
                Some(effect),
            )
            .map(Some)
        })
    }

    pub(super) fn logical_fill_constructor(
        &self,
        shape: &[usize],
        value: bool,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            self.allocate_produced_observed(
                InProcessNumericData::F64(vec![
                    if value { 1.0 } else { 0.0 };
                    logical_len_for_shape("logical_fill", shape)?
                ]),
                shape.to_vec(),
                GpuTensorStorage::Real,
                true,
                Some(effect),
            )
            .map(Some)
        })
    }

    pub(super) fn fill_like_constructor(
        &self,
        prototype: &GpuTensorHandle,
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        if value.to_bits() == 0.0f64.to_bits() {
            return self.zeros_like_constructor(prototype, prototype.shape());
        }
        execute_constructor(|effect| {
            {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
                ensure!(
                    guard.get(prototype).is_some(),
                    "fill_like: prototype is not a live owned allocation"
                );
            }
            if prototype.physical().storage() != GpuTensorStorage::Real
                || prototype.view().is_logical()
            {
                return Ok(None);
            }
            let len = logical_len_for_shape("fill_like", prototype.shape())?;
            let data = match prototype.physical().element_type() {
                NumericElementType::F32 => InProcessNumericData::F32(vec![value as f32; len]),
                NumericElementType::F64 => InProcessNumericData::F64(vec![value; len]),
                _ => return Ok(None),
            };
            self.allocate_produced_observed(
                data,
                prototype.shape().to_vec(),
                GpuTensorStorage::Real,
                false,
                Some(effect),
            )
            .map(Some)
        })
    }

    pub(super) fn ones_constructor(
        &self,
        shape: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            self.allocate_produced_observed(
                InProcessNumericData::F64(vec![1.0; logical_len_for_shape("ones", shape)?]),
                shape.to_vec(),
                GpuTensorStorage::Real,
                false,
                Some(effect),
            )
            .map(Some)
        })
    }
}
