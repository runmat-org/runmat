use super::*;

/// The original Z `Value` moves to the VM publication callback. Retain only
/// its shared numeric payload here so every device upload borrows one range
/// without constructing another full-size or tile-size host payload.
pub(super) struct RetainedSourceNoise {
    payload: NoisePayload,
    count: usize,
}

enum NoisePayload {
    Scalar(f64),
    Dense(HostNumericBuffer),
}

impl RetainedSourceNoise {
    pub(super) fn capture(
        value: &Value,
        shape: &[usize],
        class: NumericClass,
    ) -> BuiltinResult<Self> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(source_contract_error)?;
        let payload = match value {
            Value::Num(value) if class == NumericClass::Double && count == 1 => {
                NoisePayload::Scalar(*value)
            }
            Value::Tensor(tensor)
                if tensor.shape() == shape
                    && ((class == NumericClass::Single
                        && tensor.numeric_dtype() == NumericDType::F32)
                        || (class == NumericClass::Double
                            && tensor.numeric_dtype() == NumericDType::F64)) =>
            {
                NoisePayload::Dense(tensor.host_buffer().clone())
            }
            _ => return Err(source_contract_error()),
        };
        Ok(Self { payload, count })
    }

    pub(super) fn tile_view<'a>(
        &'a self,
        offset: usize,
        count: usize,
        tile_shape: &'a [usize; 2],
    ) -> BuiltinResult<HostNumericTensorView<'a>> {
        if count == 0
            || tile_shape != &[count, 1]
            || offset.checked_add(count).is_none_or(|end| end > self.count)
        {
            return Err(source_contract_error());
        }
        let data = match &self.payload {
            NoisePayload::Scalar(value) if offset == 0 && count == 1 => {
                HostNumericDataView::F64(std::slice::from_ref(value))
            }
            NoisePayload::Dense(buffer) => {
                let end = offset + count;
                if let Some(values) = buffer.as_f32_slice() {
                    HostNumericDataView::F32(
                        values.get(offset..end).ok_or_else(source_contract_error)?,
                    )
                } else if let Some(values) = buffer.as_f64_slice() {
                    HostNumericDataView::F64(
                        values.get(offset..end).ok_or_else(source_contract_error)?,
                    )
                } else {
                    return Err(source_contract_error());
                }
            }
            _ => return Err(source_contract_error()),
        };
        Ok(HostNumericTensorView {
            data,
            shape: tile_shape,
            storage: GpuTensorStorage::Real,
        })
    }

    pub(super) fn full_view<'a>(
        &'a self,
        shape: &'a [usize],
    ) -> BuiltinResult<HostNumericTensorView<'a>> {
        let count = shape
            .iter()
            .try_fold(1usize, |total, extent| total.checked_mul(*extent))
            .ok_or_else(source_contract_error)?;
        if count != self.count {
            return Err(source_contract_error());
        }
        let data = match &self.payload {
            NoisePayload::Scalar(value) if count == 1 => {
                HostNumericDataView::F64(std::slice::from_ref(value))
            }
            NoisePayload::Dense(buffer) => {
                if let Some(values) = buffer.as_f32_slice() {
                    HostNumericDataView::F32(values)
                } else if let Some(values) = buffer.as_f64_slice() {
                    HostNumericDataView::F64(values)
                } else {
                    return Err(source_contract_error());
                }
            }
            _ => return Err(source_contract_error()),
        };
        Ok(HostNumericTensorView {
            data,
            shape,
            storage: GpuTensorStorage::Real,
        })
    }
}
