//! Native floating NaN transforms used by numerical reductions. A numeric
//! mask retains the input's physical precision; it is not a logical view.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum NanMap {
    Zero,
    NumericMask,
}

fn checked_len(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow!("NaN transform shape exceeds provider capacity"))
}

fn transform<T: Copy + From<u8>>(
    values: &[T],
    is_nan: impl Fn(T) -> bool,
    mode: NanMap,
) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(values.len())?;
    for &value in values {
        output.push(match mode {
            NanMap::Zero if is_nan(value) => T::from(0),
            NanMap::Zero => value,
            NanMap::NumericMask => T::from(u8::from(!is_nan(value))),
        });
    }
    Ok(output)
}

impl InProcessProvider {
    pub(super) fn nan_map_impl(
        &self,
        input: &GpuTensorHandle,
        mode: NanMap,
    ) -> Result<GpuTensorHandle> {
        let output = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("NaN transform owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("NaN transform input is not live and owned"))?;
            if input.physical().storage() != GpuTensorStorage::Real || input.view().is_logical() {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "NaN transform requires real floating input",
                ));
            }
            let length = checked_len(input.shape())?;
            match (data, input.physical().element_type()) {
                (InProcessNumericData::F32(values), NumericElementType::F32) => {
                    ensure!(
                        values.len() == length,
                        "NaN transform payload length mismatch"
                    );
                    InProcessNumericData::F32(transform(values, f32::is_nan, mode)?)
                }
                (InProcessNumericData::F64(values), NumericElementType::F64) => {
                    ensure!(
                        values.len() == length,
                        "NaN transform payload length mismatch"
                    );
                    InProcessNumericData::F64(transform(values, f64::is_nan, mode)?)
                }
                _ => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "NaN transform requires native floating storage",
                    ));
                }
            }
        };
        self.allocate_numeric_data(output, input.shape().to_vec(), GpuTensorStorage::Real)
    }
}

#[cfg(test)]
mod tests;
