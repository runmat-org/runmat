//! Dimensional min/max: Include-NaN, signed-zero ordering, first equal extremum.
use super::*;

#[cfg(test)]
mod tests;

impl InProcessProvider {
    pub(super) fn native_dimensional_extremum(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        minimum: bool,
    ) -> Result<runmat_accelerate_api::ReduceDimResult> {
        let (values, indices, shape) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let data = guard.get(input).ok_or_else(|| {
                anyhow!("dimensional extrema input is not a live owned allocation")
            })?;
            if input.shape().len() != 2
                || dim > 1
                || input.physical().storage() != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(input)
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "dimensional extrema requires a real numeric 2D view and axis 0 or 1",
                ));
            }
            ensure!(
                data.len() == logical_len_for_shape("dimensional extrema", input.shape())?,
                "dimensional extrema payload extent mismatch"
            );
            if data.len() == 0 {
                // Empty min/max preserves the original empty shape, not an
                // artificial nonempty reduction filled with infinities.
                let values = match data {
                    InProcessNumericData::F32(_) => InProcessNumericData::F32(Vec::new()),
                    InProcessNumericData::F64(_) => InProcessNumericData::F64(Vec::new()),
                    InProcessNumericData::Integer(values) => {
                        InProcessNumericData::Integer(empty_integer_like(values))
                    }
                };
                (values, Vec::new(), input.shape().to_vec())
            } else {
                let shape = if dim == 0 {
                    vec![1, input.shape()[1]]
                } else {
                    vec![input.shape()[0], 1]
                };
                match data {
                    InProcessNumericData::F32(values) => {
                        let (values, indices) =
                            reduce(values, input.shape(), dim, minimum, f32::is_nan, |v| {
                                (v == 0.0).then(|| v.is_sign_negative())
                            });
                        (InProcessNumericData::F32(values), indices, shape)
                    }
                    InProcessNumericData::F64(values) => {
                        let (values, indices) =
                            reduce(values, input.shape(), dim, minimum, f64::is_nan, |v| {
                                (v == 0.0).then(|| v.is_sign_negative())
                            });
                        (InProcessNumericData::F64(values), indices, shape)
                    }
                    InProcessNumericData::Integer(values) => {
                        let (values, indices, shape) =
                            reduce_integer_extrema_data_dim(values, input.shape(), dim, minimum)?;
                        (InProcessNumericData::Integer(values), indices, shape)
                    }
                }
            }
        };
        // Publication is outside the map lock; if indices fail, values' owned
        // lease is dropped normally. Neither output can retire the input.
        let values = self.allocate_numeric_data(values, shape.clone(), GpuTensorStorage::Real)?;
        let indices = self.allocate_tensor(indices, shape)?;
        Ok(runmat_accelerate_api::ReduceDimResult { values, indices })
    }
}

fn reduce<T: Copy + PartialOrd>(
    data: &[T],
    shape: &[usize],
    dim: usize,
    minimum: bool,
    is_nan: fn(T) -> bool,
    zero_sign: fn(T) -> Option<bool>,
) -> (Vec<T>, Vec<f64>) {
    let rows = shape[0];
    let count = if dim == 0 { shape[1] } else { rows };
    let extent = shape[dim];
    let mut values = Vec::with_capacity(count);
    let mut indices = Vec::with_capacity(count);
    for output in 0..count {
        let source = |position| {
            if dim == 0 {
                position + output * rows
            } else {
                output + position * rows
            }
        };
        let mut best = data[source(0)];
        let mut best_index = 0;
        for position in 1..extent {
            if is_nan(best) {
                break;
            }
            let candidate = data[source(position)];
            // Owning min/max domains order -0 below +0; only equal signed
            // zeros are ties. Preserve the first preferred zero's index.
            let preferred_zero = match (zero_sign(best), zero_sign(candidate)) {
                (Some(current), Some(candidate)) => {
                    if minimum {
                        candidate && !current
                    } else {
                        !candidate && current
                    }
                }
                _ => false,
            };
            if is_nan(candidate)
                || preferred_zero
                || (minimum && candidate < best)
                || (!minimum && candidate > best)
            {
                best = candidate;
                best_index = position;
            }
        }
        values.push(best);
        indices.push((best_index + 1) as f64);
    }
    (values, indices)
}

fn empty_integer_like(data: &HostIntegerDataOwned) -> HostIntegerDataOwned {
    macro_rules! empty { ($($class:ident),*) => { match data { $(HostIntegerDataOwned::$class(_) => HostIntegerDataOwned::$class(Vec::new()),)* } }; }
    empty!(I8, I16, I32, I64, U8, U16, U32, U64)
}
