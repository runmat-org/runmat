//! Native floating reference medians over the authenticated payload owner.
use super::*;

impl InProcessProvider {
    pub(super) fn native_median(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let (output, shape) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("median input is not a live owned allocation"))?;
            if input.physical().storage() != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(input)
                || runmat_accelerate_api::handle_is_transposed(input)
                || matches!(data, InProcessNumericData::Integer(_))
                || dim.is_some_and(|axis| input.shape().len() != 2 || axis > 1)
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "median requires contiguous native real floating storage; dimensional hooks support 2D axes 0 and 1",
                ));
            }
            ensure!(
                data.len() == logical_len_for_shape("median", input.shape())?,
                "median payload extent mismatch"
            );
            let shape = match dim {
                None => vec![1, 1],
                Some(0) => vec![1, input.shape()[1]],
                Some(_) => vec![input.shape()[0], 1],
            };
            let output = match data {
                // Empty Single host reduction currently promotes to Double;
                // leave that domain rule to the established fallback owner.
                InProcessNumericData::F32(values) if values.is_empty() => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "empty Single median requires host class adaptation",
                    ));
                }
                InProcessNumericData::F32(values) => InProcessNumericData::F32(reduce(
                    values,
                    input.shape(),
                    dim,
                    f32::NAN,
                    f32::is_nan,
                    2.0,
                )),
                InProcessNumericData::F64(values) => InProcessNumericData::F64(reduce(
                    values,
                    input.shape(),
                    dim,
                    f64::NAN,
                    f64::is_nan,
                    2.0,
                )),
                InProcessNumericData::Integer(_) => {
                    unreachable!("integer admission rejected above")
                }
            };
            (output, shape)
        };
        self.allocate_numeric_data(output, shape, GpuTensorStorage::Real)
    }
}

fn reduce<T>(
    data: &[T],
    shape: &[usize],
    dim: Option<usize>,
    nan: T,
    is_nan: fn(T) -> bool,
    two: T,
) -> Vec<T>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
{
    let (count, extent) = match dim {
        None => (1, data.len()),
        Some(axis) => (shape[1 - axis], shape[axis]),
    };
    let mut output = Vec::with_capacity(count);
    let mut scratch = Vec::with_capacity(extent);
    for index in 0..count {
        scratch.clear();
        for position in 0..extent {
            let source = match dim {
                None => position,
                Some(0) => position + index * shape[0],
                Some(_) => index + position * shape[0],
            };
            scratch.push(data[source]);
        }
        if scratch.is_empty() || scratch.iter().copied().any(is_nan) {
            output.push(nan);
            continue;
        }
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let middle = scratch.len() / 2;
        output.push(if scratch.len() % 2 == 1 {
            scratch[middle]
        } else {
            (scratch[middle - 1] + scratch[middle]) / two
        });
    }
    output
}

#[cfg(test)]
mod tests {
    use super::reduce;

    #[test]
    fn native_median_preserves_rounding_nan_and_signed_zero_rules() {
        // A widened intermediate would avoid native Single overflow here.
        let single = reduce(
            &[f32::MAX, f32::MAX],
            &[2, 1],
            None,
            f32::NAN,
            f32::is_nan,
            2.0,
        );
        assert_eq!(single, vec![f32::INFINITY]);
        let zero = reduce(&[-0.0f32], &[1, 1], None, f32::NAN, f32::is_nan, 2.0);
        assert_eq!(zero[0].to_bits(), (-0.0f32).to_bits());
        let nan = reduce(
            &[1.0f64, f64::NAN],
            &[2, 1],
            None,
            f64::NAN,
            f64::is_nan,
            2.0,
        );
        assert!(nan[0].is_nan());
        assert_eq!(
            reduce(
                &[1., 3., 2., 4., 100., 5.],
                &[2, 3],
                Some(1),
                f64::NAN,
                f64::is_nan,
                2.0
            ),
            vec![2., 4.]
        );
        assert_eq!(
            reduce(
                &[1., 3., 2., 4., 100., 5.],
                &[2, 3],
                None,
                f64::NAN,
                f64::is_nan,
                2.0
            ),
            vec![3.5]
        );
    }
}
