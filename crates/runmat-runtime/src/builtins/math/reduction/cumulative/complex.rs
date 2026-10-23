//! Native complex prefix arithmetic over borrowed canonical storage.
use super::options::Options;
use crate::{context::RuntimeContext, execution::CpuCooperationBudget, BuiltinResult};
use num_traits::Float;
use runmat_builtins::{CumulativeKind, CumulativeMissingMode, CumulativeScanDirection};
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor};

async fn tick(
    kind: CumulativeKind,
    runtime: Option<&RuntimeContext>,
    budget: &mut Option<CpuCooperationBudget>,
) -> BuiltinResult<()> {
    if let Some(budget) = budget {
        budget.checkpoint(1).await.map_err(|source| {
            let mut error = source.into_runtime_error_in_context(runtime);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    }
    super::admission::host(kind, runtime)
}
pub(in crate::builtins::math::reduction) async fn cumulative(
    tensor: &ComplexTensor,
    dimension: usize,
    options: &Options,
    kind: CumulativeKind,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<ComplexTensor> {
    super::admission::host(kind, runtime)?;
    if dimension == 0 {
        return Err(super::failure(kind, "dimension must be positive"));
    }
    if tensor.is_empty() || dimension > tensor.shape().len() {
        return Ok(tensor.clone());
    }
    let storage = if let Some(values) = tensor.as_f32_slice() {
        ComplexStorage::F32(
            scan(values, tensor.shape(), dimension, options, kind, runtime)
                .await?
                .into(),
        )
    } else if let Some(values) = tensor.as_f64_slice() {
        ComplexStorage::F64(
            scan(values, tensor.shape(), dimension, options, kind, runtime)
                .await?
                .into(),
        )
    } else {
        return Err(super::failure(
            kind,
            "complex integer storage is unsupported",
        ));
    };
    ComplexTensor::from_complex_storage(storage, tensor.shape().to_vec())
        .map_err(|e| super::failure(kind, e))
}
async fn scan<T: Float>(
    values: &[ComplexElement<T>],
    shape: &[usize],
    dimension: usize,
    options: &Options,
    kind: CumulativeKind,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<ComplexElement<T>>> {
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    let mut output = Vec::new();
    output.try_reserve_exact(values.len()).map_err(|_| {
        super::admission::control(
            kind,
            runtime,
            crate::context::host_allocation::HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ),
        )
    })?;
    for _ in values {
        tick(kind, runtime, &mut budget).await?;
        output.push(ComplexElement(T::zero(), T::zero()));
    }
    let axis = dimension - 1;
    let segment = shape[axis];
    let before = shape[..axis].iter().product::<usize>();
    let after = shape[dimension..].iter().product::<usize>();
    let block = before * segment;
    for outer in 0..after {
        for inner in 0..before {
            let mut accumulator = ComplexElement(
                if kind == CumulativeKind::Product {
                    T::one()
                } else {
                    T::zero()
                },
                T::zero(),
            );
            let mut poisoned = false;
            for offset in 0..segment {
                let offset = match options.direction {
                    CumulativeScanDirection::Forward => offset,
                    CumulativeScanDirection::Reverse => segment - 1 - offset,
                };
                let at = outer * block + inner + offset * before;
                let value = values[at];
                let missing = value.0.is_nan() || value.1.is_nan();
                if options.missing == CumulativeMissingMode::Include && (missing || poisoned) {
                    poisoned = true;
                    output[at] = ComplexElement(T::nan(), T::nan());
                } else {
                    if !missing {
                        accumulator = match kind {
                            CumulativeKind::Sum => {
                                ComplexElement(accumulator.0 + value.0, accumulator.1 + value.1)
                            }
                            CumulativeKind::Product => ComplexElement(
                                accumulator.0 * value.0 - accumulator.1 * value.1,
                                accumulator.0 * value.1 + accumulator.1 * value.0,
                            ),
                            _ => {
                                return Err(super::failure(
                                    kind,
                                    "invalid complex arithmetic operation",
                                ))
                            }
                        };
                    }
                    output[at] = accumulator;
                }
                tick(kind, runtime, &mut budget).await?;
            }
        }
    }
    Ok(output)
}
