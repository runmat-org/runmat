//! Checked, cooperative host materialization for admitted IIR operands.

use super::*;
use runmat_value::{ComplexTensor, LogicalArray};

const PREPARATION_CHUNK: usize = 4096;

fn reservation_error(source: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error("filter: operand storage reservation failed")
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = FILTER_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn materialize(
    len: usize,
    cooperation: &mut SignalCooperation,
    mut at: impl FnMut(usize) -> Option<Complex<f64>>,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(reservation_error)?;
    while data.len() < len {
        let start = data.len();
        let end = start.saturating_add(PREPARATION_CHUNK).min(len);
        for index in start..end {
            data.push(at(index).ok_or_else(|| filter_error(&FILTER_ERROR_INTERNAL))?);
        }
        cooperation.completed((end - start) as u64).await?;
    }
    Ok(data)
}

pub(super) async fn real_tensor(
    tensor: &Tensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(tensor.len(), cooperation, |index| {
        tensor
            .numeric_value_at(index)
            .map(|value| Complex::new(value.materialize_f64(), 0.0))
    })
    .await
}

pub(super) async fn complex_tensor(
    tensor: &ComplexTensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(tensor.len(), cooperation, |index| {
        tensor.numeric_value_at(index).map(|(real, imaginary)| {
            Complex::new(real.materialize_f64(), imaginary.materialize_f64())
        })
    })
    .await
}

pub(super) async fn logical_array(
    logical: &LogicalArray,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(logical.data.len(), cooperation, |index| {
        logical
            .data
            .get(index)
            .map(|value| Complex::new(if *value == 0 { 0.0 } else { 1.0 }, 0.0))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_value::LogicalArray;
    use std::num::NonZeroU64;

    #[test]
    fn logical_materialization_preserves_shape_order_and_boolean_values() {
        let logical = LogicalArray::new(vec![2, 0, 1, 0], vec![2, 2]).unwrap();
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let values = block_on(logical_array(&logical, &mut cooperation)).unwrap();
        assert_eq!(
            values,
            [2.0, 0.0, 1.0, 0.0]
                .map(|value| Complex::new(if value == 0.0 { 0.0 } else { 1.0 }, 0.0))
        );
    }

    #[test]
    fn large_operand_preparation_retains_service_cause_and_never_retries_gather() {
        let tensor = Tensor::new(
            vec![1.0; PREPARATION_CHUNK + 1],
            vec![PREPARATION_CHUNK + 1, 1],
        )
        .unwrap();
        let cancellation = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let Err(error) = block_on(real_tensor(&tensor, &mut cooperation)) else {
            panic!("failed cooperation turn must stop preparation")
        };
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error)
            .is_some_and(|cause| cause.is::<ExecutionServiceError>()));
    }
}
