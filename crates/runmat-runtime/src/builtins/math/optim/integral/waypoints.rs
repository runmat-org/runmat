//! Typed waypoint admission; path traversal is owned by the integration domain.

use num_complex::Complex;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;
use runmat_value::{NumericScalar, Value};

use super::integral_error_with_detail;
use crate::builtins::math::optim::quadrature::QuadratureCooperation;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

const WAYPOINT_CHUNK: usize = 4096;

fn invalid(detail: &str) -> crate::RuntimeError {
    integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, detail)
}

fn vector_shape(shape: &[usize]) -> bool {
    let (rows, cols) = match shape {
        [] => return true,
        [len] => (1, *len),
        [rows, cols, ..] => (*rows, *cols),
    };
    (rows <= 1 || cols <= 1 || rows == 0 || cols == 0)
        && shape.iter().skip(2).all(|&axis| axis == 1)
}

pub(super) fn value_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        integer => integer
            .into_int_value()
            .expect("nonfloating scalar is an integer")
            .to_f64(),
    }
}

fn finite(real: f64, imaginary: f64) -> BuiltinResult<Complex<f64>> {
    if real.is_finite() && imaginary.is_finite() {
        Ok(Complex::new(real, imaginary))
    } else {
        Err(invalid("Waypoints must be finite"))
    }
}

fn reserve(len: usize) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        build_runtime_error("integral: could not reserve waypoint sequence")
            .with_identifier("RunMat:Internal")
            .with_builtin("integral")
            .with_source(source)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build()
    })?;
    Ok(values)
}

pub(super) fn admit_shape(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => return Ok(()),
        Value::Tensor(tensor) => tensor.shape(),
        Value::ComplexTensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => {
            return Err(invalid(
                "Waypoints must be a real or complex numeric vector",
            ));
        }
    };
    if vector_shape(shape) {
        Ok(())
    } else {
        Err(invalid("Waypoints must be a vector"))
    }
}

pub(super) async fn decode(
    value: &Value,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    admit_shape(value)?;
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        let prepared = cooperation.prepare_integer_boundary(value).await?;
        if !prepared.is_exact() {
            return Err(invalid(
                "Waypoints integer values must convert to double exactly",
            ));
        }
    }
    decode_exact(value, cooperation).await
}

pub(super) async fn decode_exact(
    value: &Value,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    admit_shape(value)?;
    cooperation.completed(0).await?;
    let values = match value {
        Value::Num(number) => vec![finite(*number, 0.0)?],
        Value::Int(integer) => vec![finite(integer.to_f64(), 0.0)?],
        Value::Bool(flag) => vec![finite(if *flag { 1.0 } else { 0.0 }, 0.0)?],
        Value::Complex(real, imaginary) => vec![finite(*real, *imaginary)?],
        Value::Tensor(tensor) => {
            let mut values = reserve(tensor.len())?;
            for index in 0..tensor.len() {
                let number = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid Waypoints storage"))?;
                values.push(finite(value_f64(number), 0.0)?);
                if (index + 1).is_multiple_of(WAYPOINT_CHUNK) {
                    cooperation.completed(WAYPOINT_CHUNK as u64).await?;
                }
            }
            cooperation
                .completed((tensor.len() % WAYPOINT_CHUNK) as u64)
                .await?;
            values
        }
        Value::ComplexTensor(tensor) => {
            let mut values = reserve(tensor.len())?;
            for index in 0..tensor.len() {
                let (real, imaginary) = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid Waypoints storage"))?;
                values.push(finite(value_f64(real), value_f64(imaginary))?);
                if (index + 1).is_multiple_of(WAYPOINT_CHUNK) {
                    cooperation.completed(WAYPOINT_CHUNK as u64).await?;
                }
            }
            cooperation
                .completed((tensor.len() % WAYPOINT_CHUNK) as u64)
                .await?;
            values
        }
        Value::LogicalArray(array) => {
            let mut values = reserve(array.data.len())?;
            for (index, &bit) in array.data.iter().enumerate() {
                values.push(finite(if bit == 0 { 0.0 } else { 1.0 }, 0.0)?);
                if (index + 1).is_multiple_of(WAYPOINT_CHUNK) {
                    cooperation.completed(WAYPOINT_CHUNK as u64).await?;
                }
            }
            cooperation
                .completed((array.data.len() % WAYPOINT_CHUNK) as u64)
                .await?;
            values
        }
        Value::GpuTensor(_) => {
            return Err(invalid("Waypoints must be materialized by their owner"));
        }
        _ => {
            return Err(invalid(
                "Waypoints must be a real or complex numeric vector",
            ));
        }
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::task::noop_waker;
    use runmat_value::{ComplexStorage, ComplexTensor, Tensor};
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct PendingTurn(Rc<Cell<bool>>);

    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[test]
    fn waypoint_values_keep_complex_order_and_exact_integer_boundary() {
        let mut cooperation =
            QuadratureCooperation::new(super::super::QuadratureIdentity::Integral, None);
        let complex = ComplexTensor::from_complex_storage(
            ComplexStorage::F64(vec![(1.0, 1.0), (2.0, -1.0)].into()),
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            futures::executor::block_on(decode(&Value::ComplexTensor(complex), &mut cooperation))
                .unwrap(),
            vec![Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)]
        );
        let wide = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .unwrap();
        assert!(
            futures::executor::block_on(decode(&Value::Tensor(wide), &mut cooperation)).is_err()
        );
    }

    #[test]
    fn large_host_waypoint_decode_releases_pending_turn_on_abandonment() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let witness = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(witness.clone())),
        );
        let mut cooperation =
            QuadratureCooperation::with_budget(super::super::QuadratureIdentity::Integral, budget);
        let host =
            Value::Tensor(Tensor::new(vec![0.5; WAYPOINT_CHUNK], vec![1, WAYPOINT_CHUNK]).unwrap());
        let mut decode_future = Box::pin(decode(&host, &mut cooperation));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(decode_future.as_mut().poll(&mut context).is_pending());
        drop(decode_future);
        assert!(dropped.get());
        assert!(!source.token().is_cancelled());
    }
}
