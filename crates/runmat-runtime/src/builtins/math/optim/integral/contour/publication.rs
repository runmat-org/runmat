//! Checked, cooperative publication of complex-capable quadrature estimates.

use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, HostComplexBuffer, Tensor, Value,
};

use super::{invalid, resource_error, COMPONENT_CHUNK};
use crate::builtins::math::optim::quadrature::{QuadratureCooperation, ScaledComplex};
use crate::BuiltinResult;

pub(super) async fn zero(
    len: usize,
    shape: Vec<usize>,
    complex: bool,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    if complex {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(resource_error)?;
        while values.len() < len {
            let next = values.len().saturating_add(COMPONENT_CHUNK).min(len);
            let completed = next - values.len();
            values.resize(next, ComplexElement(0.0, 0.0));
            cooperation.completed(completed as u64).await?;
        }
        cooperation.completed(0).await?;
        complex_tensor(values, shape)
    } else {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(resource_error)?;
        while values.len() < len {
            let next = values.len().saturating_add(COMPONENT_CHUNK).min(len);
            let completed = next - values.len();
            values.resize(next, 0.0);
            cooperation.completed(completed as u64).await?;
        }
        cooperation.completed(0).await?;
        real_tensor(values, shape)
    }
}

pub(super) async fn estimate(
    estimates: &[ScaledComplex],
    shape: Vec<usize>,
    sign: f64,
    array_valued: bool,
    complex: bool,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    if !array_valued {
        cooperation.completed(0).await?;
        let value = estimates
            .first()
            .ok_or_else(|| invalid("missing scalar result"))?;
        let (real, imaginary) = value
            .to_pair()
            .ok_or_else(|| invalid("quadrature result is not representable as finite double"))?;
        return Ok(if complex {
            Value::Complex(sign * real, sign * imaginary)
        } else {
            Value::Num(sign * real)
        });
    }
    if complex {
        let mut values = Vec::new();
        values
            .try_reserve_exact(estimates.len())
            .map_err(resource_error)?;
        for (index, value) in estimates.iter().enumerate() {
            let (real, imaginary) = value.to_pair().ok_or_else(|| {
                invalid("quadrature result is not representable as finite double")
            })?;
            values.push(ComplexElement(sign * real, sign * imaginary));
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                cooperation.completed(COMPONENT_CHUNK as u64).await?;
            }
        }
        cooperation
            .completed((estimates.len() % COMPONENT_CHUNK) as u64)
            .await?;
        complex_tensor(values, shape)
    } else {
        let mut values = Vec::new();
        values
            .try_reserve_exact(estimates.len())
            .map_err(resource_error)?;
        for (index, value) in estimates.iter().enumerate() {
            let (real, _) = value.to_pair().ok_or_else(|| {
                invalid("quadrature result is not representable as finite double")
            })?;
            values.push(sign * real);
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                cooperation.completed(COMPONENT_CHUNK as u64).await?;
            }
        }
        cooperation
            .completed((estimates.len() % COMPONENT_CHUNK) as u64)
            .await?;
        real_tensor(values, shape)
    }
}

fn complex_tensor(values: Vec<ComplexElement<f64>>, shape: Vec<usize>) -> BuiltinResult<Value> {
    ComplexTensor::from_complex_storage(
        ComplexStorage::F64(HostComplexBuffer::from_elements(values)),
        shape,
    )
    .map(Value::ComplexTensor)
    .map_err(|cause| invalid(format!("complex result geometry: {cause}")))
}

fn real_tensor(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|cause| invalid(format!("real result geometry: {cause}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::optim::quadrature::QuadratureIdentity;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
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
    fn complex_and_real_publication_retire_pending_turn_on_cancel_or_drop() {
        for complex in [false, true] {
            for zeros in [false, true] {
                for cancel in [false, true] {
                    let source = CancellationSource::new();
                    let dropped = Rc::new(Cell::new(false));
                    let witness = dropped.clone();
                    let budget = CpuCooperationBudget::with_test_turn(
                        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
                        source.token(),
                        move || Box::pin(PendingTurn(witness.clone())),
                    );
                    let mut cooperation =
                        QuadratureCooperation::with_budget(QuadratureIdentity::Integral, budget);
                    let estimates = vec![ScaledComplex::new(1.0, 1.0); COMPONENT_CHUNK + 1];
                    let shape = vec![1, estimates.len()];
                    let mut operation = Box::pin(async {
                        if zeros {
                            zero(estimates.len(), shape, complex, &mut cooperation).await
                        } else {
                            estimate(&estimates, shape, 1.0, true, complex, &mut cooperation).await
                        }
                    });
                    let waker = noop_waker();
                    let mut context = Context::from_waker(&waker);
                    assert!(operation.as_mut().poll(&mut context).is_pending());
                    if cancel {
                        source.cancel(CancellationReason::User);
                        let Poll::Ready(Err(error)) = operation.as_mut().poll(&mut context) else {
                            panic!("cancelled publication must not return a value");
                        };
                        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                    }
                    drop(operation);
                    assert!(dropped.get());
                }
            }
        }
    }
}
