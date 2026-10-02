//! Array-valued integral callbacks share the adaptive traversal with scalar callbacks.

use runmat_value::{Tensor, Value};
use std::sync::Arc;

use super::{
    integral_error_with_detail, integral_map_error, quadrature_arithmetic_error, IntegralOptions,
    MAX_DEPTH, NAME,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::call_function;
use crate::builtins::math::optim::quadrature::{
    integrate_array, QuadratureArithmeticError, QuadratureCooperation, QuadratureEvaluator,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::INTEGRAL_ERROR_INVALID_INPUT;

const ELEMENT_CHUNK: usize = 4096;

struct ArrayEvaluator<'a> {
    function: &'a Value,
    options: &'a IntegralOptions,
    cooperation: &'a mut QuadratureCooperation,
    shape: Option<Vec<usize>>,
}

impl ArrayEvaluator<'_> {
    async fn sample(&mut self, x: f64) -> BuiltinResult<Arc<Vec<f64>>> {
        self.cooperation.completed(0).await?;
        let value = call_function(self.function, [Value::Num(x)]).await?;
        self.cooperation.completed(1).await?;
        admit_callback_value(&value, self.shape.as_deref())?;
        let value = crate::dispatcher::gather_if_needed_async(&value)
            .await
            .map_err(|error| integral_map_error(error, &INTEGRAL_ERROR_INVALID_INPUT))?;
        let (shape, values) = match value {
            Value::Num(value) if value.is_finite() => (vec![1, 1], vec![value]),
            Value::Num(_) => return Err(nonfinite_error()),
            Value::Int(value) => (vec![1, 1], vec![value.to_f64()]),
            Value::Bool(value) => (vec![1, 1], vec![if value { 1.0 } else { 0.0 }]),
            Value::Tensor(value) => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(value.len())
                    .map_err(resource_error)?;
                for index in 0..value.len() {
                    let element = tensor::tensor_value_f64(&value, index);
                    if !element.is_finite() {
                        return Err(nonfinite_error());
                    }
                    values.push(element);
                    if (index + 1).is_multiple_of(ELEMENT_CHUNK) {
                        self.cooperation.completed(ELEMENT_CHUNK as u64).await?;
                    }
                }
                self.cooperation
                    .completed((value.len() % ELEMENT_CHUNK) as u64)
                    .await?;
                (value.shape, values)
            }
            Value::LogicalArray(value) => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(value.data.len())
                    .map_err(resource_error)?;
                for (index, item) in value.data.into_iter().enumerate() {
                    values.push(if item != 0 { 1.0 } else { 0.0 });
                    if (index + 1).is_multiple_of(ELEMENT_CHUNK) {
                        self.cooperation.completed(ELEMENT_CHUNK as u64).await?;
                    }
                }
                self.cooperation
                    .completed((values.len() % ELEMENT_CHUNK) as u64)
                    .await?;
                (value.shape, values)
            }
            other => {
                return Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_INPUT,
                    format!(
                        "array-valued callback must return a real numeric array, got {other:?}"
                    ),
                ))
            }
        };
        if let Some(expected) = &self.shape {
            if expected != &shape {
                return Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_INPUT,
                    "array-valued callback changed its output size",
                ));
            }
        } else {
            self.shape = Some(shape);
        }
        Ok(Arc::new(values))
    }
}

fn nonfinite_error() -> RuntimeError {
    integral_error_with_detail(
        &INTEGRAL_ERROR_INVALID_INPUT,
        "array-valued callback must return finite real values",
    )
}

fn admit_callback_value(value: &Value, expected: Option<&[usize]>) -> BuiltinResult<()> {
    use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
    let shape: &[usize] = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1],
        Value::Tensor(tensor) => &tensor.shape,
        Value::LogicalArray(logical) => &logical.shape,
        Value::GpuTensor(handle) => {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| {
                    build_runtime_error(
                        "integral: array-valued callback returned an invalid resident lease",
                    )
                    .with_builtin(NAME)
                    .with_source(source)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
                })?;
            if handle_storage(handle) != GpuTensorStorage::Real {
                return Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_INPUT,
                    "array-valued callback must return a real numeric array",
                ));
            }
            handle.shape()
        }
        _ => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                "array-valued callback must return a real numeric array",
            ));
        }
    };
    if expected.is_some_and(|expected| expected != shape) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "array-valued callback changed its output size",
        ));
    }
    Ok(())
}

impl QuadratureEvaluator for ArrayEvaluator<'_> {
    type Sample = Arc<Vec<f64>>;

    async fn checkpoint(&mut self) -> BuiltinResult<()> {
        self.cooperation.completed(0).await
    }

    async fn charge(&mut self, units: u64) -> BuiltinResult<()> {
        self.cooperation.completed(units).await
    }

    async fn evaluate(&mut self, x: f64) -> BuiltinResult<Self::Sample> {
        self.sample(x).await
    }

    fn midpoint(&self, a: f64, b: f64) -> f64 {
        crate::builtins::math::optim::quadrature::midpoint(a, b)
    }

    fn tolerances(&self) -> (f64, f64) {
        (self.options.abs_tol, self.options.rel_tol)
    }

    fn max_evaluations(&self) -> usize {
        self.options.max_fun_evals
    }

    fn max_depth(&self) -> usize {
        MAX_DEPTH
    }

    fn evaluation_limit_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "exceeded maximum function evaluations",
        )
    }

    fn convergence_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "adaptive quadrature did not converge",
        )
    }

    fn nonrepresentable_error(&self) -> RuntimeError {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "quadrature estimate is not representable as a finite double",
        )
    }

    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
        quadrature_arithmetic_error(error)
    }
}

fn resource_error(source: std::collections::TryReserveError) -> RuntimeError {
    build_runtime_error("integral: could not reserve array-valued callback workspace")
        .with_identifier("RunMat:Internal")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(super) async fn integrate_array_valued(
    function: &Value,
    lo: f64,
    hi: f64,
    sign: f64,
    options: &IntegralOptions,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    let mut evaluator = ArrayEvaluator {
        function,
        options,
        cooperation,
        shape: None,
    };
    let estimate = if lo == hi {
        let sample = evaluator.sample(lo).await?;
        let mut zero = Vec::new();
        zero.try_reserve_exact(sample.len())
            .map_err(resource_error)?;
        zero.resize(sample.len(), 0.0);
        return finish(evaluator.shape.expect("sample establishes shape"), zero);
    } else {
        integrate_array(&mut evaluator, lo, hi)
            .await
            .map_err(|error| integral_map_error(error, &INTEGRAL_ERROR_INVALID_INPUT))?
    };
    let shape = evaluator
        .shape
        .expect("integration evaluates at least one sample");
    let mut values = Vec::new();
    values
        .try_reserve_exact(estimate.0.len())
        .map_err(resource_error)?;
    for &value in estimate.0.iter() {
        let signed = value.multiply_f64(sign);
        values.push(
            signed
                .to_f64()
                .ok_or_else(|| evaluator.nonrepresentable_error())?,
        );
    }
    finish(shape, values)
}

fn finish(shape: Vec<usize>, values: Vec<f64>) -> BuiltinResult<Value> {
    if values.len() == 1 {
        Ok(Value::Num(values[0]))
    } else {
        Tensor::new(values, shape)
            .map(Value::Tensor)
            .map_err(|error| {
                integral_error_with_detail(&INTEGRAL_ERROR_INVALID_INPUT, error.to_string())
            })
    }
}
