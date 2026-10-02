//! Ordered finite complex-contour integration through one open-rule traversal.

use std::sync::Arc;

use num_complex::Complex;
use runmat_builtins::{INTEGRAL_ERROR_INVALID_INPUT, IntegralCallbackMode};
use runmat_value::{ComplexStorage, ComplexTensor, Tensor, Value};

use super::{
    IntegralOptions, MAX_DEPTH, NAME, integral_error_with_detail, integral_map_error,
    quadrature_arithmetic_error, waypoints,
};
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::optim::common::call_function;
use crate::builtins::math::optim::quadrature::{
    OpenDomain, OpenSegment, QuadratureArithmeticError, QuadratureCooperation, QuadratureEvaluator,
    ScaledComplex, integrate_open,
};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError, build_runtime_error};

mod admission;
mod publication;

pub(super) use admission::{ordered_segments, scalar_bound, scalar_bound_host};

const COMPONENT_CHUNK: usize = 4096;

pub(super) fn has_complex_class(value: &Value) -> bool {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Complex
        }
        _ => false,
    }
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    integral_error_with_detail(&INTEGRAL_ERROR_INVALID_INPUT, detail)
}

fn resource_error(source: std::collections::TryReserveError) -> RuntimeError {
    build_runtime_error("integral: could not reserve complex contour workspace")
        .with_identifier("RunMat:Internal")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

async fn materialize(value: &Value) -> BuiltinResult<std::borrow::Cow<'_, Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(std::borrow::Cow::Borrowed(value));
    };
    let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        build_runtime_error("integral: resident value has no owning provider")
            .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
            .with_builtin(NAME)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build()
    })?;
    let downloaded = gpu_helpers::download_value_preserving_residency_async(owner, handle)
        .await
        .map_err(|source| integral_map_error(source, &INTEGRAL_ERROR_INVALID_INPUT))?;
    Ok(std::borrow::Cow::Owned(downloaded))
}

fn callback_shape(value: &Value) -> BuiltinResult<&[usize]> {
    use runmat_accelerate_api::{GpuTensorStorage, handle_storage};
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(&[1, 1]),
        Value::Tensor(array) => Ok(&array.shape),
        Value::ComplexTensor(array) => Ok(&array.shape),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::GpuTensor(handle) => {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| {
                    build_runtime_error("integral: callback returned an invalid resident lease")
                        .with_builtin(NAME)
                        .with_source(source)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                })?;
            if !matches!(
                handle_storage(handle),
                GpuTensorStorage::Real | GpuTensorStorage::Complex
            ) {
                return Err(invalid(
                    "callback must return real or complex numeric values",
                ));
            }
            Ok(handle.shape())
        }
        _ => Err(invalid(
            "callback must return real or complex numeric values",
        )),
    }
}

async fn decode_callback(
    value: &Value,
    expected: &[usize],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<ScaledComplex>> {
    let shape = callback_shape(value)?;
    if shape != expected {
        return Err(invalid("complex callback changed its output size"));
    }
    let count = expected
        .iter()
        .try_fold(1_usize, |product, &axis| product.checked_mul(axis))
        .ok_or_else(|| invalid("callback output size overflows"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(resource_error)?;
    for index in 0..count {
        let item = match value {
            Value::Num(real) => ScaledComplex::new(*real, 0.0),
            Value::Int(integer) => ScaledComplex::new(integer.to_f64(), 0.0),
            Value::Bool(flag) => ScaledComplex::new(f64::from(*flag as u8), 0.0),
            Value::Complex(real, imaginary) => ScaledComplex::new(*real, *imaginary),
            Value::Tensor(array) => {
                let element = array
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid real callback storage"))?;
                ScaledComplex::new(waypoints::value_f64(element), 0.0)
            }
            Value::ComplexTensor(array) => {
                let (real, imaginary) = array
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid complex callback storage"))?;
                ScaledComplex::new(waypoints::value_f64(real), waypoints::value_f64(imaginary))
            }
            Value::LogicalArray(array) => {
                let bit = array
                    .data
                    .get(index)
                    .ok_or_else(|| invalid("invalid logical callback storage"))?;
                ScaledComplex::new(f64::from((*bit != 0) as u8), 0.0)
            }
            Value::GpuTensor(_) => return Err(invalid("resident callback was not materialized")),
            _ => {
                return Err(invalid(
                    "callback must return real or complex numeric values",
                ));
            }
        };
        if item.to_pair().is_none() {
            return Err(invalid("complex callback must return finite values"));
        }
        values.push(item);
        if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
            cooperation.completed(COMPONENT_CHUNK as u64).await?;
        }
    }
    cooperation
        .completed((count % COMPONENT_CHUNK) as u64)
        .await?;
    Ok(values)
}

struct ContourEvaluator<'a> {
    function: &'a Value,
    options: &'a IntegralOptions,
    cooperation: &'a mut QuadratureCooperation,
    shape: Option<Vec<usize>>,
    complex_output: bool,
}

impl ContourEvaluator<'_> {
    async fn sample(
        &mut self,
        point: Value,
        delta: ScaledComplex,
    ) -> BuiltinResult<Arc<Vec<ScaledComplex>>> {
        self.cooperation.completed(0).await?;
        let value = call_function(self.function, [point]).await?;
        self.cooperation.completed(1).await?;
        let shape = callback_shape(&value)?;
        if !self.options.array_valued && shape != [1, 1] {
            return Err(invalid(
                "non-array-valued callback must return one value per sample",
            ));
        }
        if self
            .shape
            .as_deref()
            .is_some_and(|expected| expected != shape)
        {
            return Err(invalid("complex callback changed its output size"));
        }
        let owned_shape = shape.to_vec();
        self.complex_output |= has_complex_class(&value);
        let host = materialize(&value).await?;
        let mut values = decode_callback(host.as_ref(), &owned_shape, self.cooperation).await?;
        for (index, value) in values.iter_mut().enumerate() {
            *value = value.multiply_complex(delta);
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                self.cooperation.completed(COMPONENT_CHUNK as u64).await?;
            }
        }
        self.cooperation
            .completed((values.len() % COMPONENT_CHUNK) as u64)
            .await?;
        if self.shape.is_none() {
            self.shape = Some(owned_shape);
        }
        Ok(Arc::new(values))
    }

    async fn batch<const N: usize>(
        &mut self,
        start: Complex<f64>,
        end: Complex<f64>,
        points: [f64; N],
        delta: ScaledComplex,
        real_axis: bool,
    ) -> BuiltinResult<[Arc<Vec<ScaledComplex>>; N]> {
        let mut arguments = Vec::new();
        arguments.try_reserve_exact(N).map_err(resource_error)?;
        for &point in &points {
            let real = start.re * (1.0 - point) + end.re * point;
            let imaginary = start.im * (1.0 - point) + end.im * point;
            arguments.push((real, imaginary));
        }
        let input = if real_axis {
            Value::Tensor(
                Tensor::new(
                    arguments.iter().map(|(real, _)| *real).collect(),
                    vec![1, N],
                )
                .map_err(|cause| invalid(format!("callback batch geometry: {cause}")))?,
            )
        } else {
            Value::ComplexTensor(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(arguments.into()),
                    vec![1, N],
                )
                .map_err(|cause| invalid(format!("complex callback batch geometry: {cause}")))?,
            )
        };
        self.cooperation.completed(0).await?;
        let value = call_function(self.function, [input]).await?;
        self.cooperation.completed(N as u64).await?;
        if callback_shape(&value)? != [1, N] {
            return Err(invalid(
                "vectorized callback must return one value per sample",
            ));
        }
        self.complex_output |= has_complex_class(&value);
        let host = materialize(&value).await?;
        let values = decode_callback(host.as_ref(), &[1, N], self.cooperation).await?;
        Ok(std::array::from_fn(|index| {
            Arc::new(vec![values[index].multiply_complex(delta)])
        }))
    }
}

impl QuadratureEvaluator for ContourEvaluator<'_> {
    type Sample = Arc<Vec<ScaledComplex>>;

    async fn checkpoint(&mut self) -> BuiltinResult<()> {
        self.cooperation.completed(0).await
    }

    async fn charge(&mut self, units: u64) -> BuiltinResult<()> {
        self.cooperation.completed(units).await
    }

    async fn evaluate(&mut self, _x: f64) -> BuiltinResult<Self::Sample> {
        Err(invalid("contour evaluation requires segment geometry"))
    }

    async fn evaluate_open_many<const N: usize>(
        &mut self,
        domain: OpenDomain,
        points: [f64; N],
    ) -> BuiltinResult<[Self::Sample; N]> {
        let (start, end, delta, real_axis) = match domain {
            OpenDomain::Contour { start, end } => (
                start,
                end,
                ScaledComplex::new(end.re, end.im).subtract(ScaledComplex::new(start.re, start.im)),
                false,
            ),
            OpenDomain::Finite
            | OpenDomain::RightInfinite { .. }
            | OpenDomain::LeftInfinite { .. } => (
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                ScaledComplex::new(1.0, 0.0),
                true,
            ),
        };
        if !self.options.array_valued
            && self.options.callback_mode == IntegralCallbackMode::Vectorized
        {
            return self.batch(start, end, points, delta, real_axis).await;
        }
        let mut values: [Option<Self::Sample>; N] = std::array::from_fn(|_| None);
        for (index, point) in points.into_iter().enumerate() {
            let coordinate = Complex::new(
                start.re * (1.0 - point) + end.re * point,
                start.im * (1.0 - point) + end.im * point,
            );
            let argument = if real_axis {
                Value::Num(point)
            } else {
                Value::Complex(coordinate.re, coordinate.im)
            };
            values[index] = Some(self.sample(argument, delta).await?);
        }
        Ok(values.map(|value| value.expect("all contour nodes evaluated")))
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

    fn max_intervals(&self) -> Option<usize> {
        self.options.max_interval_count.map(|limit| limit.get())
    }

    fn interval_limit_error(&self) -> RuntimeError {
        invalid("exceeded MaxIntervalCount adaptive subintervals")
    }

    fn max_depth(&self) -> usize {
        MAX_DEPTH
    }

    fn evaluation_limit_error(&self) -> RuntimeError {
        invalid("exceeded maximum function evaluations")
    }

    fn convergence_error(&self) -> RuntimeError {
        invalid("adaptive complex contour quadrature did not converge")
    }

    fn nonrepresentable_error(&self) -> RuntimeError {
        invalid("complex quadrature result is not representable as finite double")
    }

    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
        quadrature_arithmetic_error(error)
    }
}

pub(super) async fn integrate_contour(
    function: &Value,
    options: &IntegralOptions,
    segments: &[OpenSegment],
    sign: f64,
    real_axis: bool,
    empty_point: Complex<f64>,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    let mut evaluator = ContourEvaluator {
        function,
        options,
        cooperation,
        shape: None,
        complex_output: false,
    };
    if segments.is_empty() {
        if !options.array_valued {
            return Ok(if real_axis {
                Value::Num(0.0)
            } else {
                Value::Complex(0.0, 0.0)
            });
        }
        let argument = if real_axis {
            Value::Num(empty_point.re)
        } else {
            Value::Complex(empty_point.re, empty_point.im)
        };
        let sample = evaluator
            .sample(argument, ScaledComplex::new(0.0, 0.0))
            .await?;
        let shape = evaluator
            .shape
            .take()
            .ok_or_else(|| invalid("missing zero-contour callback shape"))?;
        return publication::zero(
            sample.len(),
            shape,
            !real_axis || evaluator.complex_output,
            evaluator.cooperation,
        )
        .await;
    }
    let result = integrate_open(&mut evaluator, segments).await?;
    let shape = evaluator.shape.unwrap_or_else(|| vec![1, 1]);
    publication::estimate(
        &result.0,
        shape,
        sign,
        options.array_valued,
        !real_axis || evaluator.complex_output,
        evaluator.cooperation,
    )
    .await
}
