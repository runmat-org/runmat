//! Data normalization helpers.

use runmat_value::NumericDType;

use runmat_builtins::{
    BuiltinErrorDescriptor, NORMALIZE_ERROR_INTERNAL, NORMALIZE_ERROR_INVALID_ARGUMENT,
    NORMALIZE_INTEGER_DATA_EXTENSION, NORMALIZE_LOGICAL_DATA_EXTENSION,
    NORMALIZE_RESIDENT_CONTROL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod arguments;
mod evaluation;
use arguments::{parse_args, value_to_input};
use evaluation::evaluate;

#[derive(Clone)]
enum NumericInput {
    Real {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    Complex {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
}

#[derive(Clone)]
enum ParamReal {
    Computed { data: Vec<f64>, shape: Vec<usize> },
    Explicit(Tensor),
}

#[derive(Clone)]
enum ParamComplex {
    Computed {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
    ExplicitReal(Tensor),
    ExplicitComplex(ComplexTensor),
}

#[derive(Clone)]
enum RealPlan {
    CenterScale {
        center: CenterSpec,
        scale: ScaleSpec,
    },
    Range {
        bounds: RangeBounds,
    },
}

#[derive(Clone)]
enum ComplexPlan {
    CenterScale {
        center: ComplexCenterSpec,
        scale: ScaleSpec,
    },
}

#[derive(Clone)]
enum CenterSpec {
    None,
    Mean,
    Median,
    Explicit(Tensor),
}

#[derive(Clone)]
enum ComplexCenterSpec {
    None,
    Mean,
    Median,
    ExplicitReal(Tensor),
    ExplicitComplex(ComplexTensor),
}

#[derive(Clone)]
enum ScaleSpec {
    One,
    Std,
    Mad,
    Iqr,
    First,
    Norm(f64),
    Explicit(Tensor),
}

#[derive(Clone, Copy)]
struct RangeBounds {
    lower: f64,
    upper: f64,
}

struct ParsedArgs {
    dim: Option<usize>,
    real_plan: RealPlan,
    complex_plan: ComplexPlan,
}

struct NormalizeEval {
    n: Value,
    c: Value,
    s: Value,
}

struct NormalizeOutputPlan {
    single: bool,
    gpu_source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl NormalizeOutputPlan {
    fn inspect(value: &Value, _rest: &[Value]) -> BuiltinResult<Self> {
        let single = matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            || matches!(value, Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle)
                    && runmat_accelerate_api::handle_precision(handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F32));
        let gpu_source = gpu_helpers::select_resident_output_source(
            std::iter::once(value).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "normalize",
        )?;
        Ok(Self { single, gpu_source })
    }

    fn finish(self, eval: NormalizeEval) -> BuiltinResult<NormalizeEval> {
        Ok(NormalizeEval {
            n: self.finish_value(eval.n)?,
            c: self.finish_value(eval.c)?,
            s: self.finish_value(eval.s)?,
        })
    }

    fn finish_value(&self, value: Value) -> BuiltinResult<Value> {
        let value = if self.single {
            value_to_single(value)?
        } else {
            value
        };
        match &self.gpu_source {
            Some(source) => {
                let restored =
                    gpu_helpers::restore_class_preserving_value(source, value, "normalize")?;
                if runmat_accelerate_api::handle_is_explicit(source)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(normalize_internal(
                        "normalize: provider cannot preserve explicit gpuArray output residency",
                    ));
                }
                Ok(restored)
            }
            None => Ok(value),
        }
    }
}

fn value_to_single(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Tensor::from_f32(vec![value as f32], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| normalize_internal(format!("normalize: {err}"))),
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            Ok(Value::Tensor(tensor))
        }
        Value::Tensor(tensor) => Tensor::from_f32(
            tensor
                .materialize_f64()
                .iter()
                .map(|value| *value as f32)
                .collect(),
            tensor.shape.clone(),
        )
        .map(Value::Tensor)
        .map_err(|err| normalize_internal(format!("normalize: {err}"))),
        Value::Complex(re, im) => ComplexTensor::from_f32(vec![(re as f32, im as f32)], vec![1, 1])
            .map(Value::ComplexTensor)
            .map_err(|err| normalize_internal(format!("normalize: {err}"))),
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            Ok(Value::ComplexTensor(tensor))
        }
        Value::ComplexTensor(tensor) => ComplexTensor::from_f32(
            tensor
                .materialize_f64()
                .iter()
                .map(|(re, im)| (*re as f32, *im as f32))
                .collect(),
            tensor.shape.clone(),
        )
        .map(Value::ComplexTensor)
        .map_err(|err| normalize_internal(format!("normalize: {err}"))),
        other => Ok(other),
    }
}

const MAX_NORMALIZE_DIM: usize = 64;

#[runtime_builtin(
    name = "normalize",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::normalize"
)]
async fn normalize_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_compatibility(&value, &rest)?;
    let output = NormalizeOutputPlan::inspect(&value, &rest)?;
    let input = value_to_input(value).await?;
    let parsed = parse_args(&input, rest).await?;
    let eval = output.finish(evaluate(input, parsed)?)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.n]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.n, eval.c, eval.s],
        ));
    }
    Ok(eval.n)
}

fn ensure_compatibility(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NORMALIZE_INTEGER_DATA_EXTENSION,
            "normalize",
        )?;
    }
    if is_logical(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NORMALIZE_LOGICAL_DATA_EXTENSION,
            "normalize",
        )?;
    }
    if rest.iter().any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NORMALIZE_RESIDENT_CONTROL_EXTENSION,
            "normalize",
        )?;
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn normalize_error(message: impl Into<String>) -> RuntimeError {
    normalize_descriptor_error(message, &NORMALIZE_ERROR_INVALID_ARGUMENT)
}

fn normalize_internal(message: impl Into<String>) -> RuntimeError {
    normalize_descriptor_error(message, &NORMALIZE_ERROR_INTERNAL)
}

fn normalize_descriptor_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("normalize");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn real_param_values(param: &ParamReal) -> Vec<f64> {
    match param {
        ParamReal::Computed { data, .. } => data.clone(),
        ParamReal::Explicit(tensor) => tensor::tensor_values_f64(tensor),
    }
}

fn real_param_shape(param: &ParamReal, len: usize) -> Vec<usize> {
    match param {
        ParamReal::Computed { shape, .. } => shape.clone(),
        ParamReal::Explicit(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::tensor_element_len(tensor))
        }
    }
    .shape_fallback(len)
}

fn complex_param_values(param: &ParamComplex) -> Vec<(f64, f64)> {
    match param {
        ParamComplex::Computed { data, .. } => data.clone(),
        ParamComplex::ExplicitReal(tensor) => tensor::tensor_values_f64(tensor)
            .into_iter()
            .map(|value| (value, 0.0))
            .collect(),
        ParamComplex::ExplicitComplex(tensor) => tensor::complex_tensor_values_complex64(tensor)
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
    }
}

fn complex_param_shape(param: &ParamComplex, len: usize) -> Vec<usize> {
    match param {
        ParamComplex::Computed { shape, .. } => shape.clone(),
        ParamComplex::ExplicitReal(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::tensor_element_len(tensor))
        }
        ParamComplex::ExplicitComplex(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::complex_tensor_element_len(tensor))
        }
    }
    .shape_fallback(len)
}

trait ShapeFallback {
    fn shape_fallback(self, len: usize) -> Vec<usize>;
}

impl ShapeFallback for Vec<usize> {
    fn shape_fallback(self, len: usize) -> Vec<usize> {
        if self.is_empty() {
            vec![len, 1]
        } else {
            self
        }
    }
}

fn real_param_value(param: ParamReal) -> BuiltinResult<Value> {
    let data = real_param_values(&param);
    let shape = real_param_shape(&param, data.len());
    tensor_value(data, shape)
}

fn complex_param_value(param: ParamComplex) -> BuiltinResult<Value> {
    let data = complex_param_values(&param);
    let shape = complex_param_shape(&param, data.len());
    complex_value(data, shape)
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| normalize_internal(format!("normalize: {err}")))
}

fn complex_value(data: Vec<(f64, f64)>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 && normalize_shape_for(&shape, data.len()) == vec![1, 1] {
        Ok(Value::Complex(data[0].0, data[0].1))
    } else {
        ComplexTensor::new(data, shape)
            .map(Value::ComplexTensor)
            .map_err(|err| normalize_internal(format!("normalize: {err}")))
    }
}

fn normalize_shape_for(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        tensor::default_shape_for(shape, len)
    } else {
        shape.to_vec()
    }
}

fn real_part_tensor(tensor: &ComplexTensor) -> BuiltinResult<Tensor> {
    Tensor::new(
        tensor::complex_tensor_values_complex64(tensor)
            .into_iter()
            .map(|value| value.re)
            .collect(),
        tensor.shape.clone(),
    )
    .map_err(|err| normalize_internal(format!("normalize: {err}")))
}

#[cfg(test)]
mod tests;
