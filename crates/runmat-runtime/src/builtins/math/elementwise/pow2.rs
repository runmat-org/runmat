//! MATLAB-compatible `pow2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    broadcast::BroadcastPlan, gpu_helpers, map_control_flow_with_builtin, tensor,
};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const LN_2: f64 = std::f64::consts::LN_2;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pow2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_pow2" },
        ProviderHook::Binary {
            name: "pow2_scale",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_pow2 and pow2_scale to keep tensors on-device; the runtime gathers to host when hooks are unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pow2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input} * {:.17})", LN_2))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits `exp(x * ln2)` for unary pow2; binary scaling currently falls back to the host when implicit expansion is required.",
};

const BUILTIN_NAME: &str = "pow2";

const POW2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Binary power-of-two result.",
}];

const POW2_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exponent input for 2.^X.",
}];

const POW2_INPUTS_F_E: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mantissa input.",
    },
    BuiltinParamDescriptor {
        name: "E",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Binary exponent input.",
    },
];

const POW2_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pow2(X)",
        inputs: &POW2_INPUTS_X,
        outputs: &POW2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pow2(F, E)",
        inputs: &POW2_INPUTS_F_E,
        outputs: &POW2_OUTPUT,
    },
];

const POW2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INVALID_ARGUMENT",
    identifier: Some("RunMat:pow2:InvalidArgument"),
    when: "Argument arity is invalid.",
    message: "pow2: invalid argument",
};

const POW2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INVALID_INPUT",
    identifier: Some("RunMat:pow2:InvalidInput"),
    when: "Input value cannot be converted to supported numeric form.",
    message: "pow2: invalid input",
};

const POW2_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.SIZE_MISMATCH",
    identifier: Some("RunMat:pow2:SizeMismatch"),
    when: "Binary operands are not broadcast-compatible.",
    message: "pow2: size mismatch",
};

const POW2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INTERNAL",
    identifier: Some("RunMat:pow2:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "pow2: internal error",
};

const POW2_ERRORS: [BuiltinErrorDescriptor; 4] = [
    POW2_ERROR_INVALID_ARGUMENT,
    POW2_ERROR_INVALID_INPUT,
    POW2_ERROR_SIZE_MISMATCH,
    POW2_ERROR_INTERNAL,
];

pub const POW2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POW2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POW2_ERRORS,
};

fn pow2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "pow2",
    category = "math/elementwise",
    summary = "Compute powers of two or scale mantissas by exponents.",
    keywords = "pow2,ldexp,binary scaling,gpu",
    accel = "unary",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::elementwise::pow2::POW2_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::pow2"
)]
async fn pow2_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => pow2_unary(first).await,
        1 => pow2_binary(first, rest.into_iter().next().unwrap()).await,
        _ => Err(pow2_error_with_detail(
            &POW2_ERROR_INVALID_ARGUMENT,
            "expected at most two arguments",
        )),
    }
}

async fn pow2_unary(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => pow2_gpu(handle).await,
        Value::Complex(re, im) => {
            let (rr, ii) = pow2_complex(re, im);
            Ok(Value::Complex(rr, ii))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(
                &ct,
                BUILTIN_NAME,
            )?;
            pow2_complex_tensor(ct)
        }
        Value::CharArray(ca) => pow2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(pow2_error_with_detail(
            &POW2_ERROR_INVALID_INPUT,
            "expected numeric input, got string",
        )),
        other => pow2_real(other),
    }
}

async fn pow2_binary(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&mantissa, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&exponent, BUILTIN_NAME)?;
    match (mantissa, exponent) {
        (Value::GpuTensor(mh), Value::GpuTensor(eh)) => pow2_gpu_scale(mh, eh).await,
        (Value::GpuTensor(mh), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&mh)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(eh)) => {
            let gathered = gpu_helpers::gather_tensor_async(&eh)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(other, Value::Tensor(gathered))
        }
        (m, e) => pow2_host_scale(m, e),
    }
}

async fn pow2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let provider = runmat_accelerate_api::provider_for_handle(&handle);
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let output = pow2_tensor(tensor)?;
        if let Some(provider) = provider {
            if let Ok(handle) = gpu_helpers::upload_tensor(provider, &output) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        return Ok(tensor::tensor_into_value(output));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_pow2(&handle).await {
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

async fn pow2_gpu_scale(
    mantissa: GpuTensorHandle,
    exponent: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let has_integer_input = runmat_accelerate_api::handle_integer_type(&mantissa).is_some()
        || runmat_accelerate_api::handle_integer_type(&exponent).is_some();
    if !has_integer_input && mantissa.device_id == exponent.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&mantissa) {
            if mantissa.shape == exponent.shape {
                if let Ok(out) = provider.pow2_scale(&mantissa, &exponent) {
                    return Ok(gpu_helpers::resident_gpu_value(out));
                }
            }
        }
    }
    let m = gpu_helpers::gather_tensor_async(&mantissa)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let e = gpu_helpers::gather_tensor_async(&exponent)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    pow2_host_scale(Value::Tensor(m), Value::Tensor(e))
}

fn pow2_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("pow2", value)
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

fn pow2_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
    let storage = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::exp2).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::exp2).collect())
        }
        storage => NumericStorage::F64(
            promote_integer_storage_to_pow2_double_domain(storage)
                .into_iter()
                .map(f64::exp2)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))
}

fn pow2_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| pow2_complex(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| pow2_complex_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(pow2_error_with_detail(
                &POW2_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_pow2_double_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("pow2 integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn pow2_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp2())
        .collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn pow2_host_scale(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    if let Some(result) = scalar_pow2_value(&mantissa, &exponent) {
        return Ok(result);
    }
    let mantissa_array = value_into_numeric_array(mantissa, "pow2")?;
    let exponent_array = value_into_numeric_array(exponent, "pow2")?;
    let plan = BroadcastPlan::new(mantissa_array.shape(), exponent_array.shape())
        .map_err(|e| pow2_error_with_detail(&POW2_ERROR_SIZE_MISMATCH, e))?;
    let output_shape = plan.output_shape().to_vec();
    let output_is_complex = mantissa_array.is_complex() || exponent_array.is_complex();
    let use_single = mantissa_array.uses_single() || exponent_array.uses_single();
    if use_single {
        let mantissa = pow2_array_into_f32_components(mantissa_array)?;
        let exponent = pow2_array_into_f32_components(exponent_array)?;
        let output = plan
            .iter()
            .map(|(_, idx_m, idx_e)| {
                let (mr, mi) = mantissa[idx_m];
                let (er, ei) = exponent[idx_e];
                let power = pow2_complex_f32(er, ei);
                complex_mul_f32(mr, mi, power.0, power.1)
            })
            .collect::<Vec<_>>();
        if output_is_complex {
            let tensor =
                ComplexTensor::from_complex_storage(ComplexStorage::F32(output), output_shape)
                    .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(tensor))
        } else {
            let values = output.into_iter().map(|(real, _)| real).collect();
            let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), output_shape)
                .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            Ok(tensor::tensor_into_value(tensor))
        }
    } else {
        let mantissa = pow2_array_into_f64_components(mantissa_array)?;
        let exponent = pow2_array_into_f64_components(exponent_array)?;
        let output = plan
            .iter()
            .map(|(_, idx_m, idx_e)| {
                let (mr, mi) = mantissa[idx_m];
                let (er, ei) = exponent[idx_e];
                let power = pow2_complex(er, ei);
                complex_mul(mr, mi, power.0, power.1)
            })
            .collect::<Vec<_>>();
        if output_is_complex {
            let tensor =
                ComplexTensor::from_complex_storage(ComplexStorage::F64(output), output_shape)
                    .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(tensor))
        } else {
            let values = output.into_iter().map(|(real, _)| real).collect();
            let tensor = Tensor::from_numeric_storage(NumericStorage::F64(values), output_shape)
                .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            Ok(tensor::tensor_into_value(tensor))
        }
    }
}

fn pow2_array_into_f32_components(array: NumericArray) -> BuiltinResult<Vec<(f32, f32)>> {
    match array {
        NumericArray::Real(tensor) => {
            let storage = tensor
                .into_numeric_storage()
                .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            Ok(storage
                .materialize_f32()
                .into_iter()
                .map(|value| (value, 0.0))
                .collect())
        }
        NumericArray::Complex(tensor) => match tensor.into_complex_storage() {
            ComplexStorage::F32(values) => Ok(values),
            ComplexStorage::F64(values) => Ok(values
                .into_iter()
                .map(|(real, imag)| (real as f32, imag as f32))
                .collect()),
            ComplexStorage::Integer(_) => Err(pow2_error_with_detail(
                &POW2_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            )),
        },
    }
}

fn pow2_array_into_f64_components(array: NumericArray) -> BuiltinResult<Vec<(f64, f64)>> {
    match array {
        NumericArray::Real(tensor) => {
            let storage = tensor
                .into_numeric_storage()
                .map_err(|e| pow2_error_with_detail(&POW2_ERROR_INTERNAL, e))?;
            let values = match storage {
                NumericStorage::F64(values) => values,
                NumericStorage::F32(_) => {
                    unreachable!("single storage selects the native-single pow2 domain")
                }
                storage => promote_integer_storage_to_pow2_double_domain(storage),
            };
            Ok(values.into_iter().map(|value| (value, 0.0)).collect())
        }
        NumericArray::Complex(tensor) => match tensor.into_complex_storage() {
            ComplexStorage::F64(values) => Ok(values),
            ComplexStorage::F32(_) => {
                unreachable!("complex single selects the native-single pow2 domain")
            }
            ComplexStorage::Integer(_) => Err(pow2_error_with_detail(
                &POW2_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            )),
        },
    }
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        _ => None,
    }
}

fn scalar_pow2_value(mantissa: &Value, exponent: &Value) -> Option<Value> {
    let base =
        scalar_complex_value(mantissa).or_else(|| scalar_real_value(mantissa).map(|v| (v, 0.0)))?;
    let exp =
        scalar_complex_value(exponent).or_else(|| scalar_real_value(exponent).map(|v| (v, 0.0)))?;
    let (mr, mi) = base;
    let (er, ei) = exp;
    if mi != 0.0 || ei != 0.0 {
        let (re_pow, im_pow) = pow2_complex(er, ei);
        let (re, im) = complex_mul(mr, mi, re_pow, im_pow);
        return Some(Value::Complex(re, im));
    }
    let scale = er.exp2();
    Some(Value::Num(mr * scale))
}

fn pow2_complex(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        return (re.exp2(), 0.0);
    }
    let scale = (re * LN_2).exp();
    let angle = im * LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

fn pow2_complex_f32(re: f32, im: f32) -> (f32, f32) {
    if im == 0.0 {
        return (re.exp2(), 0.0);
    }
    let scale = (re * std::f32::consts::LN_2).exp();
    let angle = im * std::f32::consts::LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_mul_f32(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn value_into_numeric_array(value: Value, name: &str) -> BuiltinResult<NumericArray> {
    match value {
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| {
                pow2_error_with_detail(&POW2_ERROR_INTERNAL, format!("{name}: {e}"))
            })?;
            Ok(NumericArray::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(NumericArray::Complex(ct)),
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| {
                pow2_error_with_detail(&POW2_ERROR_INTERNAL, format!("{name}: {e}"))
            })?;
            Ok(NumericArray::Real(tensor))
        }
        Value::String(_) | Value::StringArray(_) => Err(pow2_error_with_detail(
            &POW2_ERROR_INVALID_INPUT,
            format!("{name}: expected numeric input, got string"),
        )),
        Value::GpuTensor(_) => Err(pow2_error_with_detail(
            &POW2_ERROR_INTERNAL,
            format!("{name}: internal error converting GPU tensor"),
        )),
        other => {
            let tensor = tensor::value_into_tensor_for(name, other).map_err(|e| {
                pow2_error_with_detail(&POW2_ERROR_INVALID_INPUT, format!("{name}: {e}"))
            })?;
            Ok(NumericArray::Real(tensor))
        }
    }
}

enum NumericArray {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericArray {
    fn shape(&self) -> &[usize] {
        match self {
            NumericArray::Real(t) => &t.shape,
            NumericArray::Complex(t) => &t.shape,
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, NumericArray::Complex(_))
    }

    fn uses_single(&self) -> bool {
        match self {
            NumericArray::Real(tensor) => {
                tensor.numeric_dtype() == runmat_builtins::NumericDType::F32
            }
            NumericArray::Complex(tensor) => {
                tensor.numeric_dtype() == runmat_builtins::NumericDType::F32
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        IntValue, IntegerComplexStorage, IntegerStorage, ResolveContext, Tensor, Type,
    };

    fn pow2_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pow2_builtin(first, rest))
    }

    #[test]
    fn pow2_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = POW2_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = pow2(X)"));
        assert!(labels.contains(&"Y = pow2(F, E)"));
    }

    #[test]
    fn pow2_string_input_has_stable_identifier() {
        let err = pow2_builtin(Value::from("bad"), vec![]).expect_err("expected error");
        assert_eq!(err.identifier(), POW2_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn pow2_rejects_typed_complex_integer_inputs() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");

        let unary = pow2_builtin(Value::ComplexTensor(complex.clone()), vec![])
            .expect_err("typed complex integer input must reject");
        assert!(unary
            .message()
            .contains("complex numbers with integer types"));

        let binary = pow2_builtin(Value::Num(1.0), vec![Value::ComplexTensor(complex)])
            .expect_err("typed complex integer exponent must reject");
        assert!(binary
            .message()
            .contains("complex numbers with integer types"));
    }

    #[test]
    fn pow2_unary_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![1, 2]).expect("tensor");

        let result = pow2_builtin(Value::Tensor(tensor), vec![]).expect("pow2");

        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![2.0, 8.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn pow2_binary_scalar_reads_typed_integer_storage_exactly() {
        let mantissa =
            Tensor::new_integer(IntegerStorage::I32(vec![5]), vec![1, 1]).expect("mantissa");
        let exponent =
            Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("exponent");

        let result = pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)])
            .expect("pow2 scale");

        assert_eq!(result, Value::Num(40.0));
    }

    #[test]
    fn pow2_binary_arrays_ignore_poisoned_integer_mirrors() {
        let mantissa = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 3]),
            vec![1, 2],
        )
        .expect("mantissa");
        let exponent =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 4]), vec![1, 2]).expect("exponent");

        let result = pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)])
            .expect("pow2 scale");

        let Value::Tensor(result) = result else {
            panic!("expected double tensor result");
        };
        assert_eq!(
            result.materialize_f64(),
            vec![18_014_398_509_481_986.0, 48.0]
        );
        assert!(result.integer_storage().is_none());
    }

    #[test]
    fn pow2_unary_preserves_native_single_real_complex_scalar_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 3.0], vec![1, 2]).unwrap();
        let Value::Tensor(output) = pow2_builtin(Value::Tensor(tensor), vec![]).expect("pow2")
        else {
            panic!("expected single tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 8.0])
        );

        let scalar = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
        let Value::Tensor(output) = pow2_builtin(Value::Tensor(scalar), vec![]).expect("pow2")
        else {
            panic!("single scalar must retain tensor class");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![4.0])
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 0.5)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) =
            pow2_builtin(Value::ComplexTensor(complex), vec![]).expect("complex pow2")
        else {
            panic!("complex single scalar must retain tensor class");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(&[pow2_complex_f32(1.0, 0.5)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) =
            pow2_builtin(Value::ComplexTensor(empty), vec![]).expect("empty pow2")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn pow2_binary_preserves_mixed_and_complex_single_storage() {
        let mantissa = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
        let exponent = Tensor::from_f32(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let Value::Tensor(output) =
            pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)])
                .expect("mixed pow2")
        else {
            panic!("expected mixed result to retain single");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![4.0, 24.0])
        );

        let mantissa = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let exponent = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) = pow2_builtin(
            Value::ComplexTensor(mantissa),
            vec![Value::Tensor(exponent)],
        )
        .expect("complex mixed pow2") else {
            panic!("complex single scalar must retain tensor class");
        };
        assert_eq!(output.as_f32_slice(), Some(&[(4.0, 4.0)][..]));

        let mantissa = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let exponent = Tensor::new(Vec::new(), vec![0, 2]).unwrap();
        let Value::ComplexTensor(output) = pow2_builtin(
            Value::ComplexTensor(mantissa),
            vec![Value::Tensor(exponent)],
        )
        .expect("empty mixed pow2") else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn pow2_integer_gpu_paths_bypass_floating_provider_hooks() {
        test_support::with_test_provider(|provider| {
            let unary = Tensor::new_integer(IntegerStorage::U64(vec![0, 64]), vec![1, 2]).unwrap();
            let unary = gpu_helpers::upload_tensor(provider, &unary).expect("upload unary");
            let result =
                pow2_builtin(Value::GpuTensor(unary), vec![]).expect("integer gpu unary pow2");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather unary result");
            assert_eq!(
                gathered.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![1.0, 2.0_f64.powi(64)])
            );

            let wide = 9_007_199_254_740_993_u64;
            let mantissa =
                Tensor::new_integer(IntegerStorage::U64(vec![wide, 3]), vec![1, 2]).unwrap();
            let exponent =
                Tensor::new_integer(IntegerStorage::U64(vec![1, 4]), vec![1, 2]).unwrap();
            let mantissa =
                gpu_helpers::upload_tensor(provider, &mantissa).expect("upload mantissa");
            let exponent =
                gpu_helpers::upload_tensor(provider, &exponent).expect("upload exponent");
            let Value::Tensor(output) =
                pow2_builtin(Value::GpuTensor(mantissa), vec![Value::GpuTensor(exponent)])
                    .expect("integer gpu binary pow2")
            else {
                panic!("binary integer fallback must return host double tensor");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![(wide as f64) * 2.0, 48.0])
            );
        });
    }

    #[test]
    fn pow2_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn pow2_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_scalar_exponent() {
        let result = pow2_builtin(Value::Num(3.0), Vec::new()).expect("pow2");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_tensor_exponent() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let result = pow2_builtin(Value::Tensor(tensor), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [0.5, 1.0, 2.0, 4.0];
                for (a, b) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_scaling() {
        let mantissa = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
        let exponent = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![4.0, 24.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_exponent_scalar() {
        let result = pow2_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("pow2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = pow2_complex(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_mantissa_real_exponent() {
        let mantissa =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -0.5)], vec![2, 1]).expect("complex tensor");
        let exponent = Tensor::new(vec![2.0, -1.0], vec![2, 1]).unwrap();
        let result = pow2_builtin(
            Value::ComplexTensor(mantissa),
            vec![Value::Tensor(exponent)],
        )
        .expect("pow2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                let scale0 = 2.0f64.exp2();
                let scale1 = (-1.0f64).exp2();
                assert!((out.materialize_f64()[0].0 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.materialize_f64()[1].0 - (2.0 * scale1)).abs() < 1e-12);
                assert!((out.materialize_f64()[1].1 - (-0.5 * scale1)).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_char_array() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = pow2_builtin(Value::CharArray(chars), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert!((out.materialize_f64()[0] - (65.0f64).exp2()).abs() < 1e-6);
                assert!((out.materialize_f64()[1] - (66.0f64).exp2()).abs() < 1e-6);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_rejects_strings() {
        let err = pow2_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pow2_builtin(Value::GpuTensor(handle), Vec::new()).expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected = vec![1.0, 2.0, 4.0];
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_scale_roundtrip() {
        test_support::with_test_provider(|provider| {
            let mantissa = Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap();
            let exponent = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let m_view = runmat_accelerate_api::HostTensorView {
                data: &mantissa.materialize_f64(),
                shape: &mantissa.shape,
            };
            let e_view = runmat_accelerate_api::HostTensorView {
                data: &exponent.materialize_f64(),
                shape: &exponent.shape,
            };
            let m_handle = provider.upload(&m_view).expect("upload m");
            let e_handle = provider.upload(&e_view).expect("upload e");
            let result = pow2_builtin(Value::GpuTensor(m_handle), vec![Value::GpuTensor(e_handle)])
                .expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![4.0, 24.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_broadcast_host() {
        let mantissa = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let exponent = Value::Int(IntValue::I32(2));
        let result = pow2_builtin(Value::Tensor(mantissa), vec![exponent]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![4.0, 8.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_matches_cpu_unary() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.5, -1.0, 0.0, 2.0, 4.25], vec![5, 1]).unwrap();
        let cpu_value = pow2_real(Value::Tensor(tensor.clone())).expect("pow2 cpu");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result from cpu path, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(pow2_gpu(handle)).expect("pow2 gpu");
        let gpu = test_support::gather(gpu_value).expect("gather gpu result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (g, c) in gpu
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert!((g - c).abs() <= tol, "mismatch: gpu={g} cpu={c} tol={tol}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_scale_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let mantissa = Tensor::new(vec![0.5, 1.5, 3.0], vec![3, 1]).unwrap();
        let exponent = Tensor::new(vec![3.0, -2.0, 5.5], vec![3, 1]).unwrap();

        let cpu_value = pow2_host_scale(
            Value::Tensor(mantissa.clone()),
            Value::Tensor(exponent.clone()),
        )
        .expect("pow2 host scale");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor from cpu scale, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let m_view = runmat_accelerate_api::HostTensorView {
            data: &mantissa.materialize_f64(),
            shape: &mantissa.shape,
        };
        let e_view = runmat_accelerate_api::HostTensorView {
            data: &exponent.materialize_f64(),
            shape: &exponent.shape,
        };
        let m_handle = provider.upload(&m_view).expect("upload mantissa");
        let e_handle = provider.upload(&e_view).expect("upload exponent");
        let gpu_value = block_on(pow2_gpu_scale(m_handle, e_handle)).expect("pow2 gpu scale");
        let gpu = test_support::gather(gpu_value).expect("gather gpu scale result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-3,
        };
        for (g, c) in gpu
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert!(
                (g - c).abs() <= tol,
                "scale mismatch: gpu={g} cpu={c} tol={tol}"
            );
        }
    }
}
