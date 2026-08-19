//! MATLAB-compatible `factorial` builtin with GPU-aware semantics for RunMat.
//!
//! Implements element-wise factorial for numerical inputs, mirroring MATLAB’s
//! restrictions to non-negative integers while providing documented fallbacks
//! when GPU providers lack a dedicated kernel.

use once_cell::sync::Lazy;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const MAX_FACTORIAL_N: usize = 170;

static FACT_TABLE: Lazy<[f64; MAX_FACTORIAL_N + 1]> = Lazy::new(|| {
    let mut table = [1.0f64; MAX_FACTORIAL_N + 1];
    let mut acc = 1.0;
    for (n, slot) in table.iter_mut().enumerate().skip(1) {
        acc *= n as f64;
        *slot = acc;
    }
    table
});

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::factorial")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "factorial",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_factorial",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_factorial; otherwise the runtime gathers to host and mirrors MATLAB overflow/NaN behaviour.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "factorial",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Factorial is evaluated as a scalar helper; fusion currently bypasses it and executes the standalone host or provider kernel.",
};

const BUILTIN_NAME: &str = "factorial";

const FACTORIAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise factorial result.",
}];

const FACTORIAL_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric input.",
}];

const FACTORIAL_INPUTS_X_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real numeric input.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output class/device prototype.",
    },
];

const FACTORIAL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = factorial(X)",
        inputs: &FACTORIAL_INPUTS_X,
        outputs: &FACTORIAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = factorial(X, \"like\", prototype)",
        inputs: &FACTORIAL_INPUTS_X_LIKE,
        outputs: &FACTORIAL_OUTPUT,
    },
];

const FACTORIAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:factorial:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "factorial: invalid argument",
};

const FACTORIAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INVALID_INPUT",
    identifier: Some("RunMat:factorial:InvalidInput"),
    when: "Input value or prototype cannot be converted to supported real numeric form.",
    message: "factorial: invalid input",
};

const FACTORIAL_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.GPU_UNSUPPORTED",
    identifier: Some("RunMat:factorial:GpuUnsupported"),
    when: "GPU output via \"like\" is requested but no compatible provider is active.",
    message: "factorial: gpu output not supported",
};

const FACTORIAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INTERNAL",
    identifier: Some("RunMat:factorial:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "factorial: internal error",
};

const FACTORIAL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FACTORIAL_ERROR_INVALID_ARGUMENT,
    FACTORIAL_ERROR_INVALID_INPUT,
    FACTORIAL_ERROR_GPU_UNSUPPORTED,
    FACTORIAL_ERROR_INTERNAL,
];

pub const FACTORIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FACTORIAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FACTORIAL_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every element must be a real nonnegative integer value.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = factorial(X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Ordinary output preserves input class and saturates at its documented threshold; interactive GPU input excludes int64 and uint64, while an explicit prototype can select output class/residency.",
    }];

fn factorial_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    if std::ptr::eq(error, &FACTORIAL_ERROR_GPU_UNSUPPORTED)
        || std::ptr::eq(error, &FACTORIAL_ERROR_INVALID_INPUT)
    {
        builder = builder.with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    }
    builder.build()
}

#[runtime_builtin(
    name = "factorial",
    category = "math/elementwise",
    summary = "Compute element-wise factorial values.",
    keywords = "factorial,n!,permutation,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::factorial::FACTORIAL_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::factorial::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
async fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => factorial_gpu(handle).await?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(factorial_error_with_detail(
                &FACTORIAL_ERROR_INVALID_INPUT,
                "complex inputs are not supported; use gamma(z + 1) instead",
            ))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err(factorial_error_with_detail(
                &FACTORIAL_ERROR_INVALID_INPUT,
                "expected numeric or logical input",
            ))
        }
        other => {
            let tensor = tensor::value_into_tensor_for("factorial", other)
                .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INVALID_INPUT, e))?;
            factorial_tensor(tensor).map(tensor::tensor_into_value)?
        }
    };
    apply_output_template(base, &output).await
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(factorial_error_with_detail(
                    &FACTORIAL_ERROR_INVALID_ARGUMENT,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(factorial_error_with_detail(
                    &FACTORIAL_ERROR_INVALID_ARGUMENT,
                    "unrecognised option; only 'like' is supported",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(factorial_error_with_detail(
                    &FACTORIAL_ERROR_INVALID_ARGUMENT,
                    "unrecognised option; only 'like' is supported",
                ))
            }
        }
        _ => Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
    }
}

async fn factorial_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(&handle) {
        if matches!(
            integer_type,
            runmat_accelerate_api::IntegerElementType::I64
                | runmat_accelerate_api::IntegerElementType::U64
        ) {
            return Err(factorial_error_with_detail(
                &FACTORIAL_ERROR_INVALID_INPUT,
                "64-bit integer GPU inputs are not supported",
            ));
        }
        let provider = runmat_accelerate_api::provider_for_handle(&handle);
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let output = factorial_tensor(tensor)?;
        if let Some(provider) = provider {
            if let Ok(handle) = gpu_helpers::upload_tensor(provider, &output) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        return Ok(tensor::tensor_into_value(output));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_factorial(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(factorial_tensor(tensor)?))
}

fn factorial_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INTERNAL, e))?;
    let storage = match storage {
        NumericStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(factorial_scalar)
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(factorial_scalar_f32)
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericStorage::I8(values) => NumericStorage::I8(factorial_signed_values(values)?),
        NumericStorage::I16(values) => NumericStorage::I16(factorial_signed_values(values)?),
        NumericStorage::I32(values) => NumericStorage::I32(factorial_signed_values(values)?),
        NumericStorage::I64(values) => NumericStorage::I64(factorial_signed_values(values)?),
        NumericStorage::U8(values) => NumericStorage::U8(factorial_unsigned_values(values)),
        NumericStorage::U16(values) => NumericStorage::U16(factorial_unsigned_values(values)),
        NumericStorage::U32(values) => NumericStorage::U32(factorial_unsigned_values(values)),
        NumericStorage::U64(values) => NumericStorage::U64(factorial_unsigned_values(values)),
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INTERNAL, e))
}

fn factorial_scalar(value: f64) -> BuiltinResult<f64> {
    let n = validate_factorial_f64(value)?;
    if n > MAX_FACTORIAL_N {
        return Ok(f64::INFINITY);
    }
    Ok(FACT_TABLE[n])
}

fn factorial_scalar_f32(value: f32) -> BuiltinResult<f32> {
    let n = validate_factorial_f32(value)?;
    if n >= 35 {
        return Ok(f32::INFINITY);
    }
    Ok(FACT_TABLE[n] as f32)
}

fn validate_factorial_f64(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_INPUT,
            "input values must be real, finite, nonnegative integers",
        ));
    }
    if value > MAX_FACTORIAL_N as f64 {
        return Ok(MAX_FACTORIAL_N + 1);
    }
    Ok(value as usize)
}

fn validate_factorial_f32(value: f32) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_INPUT,
            "input values must be real, finite, nonnegative integers",
        ));
    }
    if value >= 35.0 {
        return Ok(35);
    }
    Ok(value as usize)
}

trait FactorialUnsigned: Copy {
    const MAX_U128: u128;
    fn into_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

trait FactorialSigned: Copy {
    const MAX_U128: u128;
    fn is_negative(self) -> bool;
    fn into_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

macro_rules! impl_factorial_unsigned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FactorialUnsigned for $ty {
                const MAX_U128: u128 = <$ty>::MAX as u128;
                fn into_u128(self) -> u128 { self as u128 }
                fn from_u128(value: u128) -> Self { value as $ty }
            }
        )+
    };
}

macro_rules! impl_factorial_signed {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FactorialSigned for $ty {
                const MAX_U128: u128 = <$ty>::MAX as u128;
                fn is_negative(self) -> bool { self < 0 }
                fn into_u128(self) -> u128 { self as u128 }
                fn from_u128(value: u128) -> Self { value as $ty }
            }
        )+
    };
}

impl_factorial_unsigned!(u8, u16, u32, u64);
impl_factorial_signed!(i8, i16, i32, i64);

fn factorial_unsigned_values<T: FactorialUnsigned>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_u128(factorial_integer_saturating(value.into_u128(), T::MAX_U128)))
        .collect()
}

fn factorial_signed_values<T: FactorialSigned>(values: Vec<T>) -> BuiltinResult<Vec<T>> {
    if values.iter().any(|value| value.is_negative()) {
        return Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_INPUT,
            "input values must be real, finite, nonnegative integers",
        ));
    }
    Ok(values
        .into_iter()
        .map(|value| T::from_u128(factorial_integer_saturating(value.into_u128(), T::MAX_U128)))
        .collect())
}

fn factorial_integer_saturating(n: u128, maximum: u128) -> u128 {
    let mut value = 1_u128;
    let mut factor = 2_u128;
    while factor <= n {
        let Some(next) = value.checked_mul(factor) else {
            return maximum;
        };
        if next > maximum {
            return maximum;
        }
        value = next;
        factor += 1;
    }
    value
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => {
            let analysis = analyse_like_prototype(proto).await?;
            match analysis.device {
                DevicePreference::Host => convert_to_host_like(value).await,
                DevicePreference::Gpu => convert_to_gpu_like(value),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

struct LikeAnalysis {
    device: DevicePreference,
}

#[async_recursion::async_recursion(?Send)]
async fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
        }),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::LogicalArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_INPUT,
            "complex prototypes for 'like' are not supported; results are always real",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(factorial_error_with_detail(
                &FACTORIAL_ERROR_INVALID_INPUT,
                "prototype must be numeric or a gpuArray",
            ))
        }
        other => {
            let gathered = gpu_helpers::gather_value_async(other)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            analyse_like_prototype(&gathered).await
        }
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME)),
        other => Ok(other),
    }
}

fn convert_to_gpu_like(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        factorial_error_with_detail(
            &FACTORIAL_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => upload_tensor(provider, tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INTERNAL, e))?;
            upload_tensor(provider, tensor)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, Value::Int(i))
                .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INTERNAL, e))?;
            upload_tensor(provider, tensor)
        }
        Value::Bool(b) => convert_to_gpu_like(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| factorial_error_with_detail(&FACTORIAL_ERROR_INTERNAL, e))?;
            upload_tensor(provider, tensor)
        }
        other => Err(factorial_error_with_detail(
            &FACTORIAL_ERROR_INVALID_INPUT,
            format!("cannot place value {other:?} on the GPU via 'like'"),
        )),
    }
}

fn upload_tensor(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, &tensor).map_err(|e| {
        factorial_error_with_detail(
            &FACTORIAL_ERROR_INTERNAL,
            format!("failed to upload GPU result: {e}"),
        )
    })?;
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};

    fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::factorial_builtin(value, rest))
    }

    #[test]
    fn factorial_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FACTORIAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = factorial(X)"));
        assert!(labels.contains(&"Y = factorial(X, \"like\", prototype)"));
    }

    #[test]
    fn factorial_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
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
    fn factorial_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_scalar_positive() {
        let result = factorial_builtin(Value::Num(5.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_zero_is_one() {
        let result = factorial_builtin(Value::Num(0.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_vector_inputs() {
        let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_preserves_uint64_storage_and_saturates() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![3, 5, 171]), vec![3, 1])
            .expect("integer tensor");

        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(
                    out.into_numeric_storage().unwrap(),
                    NumericStorage::U64(vec![6, 120, u64::MAX])
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn factorial_preserves_all_native_classes_and_documented_saturation_thresholds() {
        let cases = [
            (
                IntegerStorage::I8(vec![5, 6]),
                IntegerStorage::I8(vec![120, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![7, 8]),
                IntegerStorage::I16(vec![5_040, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![12, 13]),
                IntegerStorage::I32(vec![479_001_600, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![20, 21]),
                IntegerStorage::I64(vec![2_432_902_008_176_640_000, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![5, 6]),
                IntegerStorage::U8(vec![120, u8::MAX]),
            ),
            (
                IntegerStorage::U16(vec![8, 9]),
                IntegerStorage::U16(vec![40_320, u16::MAX]),
            ),
            (
                IntegerStorage::U32(vec![12, 13]),
                IntegerStorage::U32(vec![479_001_600, u32::MAX]),
            ),
            (
                IntegerStorage::U64(vec![20, 21]),
                IntegerStorage::U64(vec![2_432_902_008_176_640_000, u64::MAX]),
            ),
        ];
        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![1, 2]).unwrap();
            let Value::Tensor(output) =
                factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial")
            else {
                panic!("expected typed tensor");
            };
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let single = Tensor::from_f32(vec![5.0, 34.0, 35.0], vec![1, 3]).unwrap();
        let Value::Tensor(output) =
            factorial_builtin(Value::Tensor(single), Vec::new()).expect("single factorial")
        else {
            panic!("expected single tensor");
        };
        let NumericStorage::F32(values) = output.into_numeric_storage().unwrap() else {
            panic!("expected native single storage");
        };
        assert_eq!(values[0], 120.0);
        assert!(values[1].is_finite());
        assert!(values[2].is_infinite());

        let empty = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) =
            factorial_builtin(Value::Tensor(empty), Vec::new()).expect("empty factorial")
        else {
            panic!("expected empty single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[test]
    fn factorial_rejects_negative_signed_integer_and_64_bit_integer_gpu_inputs() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 3]), vec![1, 2]).unwrap();
        let err = factorial_builtin(Value::Tensor(tensor), Vec::new())
            .expect_err("negative integer must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![5, 20]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let err = factorial_builtin(Value::GpuTensor(handle), Vec::new())
                .expect_err("uint64 GPU factorial must reject");
            assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
            assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
            assert!(err.message().contains("64-bit integer GPU"));
        });
    }

    #[test]
    fn public_dispatch_preserves_64_bit_integer_gpu_factorial_rejection() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![5, 20]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let error = crate::dispatcher::call_builtin("factorial", &[Value::GpuTensor(handle)])
                .expect_err("public dispatch must not gather unsupported uint64 GPU input");
            assert_eq!(error.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
            assert!(error.message().contains("64-bit integer GPU"));
        });
    }

    #[test]
    fn factorial_integer_gpu_preserves_class_and_residency_without_floating_hook() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U32(vec![5, 13]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("factorial");
            assert!(matches!(result, Value::GpuTensor(_)));
            let output = test_support::gather(result).expect("gather");
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U32(vec![120, u32::MAX]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_non_integer_errors() {
        let err =
            factorial_builtin(Value::Num(2.5), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_negative_errors() {
        let tensor = Tensor::new(vec![-1.0, 3.0], vec![2, 1]).unwrap();
        let err = factorial_builtin(Value::Tensor(tensor), Vec::new())
            .expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_small_positive_non_integer_errors() {
        let err =
            factorial_builtin(Value::Num(1e-12), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_overflow_returns_inf() {
        let result = factorial_builtin(Value::Num(171.0), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_infinite()),
            other => panic!("expected scalar Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_missing_prototype_errors() {
        let err = factorial_builtin(Value::Num(3.0), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.message().contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_gpu_prototype_uploads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = factorial_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("factorial");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![6.0, 24.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn factorial_like_gpu_prototype_preserves_integer_scalar_class() {
        test_support::with_test_provider(|provider| {
            let prototype = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("upload prototype");
            let result = factorial_builtin(
                Value::Int(IntValue::U16(5)),
                vec![Value::from("like"), Value::GpuTensor(prototype)],
            )
            .expect("factorial like");
            let output = test_support::gather(result).expect("gather");
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U16(vec![120]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("fact");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0, 6.0, 120.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("factorial");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.materialize_f64(), vec![6.0, 24.0]);
                }
                other => panic!("expected host tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_logical_input_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = factorial_builtin(Value::LogicalArray(logical), Vec::new()).expect("fact");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_int_input_preserves_class() {
        let value = Value::Int(IntValue::U16(5));
        let result = factorial_builtin(value, Vec::new()).expect("factorial");
        assert_eq!(result, Value::Int(IntValue::U16(120)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_nan_errors() {
        let err =
            factorial_builtin(Value::Num(f64::NAN), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_complex_input_errors() {
        let err = factorial_builtin(Value::Complex(1.0, 0.5), Vec::new())
            .expect_err("expected complex rejection");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_string_input_errors() {
        let err = factorial_builtin(Value::from("hello"), Vec::new())
            .expect_err("expected string rejection");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_complex_prototype_rejected() {
        let err = factorial_builtin(
            Value::Num(3.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected complex prototype rejection");
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn factorial_wgpu_matches_cpu_after_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0], vec![3, 1]).unwrap();
        let cpu = factorial_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(factorial_gpu(handle)).expect("gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
    }
}
