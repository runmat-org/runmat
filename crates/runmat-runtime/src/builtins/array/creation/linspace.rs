//! MATLAB-compatible `linspace` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linspace")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may generate sequences directly; the runtime uploads host-generated data when hooks are absent.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("linspace")
        .build()
}

fn linspace_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    linspace_error_with_message(error.message, error)
}

fn linspace_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    linspace_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn linspace_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("linspace");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

fn linspace_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    row_vector_type(ctx)
}

const LINSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of linearly spaced values.",
}];

const LINSPACE_SIG_2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
];

const LINSPACE_SIG_3_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("100"),
        description: "Number of points.",
    },
];

const LINSPACE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop)",
        inputs: &LINSPACE_SIG_2_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop, n)",
        inputs: &LINSPACE_SIG_3_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
];

const LINSPACE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.ARG_COUNT",
    identifier: None,
    when: "More than three input arguments are provided.",
    message: "linspace: expected at most three input arguments",
};

const LINSPACE_ERROR_COUNT_NOT_SCALAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_SCALAR",
    identifier: None,
    when: "The count argument is not a numeric scalar value.",
    message: "linspace: number of points must be a scalar",
};

const LINSPACE_ERROR_COUNT_NOT_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_INTEGER",
    identifier: None,
    when: "The count argument is not an integer value.",
    message: "linspace: number of points must be an integer",
};

const LINSPACE_ERROR_COUNT_NEGATIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NEGATIVE",
    identifier: None,
    when: "The count argument is negative.",
    message: "linspace: number of points must be >= 0",
};

const LINSPACE_ERROR_COUNT_NOT_FINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_FINITE",
    identifier: None,
    when: "The count argument is not finite.",
    message: "linspace: number of points must be finite",
};

const LINSPACE_ERROR_COUNT_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_TOO_LARGE",
    identifier: None,
    when: "The count argument exceeds platform limits.",
    message: "linspace: number of points is too large for this platform",
};

const LINSPACE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    LINSPACE_ERROR_ARG_COUNT,
    LINSPACE_ERROR_COUNT_NOT_SCALAR,
    LINSPACE_ERROR_COUNT_NOT_INTEGER,
    LINSPACE_ERROR_COUNT_NEGATIVE,
    LINSPACE_ERROR_COUNT_NOT_FINITE,
    LINSPACE_ERROR_COUNT_TOO_LARGE,
];

pub const LINSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINSPACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINSPACE_ERRORS,
};

#[runtime_builtin(
    name = "linspace",
    category = "array/creation",
    summary = "Generate linearly spaced row vectors.",
    keywords = "linspace,range,vector,gpu",
    examples = "x = linspace(0, 1, 5)  % [0 0.25 0.5 0.75 1]",
    accel = "array_construct",
    type_resolver(linspace_type),
    descriptor(crate::builtins::array::creation::linspace::LINSPACE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::linspace"
)]
async fn linspace_builtin(
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(linspace_error(&LINSPACE_ERROR_ARG_COUNT));
    }

    let (start_scalar, start_gpu) = parse_scalar("linspace", start).await?;
    let (stop_scalar, stop_gpu) = parse_scalar("linspace", stop).await?;

    let count = if rest.is_empty() {
        100usize
    } else {
        parse_count(&rest[0]).await?
    };

    let residency = sequence_gpu_preference(count, SequenceIntent::Linspace, start_gpu || stop_gpu);
    if log::log_enabled!(log::Level::Trace) {
        trace!(
            "linspace: len={} prefer_gpu={} reason={:?}",
            count,
            residency.prefer_gpu,
            residency.reason
        );
    }
    let prefer_gpu = residency.prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu)
}

#[derive(Clone, Copy)]
enum Scalar {
    Real(f64),
    Complex { re: f64, im: f64 },
}

impl Scalar {
    fn parts(&self) -> (f64, f64) {
        match *self {
            Scalar::Real(r) => (r, 0.0),
            Scalar::Complex { re, im } => (re, im),
        }
    }
}

async fn parse_scalar(name: &str, value: Value) -> crate::BuiltinResult<(Scalar, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let scalar = tensor_scalar(name, &tensor)?;
            Ok((scalar, true))
        }
        other => parse_scalar_host(name, other),
    }
}

fn parse_scalar_host(name: &str, value: Value) -> crate::BuiltinResult<(Scalar, bool)> {
    match value {
        Value::Num(n) => Ok((Scalar::Real(n), false)),
        Value::Int(i) => Ok((Scalar::Real(i.to_f64()), false)),
        Value::Bool(b) => Ok((Scalar::Real(if b { 1.0 } else { 0.0 }), false)),
        Value::Complex(re, im) => Ok((Scalar::Complex { re, im }, false)),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_scalar"),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(builtin_error(
            format!("{name}: endpoints must be numeric scalars; received a string-like value"),
        )),
        other => Err(builtin_error(format!(
            "{name}: endpoints must be numeric scalars; received {other:?}"
        ))),
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<Scalar> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    Ok(Scalar::Real(tensor::tensor_value_f64(tensor, 0)))
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<Scalar> {
    if let Some(storage) = tensor.integer_storage() {
        if storage.len() != 1 {
            return Err(builtin_error(format!("{name}: expected scalar input")));
        }
        let re = storage
            .real
            .value_at(0)
            .expect("scalar complex integer tensor has one real value")
            .to_f64();
        let im = storage
            .imag
            .value_at(0)
            .expect("scalar complex integer tensor has one imaginary value")
            .to_f64();
        return Ok(Scalar::Complex { re, im });
    }
    if tensor.materialize_f64().len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let (re, im) = tensor.materialize_f64()[0];
    Ok(Scalar::Complex { re, im })
}

async fn parse_count(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(&tensor)
        }
        other => parse_count_host(other),
    }
}

fn parse_count_host(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(i) => i
            .try_to_usize()
            .ok_or_else(|| linspace_error(&LINSPACE_ERROR_COUNT_NEGATIVE)),
        Value::Num(n) => parse_numeric_count(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(t)
        }
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_count"),
        other => Err(linspace_error_with_detail(
            &LINSPACE_ERROR_COUNT_NOT_SCALAR,
            format!("got {other:?}"),
        )),
    }
}

fn parse_tensor_count(tensor: &Tensor) -> crate::BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .expect("scalar integer tensor has one storage value");
        return parse_integer_count(&value);
    }
    parse_numeric_count(tensor::tensor_value_f64(tensor, 0))
}

fn parse_integer_count(value: &IntValue) -> crate::BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| linspace_error(&LINSPACE_ERROR_COUNT_NEGATIVE))
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_FINITE));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_INTEGER));
    }
    if rounded < 0.0 {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_NEGATIVE));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_TOO_LARGE));
    }
    Ok(rounded as usize)
}

fn build_sequence(
    start: Scalar,
    stop: Scalar,
    count: usize,
    prefer_gpu: bool,
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start_im != 0.0 || stop_im != 0.0;

    if complex {
        let data = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
        let tensor = ComplexTensor::new(data, vec![1, count])
            .map_err(|e| builtin_error(format!("linspace: {e}")))?;
        return Ok(Value::ComplexTensor(tensor));
    }

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            if count > 0 {
                if log::log_enabled!(log::Level::Trace) {
                    trace!(
                        "linspace: attempting provider.linspace start={} stop={} count={}",
                        start_re,
                        stop_re,
                        count
                    );
                }
                match provider.linspace(start_re, stop_re, count) {
                    Ok(handle) => {
                        trace!("linspace: provider.linspace succeeded");
                        return Ok(Value::GpuTensor(handle));
                    }
                    Err(err) => {
                        trace!("linspace: provider.linspace failed: {err}");
                    }
                }
            }
        }
    }

    let data = generate_real_sequence(start_re, stop_re, count);
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let shape = [1usize, count];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let tensor =
        Tensor::new(data, vec![1, count]).map_err(|e| builtin_error(format!("linspace: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn generate_real_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![stop];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        data.push(start + (idx as f64) * step);
    }
    if let Some(last) = data.last_mut() {
        *last = stop;
    }
    data
}

fn generate_complex_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(stop_re, stop_im)];
    }
    let mut data = Vec::with_capacity(count);
    let step_re = (stop_re - start_re) / ((count - 1) as f64);
    let step_im = (stop_im - start_im) / ((count - 1) as f64);
    for idx in 0..count {
        let re = start_re + (idx as f64) * step_re;
        let im = start_im + (idx as f64) * step_im;
        data.push((re, im));
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn count_parser_preserves_representable_uint64() {
        assert_eq!(
            parse_count_host(&Value::Int(IntValue::U64(u64::MAX))).ok(),
            usize::try_from(u64::MAX).ok()
        );
        assert!(parse_count_host(&Value::Int(IntValue::I64(-1))).is_err());
        assert!(parse_count_host(&Value::Num(usize::MAX as f64)).is_err());
        assert!(parse_count_host(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
    }
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    fn linspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::linspace_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_basic() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_count_parser_preserves_typed_integer_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let count = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap();

        assert_eq!(
            parse_count_host(&Value::Tensor(count)).unwrap(),
            large as usize
        );
    }

    #[test]
    fn linspace_count_parser_ignores_poisoned_f64_mirrors_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let count = Tensor::new_integer(storage, vec![1, 1]).expect("integer count");
            assert_eq!(parse_count_host(&Value::Tensor(count)).unwrap(), 2);
        }
    }

    #[test]
    fn linspace_count_parser_rejects_negative_typed_integer_tensors() {
        let count = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();

        assert!(parse_count_host(&Value::Tensor(count)).is_err());
    }

    #[test]
    fn linspace_real_integer_tensor_endpoints_read_exact_storage() {
        let start =
            Tensor::new_integer(IntegerStorage::I64(vec![-3]), vec![1, 1]).expect("start tensor");
        let stop =
            Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1]).expect("stop tensor");

        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![-3.0, 3.0, 9.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_type_is_row_vector() {
        assert_eq!(
            linspace_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_default_count() {
        let result =
            linspace_builtin(Value::Num(-1.0), Value::Num(1.0), Vec::new()).expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 100]);
                assert!((t.materialize_f64().first().copied().unwrap() + 1.0).abs() < 1e-12);
                assert!((t.materialize_f64().last().copied().unwrap() - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_zero_count() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(10.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_single_point() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(9.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.materialize_f64()[0] - 9.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_integer_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(3.5)])
            .expect_err("expected error");
        assert!(err.message().contains("integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_negative_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-2))],
        )
        .expect_err("expected error");
        assert!(err.message().contains(">= 0"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_infinite_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_nan_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_scalar_count_errors() {
        let sz = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Tensor(sz)])
            .expect_err("expected error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_complex_sequence() {
        let result = linspace_builtin(
            Value::Complex(1.0, 1.0),
            Value::Complex(-3.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (idx, &(re, im)) in expected.iter().enumerate() {
                    let (r, i) = t.materialize_f64()[idx];
                    assert!((r - re).abs() < 1e-9);
                    assert!((i - im).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn linspace_complex_integer_tensor_endpoints_read_exact_storage() {
        let start_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .expect("start storage");
        let start = ComplexTensor::new_integer(start_storage, vec![1, 1]).expect("start tensor");
        let stop_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![-3]), IntegerStorage::I16(vec![2]))
                .expect("stop storage");
        let stop = ComplexTensor::new_integer(stop_storage, vec![1, 1]).expect("stop tensor");

        let result = linspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");

        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (actual, expected) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual.0 - expected.0).abs() < 1e-9);
                    assert!((actual.1 - expected.1).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_boolean_arguments_are_promoted() {
        let result = linspace_builtin(
            Value::Bool(true),
            Value::Bool(false),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0, 0.5, 0.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_boolean_count_supported() {
        let result = linspace_builtin(Value::Num(3.0), Value::Num(7.0), vec![Value::Bool(true)])
            .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.materialize_f64()[0] - 7.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_tensor_scalar_arguments() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [2.0, 3.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_typed_integer_tensor_endpoints_read_exact_storage() {
        let start = Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("start");
        let stop = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).expect("stop");

        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [-2.0, 0.0, 2.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_equal_endpoints_fill_with_endpoint() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(5.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t.materialize_f64().iter().all(|v| (*v - 5.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(5))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                    assert_eq!(gathered.shape, vec![1, 5]);
                    for (idx, expected_val) in expected.iter().enumerate() {
                        assert!((gathered.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                    }
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_zero_count_produces_gpu_empty_vector() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 0]);
                    assert!(gathered.materialize_f64().is_empty());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn linspace_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        let provider =
            register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");

        let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.materialize_f64(),
            shape: &start.shape,
        };
        let stop_view = HostTensorView {
            data: &stop.materialize_f64(),
            shape: &stop.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let stop_handle = provider.upload(&stop_view).expect("upload stop");

        let result = linspace_builtin(
            Value::GpuTensor(start_handle),
            Value::GpuTensor(stop_handle),
            vec![Value::Int(IntValue::I32(9))],
        )
        .expect("linspace");
        let gathered = test_support::gather(result).expect("gather");
        let expected = generate_real_sequence(0.0, 1.0, 9);

        let precision = runmat_accelerate_api::provider()
            .expect("provider")
            .precision();
        let tol = match precision {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };

        assert_eq!(gathered.shape, vec![1, 9]);
        for (idx, expected_value) in expected.iter().enumerate() {
            let actual = gathered.materialize_f64()[idx];
            assert!(
                (actual - expected_value).abs() <= tol,
                "mismatch at {idx}: gpu={} expected={}",
                actual,
                expected_value
            );
        }
    }
}
