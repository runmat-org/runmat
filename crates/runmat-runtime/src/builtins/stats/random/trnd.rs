//! Student's t random variates.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "trnd";

const TRND_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trnd-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trnd with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrndIntegerDegreesOfFreedomExtension"),
};

const TRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrndIntegerSizeExtension"),
};

pub const TRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [TRND_INTEGER_NU_EXTENSION, TRND_INTEGER_SIZE_EXTENSION];

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from the Student's t distribution.",
}];

const INPUT_NU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Degrees of freedom parameter.",
};

const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size arguments.",
};

const INPUTS_NU: [BuiltinParamDescriptor; 1] = [INPUT_NU];
const INPUTS_NU_SZ: [BuiltinParamDescriptor; 2] = [INPUT_NU, INPUT_SZ];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu)",
        inputs: &INPUTS_NU,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz1, sz2, ...)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:trnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing, malformed, or incompatible.",
    message: "trnd: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INTERNAL",
    identifier: Some("RunMat:trnd:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "trnd: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer degrees of freedom are gated by trnd-integer-degrees-of-freedom and enter the floating Student-t computation domain.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer scalar/vector size arguments are gated by trnd-integer-size and parsed exactly before allocation.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = trnd(nu, sz1, sz2, ...)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer parameters are RunMat-only; integer nu produces double samples, and a resident source receives host-generated fallback output re-uploaded with matching precision.",
    }];

fn trnd_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn trnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args {
        [_] => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "trnd",
    category = "stats/random",
    summary = "Generate Student's t random numbers.",
    keywords = "trnd,student t,random,statistics,distribution",
    type_resolver(trnd_type),
    descriptor(crate::builtins::stats::random::trnd::TRND_DESCRIPTOR),
    extensions(crate::builtins::stats::random::trnd::TRND_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::trnd::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::trnd"
)]
pub(crate) async fn trnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.first().is_some_and(is_resident_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRND_INTEGER_NU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().skip(1).any(is_resident_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRND_INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (nu, shape, output_precision, gpu_source, integer_nu, integer_size) =
        parse_args(args).await?;
    if integer_nu {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRND_INTEGER_NU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if integer_size {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRND_INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_student_t(&nu, len, BUILTIN_NAME)?;
    build_output(data, shape, output_precision, gpu_source)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPrecision {
    Double,
    Single,
}

fn output_precision(value: &Value) -> OutputPrecision {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            OutputPrecision::Single
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(ProviderPrecision::F32) =>
        {
            OutputPrecision::Single
        }
        _ => OutputPrecision::Double,
    }
}

fn build_output(
    data: Vec<f64>,
    shape: Vec<usize>,
    output_precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = match output_precision {
        OutputPrecision::Double => Tensor::new(data, shape),
        OutputPrecision::Single => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
    }
    .map_err(|err| trnd_error(&ERROR_INTERNAL, format!("trnd: {err}")))?;

    if let Some(source) = gpu_source {
        let provider = runmat_accelerate_api::provider_for_handle(&source)
            .or_else(runmat_accelerate_api::provider)
            .ok_or_else(|| {
                trnd_error(
                    &ERROR_INTERNAL,
                    "trnd: no acceleration provider registered for GPU output",
                )
            })?;
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|err| trnd_error(&ERROR_INTERNAL, format!("trnd: {err}")))?;
        runmat_accelerate_api::set_handle_precision(
            &handle,
            match output_precision {
                OutputPrecision::Double => ProviderPrecision::F64,
                OutputPrecision::Single => ProviderPrecision::F32,
            },
        );
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }

    match output_precision {
        OutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
        OutputPrecision::Single => Ok(Value::Tensor(tensor)),
    }
}

async fn parse_args(
    args: Vec<Value>,
) -> BuiltinResult<(
    Vec<f64>,
    Vec<usize>,
    OutputPrecision,
    Option<GpuTensorHandle>,
    bool,
    bool,
)> {
    if args.is_empty() {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: nu argument is required",
        ));
    }
    let integer_nu = is_typed_integer_value(&args[0]);
    let integer_size = args[1..].iter().any(is_typed_integer_value);
    let output_precision = output_precision(&args[0]);
    let gpu_source = match &args[0] {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let nu_value = gather_if_needed_async(&args[0])
        .await
        .map_err(|err| trnd_error(&ERROR_INVALID_ARGUMENT, format!("trnd: {err}")))?;
    let nu = tensor::value_into_tensor_for(BUILTIN_NAME, nu_value)
        .map_err(|err| trnd_error(&ERROR_INVALID_ARGUMENT, format!("trnd: {err}")))?;
    let nu_shape = nu.shape.clone();
    let nu = tensor::tensor_into_values_f64(nu);
    if nu.iter().any(|value| value.is_nan() || *value <= 0.0) {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: nu must contain positive degrees of freedom",
        ));
    }

    let shape = if args.len() == 1 {
        normalize_shape(nu_shape.clone())
    } else {
        parse_shape_args(&args[1..]).await?
    };
    if nu.len() != 1 && normalize_shape(nu_shape) != shape {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: requested size must match non-scalar nu",
        ));
    }
    Ok((
        nu,
        shape,
        output_precision,
        gpu_source,
        integer_nu,
        integer_size,
    ))
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_resident_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, BUILTIN_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => {
                return Err(trnd_error(
                    &ERROR_INVALID_ARGUMENT,
                    format!("trnd: invalid size argument: {arg:?}"),
                ));
            }
            Err(err) => return Err(trnd_error(&ERROR_INVALID_ARGUMENT, err)),
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        normalize_shape(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    #[test]
    fn trnd_scalar_is_deterministic_and_finite() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result = block_on(trnd_builtin(vec![Value::Num(10.0)])).expect("trnd");
        match result {
            Value::Num(value) => assert!(value.is_finite()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn trnd_accepts_size_forms() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let out = block_on(trnd_builtin(vec![
            Value::Num(5.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ]))
        .expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let size = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let out = block_on(trnd_builtin(vec![Value::Num(5.0), Value::Tensor(size)]))
            .expect("trnd size vector");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn trnd_non_scalar_nu_shape_must_match_requested_shape() {
        let nu = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let out = block_on(trnd_builtin(vec![Value::Tensor(nu.clone())])).expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![3, 1]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let err = block_on(trnd_builtin(vec![
            Value::Tensor(nu),
            Value::Num(1.0),
            Value::Num(3.0),
        ]))
        .expect_err("mismatched shape should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trnd_reads_typed_integer_nu_and_size_exactly() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset();
        let nu = integer_tensor(IntegerStorage::U16(vec![5, 6, 7]), vec![3, 1]);
        let out = block_on(trnd_builtin(vec![Value::Tensor(nu)])).expect("trnd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert!(tensor
                    .materialize_f64()
                    .iter()
                    .all(|value| value.is_finite()));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let nu = integer_tensor(IntegerStorage::I16(vec![5]), vec![1, 1]);
        let size = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let out =
            block_on(trnd_builtin(vec![Value::Tensor(nu), Value::Tensor(size)])).expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn trnd_typed_integer_arguments_follow_compatibility_mode() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(trnd_builtin(vec![Value::Int(
                runmat_builtins::IntValue::U16(0),
            )]))
            .expect_err("invalid typed-integer nu retains ordinary validation");
            assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);

            let error = block_on(trnd_builtin(vec![Value::Int(
                runmat_builtins::IntValue::U16(5),
            )]))
            .expect_err("MATLAB mode rejects typed-integer degrees of freedom");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:TrndIntegerDegreesOfFreedomExtension")
            );

            let size = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
            let error = block_on(trnd_builtin(vec![Value::Num(5.0), Value::Tensor(size)]))
                .expect_err("MATLAB mode rejects typed-integer size controls");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:TrndIntegerSizeExtension")
            );

            let resident_nu = GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 9_305_001,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &resident_nu,
                runmat_accelerate_api::IntegerElementType::U16,
            );
            let error = block_on(trnd_builtin(vec![Value::GpuTensor(resident_nu.clone())]))
                .expect_err("MATLAB mode rejects resident integer nu before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:TrndIntegerDegreesOfFreedomExtension")
            );
            runmat_accelerate_api::clear_handle_integer_type(&resident_nu);

            let resident_size = GpuTensorHandle {
                shape: vec![1, 2],
                device_id: 0,
                buffer_id: 9_305_002,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &resident_size,
                runmat_accelerate_api::IntegerElementType::U16,
            );
            let error = block_on(trnd_builtin(vec![
                Value::Num(5.0),
                Value::GpuTensor(resident_size.clone()),
            ]))
            .expect_err("MATLAB mode rejects resident integer size before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:TrndIntegerSizeExtension")
            );
            runmat_accelerate_api::clear_handle_integer_type(&resident_size);
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let size = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
            let out = block_on(trnd_builtin(vec![
                Value::Int(runmat_builtins::IntValue::U16(5)),
                Value::Tensor(size),
            ]))
            .expect("RunMat mode accepts typed-integer trnd arguments");
            let Value::Tensor(tensor) = out else {
                panic!("expected tensor output");
            };
            assert_eq!(tensor.shape, vec![2, 3]);
        }
    }

    #[test]
    fn trnd_preserves_native_single_output() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let nu = Tensor::from_f32(vec![5.0, 6.0], vec![2, 1]).unwrap();
        let out = block_on(trnd_builtin(vec![Value::Tensor(nu)])).expect("single trnd");
        let Value::Tensor(tensor) = out else {
            panic!("expected native-single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.shape, vec![2, 1]);
        assert!(tensor
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn trnd_gpu_input_returns_resident_output_with_matching_precision() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let nu = Tensor::from_f32(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &nu).expect("upload single parameter");
            runmat_accelerate_api::set_handle_precision(&input, ProviderPrecision::F32);
            let out = block_on(trnd_builtin(vec![Value::GpuTensor(input)]))
                .expect("resident single trnd");
            let Value::GpuTensor(handle) = out else {
                panic!("expected resident GPU output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(ProviderPrecision::F32)
            );
            assert_eq!(handle.shape, vec![2, 1]);
        });
    }

    #[test]
    fn trnd_rejects_nonpositive_degrees_of_freedom() {
        let err =
            block_on(trnd_builtin(vec![Value::Num(0.0)])).expect_err("nonpositive nu should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trnd_distribution_has_heavier_tails_than_normal() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let n = 20_000;
        let out = block_on(trnd_builtin(vec![
            Value::Num(3.0),
            Value::Num(n as f64),
            Value::Num(1.0),
        ]))
        .expect("trnd");
        let data = match out {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.1, "sample mean {mean}");
        assert!((variance - 3.0).abs() < 0.35, "sample variance {variance}");
    }
}
