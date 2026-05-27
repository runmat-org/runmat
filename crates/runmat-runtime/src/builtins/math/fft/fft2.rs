//! MATLAB-compatible `fft2` builtin with GPU-aware semantics for RunMat.

use super::common::{
    download_provider_complex_tensor, gather_gpu_complex_tensor, parse_2d_lengths_from_data,
    parse_length, transform_axes_complex_tensor, value_to_complex_tensor, TransformDirection,
};
use super::fft::fft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::fft::type_resolvers::fft2_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fft2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fft2",
    op_kind: GpuOpKind::Custom("fft2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs two sequential `fft_dim` passes (dimensions 0 and 1); falls back to host execution when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fft2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fft2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "fft2 terminates fusion plans; fused kernels are not generated for multi-dimensional FFTs.",
};

const BUILTIN_NAME: &str = "fft2";

const FFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D complex Fourier spectrum output.",
}];

const FFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const FFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Scalar N or two-element [M N] size vector.",
    },
];

const FFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
];

const FFT2_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X)",
        inputs: &FFT2_INPUTS_CORE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, SIZE)",
        inputs: &FFT2_INPUTS_SIZE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, M, N)",
        inputs: &FFT2_INPUTS_M_N,
        outputs: &FFT2_OUTPUT,
    },
];

const FFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.ARG_COUNT",
    identifier: Some("RunMat:fft2:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "fft2: invalid argument count",
};

const FFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_LENGTH",
    identifier: Some("RunMat:fft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "fft2: invalid transform length argument",
};

const FFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:fft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "fft2: invalid size vector argument",
};

const FFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_INPUT",
    identifier: Some("RunMat:fft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fft2: invalid input",
};

const FFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INTERNAL",
    identifier: Some("RunMat:fft2:Internal"),
    when: "FFT2 execution or tensor shaping fails.",
    message: "fft2: internal error",
};

const FFT2_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFT2_ERROR_ARG_COUNT,
    FFT2_ERROR_INVALID_LENGTH,
    FFT2_ERROR_INVALID_SIZE_VECTOR,
    FFT2_ERROR_INVALID_INPUT,
    FFT2_ERROR_INTERNAL,
];

pub const FFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFT2_ERRORS,
};

fn fft2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fft2_error_with_message(error.message, error)
}

fn fft2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fft2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fft2_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fft2_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fft2",
    category = "math/fft",
    summary = "Compute the two-dimensional discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fft2,2d fft,two-dimensional fourier transform,gpu",
    type_resolver(fft2_type),
    descriptor(crate::builtins::math::fft::fft2::FFT2_DESCRIPTOR),
    builtin_path = "crate::builtins::math::fft::fft2"
)]
async fn fft2_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let lengths = parse_fft2_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => fft2_gpu(handle, lengths).await,
        other => fft2_host(other, lengths),
    }
}

fn fft2_host(value: Value, lengths: (Option<usize>, Option<usize>)) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        fft2_error_with_source(&FFT2_ERROR_INVALID_INPUT, "input conversion failed", source)
    })?;
    let transformed = fft2_complex_tensor(tensor, lengths)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<Value> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return fft2_gpu_fallback(handle, lengths).await;
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(first) = provider.fft_dim(&handle, lengths.0, 0).await {
            match provider.fft_dim(&first, lengths.1, 1).await {
                Ok(second) => {
                    if first.buffer_id != second.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    return Ok(Value::GpuTensor(second));
                }
                Err(_) => {
                    let partial =
                        download_provider_complex_tensor(provider, &first, BUILTIN_NAME, true)
                            .await
                            .map_err(|source| {
                                fft2_error_with_source(
                                    &FFT2_ERROR_INVALID_INPUT,
                                    "provider download failed",
                                    source,
                                )
                            })?;
                    let completed = fft_complex_tensor(partial, lengths.1, Some(2))?;
                    return Ok(complex_tensor_into_value(completed));
                }
            }
        }
    }

    fft2_gpu_fallback(handle, lengths).await
}

async fn fft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<Value> {
    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            fft2_error_with_source(&FFT2_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = fft2_complex_tensor(complex, lengths)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<ComplexTensor> {
    let (len_rows, len_cols) = lengths;
    transform_axes_complex_tensor(
        tensor,
        &[len_rows, len_cols],
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| fft2_error_with_source(&FFT2_ERROR_INTERNAL, "transform failed", source))
}

fn parse_fft2_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match args.len() {
        0 => Ok((None, None)),
        1 => parse_fft2_single(&args[0]),
        2 => {
            let rows = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(
                    &FFT2_ERROR_INVALID_LENGTH,
                    "row-length parse failed",
                    source,
                )
            })?;
            let cols = parse_length(&args[1], BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(
                    &FFT2_ERROR_INVALID_LENGTH,
                    "column-length parse failed",
                    source,
                )
            })?;
            Ok((rows, cols))
        }
        _ => Err(fft2_error(&FFT2_ERROR_ARG_COUNT)),
    }
}

fn parse_fft2_single(value: &Value) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match value {
        Value::Tensor(tensor) => {
            parse_2d_lengths_from_data(&tensor.data, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_detail(
                    &FFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical).map_err(|source| {
                fft2_error_with_detail(
                    &FFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("logical size-vector conversion failed: {source}"),
                )
            })?;
            parse_2d_lengths_from_data(&tensor.data, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_detail(
                    &FFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::Num(_) | Value::Int(_) => {
            let len = parse_length(value, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(&FFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(fft2_error(&FFT2_ERROR_INVALID_LENGTH));
            }
            let scalar = Value::Num(*re);
            let len = parse_length(&scalar, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(&FFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::ComplexTensor(_) => Err(fft2_error(&FFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::GpuTensor(_) => Err(fft2_error(&FFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::Bool(_) => Err(fft2_error(&FFT2_ERROR_INVALID_LENGTH)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(fft2_error(&FFT2_ERROR_INVALID_LENGTH)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::math::fft::common;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{builtin_function_by_name, IntValue, ResolveContext, Tensor, Type};

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    fn value_to_host_complex(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu fft2 output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[test]
    fn fft2_type_pads_rank() {
        let out = fft2_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn fft2_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fft2 builtin");
        let descriptor = builtin.descriptor.expect("fft2 descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fft2(X)"));
        assert!(labels.contains(&"Y = fft2(X, SIZE)"));
        assert!(labels.contains(&"Y = fft2(X, M, N)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFT2.INVALID_LENGTH"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_matches_sequential_fft() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("fft2");
        let sequential = {
            let complex = value_to_complex_tensor(Value::Tensor(tensor), "fft2").unwrap();
            let first = fft_complex_tensor(complex, None, Some(1)).unwrap();
            fft_complex_tensor(first, None, Some(2)).unwrap()
        };
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, sequential.shape);
                for (lhs, rhs) in out.data.iter().zip(sequential.data.iter()) {
                    assert!(approx_eq(*lhs, *rhs, 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_scalar_length() {
        let tensor = Tensor::new((0..9).map(|v| v as f64).collect(), vec![3, 3]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 4]);
                assert_eq!(out.data.len(), 16);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_size_vector() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(size)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_empty_length_vector() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(empty)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, tensor.shape),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_zero_length_returns_empty() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(3))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft2_builtin(Value::GpuTensor(handle), Vec::new()).expect("fft2 gpu");
            let cpu = fft2_builtin(Value::Tensor(tensor), Vec::new()).expect("fft2 cpu");
            let g = value_to_host_complex(gpu);
            let c = value_to_host_complex(cpu);
            assert_eq!(g.shape, c.shape);
            let tol = 1e-10;
            for (lhs, rhs) in g.data.iter().zip(c.data.iter()) {
                assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_rejects_size_vector_with_more_than_two_entries() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0, 1.0], vec![1, 3]).unwrap();
        let err = error_message(
            fft2_builtin(Value::Tensor(tensor), vec![Value::Tensor(size)]).unwrap_err(),
        );
        assert!(err.contains("fft2"));
        assert!(err.contains("two elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_rejects_boolean_length_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fft2_builtin(Value::Tensor(tensor), vec![Value::Bool(true)]).unwrap_err();
        assert_eq!(error_identifier(&err), FFT2_ERROR_INVALID_LENGTH.identifier);
        assert!(error_message(err).contains(FFT2_ERROR_INVALID_LENGTH.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_mixed_empty_and_length_arguments() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty), Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![2, 4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fft2_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap_err();
        assert_eq!(error_identifier(&err), FFT2_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(FFT2_ERROR_ARG_COUNT.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fft2_wgpu_matches_cpu() {
        let provider = match std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
        }) {
            Ok(Ok(Some(provider))) => provider,
            _ => return,
        };

        let tensor = Tensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).expect("tensor");
        let tensor_cpu = tensor.clone();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value =
            fft2_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft2 gpu");
        let cpu_value = fft2_builtin(Value::Tensor(tensor_cpu), Vec::new()).expect("fft2 cpu");
        let gpu_ct = value_to_host_complex(gpu_value);
        let cpu_ct = value_to_host_complex(cpu_value);
        assert_eq!(gpu_ct.shape, cpu_ct.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (lhs, rhs) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
            assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
        }
        provider.free(&handle).ok();
        runmat_accelerate_api::clear_residency(&handle);
    }

    fn fft2_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fft2_builtin(value, rest))
    }
}
