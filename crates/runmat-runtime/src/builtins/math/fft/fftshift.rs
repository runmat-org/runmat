//! MATLAB-compatible `fftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `fftshift` recenters zero-frequency components for outputs produced by FFTs.

use super::common::{apply_shift, build_shift_plan, compute_shift_dims, ShiftKind};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::fftshift_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftshift",
    op_kind: GpuOpKind::Custom("fftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider circshift kernels when available; otherwise gathers once and shifts on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not currently fused; treated as an explicit data shuffling operation.",
};

const BUILTIN_NAME: &str = "fftshift";

const FFTSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted array with the same size and type family as X.",
}];

const FFTSHIFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, complex, logical, or gpuArray input.",
}];

const FFTSHIFT_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, complex, logical, or gpuArray input.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector (scalar, numeric vector, or logical mask vector).",
    },
];

const FFTSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X)",
        inputs: &FFTSHIFT_INPUTS_CORE,
        outputs: &FFTSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X, DIM)",
        inputs: &FFTSHIFT_INPUTS_DIMS,
        outputs: &FFTSHIFT_OUTPUT,
    },
];

const FFTSHIFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.ARG_COUNT",
    identifier: Some("RunMat:fftshift:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftshift: invalid argument count",
};

const FFTSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:fftshift:InvalidDimensions"),
    when: "DIM argument is malformed or out of range.",
    message: "fftshift: invalid dimension argument",
};

const FFTSHIFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_INPUT",
    identifier: Some("RunMat:fftshift:InvalidInput"),
    when: "X is not a supported numeric/logical input type.",
    message: "fftshift: expected numeric or logical input",
};

const FFTSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:fftshift:UnsupportedInput"),
    when: "X is an unsupported object/cell/function/meta runtime type.",
    message: "fftshift: unsupported input type",
};

const FFTSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INTERNAL",
    identifier: Some("RunMat:fftshift:Internal"),
    when: "Shifting, tensor reconstruction, or GPU transfer operations fail.",
    message: "fftshift: internal error",
};

const FFTSHIFT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFTSHIFT_ERROR_ARG_COUNT,
    FFTSHIFT_ERROR_INVALID_DIMS,
    FFTSHIFT_ERROR_INVALID_INPUT,
    FFTSHIFT_ERROR_UNSUPPORTED_INPUT,
    FFTSHIFT_ERROR_INTERNAL,
];

pub const FFTSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTSHIFT_ERRORS,
};

fn fftshift_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fftshift_error_with_message(error.message, error)
}

fn fftshift_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fftshift_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fftshift_error_with_source(
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

fn fftshift_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn compute_fftshift_dims(shape: &[usize], dims_arg: Option<&Value>) -> BuiltinResult<Vec<usize>> {
    compute_shift_dims(shape, dims_arg, BUILTIN_NAME).map_err(|source| {
        fftshift_error_with_source(
            &FFTSHIFT_ERROR_INVALID_DIMS,
            "dimension parsing failed",
            source,
        )
    })
}

#[runtime_builtin(
    name = "fftshift",
    category = "math/fft",
    summary = "Shift zero-frequency components to spectrum centers.",
    keywords = "fftshift,fourier transform,frequency centering,spectrum,gpu",
    accel = "custom",
    type_resolver(fftshift_type),
    descriptor(crate::builtins::math::fft::fftshift::FFTSHIFT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::fft::fftshift"
)]
async fn fftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fftshift_error(&FFTSHIFT_ERROR_ARG_COUNT));
    }
    let dims_arg = rest.first();

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_fftshift_dims(&ct.shape, dims_arg)?;
            Ok(fftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_fftshift_dims(&array.shape, dims_arg)?;
            Ok(fftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|source| {
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INTERNAL,
                    format!("complex tensor construction failed: {source}"),
                )
            })?;
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_complex_tensor(tensor, &dims).map(|result| {
                if result.data.len() == 1 {
                    let (r, i) = result.data[0];
                    Value::Complex(r, i)
                } else {
                    Value::ComplexTensor(result)
                }
            })?)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(|detail| {
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INVALID_INPUT,
                    format!("scalar/tensor conversion failed: {detail}"),
                )
            })?;
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_fftshift_dims(&handle.shape, dims_arg)?;
            Ok(fftshift_gpu(handle, &dims).await?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            Err(fftshift_error(&FFTSHIFT_ERROR_INVALID_INPUT))
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            Err(fftshift_error(&FFTSHIFT_ERROR_INVALID_INPUT))
        }
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::SparseTensor(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(fftshift_error(&FFTSHIFT_ERROR_UNSUPPORTED_INPUT)),
    }
}

fn fftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|source| {
            fftshift_error_with_detail(
                &FFTSHIFT_ERROR_INTERNAL,
                format!("tensor reconstruction failed: {source}"),
            )
        });
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    Tensor::new(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("tensor reconstruction failed: {source}"),
        )
    })
}

fn fftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape).map_err(|source| {
            fftshift_error_with_detail(
                &FFTSHIFT_ERROR_INTERNAL,
                format!("complex tensor reconstruction failed: {source}"),
            )
        });
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    ComplexTensor::new(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("complex tensor reconstruction failed: {source}"),
        )
    })
}

fn fftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape).map_err(|source| {
            fftshift_error_with_detail(
                &FFTSHIFT_ERROR_INTERNAL,
                format!("logical array reconstruction failed: {source}"),
            )
        });
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("logical array reconstruction failed: {source}"),
        )
    })
}

async fn fftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Fft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return fftshift_gpu_fallback(handle, dims).await,
            }
        }
        if let Ok(mut out) = provider.circshift(&working, &plan.provider) {
            if plan.ext_shape != handle.shape {
                match provider.reshape(&out, &handle.shape) {
                    Ok(restored) => out = restored,
                    Err(_) => {
                        let mut coerced = out.clone();
                        coerced.shape = handle.shape.clone();
                        out = coerced;
                    }
                }
            }
            return Ok(Value::GpuTensor(out));
        }
    }

    fftshift_gpu_fallback(handle, dims).await
}

async fn fftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let host_tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            fftshift_error_with_source(&FFTSHIFT_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    let shifted = fftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &shifted.data,
            shape: &shifted.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|source| {
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INTERNAL,
                    format!("gpu upload failed: {source}"),
                )
            });
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        builtin_function_by_name, ComplexTensor, IntValue, LogicalArray, ResolveContext, Tensor,
        Type,
    };

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn fftshift_type_preserves_tensor_shape() {
        let out = fftshift_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(5)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(5)])
            }
        );
    }

    #[test]
    fn fftshift_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftshift builtin");
        let descriptor = builtin.descriptor.expect("fftshift descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fftshift(X)"));
        assert!(labels.contains(&"Y = fftshift(X, DIM)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFTSHIFT.INVALID_DIMS"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![8, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_columns_only_via_vector_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![3.0, 6.0, 1.0, 4.0, 2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only_logical_mask() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_all_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![6.0, 3.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_with_empty_dimension_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_dimension_beyond_rank_is_ignored() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_logical_array_input_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result = fftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("fftshift");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = fftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(
                    out.data,
                    vec![(2.0, 2.0), (3.0, 3.0), (0.0, 0.0), (1.0, 1.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_complex_scalar_passthrough() {
        let result = fftshift_builtin(Value::Complex(1.0, -2.0), Vec::new()).expect("fftshift");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_zero_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_integer_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_vector_dimension_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(gathered.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_gpu_with_explicit_dims() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let dims = Value::Int(IntValue::I32(1));
            let result = fftshift_builtin(Value::GpuTensor(handle), vec![dims]).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(gathered.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu =
            fftshift_tensor(tensor.clone(), &(0..tensor.shape.len()).collect::<Vec<_>>()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    fn fftshift_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fftshift_builtin(value, rest))
    }
}
