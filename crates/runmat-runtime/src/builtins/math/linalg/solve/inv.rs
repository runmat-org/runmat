//! MATLAB-compatible `inv` builtin with GPU-aware fallbacks.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderInvOptions};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::matrix_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "inv";
const PROVIDER_UNSUPPORTED: &str = "inv not supported by provider";

const INV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Inverse of A.",
}];

const INV_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input square matrix.",
}];

const INV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = inv(A)",
    inputs: &INV_INPUTS,
    outputs: &INV_OUTPUT,
}];

const INV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INV.INVALID_INPUT",
    identifier: Some("RunMat:inv:InvalidInput"),
    when: "Input shape/type is unsupported or matrix is singular for inversion.",
    message: "inv: invalid input",
};

const INV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INV.INTERNAL",
    identifier: Some("RunMat:inv:Internal"),
    when: "Runtime fails while executing inversion or fallback/upload paths.",
    message: "inv: internal runtime failure",
};

const INV_ERRORS: [BuiltinErrorDescriptor; 2] = [INV_ERROR_INVALID_INPUT, INV_ERROR_INTERNAL];

pub const INV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INV_ERRORS,
};

const INV_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inv-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inv with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InvIntegerInputExtension"),
};
const INV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inv with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InvLogicalInputExtension"),
};
pub const INV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INV_INTEGER_INPUT_EXTENSION, INV_LOGICAL_INPUT_EXTENSION];
const INV_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Every element must be exactly representable before the binary64 matrix-inversion boundary.",
}];
pub const INV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "X = inv(integer_A)",
        inputs: &INV_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only integer matrices cross one checked binary64 boundary. Resident integer input is classified before provider execution and the double result is restored only when the owning provider can represent it truthfully.",
    }];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("inv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("inv")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a native inverse; the reference WGPU backend gathers to the host implementation and re-uploads the result.",
};

fn inv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    inv_error_with_message(message, &INV_ERROR_INVALID_INPUT)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Matrix inversion is a terminal operation and does not participate in fusion pipelines.",
};

#[runtime_builtin(
    name = "inv",
    category = "math/linalg/solve",
    summary = "Compute the inverse of a square matrix.",
    keywords = "inv,matrix inverse,linear solve,gpu",
    accel = "inv",
    type_resolver(matrix_unary_type),
    descriptor(crate::builtins::math::linalg::solve::inv::INV_DESCRIPTOR),
    extensions(crate::builtins::math::linalg::solve::inv::INV_EXTENSIONS),
    integer_capabilities(crate::builtins::math::linalg::solve::inv::INV_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::linalg::solve::inv"
)]
async fn inv_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INV_INTEGER_INPUT_EXTENSION, NAME)?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(builtin_error(
                "inv: integer input must be exactly representable as double",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INV_LOGICAL_INPUT_EXTENSION, NAME)?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
    match value {
        Value::GpuTensor(handle) => inv_gpu(handle).await,
        Value::ComplexTensor(tensor) => inv_complex_value(tensor),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            inv_complex_value(tensor)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            inv_real_value(tensor)
        }
    }
}

async fn inv_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        inv_error_with_message(
            "inv: no acceleration provider owns the input handle",
            &INV_ERROR_INTERNAL,
        )
    })?;
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let expected_storage = runmat_accelerate_api::handle_storage(&handle);
    let expected_precision = runmat_accelerate_api::handle_precision(&handle);
    let provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    if runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
    {
        if !matches!(
            expected_storage,
            runmat_accelerate_api::GpuTensorStorage::Real
                | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        ) || expected_precision != Some(provider.precision())
            || !gpu_helpers::gpu_class_metadata_matches(&handle, expected_precision, None, false)
        {
            return Err(inv_error_with_message(
                "inv: input handle has contradictory floating metadata",
                &INV_ERROR_INTERNAL,
            ));
        }
        let options = ProviderInvOptions::default();
        match provider.inv(&handle, options).await {
            Ok(mut result)
                if valid_inv_gpu_output(
                    &handle,
                    &result,
                    provider,
                    expected_storage,
                    expected_precision,
                ) =>
            {
                gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
                runmat_accelerate_api::set_handle_provenance(&mut result, provenance);
                return Ok(gpu_helpers::resident_gpu_value(result));
            }
            Ok(result) => {
                gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
                gpu_helpers::free_unprotected_exact_owner(&result, &[&handle]);
                return Err(inv_error_with_message(
                    "inv: provider returned an invalid inverse result",
                    &INV_ERROR_INTERNAL,
                ));
            }
            Err(error) if provider_inv_is_unsupported(&error) => {
                gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
            }
            Err(error) => {
                gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
                return Err(build_runtime_error(format!(
                    "inv: provider execution failed: {error}"
                ))
                .with_builtin(NAME)
                .with_identifier("RunMat:inv:Internal")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build());
            }
        }
    }
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered.map_err(map_control_flow)?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered).map_err(builtin_error)?;
    let inv = inv_real_tensor(&tensor)?;
    gpu_helpers::restore_class_preserving_value(&handle, Value::Tensor(inv), NAME)
}

fn provider_inv_is_unsupported(error: &anyhow::Error) -> bool {
    error.to_string() == PROVIDER_UNSUPPORTED
}

fn valid_inv_gpu_output(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_helpers::same_gpu_handle(input, output)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == expected_storage
        && runmat_accelerate_api::handle_precision(output) == expected_precision
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, expected_precision, None, false)
}

fn inv_real_value(tensor: Tensor) -> BuiltinResult<Value> {
    let inv = inv_real_tensor(&tensor)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_complex_value(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let inv = inv_complex_tensor(&tensor)?;
    if inv.numeric_dtype() == NumericDType::F64 && inv.materialize_f64().len() == 1 {
        let (re, im) = inv.materialize_f64()[0];
        Ok(Value::Complex(re, im))
    } else {
        Ok(Value::ComplexTensor(inv))
    }
}

fn inv_real_tensor(matrix: &Tensor) -> BuiltinResult<Tensor> {
    inv_real_tensor_impl(matrix)
}

fn inv_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    inv_complex_tensor_impl(matrix)
}

fn inv_real_tensor_impl(matrix: &Tensor) -> BuiltinResult<Tensor> {
    let output_dtype = if matrix.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return Tensor::new_with_dtype(Vec::new(), matrix.shape.clone(), output_dtype)
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 || cols == 0 {
        return Tensor::new_with_dtype(Vec::new(), matrix.shape.clone(), output_dtype)
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let values = tensor::tensor_values_f64_cow(matrix);
    let dm = DMatrix::from_column_slice(rows, cols, &values);
    let inverse = dm.try_inverse().ok_or_else(|| {
        builtin_error(format!("{NAME}: matrix is singular to working precision."))
    })?;
    matrix_to_tensor(NAME, inverse, &matrix.shape, output_dtype)
}

fn inv_complex_tensor_impl(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let output_dtype = if matrix.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return ComplexTensor::from_f64_values_with_dtype(
            Vec::new(),
            matrix.shape.clone(),
            output_dtype,
        )
        .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 || cols == 0 {
        return ComplexTensor::from_f64_values_with_dtype(
            Vec::new(),
            matrix.shape.clone(),
            output_dtype,
        )
        .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let data: Vec<Complex64> = matrix
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let inverse = dm.try_inverse().ok_or_else(|| {
        builtin_error(format!("{NAME}: matrix is singular to working precision."))
    })?;
    matrix_to_complex_tensor(NAME, inverse, &matrix.shape, output_dtype)
}

fn matrix_dimensions(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => {
            if shape[0] == 1 {
                Ok((1, 1))
            } else {
                Err(builtin_error(format!(
                    "{NAME}: input must be a square matrix."
                )))
            }
        }
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(builtin_error(format!(
                    "{NAME}: inputs must be 2-D matrices."
                )))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

fn matrix_to_tensor(
    label: &str,
    matrix: DMatrix<f64>,
    shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(rows * cols, matrix.len());
    Tensor::new_with_dtype(matrix.as_slice().to_vec(), shape.to_vec(), dtype)
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

fn matrix_to_complex_tensor(
    label: &str,
    matrix: DMatrix<Complex64>,
    shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    debug_assert_eq!(rows * cols, matrix.len());
    ComplexTensor::from_f64_values_with_dtype(data, shape.to_vec(), dtype)
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

/// Host helper used by acceleration providers that delegate `inv` back to the CPU path.
pub fn inv_host_real_for_provider(matrix: &Tensor) -> BuiltinResult<Tensor> {
    inv_real_tensor_impl(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_scalar_num() {
        let result = inv_builtin(Value::Num(4.0)).expect("inv");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn inv_type_preserves_matrix_shape() {
        let out = matrix_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

    #[test]
    fn inv_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = INV_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"X = inv(A)"));
    }

    #[test]
    fn inv_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = INV_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.INV.INVALID_INPUT"));
        assert!(codes.contains(&"RM.INV.INTERNAL"));
    }

    #[test]
    fn inv_provider_unsupported_classification_is_exact() {
        assert!(provider_inv_is_unsupported(&anyhow::anyhow!(
            PROVIDER_UNSUPPORTED
        )));
        assert!(!provider_inv_is_unsupported(&anyhow::anyhow!(
            "inv not supported by provider: device lost"
        )));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_square_matrix() {
        let data = vec![4.0, 1.0, -2.0, 3.0];
        let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let a = DMatrix::from_column_slice(2, 2, &data);
                let inv_m = DMatrix::from_column_slice(2, 2, &out.materialize_f64());
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c { 1.0 } else { 0.0 };
                        assert!((identity[(r, c)] - expected).abs() < 1e-12);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn inv_preserves_native_single_for_real_and_complex_inputs() {
        let real = Tensor::from_f32(vec![4.0, 0.0, 0.0, 2.0], vec![2, 2]).expect("single");
        let Value::Tensor(real_output) = inv_builtin(Value::Tensor(real)).expect("single inv")
        else {
            panic!("expected single tensor")
        };
        assert_eq!(real_output.numeric_dtype(), NumericDType::F32);
        assert_eq!(real_output.materialize_f64(), vec![0.25, 0.0, 0.0, 0.5]);

        let complex =
            ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).expect("complex single");
        let Value::ComplexTensor(complex_output) =
            inv_builtin(Value::ComplexTensor(complex)).expect("complex single inv")
        else {
            panic!("expected complex single tensor")
        };
        assert_eq!(complex_output.numeric_dtype(), NumericDType::F32);
        assert_eq!(complex_output.materialize_f64(), vec![(0.5, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            || Tensor::new_integer(IntegerStorage::U64(vec![4, 1, 2, 3]), vec![2, 2]).unwrap();
        let error = inv_builtin(Value::Tensor(tensor()))
            .expect_err("compatible mode rejects integer matrices");
        assert_eq!(
            error.identifier(),
            INV_INTEGER_INPUT_EXTENSION.error_identifier
        );

        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = inv_builtin(Value::Tensor(tensor())).expect("RunMat integer inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [0.3, -0.1, -0.2, 0.4];
                for (actual, expected) in out.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_empty_matrix_returns_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor.clone())).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert!(out.materialize_f64().is_empty());
                assert_eq!(out.shape, vec![0, 0]);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_trailing_singleton_dimension_preserved() {
        let tensor =
            Tensor::new(vec![4.0, 0.0, 0.0, 2.0], vec![2, 2, 1]).expect("tensor construction");
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2, 1]);
                let expected = vec![0.25, 0.0, 0.0, 0.5];
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_complex_scalar() {
        let result = inv_builtin(Value::Complex(2.0, -1.0)).expect("inv");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 0.0) / Complex64::new(2.0, -1.0);
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_complex_matrix() {
        let raw = vec![(1.0, 2.0), (0.0, 3.0), (0.0, 0.0), (4.0, -1.0)];
        let tensor = ComplexTensor::new(raw.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::ComplexTensor(tensor)).expect("inv");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let input: Vec<Complex64> =
                    raw.iter().map(|&(re, im)| Complex64::new(re, im)).collect();
                let inv_vec: Vec<Complex64> = out
                    .materialize_f64()
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect();
                let a = DMatrix::from_column_slice(2, 2, &input);
                let inv_m = DMatrix::from_column_slice(2, 2, &inv_vec);
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let delta = identity[(r, c)] - expected;
                        assert!(delta.norm() < 1e-10, "identity mismatch at ({r},{c})");
                    }
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_rejects_higher_rank_tensor() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("2-D"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("square matrix"), "{err}");
        assert_eq!(err.identifier(), INV_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_singular_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("singular"), "{err}");
        assert_eq!(err.identifier(), INV_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let cpu = inv_real_tensor(&tensor).expect("cpu");
            assert_eq!(gathered.shape, cpu.shape);
            for (a, b) in gathered
                .materialize_f64()
                .iter()
                .zip(cpu.materialize_f64().iter())
            {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn inv_gpu_output_validation_uses_immutable_expected_metadata() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[2.0],
                    shape: &[1, 1],
                })
                .expect("upload input");
            let output = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[0.5],
                    shape: &[1, 1],
                })
                .expect("upload output");
            let expected_storage = runmat_accelerate_api::handle_storage(&input);
            let expected_precision = runmat_accelerate_api::handle_precision(&input);
            runmat_accelerate_api::set_handle_precision(
                &input,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            assert!(valid_inv_gpu_output(
                &input,
                &output,
                provider,
                expected_storage,
                expected_precision,
            ));
            runmat_accelerate_api::set_handle_class_name(&output, "uint64");
            assert!(!valid_inv_gpu_output(
                &input,
                &output,
                provider,
                expected_storage,
                expected_precision,
            ));
            provider.free(&input).ok();
            provider.free(&output).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_scalar_int_promotes() {
        let error = inv_builtin(Value::Int(IntValue::I32(2)))
            .expect_err("compatible mode rejects integer scalar");
        assert_eq!(
            error.identifier(),
            INV_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = inv_builtin(Value::Int(IntValue::I32(2))).expect("RunMat integer inv");
        match result {
            Value::Num(v) => assert!((v - 0.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn inv_wgpu_matches_cpu() {
        if provider::register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            return;
        }

        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = inv_real_tensor(&tensor).expect("cpu");

        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert!((*a - *b).abs() < tol, "expected {b}, got {a}");
        }
    }

    fn inv_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::inv_builtin(value))
    }
}
