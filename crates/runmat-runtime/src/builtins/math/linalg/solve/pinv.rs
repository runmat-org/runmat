//! MATLAB-compatible `pinv` builtin backed by SVD with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderPinvOptions};
#[cfg(test)]
use runmat_builtins::PINV_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, PINV_ERROR_INTERNAL, PINV_ERROR_INVALID_ARGUMENT,
    PINV_ERROR_INVALID_INPUT, PINV_INTEGER_INPUT_EXTENSION, PINV_INTEGER_TOLERANCE_EXTENSION,
    PINV_LOGICAL_INPUT_EXTENSION, PINV_LOGICAL_TOLERANCE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::linalg::{
    matrix_dimensions_for, parse_tolerance_arg, svd_default_tolerance,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "pinv";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::pinv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pinv",
    op_kind: GpuOpKind::Custom("pinv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("pinv")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a native GPU pseudoinverse; the reference WGPU backend gathers to host SVD and re-uploads the result.",
};

fn pinv_error_with_message(
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
    pinv_error_with_message(message, &PINV_ERROR_INVALID_INPUT)
}

fn argument_error(message: impl Into<String>) -> RuntimeError {
    pinv_error_with_message(message, &PINV_ERROR_INVALID_ARGUMENT)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::pinv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pinv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pseudoinverses are standalone solves and do not participate in fusion plans.",
};

#[runtime_builtin(
    name = "pinv",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::solve::pinv"
)]
async fn pinv_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PINV_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_logical_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PINV_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    for tolerance in &rest {
        if crate::builtins::common::validation::value_has_native_integer_class(tolerance) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PINV_INTEGER_TOLERANCE_EXTENSION,
                NAME,
            )?;
            if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(
                tolerance,
            )
            .await?
            {
                return Err(argument_error(
                    "pinv: integer tolerance must be exactly representable as double",
                ));
            }
        }
        if crate::builtins::common::validation::value_has_logical_class(tolerance) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PINV_LOGICAL_TOLERANCE_EXTENSION,
                NAME,
            )?;
        }
    }
    let tol = parse_tolerance_arg(NAME, rest.view()).map_err(argument_error)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
    if crate::builtins::common::validation::value_has_native_integer_class(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
    {
        return Err(builtin_error(
            "pinv: integer input must be exactly representable as double",
        ));
    }
    match value {
        Value::GpuTensor(handle) => pinv_gpu(handle, tol).await,
        Value::ComplexTensor(t) => pinv_complex_value(t, tol),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            pinv_complex_value(tensor, tol)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            pinv_real_value(tensor, tol)
        }
    }
}

async fn pinv_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        pinv_error_with_message(
            "pinv: resident input has no exact provider owner",
            &PINV_ERROR_INTERNAL,
        )
    })?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    if runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
    {
        let precision = runmat_accelerate_api::handle_precision(&handle);
        if precision.is_none()
            || !gpu_helpers::gpu_class_metadata_matches(&handle, precision, None, false)
        {
            return Err(pinv_error_with_message(
                "pinv: input has contradictory floating metadata",
                &PINV_ERROR_INTERNAL,
            ));
        }
        let options = ProviderPinvOptions { tolerance: tol };
        match provider.pinv(&handle, options).await {
            Ok(result) if valid_pinv_gpu_output(&handle, &result, provider) => {
                let result = gpu_helpers::propagate_owned_output_provenance(result, [&handle])?;
                return gpu_helpers::resident_gpu_value(result);
            }
            Ok(result) => {
                let mut error = pinv_error_with_message(
                    "pinv: provider returned malformed or aliased output",
                    &PINV_ERROR_INTERNAL,
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => {
                let mut failure = pinv_error_with_message(
                    format!("pinv: provider execution failed: {error}"),
                    &PINV_ERROR_INTERNAL,
                );
                failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(crate::runtime_error::with_provider_execution_source(
                    failure, error,
                ));
            }
        }
    }
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(map_control_flow)?;
    let result = match gathered {
        Value::ComplexTensor(tensor) => Value::ComplexTensor(pinv_complex_tensor(&tensor, tol)?),
        Value::Complex(real, imaginary) => {
            let tensor =
                ComplexTensor::new(vec![(real, imaginary)], vec![1, 1]).map_err(builtin_error)?;
            Value::ComplexTensor(pinv_complex_tensor(&tensor, tol)?)
        }
        real => {
            let tensor = tensor::value_into_tensor_for(NAME, real).map_err(builtin_error)?;
            Value::Tensor(pinv_real_tensor(&tensor, tol)?)
        }
    };
    gpu_helpers::restore_class_preserving_value(&handle, result, NAME)
}

fn valid_pinv_gpu_output(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let rows = input.shape().first().copied().unwrap_or(1);
    let cols = input.shape().get(1).copied().unwrap_or(1);
    gpu_helpers::unary_gpu_output_matches_shape(
        output,
        input,
        provider,
        &[cols, rows],
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::handle_storage(input),
            precision: pinv_output_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}

fn pinv_output_precision(
    input: &GpuTensorHandle,
) -> Option<runmat_accelerate_api::ProviderPrecision> {
    if runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
    {
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    } else {
        runmat_accelerate_api::handle_precision(input)
    }
}

fn pinv_real_value(tensor: Tensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let pinv = pinv_real_tensor(&tensor, tol)?;
    Ok(tensor::tensor_into_value(pinv))
}

fn pinv_complex_value(tensor: ComplexTensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let pinv = pinv_complex_tensor(&tensor, tol)?;
    Ok(complex_tensor_into_value(pinv))
}

fn pinv_real_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    pinv_real_tensor_impl(matrix, tol)
}

fn pinv_complex_tensor(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<ComplexTensor> {
    pinv_complex_tensor_impl(matrix, tol)
}

fn pinv_real_tensor_impl(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape()).map_err(builtin_error)?;
    let dtype = matrix.numeric_dtype();
    if rows == 0 || cols == 0 {
        return Tensor::new_with_dtype(
            vec![0.0; cols * rows],
            vec![cols, rows],
            result_dtype(dtype),
        )
        .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let values = tensor::tensor_values_f64_cow(matrix);
    if values.iter().any(|value| !value.is_finite()) {
        return Tensor::new_with_dtype(
            vec![f64::NAN; cols * rows],
            vec![cols, rows],
            result_dtype(dtype),
        )
        .map_err(builtin_error);
    }
    let dm = DMatrix::from_column_slice(rows, cols, &values);
    let pinv = pseudoinverse_real(&dm, tol)?;
    matrix_to_tensor(NAME, pinv, result_dtype(dtype))
}

fn result_dtype(input: NumericDType) -> NumericDType {
    if input == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn pinv_complex_tensor_impl(
    matrix: &ComplexTensor,
    tol: Option<f64>,
) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape()).map_err(builtin_error)?;
    let dtype = result_dtype(matrix.numeric_dtype());
    if rows == 0 || cols == 0 {
        return ComplexTensor::from_f64_values_with_dtype(
            vec![(0.0, 0.0); cols * rows],
            vec![cols, rows],
            dtype,
        )
        .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let source = matrix.materialize_f64();
    if source
        .iter()
        .any(|(re, im)| !re.is_finite() || !im.is_finite())
    {
        return ComplexTensor::from_f64_values_with_dtype(
            vec![(f64::NAN, f64::NAN); cols * rows],
            vec![cols, rows],
            dtype,
        )
        .map_err(builtin_error);
    }
    let data: Vec<Complex64> = source
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let pinv = pseudoinverse_complex(&dm, tol)?;
    matrix_to_complex_tensor(NAME, pinv, dtype)
}

fn pseudoinverse_real(matrix: &DMatrix<f64>, tol: Option<f64>) -> BuiltinResult<DMatrix<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let svd = SVD::new(matrix.clone(), true, true);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    svd.pseudo_inverse(cutoff)
        .map_err(|msg| builtin_error(format!("{NAME}: failed to compute pseudoinverse ({msg})")))
}

fn pseudoinverse_complex(
    matrix: &DMatrix<Complex64>,
    tol: Option<f64>,
) -> BuiltinResult<DMatrix<Complex64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let svd = SVD::new(matrix.clone(), true, true);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    let u = svd.u.ok_or_else(|| {
        builtin_error(format!(
            "{NAME}: failed to compute pseudoinverse (missing U)"
        ))
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        builtin_error(format!(
            "{NAME}: failed to compute pseudoinverse (missing Vᴴ)"
        ))
    })?;
    let mut sigma_plus = DMatrix::<Complex64>::zeros(cols, rows);
    for (idx, value) in svd.singular_values.iter().enumerate() {
        if value.is_infinite() || *value > cutoff {
            sigma_plus[(idx, idx)] = Complex64::new(1.0 / *value, 0.0);
        }
    }
    let v = v_t.adjoint();
    let u_h = u.adjoint();
    Ok(v * sigma_plus * u_h)
}

fn matrix_to_tensor(
    label: &str,
    matrix: DMatrix<f64>,
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new_with_dtype(matrix.as_slice().to_vec(), vec![rows, cols], dtype)
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

fn matrix_to_complex_tensor(
    label: &str,
    matrix: DMatrix<Complex64>,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::from_f64_values_with_dtype(data, vec![rows, cols], dtype)
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

/// Host helper used by acceleration providers that delegate `pinv` back to the CPU path.
pub fn pinv_host_real_for_provider(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    pinv_real_tensor_impl(matrix, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{CharArray, IntValue, IntegerStorage};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn approx_equal(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            assert!(
                (lhs - rhs).abs() <= tol,
                "expected {lhs} ≈ {rhs} within {tol}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_scalar_real() {
        let result = pinv_builtin(Value::Num(4.0), Vec::new()).expect("pinv");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn pinv_preserves_single_class_and_returns_nan_for_nonfinite_input() {
        let real = Tensor::from_f32(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let inverse = pinv_real_tensor(&real, None).unwrap();
        assert_eq!(inverse.numeric_dtype(), NumericDType::F32);
        approx_equal(&inverse.materialize_f64(), &[0.5, 0.0, 0.0, 0.25], 1e-6);

        let complex = ComplexTensor::from_f32(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let inverse = pinv_complex_tensor(&complex, None).unwrap();
        assert_eq!(inverse.numeric_dtype(), NumericDType::F32);

        let bad_real = Tensor::from_f32(vec![f32::INFINITY, 1.0], vec![1, 2]).unwrap();
        let nan_real = pinv_real_tensor(&bad_real, None).unwrap();
        assert_eq!(nan_real.numeric_dtype(), NumericDType::F32);
        assert_eq!(nan_real.shape(), [2, 1]);
        assert!(nan_real
            .as_f32_slice()
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));

        let bad_complex = ComplexTensor::from_f32(vec![(1.0, f32::NAN)], vec![1, 1]).unwrap();
        let nan_complex = pinv_complex_tensor(&bad_complex, None).unwrap();
        assert_eq!(nan_complex.numeric_dtype(), NumericDType::F32);
        assert!(nan_complex
            .materialize_f64()
            .iter()
            .all(|(re, im)| re.is_nan() && im.is_nan()));
    }

    #[test]
    fn pinv_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = PINV_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"X = pinv(A)"));
        assert!(labels.contains(&"X = pinv(A, tol)"));
    }

    #[test]
    fn pinv_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = PINV_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.PINV.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PINV.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PINV.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rank_deficient_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                approx_equal(&out.materialize_f64(), &[0.04, 0.08, 0.08, 0.16], 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_reads_typed_integer_tensor_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![2, 0, 0, 4]), vec![2, 2]).unwrap();

        let result = pinv_builtin(Value::Tensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                approx_equal(&out.materialize_f64(), &[0.5, 0.0, 0.0, 0.25], 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn pinv_resident_integer_gates_then_uses_checked_owner_fallback() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U16(vec![2, 0, 0, 4]), vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = pinv_builtin(Value::GpuTensor(handle.clone()), Vec::new())
                    .expect_err("strict mode must gate resident integer input");
                assert_eq!(
                    err.identifier(),
                    PINV_INTEGER_INPUT_EXTENSION.error_identifier
                );
            }
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let Value::GpuTensor(output) =
                pinv_builtin(Value::GpuTensor(handle), Vec::new()).expect("resident pinv")
            else {
                panic!("resident fallback must restore output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
            approx_equal(&gathered.materialize_f64(), &[0.5, 0.0, 0.0, 0.25], 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rectangular() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![3, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                approx_equal(
                    &out.materialize_f64(),
                    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    1e-12,
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_custom_tolerance_zeroes_small_singular_values() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-12], vec![2, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), vec![Value::Num(1e-6)]).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                approx_equal(&out.materialize_f64(), &[1.0, 0.0, 0.0, 0.0], 1e-9);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_complex_diagonal() {
        let tensor = ComplexTensor::new(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (3.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = pinv_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                let expected = [
                    (0.4, -0.2),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.23076923076923078, 0.15384615384615385),
                ];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual.0 - expected.0).abs() < 1e-12);
                    assert!((actual.1 - expected.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pinv_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu pinv");
            let gathered = test_support::gather(result).expect("gather");
            let cpu = pinv_real_tensor(&tensor, None).expect("cpu pinv");
            approx_equal(&gathered.materialize_f64(), &cpu.materialize_f64(), 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pinv_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            return;
        }
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cpu = pinv_real_tensor(&tensor, None).expect("cpu pinv");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = pinv_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu pinv");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape(), "shape mismatch");

        match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => {
                approx_equal(&gathered.materialize_f64(), &cpu.materialize_f64(), 1e-10);
            }
            runmat_accelerate_api::ProviderPrecision::F32 => {
                approx_equal(&gathered.materialize_f64(), &cpu.materialize_f64(), 5e-5);
            }
        }

        let complex = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let result = pinv_builtin(Value::GpuTensor(source), Vec::new()).expect("complex gpu pinv");
        let Value::GpuTensor(output) = result else {
            panic!("complex pseudoinverse did not restore resident output");
        };
        let gathered = futures::executor::block_on(
            gpu_helpers::download_value_preserving_residency_async(provider, &output),
        )
        .unwrap();
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("complex pseudoinverse lost complex storage");
        };
        let (re, im) = gathered.materialize_f64()[0];
        assert!((re - 0.4).abs() < 1e-5);
        assert!((im + 0.2).abs() < 1e-5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rejects_negative_tolerance() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(-1))]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance"));
        assert_eq!(err.identifier(), PINV_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_accepts_boolean() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result =
            pinv_builtin(Value::Num(4.0), vec![Value::Bool(true)]).expect("pinv with bool tol");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_rejects_non_scalar_tensor() {
        let tol_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::Tensor(tol_tensor)]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance must be a real scalar"));
        assert_eq!(err.identifier(), PINV_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_rejects_char_array() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let chars = CharArray::from_unicode_scalars("hi".chars().collect(), 1, 2).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::CharArray(chars)]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance must be a real scalar"));
        assert_eq!(err.identifier(), PINV_ERROR_INVALID_ARGUMENT.identifier);
    }

    fn pinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pinv_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
}
