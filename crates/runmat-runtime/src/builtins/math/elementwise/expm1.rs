//! MATLAB-compatible `expm1` builtin with GPU-aware semantics for RunMat.
//!
//! Provides accurate element-wise `exp(x) - 1` for documented floating inputs plus independently
//! gated RunMat integer, logical, and character extensions. GPU fallbacks preserve the owner.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
#[cfg(test)]
use runmat_builtins::EXPM1_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, EXPM1_CHARACTER_INPUT_EXTENSION, EXPM1_ERROR_INTERNAL,
    EXPM1_ERROR_INVALID_INPUT, EXPM1_INTEGER_INPUT_EXTENSION, EXPM1_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, NumericStorage, ObjectInstance,
    SparseTensor, StructValue, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "expm1",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_expm1" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement accurate real floating expm1 directly; complex values and unsupported hooks use an owner-preserving host fallback.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "expm1",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion is intentionally disabled until the fusion ABI can inject the precise expm1 helper without reducing tiny-value accuracy.",
};

const BUILTIN_NAME: &str = "expm1";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn expm1_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "expm1",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::expm1"
)]
async fn expm1_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_expm1_extensions(&value)?;
    expm1_value(value).await
}

async fn expm1_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            expm1_table(object).await
        }
        other => expm1_non_table_value(other).await,
    }
}

async fn expm1_non_table_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => expm1_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = expm1_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "expm1")?;
            expm1_complex_tensor(ct)
        }
        Value::SparseTensor(sparse) => expm1_sparse_tensor(sparse),
        Value::CharArray(ca) => expm1_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(expm1_error_with_detail(
            &EXPM1_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => expm1_real(other),
    }
}

fn ensure_expm1_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPM1_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPM1_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPM1_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn expm1_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        expm1_error_with_detail(&EXPM1_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(expm1_error_with_detail(
            &EXPM1_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        let tensor = gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let output = expm1_tensor(tensor)?;
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
            return Ok(tensor::tensor_into_value(output));
        }
        return restore_real_gpu_output(
            provider,
            &handle,
            &output,
            Some(runmat_accelerate_api::ProviderPrecision::F64),
            input_provenance,
        );
    }
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        let gathered_result =
            gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        let gathered =
            gathered_result.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = expm1_host_value(gathered)?;
        return restore_gpu_value(provider, &handle, result, input_provenance);
    }
    let provider_result = provider.unary_expm1(&handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    match provider_result {
        Ok(mut out)
            if valid_real_gpu_output(
                &out,
                &handle,
                provider,
                runmat_accelerate_api::handle_precision(&handle),
            ) =>
        {
            runmat_accelerate_api::set_handle_provenance(&mut out, input_provenance);
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
        Ok(out) => {
            gpu_helpers::free_unprotected_exact_owner(&out, &[&handle]);
            return Err(expm1_error_with_detail(
                &EXPM1_ERROR_INTERNAL,
                "provider unary_expm1 returned malformed output",
            ));
        }
        Err(err) if gpu_helpers::provider_hook_is_unsupported(&err, "unary_expm1") => {}
        Err(err) => {
            return Err(expm1_error_with_detail(
                &EXPM1_ERROR_INTERNAL,
                format!("provider unary_expm1 failed: {err}"),
            ));
        }
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let tensor = gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let output = expm1_tensor(tensor)?;
    restore_real_gpu_output(
        provider,
        &handle,
        &output,
        runmat_accelerate_api::handle_precision(&handle),
        input_provenance,
    )
}

fn expm1_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("expm1", value)
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(tensor::tensor_into_value(expm1_tensor(tensor)?))
}

fn expm1_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => {
            let (re, im) = expm1_complex_parts(re, im);
            Ok(Value::Complex(re, im))
        }
        Value::ComplexTensor(tensor) => expm1_complex_tensor(tensor),
        other => expm1_real(other),
    }
}

fn expm1_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::exp_m1).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::exp_m1).collect())
        }
        storage => NumericStorage::F64(
            promote_integer_storage_to_expm1_domain(storage)?
                .into_iter()
                .map(f64::exp_m1)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(output, shape).map_err(|e| builtin_error(format!("expm1: {e}")))
}

fn expm1_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| expm1_complex_parts(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| expm1_complex_parts_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(expm1_error_with_detail(
                &EXPM1_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn promote_integer_storage_to_expm1_domain(storage: NumericStorage) -> BuiltinResult<Vec<f64>> {
    let storage = storage
        .into_integer_storage()
        .expect("expm1 integer-promotion boundary received floating storage");
    ensure_integer_storage_exact_binary64(&storage)?;
    Ok(storage.to_f64_vec())
}

fn expm1_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp_m1())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn expm1_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let rows = sparse.rows;
    let cols = sparse.cols;
    let col_ptrs = sparse.col_ptrs.clone();
    let row_indices = sparse.row_indices.clone();
    let output = if let Some(values) = sparse.as_f64_slice() {
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.iter().copied().map(f64::exp_m1).collect(),
        )
    } else if let Some(values) = sparse.as_f32_slice() {
        SparseTensor::new_f32(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.iter().copied().map(f32::exp_m1).collect(),
        )
    } else if sparse.is_logical() {
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            vec![1.0_f64.exp_m1(); sparse.nnz()],
        )
    } else if let Some(storage) = sparse.integer_storage() {
        ensure_integer_storage_exact_binary64(storage)?;
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage.to_f64_vec().into_iter().map(f64::exp_m1).collect(),
        )
    } else {
        return Err(expm1_error_with_detail(
            &EXPM1_ERROR_INVALID_INPUT,
            "unsupported sparse storage",
        ));
    }
    .map_err(|err| expm1_error_with_detail(&EXPM1_ERROR_INTERNAL, err))?;
    Ok(Value::SparseTensor(output))
}

async fn expm1_table(object: ObjectInstance) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|err| expm1_error_with_detail(&EXPM1_ERROR_INVALID_INPUT, err.message))?;
    let mut output = StructValue::new();
    for (name, value) in variables.fields {
        ensure_expm1_extensions(&value)?;
        if matches!(value, Value::Object(_)) {
            return Err(expm1_error_with_detail(
                &EXPM1_ERROR_INVALID_INPUT,
                format!("table variable {name} does not support expm1"),
            ));
        }
        output.insert(name, expm1_non_table_value(value).await?);
    }
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|err| expm1_error_with_detail(&EXPM1_ERROR_INTERNAL, err.message))
}

const MAX_EXACT_BINARY64_INTEGER: i128 = 9_007_199_254_740_992;

fn ensure_integer_storage_exact_binary64(storage: &IntegerStorage) -> BuiltinResult<()> {
    let valid = match storage {
        IntegerStorage::I8(_) | IntegerStorage::I16(_) | IntegerStorage::I32(_) => true,
        IntegerStorage::I64(values) => values.iter().all(|&value| {
            let value = i128::from(value);
            (-MAX_EXACT_BINARY64_INTEGER..=MAX_EXACT_BINARY64_INTEGER).contains(&value)
        }),
        IntegerStorage::U8(_) | IntegerStorage::U16(_) | IntegerStorage::U32(_) => true,
        IntegerStorage::U64(values) => values
            .iter()
            .all(|&value| u128::from(value) <= MAX_EXACT_BINARY64_INTEGER as u128),
    };
    if valid {
        Ok(())
    } else {
        Err(expm1_error_with_detail(
            &EXPM1_ERROR_INVALID_INPUT,
            "integer input lies outside the inclusive exact binary64 interval [-2^53, 2^53]",
        ))
    }
}

fn restore_real_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    tensor: &Tensor,
    precision: Option<runmat_accelerate_api::ProviderPrecision>,
    provenance: runmat_accelerate_api::GpuHandleProvenance,
) -> BuiltinResult<Value> {
    let mut output = gpu_helpers::upload_tensor(provider, tensor).map_err(|err| {
        expm1_error_with_detail(
            &EXPM1_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {err}"),
        )
    })?;
    if !valid_real_gpu_output(&output, input, provider, precision) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
        return Err(expm1_error_with_detail(
            &EXPM1_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn restore_gpu_value(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
    provenance: runmat_accelerate_api::GpuHandleProvenance,
) -> BuiltinResult<Value> {
    match value {
        Value::ComplexTensor(tensor) => {
            let mut output =
                gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|err| {
                    expm1_error_with_detail(
                        &EXPM1_ERROR_INTERNAL,
                        format!("failed to restore complex result to input provider: {err}"),
                    )
                })?;
            if !valid_complex_gpu_output(&output, input, provider) {
                gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
                return Err(expm1_error_with_detail(
                    &EXPM1_ERROR_INTERNAL,
                    "provider upload returned malformed complex fallback output",
                ));
            }
            runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
            Ok(gpu_helpers::complex_gpu_value(output))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], input.shape.clone())
                .map_err(|err| expm1_error_with_detail(&EXPM1_ERROR_INTERNAL, err))?;
            restore_gpu_value(provider, input, Value::ComplexTensor(tensor), provenance)
        }
        Value::Tensor(tensor) => restore_real_gpu_output(
            provider,
            input,
            &tensor,
            runmat_accelerate_api::handle_precision(input),
            provenance,
        ),
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], input.shape.clone())
                .map_err(|err| expm1_error_with_detail(&EXPM1_ERROR_INTERNAL, err))?;
            restore_real_gpu_output(
                provider,
                input,
                &tensor,
                runmat_accelerate_api::handle_precision(input),
                provenance,
            )
        }
        other => Err(expm1_error_with_detail(
            &EXPM1_ERROR_INTERNAL,
            format!("unexpected host fallback result {other:?}"),
        )),
    }
}

fn valid_real_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    precision: Option<runmat_accelerate_api::ProviderPrecision>,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn valid_complex_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::ComplexInterleaved,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn expm1_complex_parts(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        return (re.exp_m1(), im);
    }
    if !re.is_finite() || !im.is_finite() {
        let exp_re = re.exp();
        return (exp_re * im.cos() - 1.0, exp_re * im.sin());
    }
    let half = 0.5 * im;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let cos_b_minus_one = -2.0 * sin_half * sin_half;
    let sin_b = 2.0 * sin_half * cos_half;
    let expm1_a = re.exp_m1();
    let exp_a = expm1_a + 1.0;
    let real = expm1_a + exp_a * cos_b_minus_one;
    let imag = exp_a * sin_b;
    (real, imag)
}

fn expm1_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    if im == 0.0 {
        return (re.exp_m1(), im);
    }
    if !re.is_finite() || !im.is_finite() {
        let exp_re = re.exp();
        return (exp_re * im.cos() - 1.0, exp_re * im.sin());
    }
    let half = 0.5 * im;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let cos_b_minus_one = -2.0 * sin_half * sin_half;
    let sin_b = 2.0 * sin_half * cos_half;
    let expm1_a = re.exp_m1();
    let exp_a = expm1_a + 1.0;
    let real = expm1_a + exp_a * cos_b_minus_one;
    let imag = exp_a * sin_b;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn expm1_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::expm1_builtin(value))
    }

    fn expm1_builtin_matlab(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        block_on(super::expm1_builtin(value))
    }

    #[test]
    fn expm1_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = EXPM1_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = expm1(X)"));
    }

    #[test]
    fn expm1_string_rejected_with_stable_identifier() {
        let err = expm1_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), EXPM1_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_scalar_zero() {
        let result = expm1_builtin(Value::Num(0.0)).expect("expm1");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_scalar_small_matches_high_precision() {
        let input = 1.0e-16;
        let result = expm1_builtin(Value::Num(input)).expect("expm1");
        match result {
            Value::Num(v) => {
                let naive = input.exp() - 1.0;
                let delta_precise = v - input;
                let delta_naive = naive - input;
                assert!(delta_precise.abs() <= delta_naive.abs());
                assert!(delta_precise.abs() < 1e-28);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0], vec![3, 1]).unwrap();
        let result = expm1_builtin(Value::Tensor(tensor)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.exp_m1(), (-1.0f64).exp_m1()];
                for (out, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((out - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_preserves_native_single_real_complex_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) = expm1_builtin(Value::Tensor(tensor)).expect("expm1") else {
            panic!("expected single tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 1.0_f32.exp_m1()])
        );

        let complex = ComplexTensor::from_f32(vec![(0.0, 0.0), (1.0, 0.5)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) =
            expm1_builtin(Value::ComplexTensor(complex)).expect("expm1")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                expm1_complex_parts_f32(0.0, 0.0),
                expm1_complex_parts_f32(1.0, 0.5),
            ])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) =
            expm1_builtin(Value::ComplexTensor(empty)).expect("expm1")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn expm1_integer_gpu_rejects_inexact_binary64_value_after_exact_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let err = expm1_builtin(Value::GpuTensor(handle)).expect_err("inexact integer");
            assert_eq!(err.identifier(), EXPM1_ERROR_INVALID_INPUT.identifier);
        });
    }

    #[test]
    fn expm1_resident_integer_stays_host_double_on_f32_owner() {
        test_support::with_f32_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let Value::Tensor(output) = expm1_builtin(Value::GpuTensor(handle)).expect("expm1")
            else {
                panic!("expected host double output");
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.materialize_f64(), &[0.0, 1.0_f64.exp_m1()]);
        });
    }

    #[test]
    fn expm1_independent_extensions_are_gated_with_stable_identifiers() {
        let err = expm1_builtin_matlab(Value::Int(IntValue::I8(1))).expect_err("integer gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Expm1IntegerInputExtension")
        );
        let err = expm1_builtin_matlab(Value::Bool(true)).expect_err("logical gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Expm1LogicalInputExtension")
        );
        let chars = CharArray::new(vec!['A'], 1, 1).unwrap();
        let err = expm1_builtin_matlab(Value::CharArray(chars)).expect_err("character gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Expm1CharacterInputExtension")
        );
    }

    #[test]
    fn expm1_accepts_all_integer_classes_and_exact_endpoints() {
        let storages = vec![
            IntegerStorage::I8(vec![-1, 1]),
            IntegerStorage::I16(vec![-1, 1]),
            IntegerStorage::I32(vec![-1, 1]),
            IntegerStorage::I64(vec![-9_007_199_254_740_992, 9_007_199_254_740_992]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_992]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            let Value::Tensor(output) =
                expm1_builtin(Value::Tensor(tensor)).expect("integer expm1")
            else {
                panic!("expected double tensor");
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
        }
        for storage in [
            IntegerStorage::I64(vec![-9_007_199_254_740_993]),
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let err = expm1_builtin(Value::Tensor(tensor)).expect_err("outside exact interval");
            assert_eq!(err.identifier(), EXPM1_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
            .expect("integer tensor");

        let result = expm1_builtin(Value::Tensor(tensor)).expect("expm1");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.exp_m1(), 2.0f64.exp_m1()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_int_promotes() {
        let result = expm1_builtin(Value::Int(IntValue::I32(1))).expect("expm1");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.exp_m1()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_complex_scalar() {
        let result = expm1_builtin(Value::Complex(1.0, 1.0)).expect("expm1");
        match result {
            Value::Complex(re, im) => {
                let exp_a = 1.0f64.exp();
                let expected_re = exp_a * 1.0f64.cos() - 1.0;
                let expected_im = exp_a * 1.0f64.sin();
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = expm1_builtin(Value::CharArray(chars)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).exp_m1();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_string_rejects() {
        let err = expm1_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = expm1_builtin(Value::GpuTensor(handle)).expect("expm1");
            assert!(matches!(&result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor
                .materialize_f64()
                .iter()
                .map(|&v| v.exp_m1())
                .collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn expm1_gpu_output_contract_rejects_alias_shape_and_storage_metadata() {
        test_support::with_test_provider(|provider| {
            let input = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            let mut output = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            let precision = runmat_accelerate_api::handle_precision(&input);
            assert!(valid_real_gpu_output(&output, &input, provider, precision));
            assert!(!valid_real_gpu_output(&input, &input, provider, precision));
            let mut wrong_shape = output.clone();
            wrong_shape.shape = vec![1, 2];
            assert!(!valid_real_gpu_output(
                &wrong_shape,
                &input,
                provider,
                precision
            ));
            output.descriptor.storage =
                Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
            assert!(!valid_real_gpu_output(&output, &input, provider, precision));
        });
    }

    #[test]
    fn expm1_sparse_preserves_sparse_single_storage_and_implicit_zeros() {
        let sparse =
            SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let Value::SparseTensor(output) =
            expm1_builtin(Value::SparseTensor(sparse)).expect("sparse expm1")
        else {
            panic!("expm1 must preserve sparse storage");
        };
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 2);
        assert_eq!(
            output.as_f32_slice(),
            Some(&[1.0_f32.exp_m1(), 2.0_f32.exp_m1()][..])
        );
    }

    #[test]
    fn expm1_table_maps_supported_variables_and_preserves_container() {
        let input = crate::builtins::table::table_from_columns(
            vec!["Double".into(), "Single".into()],
            vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let Value::Object(output) = expm1_builtin(input).expect("table expm1") else {
            panic!("expected table");
        };
        let variables = crate::builtins::table::table_variables(&output).unwrap();
        assert_eq!(variables.fields.len(), 2);
        assert!(matches!(
            variables.fields.get("Single"),
            Some(Value::Tensor(tensor)) if tensor.numeric_dtype() == runmat_value::NumericDType::F32
        ));
    }

    #[test]
    fn expm1_complex_real_axis_preserves_signed_zero_at_infinite_endpoint() {
        let Value::Complex(re, im) = expm1_builtin(Value::Complex(f64::INFINITY, -0.0)).unwrap()
        else {
            panic!("expected complex scalar");
        };
        assert_eq!(re, f64::INFINITY);
        assert_eq!(im, 0.0);
        assert!(im.is_sign_negative());
        let Value::Complex(re, im) = expm1_builtin(Value::Complex(f64::NAN, 0.0)).unwrap() else {
            panic!("expected complex scalar");
        };
        assert!(re.is_nan());
        assert_eq!(im, 0.0);
    }

    #[test]
    fn expm1_gpu_preserves_tiny_value_and_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0e-16], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = expm1_builtin(Value::GpuTensor(handle)).expect("expm1");
            assert!(matches!(&result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64()[0], 1.0e-16_f64.exp_m1());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn expm1_wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let t = Tensor::new(vec![0.0, -0.5, 0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = expm1_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(expm1_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
