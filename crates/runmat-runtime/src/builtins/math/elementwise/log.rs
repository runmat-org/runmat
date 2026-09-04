//! MATLAB-compatible natural logarithm (`log`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise natural logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics, including promotion of negative real values to complex outputs.
//! GPU execution uses provider hooks when available and falls back to host computation whenever
//! complex results are required or the provider lacks a dedicated kernel.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
#[cfg(test)]
use runmat_builtins::LOG_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, LOG_CHARACTER_INPUT_EXTENSION, LOG_ERROR_INTERNAL,
    LOG_ERROR_INVALID_INPUT, LOG_EXPLICIT_GPU_COMPLEX_EXTENSION, LOG_INTEGER_INPUT_EXTENSION,
    LOG_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, NumericStorage, ObjectInstance, StructValue, Tensor,
    Value,
};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::domain_probe::{
    probe_gpu_lower_bound, GpuLowerBoundResult,
};
use crate::builtins::math::symbolic::symbolic_function;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

const IMAG_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log directly on device buffers; runtimes gather to host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real inputs can require complex promotion and explicit gpuArray inputs have a distinct complex-domain contract.",
};

const BUILTIN_NAME: &str = "log";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn log_error_with_detail(
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
    name = "log",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::log"
)]
async fn log_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            log_table(object).await
        }
        other => log_non_table_value(other).await,
    }
}

async fn log_non_table_value(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Log) {
        return Ok(symbolic);
    }
    ensure_log_extensions(&value).await?;
    match value {
        Value::GpuTensor(handle) => log_gpu(handle).await,
        Value::Complex(re, im) => {
            let (r, i) = log_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "log")?;
            log_complex_tensor(ct)
        }
        Value::SparseTensor(_) => Err(log_error_with_detail(
            &LOG_ERROR_INVALID_INPUT,
            "sparse input is not currently supported",
        )),
        Value::CharArray(ca) => log_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(log_error_with_detail(
            &LOG_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => log_real(other),
    }
}

async fn log_table(object: ObjectInstance) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|error| log_error_with_detail(&LOG_ERROR_INVALID_INPUT, error.message()))?;
    let mut output = StructValue::new();
    for (name, value) in variables.fields {
        let transformed = log_non_table_value(value).await.map_err(|error| {
            log_error_with_detail(
                &LOG_ERROR_INVALID_INPUT,
                format!(
                    "table variable {name} does not support log: {}",
                    error.message()
                ),
            )
        })?;
        output.insert(name, transformed);
    }
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|error| log_error_with_detail(&LOG_ERROR_INTERNAL, error.message()))
}

async fn ensure_log_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
        {
            return Err(log_error_with_detail(
                &LOG_ERROR_INVALID_INPUT,
                "integer input lies outside the exact binary64 interval",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn log_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        log_error_with_detail(
            &LOG_ERROR_INTERNAL,
            "GPU provider unavailable for input owner",
        )
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(log_error_with_detail(
            &LOG_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        let gathered =
            gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = log_tensor_real(gathered)?;
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved
    {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        let gathered =
            gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = match gathered {
            Value::ComplexTensor(ct) => log_complex_tensor(ct)?,
            other => log_real(other)?,
        };
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    match probe_gpu_lower_bound(provider, &handle, 0.0)
        .await
        .map_err(|error| log_error_with_detail(&LOG_ERROR_INTERNAL, error.to_string()))?
    {
        GpuLowerBoundResult::Below => {
            if runmat_accelerate_api::handle_is_explicit(&handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            let gathered = gpu_helpers::gather_tensor_async(&handle).await;
            gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
            let gathered =
                gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            let result = log_tensor_real(gathered)?;
            return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
        }
        GpuLowerBoundResult::AtOrAbove => {
            let provider_result = provider.unary_log(&handle).await;
            gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
            match provider_result {
                Ok(output) => return validate_log_gpu_output(provider, &handle, output),
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(log_error_with_detail(
                        &LOG_ERROR_INTERNAL,
                        format!("provider unary_log failed: {error}"),
                    ))
                }
            }
        }
        GpuLowerBoundResult::Unknown => {}
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = log_tensor_real(gathered)?;
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_))
        && runmat_accelerate_api::handle_is_explicit(&handle)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME)
}

fn validate_log_gpu_output(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    out: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let valid = gpu_helpers::unary_gpu_output_matches(
        &out,
        source,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(source),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    );
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&out, &[source]);
        return Err(log_error_with_detail(
            &LOG_ERROR_INTERNAL,
            "provider returned malformed log output",
        ));
    }
    let mut out = out;
    runmat_accelerate_api::set_handle_provenance(
        &mut out,
        runmat_accelerate_api::handle_provenance(source)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    Ok(gpu_helpers::resident_gpu_value(out))
}

fn log_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log", value)
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

fn log_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    match storage {
        NumericStorage::F64(values) => log_real_f64_values(values, shape),
        NumericStorage::F32(values) => log_real_f32_values(values, shape),
        storage => log_real_f64_values(promote_integer_storage_to_log_domain(storage), shape),
    }
}

fn log_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = values.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for v in values {
        let (mut real_part, mut imag_part) = log_complex_parts(v, 0.0);
        if real_part.is_finite() && real_part.abs() < IMAG_EPS {
            real_part = 0.0;
        }
        if !imag_part.is_finite() || imag_part.abs() < IMAG_EPS {
            imag_part = 0.0;
        }
        if imag_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((real_part, imag_part));
    }

    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F64(complex_values.into()), shape)
                .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let data: Vec<f64> = complex_values.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut complex_values = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for value in values {
        let (mut real_part, mut imag_part) = log_complex_parts_f32(value, 0.0);
        if real_part.is_finite() && real_part.abs() < IMAG_EPS as f32 {
            real_part = 0.0;
        }
        if !imag_part.is_finite() || imag_part.abs() < IMAG_EPS as f32 {
            imag_part = 0.0;
        }
        has_imag |= imag_part != 0.0;
        complex_values.push((real_part, imag_part));
    }
    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F32(complex_values.into()), shape)
                .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = complex_values.into_iter().map(|(real, _)| real).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log_complex_parts(real, imag);
                    if real.abs() < IMAG_EPS {
                        real = 0.0;
                    }
                    if imag.abs() < IMAG_EPS {
                        imag = 0.0;
                    }
                    (real, imag)
                })
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log_complex_parts_f32(real, imag);
                    if real.abs() < IMAG_EPS as f32 {
                        real = 0.0;
                    }
                    if imag.abs() < IMAG_EPS as f32 {
                        imag = 0.0;
                    }
                    (real, imag)
                })
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(log_error_with_detail(
                &LOG_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_log_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("log integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn log_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

pub(super) fn log_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let magnitude = re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(re);
        (real_part, imag_part)
    }
}

pub(super) fn log_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let magnitude = re.hypot(im);
    if magnitude == 0.0 {
        (f32::NEG_INFINITY, 0.0)
    } else {
        (magnitude.ln(), im.atan2(re))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

    fn log_builtin(value: Value) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::log_builtin(value))
    }

    #[test]
    fn log_integer_extension_is_rejected_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::log_builtin(Value::Int(IntValue::I64(2))))
            .expect_err("integer log is a RunMat extension");
        assert_eq!(
            error.identifier(),
            LOG_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn log_fusion_is_disabled_until_complex_domain_is_representable() {
        assert!(FUSION_SPEC.elementwise.is_none());
    }

    #[test]
    fn log_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOG_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = log(X)"));
    }

    #[test]
    fn log_string_rejected_with_stable_identifier() {
        let err = log_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), LOG_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_one() {
        let result = log_builtin(Value::Num(1.0)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_zero() {
        let result = log_builtin(Value::Num(0.0)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_negative() {
        let result = log_builtin(Value::Num(-1.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_nan_remains_real() {
        let result = log_builtin(Value::Num(f64::NAN)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected real NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_bool_true() {
        let result = log_builtin(Value::Bool(true)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log_builtin(Value::LogicalArray(logical)).expect("log");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!(
                    t.materialize_f64()[1].is_infinite()
                        && t.materialize_f64()[1].is_sign_negative()
                );
                assert!((t.materialize_f64()[2] - 0.0).abs() < 1e-12);
                assert!(
                    t.materialize_f64()[3].is_infinite()
                        && t.materialize_f64()[3].is_sign_negative()
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_string_input_errors() {
        let err = log_builtin(Value::from("hello")).unwrap_err();
        assert_eq!(err.identifier(), LOG_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.materialize_f64()[0].0 - 0.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![1.0, std::f32::consts::E], vec![2, 1]).unwrap();
        let Value::Tensor(output) = log_builtin(Value::Tensor(tensor)).expect("log") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, std::f32::consts::E.ln()])
        );

        let tensor = Tensor::from_f32(vec![-1.0, 4.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::Tensor(tensor)).expect("log") else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                log_complex_parts_f32(-1.0, 0.0),
                log_complex_parts_f32(4.0, 0.0),
            ])
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::ComplexTensor(complex)).expect("log")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![log_complex_parts_f32(1.0, 1.0)])
        );
        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::ComplexTensor(empty)).expect("log")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn log_maps_table_variables_and_preserves_container_identity() {
        let input = crate::builtins::table::table_from_columns(
            vec!["Double".into(), "Single".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, std::f64::consts::E], vec![2, 1]).unwrap()),
                Value::Tensor(
                    Tensor::from_f32(vec![1.0, std::f32::consts::E], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .unwrap();
        let Value::Object(output) = log_builtin(input).expect("table log") else {
            panic!("expected table");
        };
        assert!(crate::builtins::table::is_tabular_object(&output));
        let variables = crate::builtins::table::table_variables(&output).unwrap();
        assert_eq!(variables.fields.len(), 2);
        assert!(matches!(
            variables.fields.get("Single"),
            Some(Value::Tensor(tensor)) if tensor.numeric_dtype() == runmat_value::NumericDType::F32
        ));
    }

    #[test]
    fn log_integer_gpu_gathers_exact_storage_before_floating_domain() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![1, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::GpuTensor(output) = log_builtin(Value::GpuTensor(handle)).expect("log")
            else {
                panic!("expected restored GPU tensor");
            };
            let Value::Tensor(output) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &output),
            )
            .expect("download log result") else {
                panic!("expected real tensor result");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).ln()])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_complex_scalar() {
        let result = log_builtin(Value::Complex(1.0, 2.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                let expected_re = (1.0_f64.hypot(2.0)).ln();
                let expected_im = 2.0_f64.atan2(1.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log_builtin(Value::CharArray(chars)).expect("log");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - (65.0f64).ln()).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - (90.0f64).ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log_builtin(Value::GpuTensor(handle)).expect("log");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.ln()).collect();
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result) = log_builtin(Value::GpuTensor(handle)).expect("log")
            else {
                panic!("expected restored GPU result");
            };
            let Value::ComplexTensor(ct) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &result),
            )
            .expect("download complex result") else {
                panic!("expected complex tensor");
            };
            assert_eq!(ct.shape, vec![1, 2]);
            assert!((ct.materialize_f64()[0].0 - 0.0).abs() < 1e-12);
            assert!((ct.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_with_integer_argument() {
        let result = log_builtin(Value::Int(IntValue::I32(4))).expect("log");
        match result {
            Value::Num(v) => assert!((v - (4.0f64).ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 4]), vec![3, 1])
            .expect("integer tensor");

        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 2.0f64.ln(), 4.0f64.ln()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log_rejects_integer_storage_outside_exact_binary64_interval() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");
        let error = log_builtin(Value::Tensor(tensor)).expect_err("inexact integer must fail");
        assert_eq!(error.identifier(), LOG_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("exact binary64 interval"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_negative_typed_integer_tensor_promotes_to_complex_from_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 1]), vec![1, 2])
            .expect("integer tensor");

        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64()[0].0, 0.0);
                assert!((out.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (0.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log_wgpu_matches_cpu_elementwise() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log_real(Value::Tensor(tensor.clone())).expect("cpu log");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(log_gpu(handle)).expect("gpu log");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (gpu, cpu) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(ct.materialize_f64().iter())
                {
                    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                    };
                    assert!((gpu - cpu).abs() < tol, "|{gpu} - {cpu}| >= {tol}");
                }
            }
            Value::Num(_) => panic!("expected tensor result from cpu path"),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
