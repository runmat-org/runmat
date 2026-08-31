//! MATLAB-compatible `log1p` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `log(1 + x)` with improved accuracy for small magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation whenever complex results are required or device support is
//! missing, mirroring MATLAB behavior.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
#[cfg(test)]
use runmat_builtins::LOG1P_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, LOG1P_CHARACTER_INPUT_EXTENSION, LOG1P_ERROR_INTERNAL,
    LOG1P_ERROR_INVALID_INPUT, LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION, LOG1P_INTEGER_INPUT_EXTENSION,
    LOG1P_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::IntValue;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, NumericStorage, Tensor, Value,
};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::logarithm_common::{
    probe_gpu_complex_requirement, GpuComplexRequirement,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log1p",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log1p" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers should supply unary_log1p and reduce_min; runtimes gather to host when complex outputs are required or either hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log1p",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because raw log(x+1) loses log1p accuracy and real inputs can require complex promotion.",
};

const BUILTIN_NAME: &str = "log1p";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn log1p_error_with_detail(
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
    name = "log1p",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::log1p"
)]
async fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_log1p_extensions(&value)?;
    match value {
        Value::GpuTensor(handle) => log1p_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = log1p_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "log1p")?;
            log1p_complex_tensor(ct)
        }
        Value::SparseTensor(_) => Err(log1p_error_with_detail(
            &LOG1P_ERROR_INVALID_INPUT,
            "sparse input is not currently supported",
        )),
        Value::CharArray(ca) => log1p_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(log1p_error_with_detail(
            &LOG1P_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => log1p_real(other),
    }
}

fn ensure_log1p_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn log1p_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        log1p_error_with_detail(
            &LOG1P_ERROR_INTERNAL,
            "GPU provider unavailable for input owner",
        )
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(log1p_error_with_detail(
            &LOG1P_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        let tensor = gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = log1p_tensor(tensor)?;
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
            Value::ComplexTensor(ct) => log1p_complex_tensor(ct)?,
            other => log1p_real(other)?,
        };
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    match detect_gpu_complex_requirement(provider, &handle).await? {
        GpuComplexRequirement::Required => {
            if runmat_accelerate_api::handle_is_explicit(&handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            let gathered = gpu_helpers::gather_tensor_async(&handle).await;
            gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
            let gathered =
                gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            let result = log1p_real(Value::Tensor(gathered))?;
            return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
        }
        GpuComplexRequirement::NotRequired => {
            let provider_result = provider.unary_log1p(&handle).await;
            gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
            match provider_result {
                Ok(output) => return validate_log1p_gpu_output(provider, &handle, output),
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(log1p_error_with_detail(
                        &LOG1P_ERROR_INTERNAL,
                        format!("provider unary_log1p failed: {error}"),
                    ))
                }
            }
        }
        GpuComplexRequirement::Unknown => {}
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered.map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = log1p_real(Value::Tensor(gathered))?;
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_))
        && runmat_accelerate_api::handle_is_explicit(&handle)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME)
}

fn validate_log1p_gpu_output(
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
        return Err(log1p_error_with_detail(
            &LOG1P_ERROR_INTERNAL,
            "provider returned malformed log1p output",
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

async fn detect_gpu_complex_requirement(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<GpuComplexRequirement> {
    probe_gpu_complex_requirement(provider, handle, -1.0)
        .await
        .map_err(|error| log1p_error_with_detail(&LOG1P_ERROR_INTERNAL, error.to_string()))
}

fn log1p_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log1p", value)
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    match storage {
        NumericStorage::F64(values) => log1p_real_f64_values(values, shape),
        NumericStorage::F32(values) => log1p_real_f32_values(values, shape),
        storage => log1p_real_f64_values(promote_integer_storage_to_log1p_domain(storage)?, shape),
    }
}

fn log1p_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    let mut has_imag = false;

    for v in values {
        if v.is_nan() {
            entries.push((f64::NAN, 0.0));
            continue;
        }
        if v < -1.0 {
            let (mut real_part, mut imag_part) = log1p_complex_parts(v, 0.0);
            if real_part.abs() < IMAG_EPS {
                real_part = 0.0;
            }
            if imag_part.abs() < IMAG_EPS {
                imag_part = 0.0;
            }
            if imag_part != 0.0 {
                has_imag = true;
            }
            entries.push((real_part, imag_part));
        } else {
            entries.push((v.ln_1p(), 0.0));
        }
    }

    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F64(entries.into()), shape)
                .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let data: Vec<f64> = entries.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for value in values {
        if value.is_nan() {
            entries.push((f32::NAN, 0.0));
        } else if value < -1.0 {
            let (mut real, mut imag) = log1p_complex_parts_f32(value, 0.0);
            if real.abs() < IMAG_EPS as f32 {
                real = 0.0;
            }
            if imag.abs() < IMAG_EPS as f32 {
                imag = 0.0;
            }
            has_imag |= imag != 0.0;
            entries.push((real, imag));
        } else {
            entries.push((value.ln_1p(), 0.0));
        }
    }
    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F32(entries.into()), shape)
                .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = entries.into_iter().map(|(real, _)| real).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log1p_complex_parts(real, imag);
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
                    let (mut real, mut imag) = log1p_complex_parts_f32(real, imag);
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
            return Err(log1p_error_with_detail(
                &LOG1P_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_log1p_domain(storage: NumericStorage) -> BuiltinResult<Vec<f64>> {
    let storage = storage
        .into_integer_storage()
        .expect("log1p integer-promotion boundary received floating storage");
    ensure_integer_storage_exact_binary64(&storage)?;
    Ok(storage.to_f64_vec())
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
        Err(log1p_error_with_detail(
            &LOG1P_ERROR_INVALID_INPUT,
            "integer input lies outside the inclusive exact binary64 interval [-2^53, 2^53]",
        ))
    }
}

fn log1p_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = if re.abs() < 0.5 && im.abs() < 0.5 {
            0.5 * (2.0 * re + re * re + im * im).ln_1p()
        } else {
            magnitude.ln()
        };
        let imag_part = im.atan2(shifted_re);
        (real_part, imag_part)
    }
}

fn log1p_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f32::NEG_INFINITY, 0.0)
    } else {
        let real = if re.abs() < 0.5 && im.abs() < 0.5 {
            0.5 * (2.0 * re + re * re + im * im).ln_1p()
        } else {
            magnitude.ln()
        };
        (real, im.atan2(shifted_re))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};
    use std::f64::consts::PI;

    fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::log1p_builtin(value))
    }

    #[test]
    fn log1p_integer_extension_is_rejected_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::log1p_builtin(Value::Int(IntValue::I64(1))))
            .expect_err("integer log1p is a RunMat extension");
        assert_eq!(
            error.identifier(),
            LOG1P_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn log1p_fusion_is_disabled_to_preserve_near_zero_accuracy() {
        assert!(FUSION_SPEC.elementwise.is_none());
    }

    #[test]
    fn log1p_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOG1P_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = log1p(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 3]), vec![3, 1])
            .expect("integer tensor");

        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_less_than_negative_one_typed_integer_promotes_to_complex_from_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-2, 0]), vec![1, 2])
            .expect("integer tensor");

        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
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

    #[test]
    fn log1p_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 0.5], vec![2, 1]).unwrap();
        let Value::Tensor(output) = log1p_builtin(Value::Tensor(tensor)).expect("log1p") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 0.5_f32.ln_1p()])
        );

        let tensor = Tensor::from_f32(vec![-2.0, 1.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = log1p_builtin(Value::Tensor(tensor)).expect("log1p")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                log1p_complex_parts_f32(-2.0, 0.0),
                log1p_complex_parts_f32(1.0, 0.0),
            ])
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) =
            log1p_builtin(Value::ComplexTensor(complex)).expect("log1p")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![log1p_complex_parts_f32(1.0, 1.0)])
        );
        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) =
            log1p_builtin(Value::ComplexTensor(empty)).expect("log1p")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn log1p_integer_gpu_gathers_exact_storage_before_floating_domain() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::GpuTensor(output) = log1p_builtin(Value::GpuTensor(handle)).expect("log1p")
            else {
                panic!("expected restored GPU tensor");
            };
            let Value::Tensor(output) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &output),
            )
            .expect("download log1p result") else {
                panic!("expected real tensor result");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).ln_1p()])
            );
        });
    }

    #[test]
    fn log1p_integer_gpu_rejects_inexact_binary64_value_after_exact_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let error = log1p_builtin(Value::GpuTensor(handle)).expect_err("inexact integer");
            assert_eq!(error.identifier(), LOG1P_ERROR_INVALID_INPUT.identifier);
        });
    }

    #[test]
    fn log1p_resident_integer_stays_host_double_on_f32_owner() {
        test_support::with_f32_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let Value::Tensor(output) = log1p_builtin(Value::GpuTensor(handle)).expect("log1p")
            else {
                panic!("expected host double output");
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.materialize_f64(), &[0.0, 1.0_f64.ln_1p()]);
        });
    }

    #[test]
    fn log1p_accepts_all_integer_classes_and_exact_endpoints() {
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
            let result = log1p_builtin(Value::Tensor(tensor)).expect("integer log1p");
            assert!(matches!(result, Value::Tensor(_) | Value::ComplexTensor(_)));
        }
        for storage in [
            IntegerStorage::I64(vec![-9_007_199_254_740_993]),
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let error = log1p_builtin(Value::Tensor(tensor)).expect_err("outside exact interval");
            assert_eq!(error.identifier(), LOG1P_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn log1p_string_rejected_with_stable_identifier() {
        let err = log1p_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), LOG1P_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_zero() {
        let result = log1p_builtin(Value::Num(0.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_negative_one() {
        let result = log1p_builtin(Value::Num(-1.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_less_than_negative_one_complex() {
        let result = log1p_builtin(Value::Num(-2.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_tensor_mixed_values() {
        let tensor = Tensor::new(vec![0.0, -0.5, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                let expected = [
                    (0.0, 0.0),
                    ((0.5f64).ln(), 0.0),
                    (0.0, PI),
                    ((4.0f64).ln(), 0.0),
                ];
                for ((re, im), (er, ei)) in ct.materialize_f64().iter().zip(expected.iter()) {
                    assert!((re - er).abs() < 1e-12);
                    assert!((im - ei).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_complex_input() {
        let result = log1p_builtin(Value::Complex(0.5, 1.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.5f64.hypot(1.0).ln(), 1.0f64.atan2(1.5));
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_complex_input_retains_near_zero_accuracy() {
        let input = 1.0e-20;
        let Value::Complex(real, imag) =
            log1p_builtin(Value::Complex(input, input)).expect("complex log1p")
        else {
            panic!("expected complex result");
        };
        assert!((real - input).abs() <= 1.0e-35, "real={real}");
        assert!((imag - input).abs() <= 1.0e-35, "imag={imag}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_sparse_input_is_explicitly_rejected() {
        let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], vec![1.0])
            .expect("sparse tensor");
        let error = log1p_builtin(Value::SparseTensor(sparse)).expect_err("unsupported sparse");
        assert_eq!(error.identifier(), LOG1P_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_char_array_roundtrip() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = log1p_builtin(Value::CharArray(chars)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['A', 'B', 'C'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).ln_1p();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_string_rejects() {
        let err = log1p_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, -0.25, 0.5, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor
                .materialize_f64()
                .iter()
                .map(|&v| v.ln_1p())
                .collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn log1p_gpu_output_contract_rejects_alias_shape_and_storage_metadata() {
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
            assert!(gpu_helpers::unary_gpu_output_matches(
                &output,
                &input,
                provider,
                gpu_helpers::UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&input),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                },
            ));
            assert!(!gpu_helpers::unary_gpu_output_matches(
                &input,
                &input,
                provider,
                gpu_helpers::UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&input),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                },
            ));
            output.shape = vec![1, 2];
            assert!(!gpu_helpers::unary_gpu_output_matches(
                &output,
                &input,
                provider,
                gpu_helpers::UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&input),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                },
            ));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_bool_promotes() {
        let result = log1p_builtin(Value::Bool(true)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_logical_array_converts() {
        let logical = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = log1p_builtin(Value::LogicalArray(logical)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - 2.0f64.ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_complex_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -3.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result) = log1p_builtin(Value::GpuTensor(handle)).expect("log1p")
            else {
                panic!("expected restored GPU result");
            };
            let Value::ComplexTensor(ct) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &result),
            )
            .expect("download complex result") else {
                panic!("expected complex tensor result");
            };
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(0.0, PI), ((2.0f64).ln(), PI)];
            for ((re, im), (er, ei)) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((re - er).abs() < 1e-12);
                assert!((im - ei).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log1p_wgpu_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![0.0, -0.25, 0.25, 1.0], vec![4, 1]).unwrap();
        let cpu = log1p_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(log1p_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected value kinds"),
        }
    }
}
