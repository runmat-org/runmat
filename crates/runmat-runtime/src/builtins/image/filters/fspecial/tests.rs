use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::{BuiltinIntegerComputationDomain, FSPECIAL_DESCRIPTOR};
use runmat_value::IntegerStorage;
use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, OnceLock};

async fn fspecial_builtin(kind: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::fspecial_builtin(
        None,
        kind,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

struct FspecialFailureProvider {
    device_id: u32,
    unsupported: bool,
}

impl runmat_accelerate_api::AccelProvider for FspecialFailureProvider {
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("generation failure fixture must not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("generation failure fixture must not download")
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "fspecial-failure-fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.device_id
    }
    fn fspecial(
        &self,
        _: &FspecialRequest,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        if self.unsupported {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "fspecial fixture unsupported",
            ))
        } else {
            Err(anyhow::anyhow!("fspecial driver failure sentinel"))
        }
    }
}

#[test]
fn fspecial_only_unsupported_provider_errors_fall_back_to_host() {
    let spec = FspecialFilterSpec::Average { rows: 3, cols: 3 };
    for unsupported in [true, false] {
        let provider: &'static FspecialFailureProvider =
            Box::leak(Box::new(FspecialFailureProvider {
                device_id: runmat_accelerate_api::next_device_id(),
                unsupported,
            }));
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let tensor = spec.generate_tensor().unwrap();
        let result = finalize_output_with_gpu_policy(&spec, tensor, true);
        if unsupported {
            assert!(matches!(result.unwrap(), Value::Tensor(_)));
        } else {
            let error = result.expect_err("driver failure must not replay on host");
            assert_eq!(error.identifier(), FSPECIAL_ERROR_INTERNAL.identifier);
            assert!(error.message().contains("driver failure sentinel"));
            assert!(error.source.is_some());
        }
    }
}

fn assert_close(actual: f64, expected: f64, epsilon: f64) {
    if (actual - expected).abs() > epsilon {
        panic!(
            "values differ: actual={actual:.15e}, expected={expected:.15e}, epsilon={epsilon:.3e}"
        );
    }
}

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

struct FspecialDeviceEnvRestore {
    previous: Option<OsString>,
}

impl Drop for FspecialDeviceEnvRestore {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", previous);
        } else {
            std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE");
        }
    }
}

fn fspecial_env_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fspecial_device_env_restore() -> FspecialDeviceEnvRestore {
    FspecialDeviceEnvRestore {
        previous: std::env::var_os("RUNMAT_ACCEL_FSPECIAL_DEVICE"),
    }
}

fn fspecial_host_tensor(kind: &str, args: Vec<Value>) -> Tensor {
    let _env_guard = fspecial_env_guard();
    let _restore = fspecial_device_env_restore();
    std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE");
    match block_on(fspecial_builtin(Value::from(kind), args)).unwrap() {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn all_integer_scalar_tensors(value: u8) -> Vec<Tensor> {
    vec![
        Tensor::new_integer(IntegerStorage::I8(vec![value as i8]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I16(vec![value as i16]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I32(vec![value as i32]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I64(vec![value as i64]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U8(vec![value]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U16(vec![value as u16]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U32(vec![value as u32]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U64(vec![value as u64]), vec![1, 1]).unwrap(),
    ]
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_average_default() {
    let tensor = fspecial_host_tensor("average", Vec::new());
    assert_eq!(tensor.shape(), &[3, 3]);
    for value in tensor.materialize_f64() {
        assert_close(value, 1.0 / 9.0, 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_average_scalar_size() {
    let args = vec![Value::from(5.0)];
    let tensor = fspecial_host_tensor("average", args);
    assert_eq!(tensor.shape(), &[5, 5]);
    let sum: f64 = tensor.materialize_f64().iter().sum();
    assert_close(sum, 1.0, 1e-12);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_average_rectangular_size() {
    let args = vec![Value::from(
        Tensor::new(vec![4.0, 6.0], vec![1, 2]).unwrap(),
    )];
    let tensor = fspecial_host_tensor("average", args);
    assert_eq!(tensor.shape(), &[4, 6]);
    let expected = 1.0 / (4.0 * 6.0);
    for value in tensor.materialize_f64() {
        assert_close(value, expected, 1e-12);
    }
}

#[test]
fn fspecial_lengths_preserve_typed_integer_tensor_bounds() {
    let dims = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![2, 4]), vec![1, 2])
        .expect("dims");
    assert_eq!(
        parse_lengths_strict(
            &Value::Tensor(dims),
            "fspecial: LENGTHS must be positive integers",
        )
        .unwrap(),
        vec![2, 4]
    );

    let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("negative");
    assert!(parse_lengths_strict(
        &Value::Tensor(negative),
        "fspecial: LENGTHS must be positive integers",
    )
    .is_err());
}

#[test]
fn fspecial_accepts_all_integer_classes_from_authoritative_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for tensor in all_integer_scalar_tensors(2) {
        let spec = build_filter_spec(
            &Value::from("average"),
            crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(tensor)]),
        )
        .expect("integer size");
        assert!(matches!(
            spec,
            FspecialFilterSpec::Average { rows: 2, cols: 2 }
        ));
    }
    for tensor in all_integer_scalar_tensors(1) {
        let spec = build_filter_spec(
            &Value::from("laplacian"),
            crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(tensor)]),
        )
        .expect("integer parameter");
        assert!(matches!(spec, FspecialFilterSpec::Laplacian { alpha: 1.0 }));
    }
}

#[test]
fn fspecial_rejects_inexact_wide_integer_computational_parameters() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = (1_u64 << 53) + 1;
    let radius = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    let error = build_filter_spec(
        &Value::from("disk"),
        crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(radius)]),
    )
    .expect_err("inexact radius");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn fspecial_nondouble_roles_and_runmat_forms_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let integer_size = block_on(fspecial_builtin(
        Value::from("average"),
        vec![Value::Int(IntValue::U8(2))],
    ))
    .expect_err("integer size extension");
    assert_eq!(
        integer_size.identifier(),
        FSPECIAL_NONDOUBLE_SIZE_EXTENSION.error_identifier
    );

    let integer_parameter = block_on(fspecial_builtin(
        Value::from("laplacian"),
        vec![Value::Int(IntValue::U8(1))],
    ))
    .expect_err("integer parameter extension");
    assert_eq!(
        integer_parameter.identifier(),
        FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION.error_identifier
    );

    let logical_size = block_on(fspecial_builtin(
        Value::from("average"),
        vec![Value::Bool(true)],
    ))
    .expect_err("logical size extension");
    assert_eq!(
        logical_size.identifier(),
        FSPECIAL_NONDOUBLE_SIZE_EXTENSION.error_identifier
    );

    let single_parameter =
        Tensor::from_numeric_storage(runmat_value::NumericStorage::F32(vec![1.0]), vec![1, 1])
            .unwrap();
    let single_parameter = block_on(fspecial_builtin(
        Value::from("laplacian"),
        vec![Value::Tensor(single_parameter)],
    ))
    .expect_err("single parameter extension");
    assert_eq!(
        single_parameter.identifier(),
        FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION.error_identifier
    );

    let unsharp = block_on(fspecial_builtin(Value::from("unsharp"), Vec::new()))
        .expect_err("unsharp extension");
    assert_eq!(
        unsharp.identifier(),
        FSPECIAL_UNSHARP_EXTENSION.error_identifier
    );
}

#[test]
fn fspecial_resident_output_policy_is_gated_before_provider_dispatch() {
    let _env_guard = fspecial_env_guard();
    let _restore = fspecial_device_env_restore();
    std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(fspecial_builtin(Value::from("average"), Vec::new()))
        .expect_err("resident output extension");
    assert_eq!(
        error.identifier(),
        FSPECIAL_RESIDENT_OUTPUT_EXTENSION.error_identifier
    );
}

#[test]
fn fspecial_integer_capabilities_distinguish_structural_and_floating_roles() {
    assert_eq!(FSPECIAL_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(FSPECIAL_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    assert_eq!(
        FSPECIAL_INTEGER_CAPABILITIES[0].computation_domain,
        BuiltinIntegerComputationDomain::Structural
    );
    assert_eq!(
        FSPECIAL_INTEGER_CAPABILITIES[1].computation_domain,
        BuiltinIntegerComputationDomain::FloatingPoint
    );
}

#[test]
fn fspecial_dimension_parsers_reject_unrepresentable_double_bounds() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };

    assert!(parse_numeric_dimension(boundary).is_err());

    let dims = Tensor::new(vec![boundary], vec![1, 1]).expect("dims");
    assert!(parse_lengths_strict(
        &Value::Tensor(dims),
        "fspecial: LENGTHS must be positive integers",
    )
    .is_err());
}

#[test]
fn fspecial_motion_length_preserves_typed_integer_scalar_bounds() {
    assert_eq!(
        parse_motion_length(Some(&Value::Int(IntValue::U64(17)))).unwrap(),
        17
    );

    let tensor = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![21]), vec![1, 1])
        .expect("typed scalar length");
    assert_eq!(
        parse_motion_length(Some(&Value::Tensor(tensor))).unwrap(),
        21
    );

    let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("negative");
    assert!(parse_motion_length(Some(&Value::Tensor(negative))).is_err());

    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_motion_length(Some(&Value::Num(boundary))).is_err());
}

#[test]
fn fspecial_generators_reject_overflowing_dimensions_before_allocation() {
    assert!(generate_gaussian(usize::MAX, 2, 0.5).is_err());
    assert!(generate_log(usize::MAX, 2, 0.5).is_err());
    assert!(generate_motion(usize::MAX, usize::MAX, 0.0, 8).is_err());
}

#[test]
fn fspecial_disk_rejects_unrepresentable_radius_before_size_cast() {
    let huge_radius = (isize::MAX as f64) / 2.0 + 1.0;
    assert!(parse_disk_params(Some(&Value::Num(huge_radius))).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_average_rejects_zero_size() {
    let args = vec![Value::from(0.0)];
    let err = block_on(fspecial_builtin(Value::from("average"), args))
        .expect_err("fspecial should error");
    assert!(error_message(err).contains("positive"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_gaussian_default_matches_reference() {
    let tensor = fspecial_host_tensor("gaussian", Vec::new());
    assert_eq!(tensor.shape(), &[3, 3]);
    const EXPECTED: [f64; 9] = [
        0.011_343_736_558_495,
        0.083_819_505_802_211,
        0.011_343_736_558_495,
        0.083_819_505_802_211,
        0.619_347_030_557_177,
        0.083_819_505_802_211,
        0.011_343_736_558_495,
        0.083_819_505_802_211,
        0.011_343_736_558_495,
    ];
    for (idx, value) in tensor.materialize_f64().iter().enumerate() {
        assert_close(*value, EXPECTED[idx], 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_gaussian_size_sigma() {
    let args = vec![Value::from(7.0), Value::from(2.0)];
    let tensor = fspecial_host_tensor("gaussian", args);
    assert_eq!(tensor.shape(), &[7, 7]);
    let center = tensor.rows() / 2;
    let col = center;
    let idx = col * tensor.rows() + center;
    assert!(tensor.materialize_f64()[idx] > 0.0);
    let sum: f64 = tensor.materialize_f64().iter().sum();
    assert_close(sum, 1.0, 1e-5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_laplacian_alpha() {
    let args = vec![Value::from(0.2)];
    let t = fspecial_host_tensor("laplacian", args);
    assert_eq!(t.shape(), &[3, 3]);
    let expected = [
        0.16666666666666669,
        0.6666666666666667,
        0.16666666666666669,
        0.6666666666666667,
        -3.3333333333333335,
        0.6666666666666667,
        0.16666666666666669,
        0.6666666666666667,
        0.16666666666666669,
    ];
    for (idx, value) in t.materialize_f64().iter().enumerate() {
        assert_close(*value, expected[idx], 1e-7);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_unsharp_default() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let t = fspecial_host_tensor("unsharp", Vec::new());
    assert_eq!(t.shape(), &[3, 3]);
    let sum: f64 = t.materialize_f64().iter().sum();
    assert_close(sum, 1.0, 1e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_log_basic_properties() {
    let t = fspecial_host_tensor("log", vec![Value::from(5.0), Value::from(0.5)]);
    assert_eq!(t.shape(), &[5, 5]);
    let sum: f64 = t.materialize_f64().iter().sum();
    assert_close(sum, 0.0, 1e-12);
    let center = t.rows() / 2;
    let idx = center * t.rows() + center;
    assert!(t.materialize_f64()[idx] < 0.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_disk_sum_is_one() {
    let t = fspecial_host_tensor("disk", vec![Value::from(5.0)]);
    assert_eq!(t.shape(), &[11, 11]);
    let sum: f64 = t.materialize_f64().iter().sum();
    assert_close(sum, 1.0, 1e-10);
    let idx = t.rows() * (t.cols() / 2) + t.rows() / 2;
    assert!(t.materialize_f64()[idx] > 0.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_disk_negative_radius_errors() {
    let err = block_on(fspecial_builtin(
        Value::from("disk"),
        vec![Value::from(-1.0)],
    ))
    .expect_err("fspecial should error");
    let message = err.message().to_string();
    assert!(message.contains("non-negative"));
    assert_eq!(err.identifier(), FSPECIAL_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_motion_sum_is_one() {
    let t = fspecial_host_tensor("motion", vec![Value::from(15.0), Value::from(45.0)]);
    assert_eq!(t.shape(), &[15, 15]);
    let sum: f64 = t.materialize_f64().iter().sum();
    assert_close(sum, 1.0, 1e-10);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fspecial_invalid_filter_name() {
    let err = block_on(fspecial_builtin(Value::from("notafilter"), Vec::new()))
        .expect_err("fspecial should error");
    let message = err.message().to_string();
    assert!(message.contains("not supported"));
    assert_eq!(err.identifier(), FSPECIAL_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn fspecial_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = FSPECIAL_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "H = fspecial(type)",
            "H = fspecial(type, arg1)",
            "H = fspecial(type, arg1, arg2)"
        ]
    );
}

#[test]
fn fspecial_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = FSPECIAL_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert_eq!(
        codes,
        vec![
            "RM.FSPECIAL.INVALID_ARGUMENT",
            "RM.FSPECIAL.INVALID_INPUT",
            "RM.FSPECIAL.INTERNAL",
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn fspecial_gaussian_gpu_matches_cpu() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _env_guard = fspecial_env_guard();
    let _restore = fspecial_device_env_restore();
    std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let spec = build_filter_spec(
        &Value::from("gaussian"),
        crate::call::FunctionArgumentView::from_slice(&[]),
    )
    .unwrap();
    let tensor = spec.generate_tensor().unwrap();
    let gpu_tensor = match finalize_output_with_gpu_policy(&spec, tensor, true).unwrap() {
        Value::GpuTensor(handle) => {
            test_support::gather(Value::GpuTensor(handle)).expect("gather gpu result")
        }
        Value::Tensor(t) => t,
        other => panic!("unexpected result {other:?}"),
    };
    let host_tensor = match block_on(fspecial_builtin(Value::from("gaussian"), Vec::new())).unwrap()
    {
        Value::Tensor(t) => t,
        Value::GpuTensor(handle) => {
            test_support::gather(Value::GpuTensor(handle)).expect("gather fallback")
        }
        other => panic!("unexpected result {other:?}"),
    };
    assert_eq!(gpu_tensor.shape(), host_tensor.shape());
    for (a, b) in gpu_tensor
        .materialize_f64()
        .iter()
        .zip(host_tensor.materialize_f64().iter())
    {
        assert_close(*a, *b, 1e-6);
    }
}
