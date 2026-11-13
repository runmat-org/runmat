use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostNumericTensorView, HostTensorView,
};
use runmat_value::IntegerStorage;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for CountingProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
}
use runmat_value::Tensor;

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn tensor_out(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn scaled_false_restores_intercept_and_original_scale() {
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(
        y,
        x,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape(), [2, 1]);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn ridge_accepts_typed_integer_response_design_and_k() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, f64::NAN);
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1, f64::NAN);
    let k = poisoned_int_tensor(IntegerStorage::U8(vec![0, 1]), 1, 2, f64::NAN);
    let scaled = poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1, 1.0);
    let out = block_on(ridge_builtin(
        y,
        x,
        k,
        crate::builtins::common::test_support::builtin_arguments(vec![scaled]),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape(), [2, 2]);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn scaled_true_returns_standardized_coefficients_for_each_k() {
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(
        y,
        x,
        tensor(vec![0.0, 1.0], 1, 2),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape(), [1, 2]);
    assert!(out.materialize_f64()[0] > out.materialize_f64()[1]);
    assert!((out.materialize_f64()[0] - 2.581_988_897).abs() < 1.0e-8);
}

#[test]
fn rows_with_nan_are_omitted() {
    let y = tensor(vec![1.0, f64::NAN, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(
        y,
        x,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn invalid_inputs_are_rejected() {
    let y = tensor(vec![1.0, 2.0], 2, 1);
    let x = tensor(vec![1.0, 2.0], 2, 1);
    assert!(block_on(ridge_builtin(
        y.clone(),
        x.clone(),
        Value::Num(-1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new())
    ))
    .is_err());
    assert!(block_on(ridge_builtin(
        y,
        x,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)])
    ))
    .is_err());
}

#[test]
fn wide_integer_scaled_flag_does_not_round_into_an_admitted_value() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
    for scaled in [
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        poisoned_int_tensor(IntegerStorage::U64(vec![u64::MAX]), 1, 1, 0.0),
    ] {
        assert!(block_on(ridge_builtin(
            y.clone(),
            x.clone(),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![scaled])
        ))
        .is_err());
    }
}

#[test]
fn host_visible_geometry_and_flag_are_rejected_at_admission() {
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let x = tensor(vec![0.0, 1.0], 2, 1);
    assert!(super::admission::preflight_data(&y, &x, &Value::Num(0.0)).is_err());
    assert!(super::admission::preflight_scaled(&Value::Num(2.0)).is_err());
    assert!(super::admission::preflight_data(
        &y,
        &tensor(vec![0.0, 1.0, 2.0], 3, 1),
        &Value::Num(-1.0)
    )
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ridge_preserves_preflight_and_cached_compute_rank_boundaries() {
    let scalar = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
    assert!(super::admission::preflight_data(&scalar, &scalar, &Value::Num(0.0)).is_ok());
    let error = block_on(ridge_builtin(
        scalar.clone(),
        scalar,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cached 0-by-0 response is not a vector");
    assert!(error.message().contains("y must be a vector"));

    let response = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
    let design = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
    assert!(super::admission::preflight_data(&response, &design, &Value::Num(0.0)).is_ok());
    let error = block_on(ridge_builtin(
        response,
        design,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cached rank-one design has one row");
    assert!(error
        .message()
        .contains("length(y) must match the number of rows in X"));
}

#[test]
fn invalid_host_flag_precedes_resident_integer_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = Tensor::new_integer(IntegerStorage::U64(vec![0, 1, 2]), vec![3, 1]).unwrap();
    let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &y).unwrap();
    let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
    let error = block_on(ridge_builtin(
        Value::GpuTensor(resident),
        x,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .expect_err("invalid host flag");
    assert!(error.message().contains("scaled"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn logical_response_design_and_parameter_are_admitted() {
    let y =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 1, 0, 1], vec![4, 1]).unwrap());
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(
        y,
        x,
        Value::Bool(false),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape(), [1, 1]);
    assert!(out.materialize_f64()[0].is_finite());
}

#[test]
fn rank_deficient_k_zero_uses_minimum_norm_solution() {
    let y = tensor(vec![2.0, 4.0, 6.0], 3, 1);
    let x = tensor(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 3, 2);
    let out = block_on(ridge_builtin(
        y,
        x,
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape(), [3, 1]);
    assert!(out.materialize_f64()[0].abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[2] - 0.5).abs() < 1.0e-10);
}
