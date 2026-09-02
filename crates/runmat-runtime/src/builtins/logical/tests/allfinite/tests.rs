use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_value::{
    CharArray, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
};
use std::sync::OnceLock;

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(allfinite_builtin(value))
}

fn truth(value: Value) -> bool {
    match call(value).expect("allfinite") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn scalars_dense_complex_and_empty_arrays_reduce_to_one_logical() {
    assert!(truth(Value::Num(1.0)));
    assert!(!truth(Value::Num(f64::INFINITY)));
    assert!(!truth(Value::Num(f64::NAN)));
    assert!(truth(Value::Int(IntValue::I32(7))));
    assert!(truth(Value::Complex(1.0, -2.0)));
    assert!(!truth(Value::Complex(1.0, f64::NAN)));
    assert!(truth(Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 0.0), (2.0, -3.0)], vec![1, 2]).unwrap()
    )));
    assert!(!truth(Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 0.0), (2.0, f64::INFINITY)], vec![1, 2]).unwrap()
    )));
    assert!(truth(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()
    )));
    assert!(!truth(Value::Tensor(
        Tensor::new(vec![1.0, f64::INFINITY], vec![1, 2]).unwrap()
    )));
    assert!(truth(Value::Tensor(Tensor::zeros(vec![0, 3]))));
}

#[test]
fn every_integer_storage_class_is_finite_without_floating_conversion() {
    let storages = [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
        IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
        IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
        IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
    ];
    for storage in storages {
        assert!(truth(Value::Tensor(
            Tensor::new_integer(storage, vec![1, 2]).unwrap()
        )));
    }
    let complex = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        IntegerStorage::U64(vec![0, 7]),
    )
    .unwrap();
    assert!(truth(Value::ComplexTensor(
        ComplexTensor::new_integer(complex, vec![1, 2]).unwrap()
    )));
}

#[test]
fn sparse_values_check_only_stored_payloads() {
    let finite = SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![1.0, -2.0]).unwrap();
    assert!(truth(Value::SparseTensor(finite)));
    let nonfinite =
        SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![1.0, f64::NAN]).unwrap();
    assert!(!truth(Value::SparseTensor(nonfinite)));
    assert!(truth(Value::SparseTensor(SparseTensor::zeros(4, 5))));

    let complex = SparseTensor::new_complex(
        2,
        1,
        vec![0, 2],
        vec![0, 1],
        vec![(1.0, -2.0), (3.0, f64::INFINITY)],
    )
    .unwrap();
    assert!(!truth(Value::SparseTensor(complex)));
    let integer = SparseTensor::new_integer(
        3,
        2,
        vec![0, 1, 2],
        vec![0, 2],
        IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
    )
    .unwrap();
    assert!(truth(Value::SparseTensor(integer)));
}

#[test]
fn logical_and_string_policies_are_explicit() {
    assert!(truth(Value::LogicalArray(
        LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap()
    )));
    assert!(truth(Value::CharArray(CharArray::new_row("RunMat"))));
    let text = Value::String("1".into());
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(text.clone()).expect_err("string extension is disabled");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:AllfiniteStringInputExtension")
    );
    drop(_matlab);
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    assert!(!truth(text));
    assert!(!truth(Value::StringArray(
        StringArray::new(vec!["1".to_string()], vec![1, 1]).unwrap()
    )));
    assert!(truth(Value::StringArray(
        StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap()
    )));
}

#[test]
fn errors_use_catalog_descriptors() {
    let error = call(Value::Cell(
        runmat_value::CellArray::new(Vec::new(), 0, 0).unwrap(),
    ))
    .expect_err("cell input");
    assert_eq!(error.identifier(), Some("RunMat:allfinite:InvalidInput"));

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(Value::Num(1.0)).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:allfinite:TooManyOutputs"));
}

#[derive(Clone, Copy)]
enum HookMode {
    ClassificationUnsupported,
    ReductionUnsupported,
    ClassificationFailure,
}

struct AllFiniteProvider {
    inner: InProcessProvider,
    mode: HookMode,
}

impl AllFiniteProvider {
    fn new(mode: HookMode) -> Self {
        Self {
            inner: InProcessProvider::new(),
            mode,
        }
    }
}

impl AccelProvider for AllFiniteProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(handle)
    }

    fn logical_isfinite(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        match self.mode {
            HookMode::ClassificationUnsupported => {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "finite classification is unavailable",
                ))
            }
            HookMode::ClassificationFailure => {
                anyhow::bail!("injected finite-classification failure")
            }
            HookMode::ReductionUnsupported => self.inner.logical_isfinite(input),
        }
    }

    fn reduce_all<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "logical reduction is unavailable",
            ))
        })
    }
}

fn provider(mode: HookMode) -> &'static AllFiniteProvider {
    static CLASSIFICATION_UNSUPPORTED: OnceLock<AllFiniteProvider> = OnceLock::new();
    static REDUCTION_UNSUPPORTED: OnceLock<AllFiniteProvider> = OnceLock::new();
    static CLASSIFICATION_FAILURE: OnceLock<AllFiniteProvider> = OnceLock::new();
    match mode {
        HookMode::ClassificationUnsupported => CLASSIFICATION_UNSUPPORTED
            .get_or_init(|| AllFiniteProvider::new(HookMode::ClassificationUnsupported)),
        HookMode::ReductionUnsupported => REDUCTION_UNSUPPORTED
            .get_or_init(|| AllFiniteProvider::new(HookMode::ReductionUnsupported)),
        HookMode::ClassificationFailure => CLASSIFICATION_FAILURE
            .get_or_init(|| AllFiniteProvider::new(HookMode::ClassificationFailure)),
    }
}

fn resident_input(provider: &dyn AccelProvider) -> GpuTensorHandle {
    gpu_helpers::upload_tensor(
        provider,
        &Tensor::new(vec![1.0, f64::INFINITY], vec![1, 2]).unwrap(),
    )
    .unwrap()
}

#[test]
fn only_typed_unsupported_hooks_use_host_fallback() {
    let _state = test_support::accel_test_lock();
    for mode in [
        HookMode::ClassificationUnsupported,
        HookMode::ReductionUnsupported,
    ] {
        let provider = provider(mode);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let input = resident_input(provider);
        assert!(!truth(Value::GpuTensor(input.clone())));
        provider.free(&input).unwrap();
    }

    let provider = provider(HookMode::ClassificationFailure);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let input = resident_input(provider);
    let error = call(Value::GpuTensor(input.clone())).expect_err("provider failure");
    assert_eq!(error.identifier(), Some("RunMat:allfinite:InternalError"));
    assert!(error
        .message()
        .contains("injected finite-classification failure"));
    provider.free(&input).unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_matches_host_and_returns_scalar() {
    let _state = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let input = resident_input(provider);
    assert!(!truth(Value::GpuTensor(input.clone())));
    provider.free(&input).ok();
}
