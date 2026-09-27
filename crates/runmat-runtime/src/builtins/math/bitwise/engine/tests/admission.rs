use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

struct NoDownload(AtomicUsize);
static PROVIDER: NoDownload = NoDownload(AtomicUsize::new(0));
impl AccelProvider for NoDownload {
    fn device_id(&self) -> u32 {
        u32::MAX - 220
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "bitwise-admission".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        anyhow::bail!("borrowed handle must not be freed")
    }
}

fn source() -> Value {
    Value::GpuTensor(
        GpuTensorHandle::new(vec![1, 2], PROVIDER.device_id(), 1)
            .with_numeric_descriptor(NumericElementType::U8, GpuTensorStorage::Real),
    )
}

#[test]
fn invalid_host_controls_and_shapes_fail_before_resident_download() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _provider = ThreadProviderGuard::set(Some(&PROVIDER));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    PROVIDER.0.store(0, Ordering::Relaxed);
    for (name, args) in [
        ("bitand", vec![source(), Value::Num(1.5)]),
        (
            "bitand",
            vec![source(), Value::Int(runmat_value::IntValue::U16(1))],
        ),
        ("bitor", vec![source(), Value::String("invalid".into())]),
        ("bitxor", vec![source(), Value::Num(f64::NAN)]),
        ("bitget", vec![source(), Value::Num(f64::NAN)]),
        ("bitget", vec![source(), Value::Num(0.0)]),
        (
            "bitset",
            vec![source(), Value::Num(1.0), Value::Num(f64::NAN)],
        ),
        ("bitshift", vec![source(), Value::Num(f64::NAN)]),
        ("bitcmp", vec![source(), Value::String("invalid".into())]),
        (
            "bitget",
            vec![
                source(),
                Value::Tensor(runmat_value::Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
            ],
        ),
    ] {
        assert!(
            block_on(crate::dispatcher::call_builtin_async(name, &args)).is_err(),
            "{name}"
        );
        assert_eq!(PROVIDER.0.load(Ordering::Relaxed), 0, "{name}");
    }
}

#[test]
fn every_identity_rejects_excess_outputs_before_provider_access() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _provider = ThreadProviderGuard::set(Some(&PROVIDER));
    PROVIDER.0.store(0, Ordering::Relaxed);
    for name in [
        "bitcmp",
        "bitget",
        "bitset",
        "bitshift",
        "swapbytes",
        "bitand",
        "bitor",
        "bitxor",
    ] {
        let args = if matches!(name, "bitcmp" | "swapbytes") {
            vec![source()]
        } else {
            vec![source(), Value::Num(1.0)]
        };
        assert!(block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name, &args, 2
        ))
        .is_err());
        assert_eq!(PROVIDER.0.load(Ordering::Relaxed), 0, "{name}");
        let args = if matches!(name, "bitcmp" | "swapbytes") {
            vec![Value::Num(3.0)]
        } else {
            vec![Value::Num(3.0), Value::Num(1.0)]
        };
        for count in [0, 1] {
            block_on(crate::dispatcher::call_builtin_async_with_outputs(
                name, &args, count,
            ))
            .unwrap();
        }
    }
}

#[test]
fn assumed_type_gate_uses_parsed_text_for_all_direct_forms() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let chars = || Value::CharArray(runmat_value::CharArray::new_row("uint8"));
    for (name, args) in [
        ("bitcmp", vec![source(), chars()]),
        ("bitget", vec![source(), Value::Num(1.0), chars()]),
        ("bitset", vec![source(), Value::Num(1.0), chars()]),
        (
            "bitset",
            vec![source(), Value::Num(1.0), Value::Num(0.0), chars()],
        ),
    ] {
        let error = block_on(crate::dispatcher::call_builtin_async(name, &args)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DirectBitGpuAssumedTypeExtension")
        );
    }
}

#[test]
fn complex_sparse_returns_structured_errors_not_real_access_panics() {
    let source = Value::SparseTensor(
        runmat_value::SparseTensor::new_complex(2, 2, vec![0, 1, 1], vec![0], vec![(1.0, 2.0)])
            .unwrap(),
    );
    for name in [
        "bitcmp", "bitget", "bitset", "bitshift", "bitand", "bitor", "bitxor",
    ] {
        let args = if name == "bitcmp" {
            vec![source.clone()]
        } else {
            vec![source.clone(), Value::Num(1.0)]
        };
        let error = block_on(crate::dispatcher::call_builtin_async(name, &args)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:bitwise:InvalidInput"));
    }
}

#[test]
fn dense_single_and_logical_result_classes_match_catalog_facts() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let shape = vec![1, 1, 2];
    for source in [
        Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 3.0], shape.clone()).unwrap()),
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0], shape.clone()).unwrap()),
    ] {
        for name in ["bitcmp", "bitget", "bitset", "bitshift"] {
            let args = if name == "bitcmp" {
                vec![source.clone()]
            } else {
                vec![source.clone(), Value::Num(1.0)]
            };
            let Value::Tensor(output) =
                block_on(crate::dispatcher::call_builtin_async(name, &args)).unwrap()
            else {
                panic!("{name}: dense double result")
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.shape, shape);
        }
    }
}
