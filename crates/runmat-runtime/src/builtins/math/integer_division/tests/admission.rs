use super::idivide_builtin;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct AdmissionProvider(AtomicUsize);
static PROVIDER: AdmissionProvider = AdmissionProvider(AtomicUsize::new(0));

impl AccelProvider for AdmissionProvider {
    fn device_id(&self) -> u32 {
        u32::MAX - 235
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "idivide-admission".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { anyhow::bail!("unexpected numeric download") })
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        panic!("borrowed source must not be freed")
    }
}

fn source() -> Value {
    Value::GpuTensor(
        GpuTensorHandle::new(vec![1, 2], PROVIDER.device_id(), 1)
            .with_numeric_descriptor(NumericElementType::U8, GpuTensorStorage::Real),
    )
}

#[test]
fn invalid_options_operands_shapes_and_outputs_never_download() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _provider = ThreadProviderGuard::set(Some(&PROVIDER));
    PROVIDER.0.store(0, Ordering::Relaxed);
    for arguments in [
        vec![
            source(),
            Value::Int(IntValue::U8(1)),
            Value::from("invalid"),
        ],
        vec![source(), Value::Int(IntValue::U16(1))],
        vec![source(), Value::Num(f64::NAN)],
        vec![source(), Value::from("invalid")],
        vec![
            source(),
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1; 3]), vec![1, 3]).unwrap()),
        ],
    ] {
        assert!(block_on(idivide_builtin(arguments)).is_err());
        assert_eq!(PROVIDER.0.load(Ordering::Relaxed), 0);
    }
    let _outputs = crate::output_count::push_output_count(Some(2));
    assert!(block_on(idivide_builtin(vec![source(), Value::Int(IntValue::U8(1))])).is_err());
    assert_eq!(PROVIDER.0.load(Ordering::Relaxed), 0);
}

#[test]
fn empty_nd_division_preserves_shape_without_inspecting_unused_zero_divisor() {
    let shape = vec![2, 0, 3];
    let input = Tensor::new_integer(IntegerStorage::I16(vec![]), shape.clone()).unwrap();
    let Value::Tensor(output) = block_on(idivide_builtin(vec![
        Value::Tensor(input),
        Value::Int(IntValue::I16(0)),
    ]))
    .unwrap() else {
        panic!("empty dense output");
    };
    assert_eq!(output.shape, shape);
    assert_eq!(output.integer_storage(), Some(&IntegerStorage::I16(vec![])));
}
