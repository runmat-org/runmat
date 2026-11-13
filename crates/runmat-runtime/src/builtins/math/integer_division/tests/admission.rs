use super::idivide_builtin;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct AdmissionProvider {
    downloads: AtomicUsize,
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
}

impl AccelProvider for AdmissionProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
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
        self.downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { anyhow::bail!("unexpected numeric download") })
    }
}

#[test]
fn invalid_options_operands_shapes_and_outputs_never_download() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(AdmissionProvider {
        downloads: AtomicUsize::new(0),
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
    }));
    let _provider = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::U8, GpuTensorStorage::Real)
        .unwrap();
    let source = || Value::GpuTensor(allocation.handle.clone());
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
        assert!(block_on(idivide_builtin(
            crate::builtins::common::test_support::builtin_arguments(arguments)
        ))
        .is_err());
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert_eq!(allocation.probe.retirements(), 0);
    }
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert!(block_on(idivide_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            source(),
            Value::Int(IntValue::U8(1))
        ])
    ))
    .is_err());
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn empty_nd_division_preserves_shape_without_inspecting_unused_zero_divisor() {
    let shape = vec![2, 0, 3];
    let input = Tensor::new_integer(IntegerStorage::I16(vec![]), shape.clone()).unwrap();
    let Value::Tensor(output) = block_on(idivide_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(input),
            Value::Int(IntValue::I16(0)),
        ]),
    ))
    .unwrap() else {
        panic!("empty dense output");
    };
    assert_eq!(output.shape(), shape.as_slice());
    assert_eq!(output.integer_storage(), Some(&IntegerStorage::I16(vec![])));
}
