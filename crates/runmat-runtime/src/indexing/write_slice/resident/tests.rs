use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, NumericElementType,
};
use std::sync::Mutex;

#[test]
fn slice_assignment_and_deletion_preserve_single_complex_integer_and_logical_classes() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
        let single = runmat_accelerate_api::upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0, 2.0]),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            },
            &[],
        )
        .unwrap();
        let Value::GpuTensor(result) = futures::executor::block_on(assign_gpu_slice_with_plan(
            &single,
            &plan,
            &Value::Num(9.0),
        ))
        .unwrap() else {
            panic!("resident single");
        };
        let actual = futures::executor::block_on(provider.download_numeric(&result)).unwrap();
        assert_eq!(actual.data, HostNumericDataOwned::F32(vec![1.0, 9.0]));

        let complex = runmat_accelerate_api::upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::U64(&[u64::MAX, 9_007_199_254_740_993, 2, 3]),
                shape: &[1, 2],
                storage: GpuTensorStorage::ComplexInterleaved,
            },
            &[],
        )
        .unwrap();
        let Value::GpuTensor(result) = futures::executor::block_on(assign_gpu_slice_with_plan(
            &complex,
            &plan,
            &Value::Int(runmat_value::IntValue::U64(7)),
        ))
        .unwrap() else {
            panic!("resident complex");
        };
        let actual = futures::executor::block_on(provider.download_numeric(&result)).unwrap();
        assert_eq!(
            actual.data,
            HostNumericDataOwned::U64(vec![u64::MAX, 9_007_199_254_740_993, 7, 0])
        );
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let Value::GpuTensor(result) =
            futures::executor::block_on(delete_gpu_slice_with_plan(&complex, &plan, &empty))
                .unwrap()
        else {
            panic!("resident deletion");
        };
        let actual = futures::executor::block_on(provider.download_numeric(&result)).unwrap();
        assert_eq!(
            actual.data,
            HostNumericDataOwned::U64(vec![u64::MAX, 9_007_199_254_740_993])
        );

        let logical = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1, 0],
                shape: &[1, 2],
            })
            .unwrap();
        let Value::GpuTensor(result) = futures::executor::block_on(assign_gpu_slice_with_plan(
            &logical,
            &plan,
            &Value::Num(3.0),
        ))
        .unwrap() else {
            panic!("resident logical");
        };
        assert!(runmat_accelerate_api::handle_is_logical(&result));
        let Value::LogicalArray(actual) =
            futures::executor::block_on(ownership::gather(&result)).unwrap()
        else {
            panic!("logical download");
        };
        assert_eq!(actual.data, vec![1, 1]);
    });
}

#[derive(Debug)]
struct DeviceFailure;
impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected scatter failure")
    }
}
impl std::error::Error for DeviceFailure {}

struct FailingProvider {
    owner: TestAllocationOwner,
    temporaries: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for FailingProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "failing owning scatter fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(&self, view: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        let HostNumericDataView::F64(data) = view.data else {
            panic!("F64 fixture");
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
            shape: view.shape.to_vec(),
            storage: view.storage,
        })?;
        self.temporaries.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("operational failure must not gather and retry")
    }
    fn scatter_linear(
        &self,
        _: &GpuTensorHandle,
        _: &[u32],
        _: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        Err(anyhow::Error::new(DeviceFailure))
    }
}

#[test]
fn operational_scatter_failure_retains_cause_and_retires_owned_rhs_once() {
    let provider = Box::leak(Box::new(FailingProvider {
        owner: TestAllocationOwner::new(),
        temporaries: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
    let error = futures::executor::block_on(assign_gpu_slice_with_plan(
        &input.handle,
        &plan,
        &Value::Num(4.0),
    ))
    .unwrap_err();
    let mut cursor: &(dyn std::error::Error + 'static) = &error;
    while !cursor.is::<DeviceFailure>() {
        cursor = cursor.source().expect("typed operational cause retained");
    }
    let probes = provider.temporaries.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
}
