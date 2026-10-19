use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::Mutex;

struct Provider {
    owner: TestAllocationOwner,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "solve operand ownership fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_zeroed(
            host.shape.to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.outputs.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
}

#[test]
fn prepared_solve_operand_owns_upload_but_keeps_independent_input_alias_live() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let borrowed = PreparedOperand::new(
        SolveOrientation::Left,
        &Value::GpuTensor(input.handle.clone()),
        provider,
        &[&input.handle],
    )
    .unwrap()
    .unwrap();
    drop(borrowed);
    assert_eq!(input.probe.retirements(), 0);
    let host = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let uploaded = PreparedOperand::new(SolveOrientation::Left, &host, provider, &[&input.handle])
        .unwrap()
        .unwrap();
    let published = uploaded.handle().clone();
    drop(uploaded);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
    drop(published);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
    input.handle.allocation_lease().validate_live().unwrap();
}
