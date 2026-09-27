use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "mean reduction publication fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
    fn fill(&self, shape: &[usize], value: f64) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .0
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![
                    value;
                    shape.iter().product()
                ]),
                shape: shape.to_vec(),
                storage: GpuTensorStorage::Real,
            })?
            .handle)
    }
}

#[test]
fn publication_rejects_wrong_class_shape_and_physical_alias_and_retires_output() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    for (shape, class) in [
        (vec![1, 2], NumericElementType::F64),
        (vec![2, 1], NumericElementType::F32),
    ] {
        let output = provider
            .0
            .allocate_zeroed(shape, class, GpuTensorStorage::Real)
            .unwrap();
        assert!(admit_result(
            provider,
            &input.handle,
            output.handle,
            &MeanAxes::Dim(1),
            &[0]
        )
        .is_err());
        assert_eq!(output.probe.retirements(), 1);
    }
    let alias = input
        .handle
        .with_view(input.handle.view().clone().with_shape(vec![1, 2]))
        .unwrap();
    assert!(admit_result(provider, &input.handle, alias, &MeanAxes::Dim(1), &[0]).is_err());
    assert_eq!(input.probe.retirements(), 0);
    let output = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let accepted = admit_result(
        provider,
        &input.handle,
        output.handle,
        &MeanAxes::Dim(1),
        &[0],
    )
    .unwrap();
    assert_eq!(accepted.shape(), &[1, 2]);
    let identity = admit_result(
        provider,
        &accepted,
        accepted.clone(),
        &MeanAxes::Dim(1),
        &[0],
    )
    .unwrap();
    assert!(identity.shares_allocation_with(&accepted));
    drop(identity);
    drop(accepted);
    assert_eq!(output.probe.retirements(), 1);
}

#[test]
fn empty_nd_strategy_preserves_unreduced_dimensions_without_permute() {
    let provider = Provider(TestAllocationOwner::new());
    let input = provider
        .0
        .allocate_zeroed(
            vec![0, 3, 2],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    // This provider has no permute hook. Empty shape resolution precedes it.
    let output =
        futures::executor::block_on(reduce_mean_vecdim_nd_gpu(&provider, &input.handle, &[1]))
            .unwrap();
    assert_eq!(output.shape(), &[1, 3, 2]);
    assert_eq!(input.probe.retirements(), 0);
}
