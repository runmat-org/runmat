use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderExecutionFuture, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

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
            &[0],
            Some(ProviderPrecision::F32),
        )
        .is_err());
        assert_eq!(output.probe.retirements(), 1);
    }
    let alias = input
        .handle
        .with_view(input.handle.view().clone().with_shape(vec![1, 2]))
        .unwrap();
    assert!(admit_result(
        provider,
        &input.handle,
        alias,
        &MeanAxes::Dim(1),
        &[0],
        Some(ProviderPrecision::F32)
    )
    .is_err());
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
        Some(ProviderPrecision::F32),
    )
    .unwrap();
    assert_eq!(accepted.shape(), &[1, 2]);
    let identity = admit_result(
        provider,
        &accepted,
        accepted.clone(),
        &MeanAxes::Dim(1),
        &[0],
        Some(ProviderPrecision::F32),
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

#[test]
fn omitnan_single_uses_double_reduction_then_restores_requested_class_resident() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [16_777_216.0_f32, 1.0, 1.0, f32::NAN];
        let input = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                shape: &[4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        for (controls, expected_precision) in [
            (vec![Value::from("omitnan")], ProviderPrecision::F32),
            (
                vec![Value::from("omitnan"), Value::from("double")],
                ProviderPrecision::F64,
            ),
        ] {
            provider.reset_telemetry();
            let result = futures::executor::block_on(mean_builtin(
                Value::GpuTensor(input.clone()),
                controls,
            ))
            .unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("mean should remain resident")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(expected_precision)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let host =
                crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(host.materialize_f64(), vec![5_592_406.0]);
        }
        input.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn omitnan_stage_rejects_wrong_precision_and_borrowed_alias_before_division() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let wrong = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    assert!(admit_double_omitnan_stage(
        provider,
        &original.handle,
        wrong.handle,
        &[1, 1],
        OmitNanStageAlias::Fresh,
        &[&original.handle],
    )
    .is_err());
    assert_eq!(wrong.probe.retirements(), 1);

    let borrowed = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(admit_double_omitnan_stage(
        provider,
        &original.handle,
        borrowed.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::Fresh,
        &[&borrowed.handle],
    )
    .is_err());
    assert_eq!(borrowed.probe.retirements(), 0);
    original.handle.allocation_lease().validate_live().unwrap();
    borrowed.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn omitnan_singleton_double_stage_accepts_only_its_current_identity_view() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let current = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let earlier_same_branch = current.handle.clone();
    let accepted = admit_double_omitnan_stage(
        provider,
        &original.handle,
        current.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&current.handle, &earlier_same_branch],
    )
    .unwrap();
    assert!(accepted.shares_allocation_with(&current.handle));
    assert_eq!(current.probe.retirements(), 0);

    let sibling = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let sibling_alias = admit_double_omitnan_stage(
        provider,
        &original.handle,
        sibling.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&current.handle, &sibling.handle],
    );
    assert!(matches!(
        sibling_alias,
        Err(MeanProviderFailure::Execution(_))
    ));
    assert_eq!(sibling.probe.retirements(), 0);

    let rejected = admit_double_omitnan_stage(
        provider,
        &original.handle,
        original.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&original.handle, &current.handle],
    );
    assert!(matches!(rejected, Err(MeanProviderFailure::Execution(_))));
    assert_eq!(original.probe.retirements(), 0);
    original.handle.allocation_lease().validate_live().unwrap();
}

#[derive(Clone, Copy)]
enum MalformedPreparation {
    AliasedMap,
    WrongMaskClass,
}

struct PreparationProvider {
    owner: TestAllocationOwner,
    mode: MalformedPreparation,
    mask_calls: AtomicUsize,
    reduction_calls: AtomicUsize,
}

impl AccelProvider for PreparationProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "mean preparation admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn map_nan_to_zero(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        match self.mode {
            MalformedPreparation::AliasedMap => Ok(input.clone()),
            MalformedPreparation::WrongMaskClass => Ok(self
                .owner
                .allocate_zeroed(
                    input.shape().to_vec(),
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )?
                .handle),
        }
    }

    fn not_nan_mask(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.mask_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_zeroed(
                input.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?
            .handle)
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduction_calls.fetch_add(1, Ordering::SeqCst);
            Err(
                runmat_accelerate_api::unsupported_provider_operation("unexpected reduction")
                    .into(),
            )
        })
    }
}

#[test]
fn malformed_nan_preparation_stops_before_reduction_or_later_provider_hooks() {
    for mode in [
        MalformedPreparation::AliasedMap,
        MalformedPreparation::WrongMaskClass,
    ] {
        let provider: &'static PreparationProvider = Box::leak(Box::new(PreparationProvider {
            owner: TestAllocationOwner::new(),
            mode,
            mask_calls: AtomicUsize::new(0),
            reduction_calls: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let result = futures::executor::block_on(execute(
            provider,
            &input.handle,
            &MeanAxes::Dim(1),
            ReductionNaN::Omit,
            &OutputTemplate::Default,
        ));
        assert!(result.is_err());
        assert_eq!(provider.reduction_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            provider.mask_calls.load(Ordering::SeqCst),
            usize::from(matches!(mode, MalformedPreparation::WrongMaskClass))
        );
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
