use super::*;
use runmat_accelerate_api::observation::{
    with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
    ProviderOperationReceipt,
};
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "sum output admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
}

#[test]
fn sum_receipt_requires_actual_submit_and_ignores_unrelated_outer_operation() {
    for invocation in [
        runmat_types::SumInvocation::Sum,
        runmat_types::SumInvocation::NanSum,
    ] {
        assert_sum_receipt(invocation);
    }
}

fn assert_sum_receipt(invocation: runmat_types::SumInvocation) {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sum = super::super::sum_observation_identity(invocation);
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let new_output = || {
        provider
            .0
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    };

    let mut unsubmitted = ProviderOperationReceipt::new();
    unsubmitted.require_provider_execution(provider.device_id(), sum.clone());
    let output = new_output();
    with_provider_observation(Some(unsubmitted.recorder()), || {
        super::super::publish_submitted_sum(provider, output.handle, &input.handle).unwrap()
    });
    assert!(unsubmitted.publish().is_err());
    assert!(matches!(
        unsubmitted.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    ));

    let mut upload_only = ProviderOperationReceipt::new();
    upload_only.require_provider_execution(provider.device_id(), sum.clone());
    let output = new_output();
    with_provider_observation(Some(upload_only.recorder()), || {
        // An earlier upload may submit, but a provider hook that returns a
        // plausible output without a reduction kernel has no sum witness.
        upload_only.recorder().submission().unwrap().complete();
        super::super::publish_submitted_sum(provider, output.handle, &input.handle).unwrap()
    });
    assert!(matches!(
        upload_only.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    ));

    let mut submitted = ProviderOperationReceipt::new();
    submitted.require_provider_execution(provider.device_id(), sum);
    let output = new_output();
    with_provider_observation(Some(submitted.recorder()), || {
        let checkpoint = submitted.recorder().submission_checkpoint();
        submitted.recorder().submission().unwrap().complete();
        submitted
            .recorder()
            .confirm_submitted_provider_execution_after(
                provider.device_id(),
                &super::super::sum_observation_identity(invocation),
                checkpoint,
            );
        super::super::publish_submitted_sum(provider, output.handle, &input.handle).unwrap()
    });
    submitted.publish().unwrap();
    assert!(matches!(
        submitted.status(),
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));

    let mut unrelated = ProviderOperationReceipt::new();
    unrelated.require_provider_execution(
        provider.device_id(),
        runmat_accelerate_api::ProviderOperationIdentity::new("fixture.other"),
    );
    let output = new_output();
    with_provider_observation(Some(unrelated.recorder()), || {
        super::super::publish_submitted_sum(provider, output.handle, &input.handle).unwrap()
    });
    assert!(unrelated.publish().is_err());
    assert!(matches!(
        unrelated.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    ));

    let mut fallback = ProviderOperationReceipt::new();
    fallback.require_provider_execution(
        provider.device_id(),
        super::super::sum_observation_identity(invocation),
    );
    with_provider_observation(Some(fallback.recorder()), || {
        super::super::mark_sum_runtime_fallback(invocation)
    });
    assert!(matches!(
        fallback.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::SharedRuntimeFallback)
    ));

    let other = match invocation {
        runmat_types::SumInvocation::Sum => runmat_types::SumInvocation::NanSum,
        runmat_types::SumInvocation::NanSum => runmat_types::SumInvocation::Sum,
    };
    let mut wrong_identity = ProviderOperationReceipt::new();
    wrong_identity.require_provider_execution(
        provider.device_id(),
        super::super::sum_observation_identity(invocation),
    );
    with_provider_observation(Some(wrong_identity.recorder()), || {
        super::super::mark_sum_runtime_fallback(other);
    });
    assert!(matches!(
        wrong_identity.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    ));
}

#[test]
fn reduction_rejects_shape_class_owner_and_borrowed_alias_without_retiring_input() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    for (shape, element) in [
        (vec![2, 1], NumericElementType::F64),
        (vec![1, 2], NumericElementType::F32),
    ] {
        let output = provider
            .0
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)
            .unwrap();
        let error = admit_reduction(
            provider,
            &input.handle,
            &input.handle,
            output.handle,
            Some(0),
            OutputTemplate::Default,
        )
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(output.probe.retirements(), 1);
    }
    let complex = provider
        .0
        .allocate_zeroed(
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        complex.handle,
        Some(0),
        OutputTemplate::Default,
    )
    .unwrap_err();
    assert_eq!(complex.probe.retirements(), 1);
    let other_owner = TestAllocationOwner::new();
    let foreign = other_owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        foreign.handle,
        Some(0),
        OutputTemplate::Default,
    )
    .unwrap_err();
    assert_eq!(foreign.probe.retirements(), 1);

    let alias = input
        .handle
        .with_view(input.handle.view().clone().with_shape(vec![1, 2]))
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        alias,
        Some(0),
        OutputTemplate::Default,
    )
    .unwrap_err();
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn single_reduction_admits_double_intermediate_before_public_class_adaptation() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![3, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let output = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let admitted = admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        output.handle,
        Some(0),
        OutputTemplate::Native,
    )
    .unwrap();
    assert_eq!(admitted.physical().element_type(), NumericElementType::F64);
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn native_integer_stage_requires_same_physical_integer_class() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::U64, GpuTensorStorage::Real)
        .unwrap();
    let bad = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        bad.handle,
        Some(0),
        OutputTemplate::Native,
    )
    .unwrap_err();
    assert_eq!(bad.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn ordinary_integer_stage_rejects_already_saturated_native_output() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::I8, GpuTensorStorage::Real)
        .unwrap();
    let saturated = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::I8, GpuTensorStorage::Real)
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        saturated.handle,
        Some(0),
        OutputTemplate::Default,
    )
    .unwrap_err();
    assert_eq!(saturated.probe.retirements(), 1);
    let floating = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    admit_reduction(
        provider,
        &input.handle,
        &input.handle,
        floating.handle,
        Some(0),
        OutputTemplate::Double,
    )
    .expect("double reduction belongs to ordinary integer sum");
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn nan_map_requires_fresh_same_class_storage() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let bad = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    admit_nan_map(provider, &input.handle, &input.handle, bad.handle).unwrap_err();
    assert_eq!(bad.probe.retirements(), 1);
    admit_nan_map(provider, &input.handle, &input.handle, input.handle.clone()).unwrap_err();
    assert_eq!(input.probe.retirements(), 0);
    let good = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let accepted = admit_nan_map(provider, &input.handle, &input.handle, good.handle).unwrap();
    assert_eq!(accepted.shape(), input.handle.shape());
}

#[test]
fn single_omit_widening_and_map_keep_original_borrowed() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let wrong = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    admit_input_double(provider, &original.handle, wrong.handle).unwrap_err();
    assert_eq!(wrong.probe.retirements(), 1);
    admit_input_double(provider, &original.handle, original.handle.clone()).unwrap_err();
    assert_eq!(original.probe.retirements(), 0);
    let widened = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let widened_handle = admit_input_double(provider, &original.handle, widened.handle).unwrap();
    let mapped = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mapped_handle =
        admit_nan_map(provider, &original.handle, &widened_handle, mapped.handle).unwrap();
    assert_eq!(mapped_handle.shape(), original.handle.shape());
    widened_handle.release().unwrap();
    assert_eq!(widened.probe.retirements(), 1);
    assert_eq!(original.probe.retirements(), 0);
    original.handle.allocation_lease().validate_live().unwrap();
    mapped_handle.release().unwrap();
}

#[test]
fn failed_intermediate_retirement_rejects_the_next_output_with_cause() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let intermediate = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    intermediate
        .probe
        .fail_retirement("injected retirement failure");
    let next = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = advance_reduction(
        provider,
        &original.handle,
        intermediate.handle,
        next.handle,
        1,
        OutputTemplate::Default,
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert!(
        error.source.is_some(),
        "retirement cause must remain available"
    );
    assert_eq!(intermediate.probe.retirements(), 1);
    assert_eq!(next.probe.retirements(), 1);
    assert_eq!(original.probe.retirements(), 0);
}
