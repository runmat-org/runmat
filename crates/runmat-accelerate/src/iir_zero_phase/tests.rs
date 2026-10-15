use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    provider_iir_zero_phase_query, test_support::TestAllocationOwner, GpuTensorStorage,
    GpuTensorView, HostTensorView, NumericElementType, ProviderExecutionCompletionError,
    ProviderIirFilterResult, ProviderIirZeroPhaseIndexPlan, ProviderPreparedIirCoefficients,
    ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;

fn impulse(provider: &'static InProcessProvider, shape: [usize; 2], axis: usize) {
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            shape: &shape,
        })
        .expect("resident impulse");
    let query = provider_iir_zero_phase_query(&input, 2, axis, 3).expect("typed route geometry");
    assert!(provider.query_feasibility(&query).is_supported());
    let coefficients = ProviderPreparedIirCoefficients::upload(
        provider,
        &[0.5, 0.5],
        &[1.0, 0.0],
        provider.precision(),
        &[&input],
    )
    .expect("one normalized resident coefficient set");
    let cancellation = CancellationSource::new();
    let indices =
        ProviderIirZeroPhaseIndexPlan::prepare_direct(shape, 2, axis, 3, &cancellation.token())
            .expect("sealed reflection plan");
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
        indices: &indices,
        borrowed: &[],
        initial_state: &[0.5],
        axis,
        reflection_len: 3,
        capability_revision: provider.capability_snapshot().revision,
        cancellation: cancellation.token(),
    };
    let output = block_on(provider.iir_zero_phase(&request)).expect("resident two-pass route");
    assert_eq!(output.shape(), shape);
    assert!(!output.shares_allocation_with(&input));
    let actual = block_on(provider.download(&output)).expect("numerical oracle");
    assert_eq!(actual.shape, shape);
    for (got, want) in actual
        .data
        .iter()
        .zip([0.0, 0.0, 0.25, 0.5, 0.25, 0.0, 0.0, 0.0])
    {
        assert!((got - want).abs() < 1.0e-12, "got {got}, wanted {want}");
    }
    assert!(input.allocation_lease().validate_live().is_ok());
    assert!(coefficients
        .numerator()
        .allocation_lease()
        .validate_live()
        .is_ok());
}

#[test]
fn in_process_zero_phase_impulse_preserves_row_and_column_geometry() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    impulse(provider, [8, 1], 0);
    impulse(provider, [1, 8], 1);
}

#[test]
fn pre_cancelled_zero_phase_request_does_not_publish_output() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0; 8],
            shape: &[8, 1],
        })
        .unwrap();
    let coefficients = ProviderPreparedIirCoefficients::upload(
        provider,
        &[1.0, 0.0],
        &[1.0, 0.0],
        provider.precision(),
        &[&input],
    )
    .unwrap();
    let cancellation = CancellationSource::new();
    let indices =
        ProviderIirZeroPhaseIndexPlan::prepare_direct([8, 1], 2, 0, 3, &cancellation.token())
            .unwrap();
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
        indices: &indices,
        borrowed: &[],
        initial_state: &[0.0],
        axis: 0,
        reflection_len: 3,
        capability_revision: provider.capability_snapshot().revision,
        cancellation: cancellation.token(),
    };
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(provider.iir_zero_phase(&request))
        .expect_err("pre-cancelled route must not submit");
    assert!(!error.is_unsupported());
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn rejected_borrowed_stage_alias_does_not_retire_the_caller_operand() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = || {
        provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("resident source")
    };
    let input = source();
    let numerator = source();
    let denominator = source();
    let caller_coefficient = source();
    let borrowed = [&caller_coefficient];
    let mut stages = Stages::new(provider, [&input, &numerator, &denominator], &borrowed).unwrap();
    let rejection = stages.admit(caller_coefficient.clone(), &[2, 1], StageClass::Real);
    assert!(rejection.is_err());
    let owned = stages.admit(source(), &[2, 1], StageClass::Real).unwrap();
    let identity = provider.flip(stages.get(owned), &[1]).unwrap();
    assert_eq!(
        stages.admit_flip(identity, owned, 1, &[2, 1]).unwrap(),
        owned
    );
    assert!(stages.admit_flip(None, owned, 1, &[2, 1]).is_err());
    assert!(input.allocation_lease().validate_live().is_ok());
    assert!(caller_coefficient
        .allocation_lease()
        .validate_live()
        .is_ok());
    assert_eq!(
        block_on(provider.download(&caller_coefficient))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
}

#[test]
fn malformed_filter_pair_retires_both_fresh_results_without_publication() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = |shape: &[usize]| {
        provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape,
            })
            .expect("resident source")
    };
    let input = source(&[2, 1]);
    let numerator = source(&[2, 1]);
    let denominator = source(&[2, 1]);
    let output = source(&[2, 1]);
    let malformed_state = source(&[1, 2]);
    let output_weak = output.downgrade();
    let state_weak = malformed_state.downgrade();
    let mut stages = Stages::new(provider, [&input, &numerator, &denominator], &[]).unwrap();
    let result = stages.admit_filter_result(
        ProviderIirFilterResult {
            output,
            final_state: Some(malformed_state),
        },
        &[2, 1],
        &[2, 1],
        true,
    );
    let Err(error) = result else {
        panic!("malformed sibling must reject the whole callback result")
    };
    assert!(!stages.finish(Err(error)).unwrap_err().is_unsupported());
    assert!(output_weak.acquire().is_err());
    assert!(state_weak.acquire().is_err());
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn filter_pair_alias_to_borrowed_control_preserves_caller_and_retires_sibling() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = || {
        provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("resident source")
    };
    let input = source();
    let numerator = source();
    let denominator = source();
    let caller_control = source();
    let final_state = source();
    let state_weak = final_state.downgrade();
    let borrowed = [&caller_control];
    let mut stages = Stages::new(provider, [&input, &numerator, &denominator], &borrowed).unwrap();
    let result = stages.admit_filter_result(
        ProviderIirFilterResult {
            output: caller_control.clone(),
            final_state: Some(final_state),
        },
        &[2, 1],
        &[2, 1],
        true,
    );
    let Err(error) = result else {
        panic!("borrowed callback output must not enter intermediate stages")
    };
    assert!(!stages.finish(Err(error)).unwrap_err().is_unsupported());
    assert!(state_weak.acquire().is_err());
    assert!(caller_control.allocation_lease().validate_live().is_ok());
    assert_eq!(
        block_on(provider.download(&caller_control)).unwrap().data,
        vec![1.0, 2.0]
    );
}

#[test]
fn malformed_view_cannot_be_minted_from_a_live_physical_allocation() {
    let allocation = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(allocation
        .handle
        .with_view(GpuTensorView::contiguous(vec![3, 1]))
        .is_err());
    assert!(allocation.handle.allocation_lease().validate_live().is_ok());
}

#[test]
fn rejected_filter_pair_retains_primary_and_failed_retirement_owner() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = || {
        provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("provider operand")
    };
    let input = source();
    let numerator = source();
    let denominator = source();
    let owner = TestAllocationOwner::new();
    let failing = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let sibling = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    failing
        .probe
        .fail_retirement("fixture retained first output");
    let mut stages = Stages::new(provider, [&input, &numerator, &denominator], &[]).unwrap();
    let Err(rejection) = stages.admit_filter_result(
        ProviderIirFilterResult {
            output: failing.handle,
            final_state: Some(sibling.handle),
        },
        &[2, 1],
        &[2, 1],
        true,
    ) else {
        panic!("foreign callback pair must be rejected")
    };
    let error = stages.finish(Err(rejection)).unwrap_err();
    let completion = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<ProviderExecutionCompletionError>())
        .expect("primary and cleanup must retain their typed owners");
    assert!(completion.primary().is_some());
    assert_eq!(completion.cleanup_errors().len(), 1);
    assert!(!error.is_unsupported());
    assert_eq!(failing.probe.retirements(), 1);
    assert_eq!(sibling.probe.retirements(), 1);
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn mismatched_sealed_plan_rejects_before_provider_work() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0; 8],
            shape: &[8, 1],
        })
        .unwrap();
    let coefficients = ProviderPreparedIirCoefficients::upload(
        provider,
        &[1.0, 0.0],
        &[1.0, 0.0],
        provider.precision(),
        &[&input],
    )
    .unwrap();
    let cancellation = CancellationSource::new();
    let wrong_axis =
        ProviderIirZeroPhaseIndexPlan::prepare_direct([1, 8], 2, 1, 3, &cancellation.token())
            .unwrap();
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
        indices: &wrong_axis,
        borrowed: &[],
        initial_state: &[0.0],
        axis: 0,
        reflection_len: 3,
        capability_revision: provider.capability_snapshot().revision,
        cancellation: cancellation.token(),
    };
    provider.reset_telemetry();
    let before = provider.telemetry_snapshot();
    assert!(block_on(provider.iir_zero_phase(&request)).is_err());
    assert_eq!(provider.telemetry_snapshot(), before);
    assert!(input.allocation_lease().validate_live().is_ok());
}

fn identity_case(data: &[f64; 8]) -> ProviderExecutionResult<Vec<f64>> {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data,
            shape: &[8, 1],
        })
        .unwrap();
    let coefficients = ProviderPreparedIirCoefficients::upload(
        provider,
        &[1.0, 0.0],
        &[1.0, 0.0],
        provider.precision(),
        &[&input],
    )
    .unwrap();
    let cancellation = CancellationSource::new();
    let indices =
        ProviderIirZeroPhaseIndexPlan::prepare_direct([8, 1], 2, 0, 3, &cancellation.token())
            .unwrap();
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
        indices: &indices,
        borrowed: &[],
        initial_state: &[0.0],
        axis: 0,
        reflection_len: 3,
        capability_revision: provider.capability_snapshot().revision,
        cancellation: cancellation.token(),
    };
    let output = block_on(provider.iir_zero_phase(&request))?;
    Ok(block_on(provider.download(&output))
        .map_err(ProviderExecutionError::from)?
        .data)
}

fn has_fault(error: &ProviderExecutionError, expected: ProviderIirZeroPhaseFault) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if cause.downcast_ref::<ProviderIirZeroPhaseFault>() == Some(&expected) {
            return true;
        }
        current = cause.source();
    }
    false
}

#[test]
fn extreme_constant_reflection_stays_finite_without_clipping() {
    let samples = [1.0e308; 8];
    let actual = identity_case(&samples).expect("finite constant has a finite identity result");
    assert_eq!(actual, samples);
}

#[test]
fn nonfinite_input_and_unrepresentable_reflection_have_distinct_typed_faults() {
    let nonfinite = [f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error = identity_case(&nonfinite).unwrap_err();
    assert!(has_fault(&error, ProviderIirZeroPhaseFault::NonFiniteInput));

    let overflowing = [f64::MAX, 0.0, 0.0, -f64::MAX, 0.0, 0.0, 0.0, 0.0];
    let error = identity_case(&overflowing).unwrap_err();
    assert!(has_fault(
        &error,
        ProviderIirZeroPhaseFault::UnrepresentableReflection
    ));
}
