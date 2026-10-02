use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    provider_iir_zero_phase_query, HostTensorView, ProviderPreparedIirCoefficients,
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
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
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
    let request = ProviderIirZeroPhaseRequest {
        input: &input,
        coefficients: &coefficients,
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
    let mut stages = Stages::new(provider, [&input, &numerator, &denominator]).unwrap();
    let rejection = stages.admit(input.clone(), &[2, 1], StageClass::Real);
    assert!(rejection.is_err());
    assert!(input.allocation_lease().validate_live().is_ok());
    assert_eq!(
        block_on(provider.download(&input)).unwrap().data,
        vec![1.0, 2.0]
    );
}
