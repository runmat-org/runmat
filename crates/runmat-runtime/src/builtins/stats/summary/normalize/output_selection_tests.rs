use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};
use runmat_value::Tensor;

fn eval() -> NormalizeEval {
    let scalar = || Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    NormalizeEval {
        n: scalar(),
        c: scalar(),
        s: scalar(),
    }
}

#[test]
fn requested_outputs_select_only_the_public_values() {
    for (requested, expected) in [
        (None, 1),
        (Some(0), 0),
        (Some(1), 1),
        (Some(2), 2),
        (Some(3), 3),
        (Some(4), 4),
    ] {
        let _guard = crate::output_count::push_output_count(requested).expect("output-count scope");
        let mut restorations = 0;
        let value = finish_selected(RequestedOutputs::current(), eval(), |value, _| {
            restorations += 1;
            Ok(value)
        })
        .unwrap();
        assert_eq!(restorations, expected.min(3));
        match (requested, value) {
            (None, Value::Tensor(_)) => {}
            (Some(_), Value::OutputList(values)) => assert_eq!(values.len(), expected),
            (_, other) => panic!("unexpected normalize output {other:?}"),
        }
    }
}

#[test]
fn later_restoration_failure_retires_earlier_output_but_not_borrowed_source() {
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let produced = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut first = Some(produced.handle);
    let error = finish_selected(RequestedOutputs::Explicit(2), eval(), |_, _| {
        if let Some(handle) = first.take() {
            Ok(Value::GpuTensor(handle))
        } else {
            Err(normalize_internal("later output restoration failed"))
        }
    })
    .unwrap_err();
    assert_eq!(error.message(), "later output restoration failed");
    assert_eq!(produced.probe.retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn late_failure_retains_provider_cause_and_cleanup_failure() {
    let owner = TestAllocationOwner::new();
    let produced = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    produced
        .probe
        .fail_retirement("normalization output cleanup fixture");
    let mut first = Some(produced.handle);
    let error = finish_selected(RequestedOutputs::Explicit(2), eval(), |_, _| {
        if let Some(handle) = first.take() {
            Ok(Value::GpuTensor(handle))
        } else {
            Err(output_error(
                crate::build_runtime_error("provider restoration failed")
                    .with_source(std::io::Error::other("provider fixture cause"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            ))
        }
    })
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::NORMALIZE_ERROR_INTERNAL.identifier
    );
    assert_eq!(error.context.builtin.as_deref(), Some("normalize"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("provider restoration failed"));
    assert!(error
        .message()
        .contains("normalization output cleanup fixture"));
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<std::io::Error>() {
        cause = cause.source().expect("retained provider cause");
    }
    assert_eq!(cause.to_string(), "provider fixture cause");
    assert_eq!(produced.probe.retirements(), 1);
}
