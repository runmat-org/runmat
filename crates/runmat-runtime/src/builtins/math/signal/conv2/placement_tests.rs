use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};
use runmat_execution::ExecutionCandidateKind;
use runmat_value::Value;

use crate::builtins::common::test_support;
use crate::builtins::math::signal::common::convolution::placement::tests::PlacementWitnessService;

use super::conv2_builtin;

#[test]
fn actual_inprocess_matrix_host_and_provider_routes_report_completed_work() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for prefer_provider in [false, true] {
        let service = PlacementWitnessService::new(prefer_provider);
        let signal = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap();
        let kernel = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0, 0.0, 1.0],
                shape: &[2, 2],
            })
            .unwrap();
        let result = block_on(conv2_builtin(
            Some(service.context()),
            Value::GpuTensor(signal.clone()),
            Value::GpuTensor(kernel.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("same")]),
        ))
        .expect("selected matrix convolution route");
        let Value::GpuTensor(_) = &result else {
            panic!("both complete routes must publish a resident output")
        };
        let gathered = test_support::gather(result).unwrap();
        assert_eq!(gathered.shape(), vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![5.0, 2.0, 3.0, 4.0]);
        assert_eq!(service.selections(), 1);
        assert_eq!(service.pending(), 0);
        let feedback = service.feedback();
        assert_eq!(feedback.len(), 1);
        assert!(feedback[0].succeeded);
        assert_eq!(
            feedback[0].kind,
            if prefer_provider {
                ExecutionCandidateKind::ProviderOperation
            } else {
                ExecutionCandidateKind::SharedRuntime
            }
        );
        assert!(signal.allocation_lease().validate_live().is_ok());
        assert!(kernel.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn unpolled_matrix_call_does_not_select_or_publish_feedback() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let service = PlacementWitnessService::new(false);
    let call = conv2_builtin(
        Some(service.context()),
        Value::GpuTensor(signal.clone()),
        Value::GpuTensor(kernel.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("same")]),
    );
    drop(call);
    assert_eq!(service.selections(), 0);
    assert!(service.feedback().is_empty());
    assert_eq!(service.pending(), 0);
    assert!(signal.allocation_lease().validate_live().is_ok());
    assert!(kernel.allocation_lease().validate_live().is_ok());
}
