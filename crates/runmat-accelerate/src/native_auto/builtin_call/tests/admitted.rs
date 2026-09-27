use super::*;
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};

fn logical_request() -> AdmittedBuiltinPreparation {
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name("logical").unwrap();
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: runmat_builtins::BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: Some(1),
        },
        vec![Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap())],
    )
    .unwrap()
}

#[test]
fn admitted_cpu_route_uses_existing_session_and_finishes_after_body() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "logical");
    assert!(probe.feedback.borrow().is_empty());
    futures::executor::block_on(prepared.execute(|_| async { Ok(Value::Bool(true)) })).unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn admitted_unpolled_future_and_panicking_body_abort_once() {
    for panic_body in [false, true] {
        let probe = Rc::new(Probe::default());
        let context = runtime(&probe);
        let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
        let prepared =
            super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
        if panic_body {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                futures::executor::block_on(prepared.execute(|_| async {
                    panic!("admitted body failure");
                }))
            }));
            assert!(outcome.is_err());
        } else {
            drop(prepared.execute(|_| async { Ok(Value::Bool(true)) }));
        }
        assert_eq!(probe.feedback.borrow().len(), 1);
        assert!(!probe.feedback.borrow()[0].succeeded);
    }
}

#[test]
fn installation_preserves_injected_acceleration_and_placement_authority() {
    let probe = Rc::new(Probe::default());
    let context = install_preparation_service(runtime(&probe));
    let original = context.service_ports().acceleration().unwrap().clone();
    let placement = context.service_ports().placement().unwrap().clone();
    let context = install_preparation_service(context);
    assert!(Rc::ptr_eq(
        &original,
        context.service_ports().acceleration().unwrap()
    ));
    assert!(Rc::ptr_eq(
        &placement,
        context.service_ports().placement().unwrap()
    ));
}

#[test]
fn canonical_calls_are_not_speculatively_prepared_by_legacy_route() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let request = logical_request();
    let prepared = futures::executor::block_on(prepare_builtin_call(
        "logical",
        request.operands(),
        &context,
    ))
    .unwrap();
    assert_eq!(prepared.args(), request.operands());
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn admitted_upload_failure_keeps_acceleration_error_identity() {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let error =
        match super::super::admitted::prepare_with(&auto, &runtime(&probe), logical_request()) {
            Err(error) => error,
            Ok(_) => panic!("provider rejects upload"),
        };
    assert_eq!(
        error.identifier(),
        Some("RunMat:AccelerationOperationFailed")
    );
    assert!(error.message().contains("prepare builtin args"));
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}
