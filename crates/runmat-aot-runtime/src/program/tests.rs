use super::*;
use runmat_runtime::context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts};
use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

type Events = Rc<RefCell<Vec<&'static str>>>;
struct PlacementProbe(Events);
impl RuntimePlacementService for PlacementProbe {
    fn plan(
        &self,
        _: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, runmat_runtime::RuntimeError> {
        panic!("lifecycle test must not fabricate a placement decision")
    }
    fn observe(
        &self,
        _: runmat_execution::PlacementFeedback,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        Ok(())
    }
    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
    fn drain_feedback(&self) {
        self.0.borrow_mut().push("drain");
    }
}
impl Drop for PlacementProbe {
    fn drop(&mut self) {
        self.0.borrow_mut().push("session_drop");
    }
}
fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

#[test]
fn native_runtime_failure_keeps_original_source_and_frames() {
    use std::error::Error as _;

    let original = runmat_runtime::build_runtime_error("aot source failure")
        .with_identifier("RunMat:test:AotSource")
        .with_call_frames(vec![runmat_runtime::CallFrame {
            function: "source_callback".into(),
            source_id: Some(3),
            span: Some((2, 5)),
            source_owner: None,
            diagnostic_source: None,
        }])
        .with_source(std::io::Error::other("typed source cause"))
        .build();
    let mapped = native_executor_error(runmat_native_executor::NativeExecutorError::Runtime(
        Box::new(original),
    ));
    assert_eq!(mapped.identifier(), Some("RunMat:test:AotSource"));
    assert_eq!(mapped.message, "aot source failure");
    assert_eq!(mapped.context.call_frames[0].source_id, Some(3));
    assert!(mapped.source().unwrap().is::<std::io::Error>());
}

#[test]
fn standalone_source_admission_keeps_missing_and_retired_causes() {
    use std::error::Error as _;

    for cause in [
        runmat_runtime::context::ProgramSourceSelectionError::Missing(
            runmat_types::ProgramSourceId(1),
        ),
        runmat_runtime::context::ProgramSourceSelectionError::Retired(
            runmat_types::ProgramSourceId(1),
        ),
    ] {
        let error = native_executor_error(
            runmat_native_executor::NativeExecutorError::SourceAdmission(cause),
        );
        assert_eq!(error.identifier(), Some("RunMat:AotSourceAdmission"));
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<runmat_runtime::context::ProgramSourceSelectionError>(),
            Some(&cause)
        );
    }
}

#[test]
fn installation_preserves_supplied_services_across_nested_program_contexts() {
    let events = Events::default();
    let placement: Rc<dyn RuntimePlacementService> = Rc::new(PlacementProbe(events));
    let acceleration: Rc<dyn runmat_runtime::context::RuntimeAccelerationService> =
        Rc::new(runmat_accelerate::native_auto::AdmittedAcceleration);
    let original = runtime().with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(placement.clone())
            .with_acceleration(acceleration.clone()),
    );
    let installed = install_acceleration_services(original.clone());
    let nested = isolated_runtime(installed.clone());
    for context in [&original, &installed, &nested] {
        assert!(Rc::ptr_eq(
            context.service_ports().placement().unwrap(),
            &placement
        ));
        assert!(Rc::ptr_eq(
            context.service_ports().acceleration().unwrap(),
            &acceleration
        ));
    }
    let fresh = install_acceleration_services(runtime());
    let again = install_acceleration_services(fresh.clone());
    assert!(Rc::ptr_eq(
        fresh.service_ports().placement().unwrap(),
        again.service_ports().placement().unwrap()
    ));
    let other = install_acceleration_services(runtime());
    assert!(!Rc::ptr_eq(
        fresh.service_ports().placement().unwrap(),
        other.service_ports().placement().unwrap()
    ));
}

struct BodyOwner(Events);
impl Drop for BodyOwner {
    fn drop(&mut self) {
        self.0.borrow_mut().push("body_drop");
    }
}

#[test]
fn terminal_scope_retains_session_through_teardown_rendering_and_body_destruction() {
    for failure in [false, true] {
        let events = Events::default();
        let context = install_acceleration_services(runtime().with_service_ports(
            RuntimeServicePorts::default().with_placement(Rc::new(PlacementProbe(events.clone()))),
        ));
        let owner = BodyOwner(events.clone());
        let inner_events = events.clone();
        let future = context.scope(async move {
            let _owner = owner;
            inner_events.borrow_mut().push("foreign_teardown");
            if failure {
                return Err("execution error");
            }
            crate::output::value(&Value::Num(17.0));
            inner_events.borrow_mut().push("rendered");
            Ok(())
        });
        let result = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future);
        assert_eq!(result.is_err(), failure);
        // The caller's original context still owns the session after the final
        // scope drains; terminal publication must not retire that authority.
        assert!(!events.borrow().contains(&"session_drop"));
        drop(context);
        let expected = if failure {
            vec!["foreign_teardown", "body_drop", "drain", "session_drop"]
        } else {
            vec![
                "foreign_teardown",
                "rendered",
                "body_drop",
                "drain",
                "session_drop",
            ]
        };
        assert_eq!(*events.borrow(), expected);
    }
}

#[test]
fn abandoning_polled_or_unpolled_scope_drops_body_before_final_drain() {
    for poll in [false, true] {
        let events = Events::default();
        let context = install_acceleration_services(runtime().with_service_ports(
            RuntimeServicePorts::default().with_placement(Rc::new(PlacementProbe(events.clone()))),
        ));
        let owner = BodyOwner(events.clone());
        let mut future = context.scope(async move {
            let _owner = owner;
            std::future::pending::<()>().await;
        });
        if poll {
            assert!(matches!(
                Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
        }
        drop(future);
        assert_eq!(*events.borrow(), ["body_drop", "drain"]);
        drop(context);
        assert_eq!(*events.borrow(), ["body_drop", "drain", "session_drop"]);
    }
}
