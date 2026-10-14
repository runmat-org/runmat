use super::*;
use runmat_accelerate_api::observation::{
    current_provider_observation, with_provider_observation, ProviderExecutionDisposition,
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
    ProviderOperationRecorder,
};

struct Observed {
    recorder: ProviderOperationRecorder,
    events: Rc<RefCell<Vec<&'static str>>>,
}
impl Drop for Observed {
    fn drop(&mut self) {
        // Even unpolled destruction must run inside this operation's scope.
        if let Ok(token) = current_provider_observation().unwrap().submission() {
            token.complete();
        }
        self.events.borrow_mut().push("drop");
    }
}
impl BuiltinPreparationObservation for Observed {
    fn provider_recorder(&self) -> Option<ProviderOperationRecorder> {
        Some(self.recorder.clone())
    }
    fn begin_execution(&mut self) {
        self.events.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, _: Result<&Value, &RuntimeError>) -> Result<(), RuntimeError> {
        self.events.borrow_mut().push("finish");
        Ok(())
    }
    fn abort(self: Box<Self>) {
        self.events.borrow_mut().push("abort");
    }
}

#[test]
fn prepared_body_and_unpolled_or_pending_drop_keep_their_own_recorder() {
    for poll in [false, true] {
        let mut inner = ProviderOperationReceipt::new();
        let mut outer = ProviderOperationReceipt::new();
        let events = Rc::new(RefCell::new(Vec::new()));
        let prepared = PreparedBuiltinOperands::new(
            vec![],
            Some(Box::new(Observed {
                recorder: inner.recorder(),
                events: events.clone(),
            })),
        )
        .expect("root observed operands");
        let mut future = Box::pin(execute_prepared(prepared, |_| std::future::pending()));
        with_provider_observation(Some(outer.recorder()), || {
            if poll {
                assert!(future
                    .as_mut()
                    .poll(&mut Context::from_waker(&noop_waker()))
                    .is_pending());
            }
            drop(future);
        });
        assert_eq!(
            *events.borrow(),
            if poll {
                vec!["begin", "abort", "drop"]
            } else {
                vec!["abort", "drop"]
            }
        );
        inner.publish().unwrap();
        outer.publish().unwrap();
        assert!(matches!(
            inner.status(),
            ProviderObservationStatus::Completed { submissions: 1, .. }
        ));
        assert!(matches!(
            outer.status(),
            ProviderObservationStatus::Completed { submissions: 0, .. }
        ));
    }
}

#[test]
fn internal_fallback_invalidates_body_receipt_even_if_body_succeeds() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    let events = Rc::new(RefCell::new(Vec::new()));
    let prepared = PreparedBuiltinOperands::new(
        vec![],
        Some(Box::new(Observed {
            recorder: receipt.recorder(),
            events: events.clone(),
        })),
    )
    .expect("root observed operands");
    block_on(execute_prepared(prepared, |_| async {
        current_provider_observation()
            .unwrap()
            .execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback);
        super::owned(Value::Num(1.0))
    }))
    .unwrap();
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::SharedRuntimeFallback)
    );
    assert_eq!(*events.borrow(), ["begin", "finish", "drop"]);
}
