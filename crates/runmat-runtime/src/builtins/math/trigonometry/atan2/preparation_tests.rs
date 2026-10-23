use super::*;
use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use futures::{executor::block_on, task::noop_waker};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    task::{Context, Poll},
};

type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observation(Events);
impl BuiltinPreparationObservation for Observation {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, result: Result<&Value, &RuntimeError>) -> Result<(), RuntimeError> {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "failure" });
        Ok(())
    }
    fn abort(self: Box<Self>) {
        self.0.borrow_mut().push("abort");
    }
}
#[derive(Clone, Copy)]
enum Mode {
    Pass,
    Invalid,
    Pending,
}
struct Service(Events, Mode);
impl RuntimeAccelerationService for Service {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(
            request.admission().binding.builtin,
            runmat_builtins::ATAN2_CATALOG_ENTRY.identity
        );
        assert_eq!(request.admission().binding.variant, "default");
        assert_eq!(
            request.form().operation,
            runmat_builtins::BuiltinPreparationOperation::ElementwiseBinary
        );
        assert_eq!(request.form().data_arguments, &[0, 1]);
        assert_eq!(request.admission().input_count, 2);
        assert_eq!(request.admission().signature, 0);
        assert_eq!(request.operands().len(), 2);
        self.0.borrow_mut().push("prepare");
        let mode = self.1;
        let observation = Some(Box::new(Observation(self.0.clone())) as Box<_>);
        let prepared = request
            .into_operands()
            .with_unbounded_values(|mut operands| {
                if matches!(mode, Mode::Invalid) {
                    operands[1] = Value::String("invalid".into());
                }
                PreparedBuiltinOperands::new(operands, observation)
            })
            .and_then(std::convert::identity);
        Box::pin(async move {
            if matches!(mode, Mode::Pending) {
                std::future::pending::<()>().await;
            }
            prepared
        })
    }
}
fn context(events: &Events, mode: Mode) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(Service(events.clone(), mode))),
        )
}

#[test]
fn atan2_admission_precedes_service_for_both_operands_shapes_and_outputs() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    for values in [
        vec![Value::Num(1.0), Value::String("bad".into())],
        vec![Value::Complex(1.0, 0.0), Value::Num(1.0)],
        vec![Value::Num(1.0), Value::Complex(1.0, 0.0)],
        vec![
            Value::Tensor(Tensor::new(vec![0.0; 2], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0; 3], vec![1, 3]).unwrap()),
        ],
    ] {
        assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
            &context, "atan2", &values
        ))
        .is_err());
        assert!(events.borrow().is_empty());
    }
    assert!(block_on(
        crate::dispatcher::call_builtin_async_with_outputs_in_context(
            &context,
            "atan2",
            &[Value::Num(1.0), Value::Num(1.0)],
            2
        )
    )
    .is_err());
    assert!(events.borrow().is_empty());
}

#[test]
fn atan2_preparation_preserves_broadcast_single_empty_and_zero_outputs() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    for count in [0, 1] {
        for empty in [false, true] {
            events.borrow_mut().clear();
            let y = Value::Tensor(
                Tensor::from_f32(
                    if empty { vec![] } else { vec![0.0, 1.0] },
                    if empty { vec![0, 1, 2] } else { vec![2, 1, 1] },
                )
                .unwrap(),
            );
            let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3, 1]).unwrap());
            let expected = atan2_host(y.clone(), x.clone()).unwrap();
            let result = block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &context,
                    "atan2",
                    &[y, x],
                    count,
                ),
            )
            .unwrap();
            assert_eq!(result, expected);
            assert_eq!(*events.borrow(), ["prepare", "begin", "success"]);
        }
    }
}

#[test]
fn atan2_preparation_failure_and_abandonment_are_not_success() {
    for mode in [Mode::Invalid, Mode::Pending] {
        let events = Events::default();
        let context = context(&events, mode);
        let inputs = [Value::Num(1.0), Value::Num(2.0)];
        let mut call = Box::pin(crate::dispatcher::call_builtin_async_in_context(
            &context, "atan2", &inputs,
        ));
        if matches!(mode, Mode::Pending) {
            assert!(matches!(
                call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            drop(call);
            assert_eq!(*events.borrow(), ["prepare", "abort"]);
        } else {
            assert!(block_on(call).is_err());
            assert_eq!(*events.borrow(), ["prepare", "begin", "failure"]);
        }
    }
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    let inputs = [Value::Num(1.0), Value::Num(2.0)];
    drop(crate::dispatcher::call_builtin_async_in_context(
        &context, "atan2", &inputs,
    ));
    assert!(events.borrow().is_empty());
}
