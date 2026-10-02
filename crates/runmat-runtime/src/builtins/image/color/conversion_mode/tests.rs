use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::{dispatcher, RuntimeError};
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinOperandPreparation, BuiltinPreparationOperation, BuiltinSinkPreparation,
};
use runmat_value::{IntValue, Value};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    task::{Context, Poll},
};

const NAMES: [&str; 3] = ["im2double", "im2uint8", "im2uint16"];
type Events = Rc<RefCell<Vec<&'static str>>>;

struct Observer(Events);
impl BuiltinPreparationObservation for Observer {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "failure" });
        Ok(())
    }
    fn abort(self: Box<Self>) {
        self.0.borrow_mut().push("abort");
    }
}

struct Service {
    events: Events,
    pause_after_preparation: bool,
}
impl RuntimeAccelerationService for Service {
    fn supports_operation(&self, operation: &str) -> bool {
        NAMES.contains(&operation)
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert!(NAMES.contains(&request.admission().binding.builtin.name));
        assert_eq!(
            request.form().operation,
            BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather)
        );
        assert_eq!(request.form().data_arguments, &[0]);
        self.events.borrow_mut().push("prepare");
        let operands = request.into_operands();
        let observer = Observer(self.events.clone());
        let pause = self.pause_after_preparation;
        Box::pin(async move {
            let prepared = PreparedBuiltinOperands::new(operands, Some(Box::new(observer)));
            if pause {
                std::future::pending::<()>().await;
            }
            Ok(prepared)
        })
    }
}

fn enter(events: &Events) -> RuntimeContextGuard {
    enter_with_pause(events, false)
}

fn enter_with_pause(events: &Events, pause_after_preparation: bool) -> RuntimeContextGuard {
    RuntimeContextGuard::enter(
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_acceleration(Rc::new(
                Service {
                    events: events.clone(),
                    pause_after_preparation,
                },
            ))),
    )
}

#[test]
fn all_three_declare_one_typed_gather_sink_for_both_forms() {
    for name in NAMES {
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            panic!("missing typed preparation form");
        };
        assert_eq!(forms.len(), 1);
        assert_eq!(forms[0].signatures, &[0, 1]);
        assert_eq!(forms[0].data_arguments, &[0]);
        assert_eq!(
            forms[0].operation,
            BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather)
        );
        assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
    }
}

#[test]
fn valid_calls_prepare_once_and_invalid_controls_do_not_prepare() {
    let events = Events::default();
    let _context = enter(&events);
    for name in NAMES {
        block_on(dispatcher::call_builtin_async(name, &[Value::Num(1.0)])).unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();

        block_on(dispatcher::call_builtin_async(
            name,
            &[Value::Int(IntValue::U8(1)), Value::String("indexed".into())],
        ))
        .unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();

        for args in [
            vec![Value::Num(1.0), Value::String("bad".into())],
            vec![Value::Int(IntValue::U64(u64::MAX))],
            vec![
                Value::Num(1.0),
                Value::String("indexed".into()),
                Value::Num(2.0),
            ],
        ] {
            assert!(block_on(dispatcher::call_builtin_async(name, &args)).is_err());
            assert!(events.borrow().is_empty(), "{name}: {:?}", events.borrow());
        }
    }
}

#[test]
fn unpolled_calls_and_invalid_requested_output_count_have_no_service_effect() {
    let events = Events::default();
    let _context = enter(&events);
    for name in NAMES {
        let args = [Value::Num(1.0)];
        drop(dispatcher::call_builtin_async(name, &args));
        assert!(events.borrow().is_empty());
        assert!(block_on(dispatcher::call_builtin_async_with_outputs(name, &args, 2)).is_err());
        assert!(events.borrow().is_empty());
        block_on(dispatcher::call_builtin_async_with_outputs(name, &args, 0)).unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();
    }
}

#[test]
fn malformed_indexed_control_precedes_even_an_unregistered_resident_owner() {
    let events = Events::default();
    let _context = enter(&events);
    for name in NAMES {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let handle = owner
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        let error = block_on(dispatcher::call_builtin_async(
            name,
            &[
                Value::GpuTensor(handle),
                Value::String("not-indexed".into()),
            ],
        ))
        .unwrap_err();
        let expected = format!("RunMat:{name}:InvalidInput");
        assert_eq!(error.identifier(), Some(expected.as_str()));
        assert!(events.borrow().is_empty());
    }
}

#[test]
fn dropping_polled_conversion_call_aborts_prepared_observation_once() {
    let events = Events::default();
    let _context = enter_with_pause(&events, true);
    let args = [Value::Num(0.5)];
    let mut call = Box::pin(dispatcher::call_builtin_async("im2uint8", &args));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(&*events.borrow(), &["prepare"]);
    drop(call);
    assert_eq!(&*events.borrow(), &["prepare", "abort"]);
}
