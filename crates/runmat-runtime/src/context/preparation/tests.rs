use super::*;
use crate::context::{RuntimeAccelerationService, RuntimeServiceFuture, RuntimeServicePorts};
use futures::{executor::block_on, task::noop_waker};
use runmat_builtins::*;
use std::{
    cell::RefCell,
    rc::Rc,
    task::{Context, Poll},
};

const FORM: BuiltinPreparationForm = BuiltinPreparationForm {
    id: BuiltinPreparationFormId(7),
    binding_variant: "default",
    signatures: &[0, 1],
    data_arguments: &[0],
    operation: BuiltinPreparationOperation::Unary,
    provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
};

fn admission() -> BuiltinPreparationAdmission {
    BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: SIN_CATALOG_ENTRY.identity,
            variant: "default",
        },
        form: FORM.id,
        signature: 0,
        input_count: 1,
        requested_outputs: Some(1),
    }
}

fn request_with(
    admission: BuiltinPreparationAdmission,
    values: Vec<Value>,
) -> Result<AdmittedBuiltinPreparation, RuntimeError> {
    let mut entry = SIN_CATALOG_ENTRY;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(&[FORM]);
    AdmittedBuiltinPreparation::from_entry(Box::leak(Box::new(entry)), admission, values)
}

fn request() -> AdmittedBuiltinPreparation {
    request_with(admission(), vec![Value::Num(3.0)]).unwrap()
}

fn context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

#[test]
fn declaration_witness_rejects_inconsistent_roles_before_service() {
    assert!(request_with(admission(), vec![]).is_err());
    assert!(request_with(
        BuiltinPreparationAdmission {
            form: BuiltinPreparationFormId(8),
            ..admission()
        },
        vec![Value::Num(1.0)]
    )
    .is_err());
    assert!(request_with(
        BuiltinPreparationAdmission {
            input_count: 0,
            ..admission()
        },
        vec![Value::Num(1.0)]
    )
    .is_err());
    assert!(request_with(
        BuiltinPreparationAdmission {
            signature: 99,
            ..admission()
        },
        vec![Value::Num(1.0)]
    )
    .is_err());
    assert_eq!(request().form().data_arguments, &[0]);
    assert_eq!(request().admission().requested_outputs, Some(1));
}

#[test]
fn admitted_like_controls_never_cross_the_preparation_boundary() {
    let request = request_with(
        BuiltinPreparationAdmission {
            signature: 1,
            input_count: 3,
            ..admission()
        },
        vec![Value::Num(3.0)],
    )
    .unwrap();
    assert_eq!(request.operands().len(), 1);
    assert_eq!(request.admission().input_count, 3);
    assert_eq!(request.form().data_arguments, &[0]);
}

type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observation(Events);
impl BuiltinPreparationObservation for Observation {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>) {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "error" });
    }
    fn abort(self: Box<Self>) {
        self.0.borrow_mut().push("abort");
    }
}
fn prepared(events: &Events) -> PreparedBuiltinOperands {
    PreparedBuiltinOperands::new(
        vec![Value::Num(3.0)],
        Some(Box::new(Observation(events.clone()))),
    )
}

#[test]
fn completion_observes_success_and_failure_exactly_once() {
    for succeed in [true, false] {
        let events = Events::default();
        let result = block_on(execute_prepared(
            prepared(&events),
            |mut values| async move {
                if succeed {
                    Ok(values.remove(0))
                } else {
                    Err(request::invalid("body failed"))
                }
            },
        ));
        assert_eq!(result.is_ok(), succeed);
        if let Ok(value) = result {
            assert!(matches!(value, Value::Num(3.0)));
        }
        assert_eq!(
            *events.borrow(),
            vec!["begin", if succeed { "success" } else { "error" }]
        );
    }
}

#[test]
fn cancellation_before_and_during_execution_aborts_without_success() {
    let events = Events::default();
    drop(execute_prepared(prepared(&events), |_| async {
        Ok(Value::Num(0.0))
    }));
    assert_eq!(*events.borrow(), vec!["abort"]);
    events.borrow_mut().clear();
    let mut future = Box::pin(execute_prepared(prepared(&events), |_| {
        std::future::pending()
    }));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(*events.borrow(), vec!["begin", "abort"]);
}

#[test]
fn unwind_aborts_once() {
    let events = Events::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block_on(execute_prepared(prepared(&events), |_| async {
            panic!("body panic")
        }))
    }));
    assert!(result.is_err());
    assert_eq!(*events.borrow(), vec!["begin", "abort"]);
}

struct MalformedService(Events);
impl RuntimeAccelerationService for MalformedService {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        _: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let observer = Observation(self.0.clone());
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::new(
                vec![],
                Some(Box::new(observer)),
            ))
        })
    }
}

#[test]
fn malformed_service_response_aborts_without_executing() {
    let events = Events::default();
    let context = context().with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(MalformedService(events.clone()))),
    );
    assert!(block_on(execute_admitted(&context, request(), |_| async {
        panic!("must not execute")
    }))
    .is_err());
    assert_eq!(*events.borrow(), vec!["abort"]);
}

struct DefaultService;
impl RuntimeAccelerationService for DefaultService {
    fn supports_operation(&self, _: &str) -> bool {
        false
    }
}

#[test]
fn absent_or_default_service_preserves_runtime_owned_execution() {
    let contexts = [
        context(),
        context().with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(DefaultService)),
        ),
    ];
    for context in contexts {
        let result = block_on(execute_admitted(
            &context,
            request(),
            |mut values| async move { Ok(values.remove(0)) },
        ))
        .unwrap();
        assert!(matches!(result, Value::Num(3.0)));
    }
}
