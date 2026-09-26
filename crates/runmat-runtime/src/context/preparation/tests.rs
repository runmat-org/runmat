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

#[test]
fn domain_admitted_output_count_including_padding_is_preserved() {
    for requested_outputs in [None, Some(0), Some(1), Some(4)] {
        let request = request_with(
            BuiltinPreparationAdmission {
                requested_outputs,
                ..admission()
            },
            vec![Value::Num(1.0)],
        )
        .unwrap();
        assert_eq!(request.admission().requested_outputs, requested_outputs);
    }
}

struct OutputCountService {
    admitted: Rc<RefCell<Vec<Option<usize>>>>,
    events: Events,
}

impl RuntimeAccelerationService for OutputCountService {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }

    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        self.admitted
            .borrow_mut()
            .push(request.admission().requested_outputs);
        self.events.borrow_mut().push("prepare");
        let observation = Observation(self.events.clone());
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::new(
                request.into_operands(),
                Some(Box::new(observation)),
            ))
        })
    }
}

#[test]
fn admitted_zero_and_padded_outputs_execute_body_once_with_and_without_service() {
    for installed in [false, true] {
        for requested_outputs in [None, Some(0), Some(1), Some(4)] {
            let admitted = Rc::new(RefCell::new(Vec::new()));
            let events = Events::default();
            let mut context = context();
            if installed {
                context = context.with_service_ports(
                    RuntimeServicePorts::default().with_acceleration(Rc::new(OutputCountService {
                        admitted: admitted.clone(),
                        events: events.clone(),
                    })),
                );
            }
            let request = request_with(
                BuiltinPreparationAdmission {
                    requested_outputs,
                    ..admission()
                },
                vec![Value::Num(3.0)],
            )
            .unwrap();
            // The domain, not preparation, owns output construction/padding.
            // Zero outputs must still invoke its effectful admitted body once.
            let domain_result =
                Value::OutputList(vec![Value::Num(3.0); requested_outputs.unwrap_or(1)]);
            let expected = domain_result.clone();
            let body_events = events.clone();
            let result = block_on(execute_admitted(&context, request, |operands| async move {
                body_events.borrow_mut().push("body");
                assert_eq!(operands, vec![Value::Num(3.0)]);
                Ok(domain_result)
            }))
            .unwrap();
            assert_eq!(result, expected);
            if installed {
                assert_eq!(*admitted.borrow(), vec![requested_outputs]);
                assert_eq!(
                    *events.borrow(),
                    vec!["prepare", "begin", "body", "success"]
                );
            } else {
                assert!(admitted.borrow().is_empty());
                assert_eq!(*events.borrow(), vec!["body"]);
            }
        }
    }
}

#[test]
fn ambiguous_selected_form_or_binding_is_rejected() {
    let mut entry = SIN_CATALOG_ENTRY;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(&[FORM, FORM]);
    assert!(AdmittedBuiltinPreparation::from_entry(
        Box::leak(Box::new(entry)),
        admission(),
        vec![Value::Num(1.0)]
    )
    .is_err());
    let mut entry = SIN_CATALOG_ENTRY;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(&[FORM]);
    entry.bindings = &[
        BuiltinBindingDeclaration {
            variant: "default",
            availability: BuiltinBindingAvailability::Required,
        },
        BuiltinBindingDeclaration {
            variant: "default",
            availability: BuiltinBindingAvailability::Required,
        },
    ];
    assert!(AdmittedBuiltinPreparation::from_entry(
        Box::leak(Box::new(entry)),
        admission(),
        vec![Value::Num(1.0)]
    )
    .is_err());
}

#[test]
fn malformed_selected_signature_returns_error_without_panicking() {
    let mut entry = SIN_CATALOG_ENTRY;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(&[BuiltinPreparationForm {
        signatures: &[99],
        ..FORM
    }]);
    assert!(AdmittedBuiltinPreparation::from_entry(
        Box::leak(Box::new(entry)),
        BuiltinPreparationAdmission {
            signature: 99,
            ..admission()
        },
        vec![Value::Num(1.0)]
    )
    .is_err());
}

type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observation(Events);

struct RejectedCompletion;
impl BuiltinPreparationObservation for RejectedCompletion {
    fn finish(self: Box<Self>, _: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
        Err(RuntimeError::new(
            "result ownership could not be established",
        ))
    }
    fn abort(self: Box<Self>) {}
}

#[test]
fn failed_completion_does_not_publish_a_successful_body_result() {
    let prepared = PreparedBuiltinOperands::new(vec![], Some(Box::new(RejectedCompletion)));
    let result = futures::executor::block_on(execute_prepared(prepared, |_| async {
        Ok(Value::Num(42.))
    }));
    assert!(result.is_err());
}

impl BuiltinPreparationObservation for Observation {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "error" });
        Ok(())
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
