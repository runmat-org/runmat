use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};
use std::{cell::RefCell, rc::Rc};

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
struct Service {
    events: Events,
    replace: bool,
}
impl RuntimeAccelerationService for Service {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let admission = request.admission();
        assert!(["double", "single"].contains(&admission.binding.builtin.name));
        assert_eq!(
            request.form().id.0,
            if admission.signature == 0 { 1 } else { 2 },
            "the admitted owner selects the stable plain or runtime-owned like form explicitly; overlapping legacy signatures must not select form 0"
        );
        assert_eq!(
            request.form().operation,
            if admission.signature == 0 {
                runmat_builtins::BuiltinPreparationOperation::FloatingCast(
                    if admission.binding.builtin.name == "single" {
                        runmat_types::NumericClass::Single
                    } else {
                        runmat_types::NumericClass::Double
                    },
                )
            } else {
                runmat_builtins::BuiltinPreparationOperation::RuntimeOwned
            }
        );
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(
            admission.input_count,
            if admission.signature == 0 { 1 } else { 3 }
        );
        self.events.borrow_mut().push("prepare");
        let replace = self.replace;
        let observer = Observation(self.events.clone());
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(|mut operands| {
                    assert_eq!(
                        operands.len(),
                        1,
                        "option and prototype must stay domain-local"
                    );
                    if replace {
                        operands[0] = Value::output_list(vec![]);
                    }
                    PreparedBuiltinOperands::new(operands, Some(Box::new(observer)))
                })
                .and_then(std::convert::identity)
        })
    }
}
fn runtime(events: &Events, replace: bool) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service {
                events: events.clone(),
                replace,
            })),
        )
}

#[test]
fn base_and_like_forms_prepare_only_data_and_observe_execution_once() {
    let events = Events::default();
    let runtime = runtime(&events, false);
    runtime.set_runmat_extensions_enabled(true);
    for name in ["double", "single"] {
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
        assert_eq!(
            entry.contract.async_behavior,
            runmat_builtins::BuiltinAsyncBehavior::MaySuspend
        );
        for shape in [vec![0, 2, 3], vec![1, 1, 2]] {
            let len = shape.iter().product();
            for like in [false, true] {
                let mut args = vec![Value::Tensor(
                    Tensor::new(vec![1.25; len], shape.clone()).unwrap(),
                )];
                if like {
                    args.extend([Value::from("like"), Value::Num(0.0)]);
                }
                let result = block_on(crate::dispatcher::call_builtin_async_in_context(
                    &runtime, name, &args,
                ))
                .unwrap();
                let Value::Tensor(output) = result.value() else {
                    panic!("dense shape-preserving output");
                };
                assert_eq!(output.shape(), shape.as_slice());
                assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
                events.borrow_mut().clear();
            }
        }
    }
}

#[test]
fn invalid_options_prototypes_and_output_counts_never_prepare() {
    let events = Events::default();
    let runtime = runtime(&events, false);
    runtime.set_runmat_extensions_enabled(true);
    for name in ["double", "single"] {
        for args in [
            vec![
                Value::Num(1.0),
                Value::CharArray(
                    runmat_value::CharArray::from_unicode_scalars_with_shape(
                        "like".chars().collect(),
                        vec![1, 2, 2],
                    )
                    .unwrap(),
                ),
                Value::Num(0.0),
            ],
            vec![Value::Num(1.0), Value::from("invalid"), Value::Num(0.0)],
            vec![Value::Num(1.0), Value::from("like"), Value::from("invalid")],
            vec![Value::output_list(vec![])],
        ] {
            assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
                &runtime, name, &args
            ))
            .is_err());
            assert!(events.borrow().is_empty());
        }
        assert!(block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                name,
                &[Value::Num(1.0)],
                2
            )
        )
        .is_err());
        assert!(events.borrow().is_empty());
    }
}

#[test]
fn deliberate_service_replacement_is_revalidated_by_conversion_body() {
    let events = Events::default();
    let runtime = runtime(&events, true);
    for name in ["double", "single"] {
        drop(crate::dispatcher::call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::Num(1.0)],
        ));
        assert!(events.borrow().is_empty());
        assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::Num(1.0)]
        ))
        .is_err());
        assert_eq!(&*events.borrow(), &["prepare", "begin", "failure"]);
        events.borrow_mut().clear();
    }
}

#[test]
fn pending_provider_abandonment_aborts_the_shared_observation_once() {
    use super::super::test_support::{with_conversion_hook_provider, HookBehavior};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use std::future::Future;
    use std::task::{Context, Poll};
    let events = Events::default();
    let runtime = runtime(&events, false);
    for name in ["double", "single"] {
        with_conversion_hook_provider(HookBehavior::PendingDirectHook, |provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.25],
                    shape: &[1, 1],
                })
                .unwrap();
            let args = [Value::GpuTensor(input.clone())];
            let mut call = Box::pin(crate::dispatcher::call_builtin_async_in_context(
                &runtime, name, &args,
            ));
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
            assert_eq!(&*events.borrow(), &["prepare", "begin"]);
            drop(call);
            assert_eq!(&*events.borrow(), &["prepare", "begin", "abort"]);
            assert!(!runmat_accelerate_api::handle_is_logical(&input));
            let physical = input.allocation_lease().downgrade();
            drop(args);
            input.release().expect("release caller");
            assert!(physical.acquire().is_none());
        });
        events.borrow_mut().clear();
    }
}

#[test]
fn string_conversion_inference_matches_distinct_runtime_double_and_single_policies() {
    let input = Value::from("1.25");
    for (name, target) in [
        ("double", runmat_types::NumericClass::Double),
        ("single", runmat_types::NumericClass::Single),
    ] {
        let inferred =
            runmat_types::infer_numeric_conversion(&crate::value_fact::value_fact(&input), target);
        let actual = block_on(crate::dispatcher::call_builtin_async(
            name,
            std::slice::from_ref(&input),
        ));
        match actual {
            Ok(value) => {
                assert!(inferred.diagnostics.is_empty());
                let actual = crate::value_fact::value_fact(&value);
                assert_eq!(inferred.fact.kind, actual.kind);
                assert_eq!(inferred.fact.storage, actual.storage);
            }
            Err(_) => assert!(!inferred.diagnostics.is_empty()),
        }
    }
}
