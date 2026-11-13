use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::{executor::block_on, task::noop_waker};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_value::{Tensor, Value};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    task::{Context, Poll},
};

const NAMES: &[&str] = &["sin", "cos", "tan", "atan", "sinh", "cosh", "asinh", "tanh"];
type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observer(Events);
impl BuiltinPreparationObservation for Observer {
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
struct Service {
    events: Events,
    mode: Mode,
}
impl RuntimeAccelerationService for Service {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert!(NAMES.contains(&request.admission().binding.builtin.name));
        assert_eq!(request.form().operation, BuiltinPreparationOperation::Unary);
        assert_eq!(request.form().data_arguments, &[0]);
        assert!(request.admission().signature <= 1);
        assert_eq!(
            request.admission().input_count,
            if request.admission().signature == 0 {
                1
            } else {
                3
            }
        );
        assert_eq!(request.operands().len(), 1);
        self.events.borrow_mut().push("prepare");
        let mode = self.mode;
        let observation = Some(Box::new(Observer(self.events.clone())) as Box<_>);
        let prepared = request
            .into_operands()
            .with_unbounded_values(|values| {
                let values = if matches!(mode, Mode::Invalid) {
                    vec![Value::String("invalid service result".into())]
                } else {
                    values
                };
                PreparedBuiltinOperands::new(values, observation)
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
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service {
                events: events.clone(),
                mode,
            })),
        )
}

#[test]
fn floating_trig_admission_and_output_errors_precede_preparation() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    for name in NAMES {
        for args in [
            vec![],
            vec![Value::String("bad".into())],
            vec![Value::Num(0.0), Value::Num(1.0)],
        ] {
            assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
                &context, name, &args
            ))
            .is_err());
            assert!(events.borrow().is_empty());
        }
        assert!(block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[Value::Num(0.0)],
                2
            )
        )
        .is_err());
        assert!(events.borrow().is_empty());
        for outputs in [0, 1] {
            let result = block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &[Value::Num(0.0)],
                    outputs,
                ),
            )
            .unwrap();
            if outputs == 1 {
                assert_eq!(
                    result,
                    Value::Num(if matches!(*name, "cos" | "cosh") {
                        1.0
                    } else {
                        0.0
                    })
                );
            }
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            events.borrow_mut().clear();
        }
    }
}

#[test]
fn floating_trig_preparation_preserves_empty_nd_shape_single_and_values() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    for name in NAMES {
        for (data, shape) in [(vec![], vec![0, 2, 3]), (vec![0.0, 0.5], vec![1, 1, 2])] {
            let tensor = Tensor::from_f32(data.clone(), shape.clone()).unwrap();
            let result = block_on(crate::dispatcher::call_builtin_async_in_context(
                &context,
                name,
                &[Value::Tensor(tensor)],
            ))
            .unwrap();
            let Value::Tensor(result) = result.value() else {
                panic!("typed tensor expected")
            };
            assert_eq!(result.shape(), shape.as_slice());
            assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
            for (&actual, source) in result.as_f32_slice().unwrap().iter().zip(data) {
                let expected = match *name {
                    "sin" => source.sin(),
                    "cos" => source.cos(),
                    "tan" => source.tan(),
                    "atan" => source.atan(),
                    "sinh" => source.sinh(),
                    "cosh" => source.cosh(),
                    "asinh" => source.asinh(),
                    "tanh" => source.tanh(),
                    _ => unreachable!(),
                };
                assert_eq!(actual, expected);
            }
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            events.borrow_mut().clear();
        }
    }
}

#[test]
fn floating_trig_preparation_drop_and_replacement_cannot_record_success() {
    let events = Events::default();
    for name in NAMES {
        {
            let context = context(&events, Mode::Pending);
            let args = [Value::Num(0.5)];
            drop(crate::dispatcher::call_builtin_async_in_context(
                &context, name, &args,
            ));
            assert!(events.borrow().is_empty());
            let mut call = Box::pin(crate::dispatcher::call_builtin_async_in_context(
                &context, name, &args,
            ));
            assert!(matches!(
                call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            drop(call);
            assert_eq!(&*events.borrow(), &["prepare", "abort"]);
            events.borrow_mut().clear();
        }
        let context = context(&events, Mode::Invalid);
        assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
            &context,
            name,
            &[Value::Num(0.5)]
        ))
        .is_err());
        assert_eq!(&*events.borrow(), &["prepare", "begin", "failure"]);
        events.borrow_mut().clear();
    }
}

#[test]
fn floating_trig_like_controls_are_admitted_before_service_and_remain_local() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    context.set_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        let invalid = [
            Value::Num(0.5),
            Value::String("like".into()),
            Value::String("bad prototype".into()),
        ];
        let error = block_on(crate::dispatcher::call_builtin_async_in_context(
            &context, name, &invalid,
        ))
        .unwrap_err();
        assert!(error.identifier().unwrap().ends_with(":LikePrototype"));
        assert!(events.borrow().is_empty());
        for prototype in [Value::Num(0.0), Value::Complex(0.0, 0.0)] {
            let complex = matches!(prototype, Value::Complex(_, _));
            let args = [Value::Num(0.5), Value::String("like".into()), prototype];
            let result = block_on(crate::dispatcher::call_builtin_async_in_context(
                &context, name, &args,
            ))
            .unwrap();
            let expected = match name {
                "sin" => 0.5_f64.sin(),
                "cos" => 0.5_f64.cos(),
                "tan" => 0.5_f64.tan(),
                "atan" => 0.5_f64.atan(),
                _ => unreachable!(),
            };
            assert_eq!(
                result,
                if complex {
                    Value::Complex(expected, 0.0)
                } else {
                    Value::Num(expected)
                }
            );
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            events.borrow_mut().clear();
        }
    }
}

#[test]
fn floating_trig_like_strict_mode_rejects_before_preparation() {
    let events = Events::default();
    let context = context(&events, Mode::Pass);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (name, identifier) in [
        ("sin", "RunMat:compatibility:SinLikeOutputExtension"),
        ("cos", "RunMat:compatibility:CosLikeOutputExtension"),
        ("tan", "RunMat:compatibility:TanLikeOutputExtension"),
        (
            "atan",
            "RunMat:compatibility:AtanLikeOutputTemplateExtension",
        ),
    ] {
        let args = [
            Value::Num(0.5),
            Value::String("like".into()),
            Value::Num(0.0),
        ];
        let error = block_on(crate::dispatcher::call_builtin_async_in_context(
            &context, name, &args,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some(identifier));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(events.borrow().is_empty());
    }
}
