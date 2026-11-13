use super::*;
use runmat_value::ObjectInstance;
use std::{cell::RefCell, rc::Rc};

#[derive(Default)]
struct RecordingForeign {
    calls: Rc<RefCell<Vec<crate::context::ForeignCall>>>,
}

impl crate::context::RuntimeForeignService for RecordingForeign {
    fn invoke(
        &self,
        _context: crate::context::RuntimeContext,
        call: crate::context::ForeignCall,
    ) -> crate::context::RuntimeServiceFuture<crate::context::ForeignCallCompletion> {
        self.calls.borrow_mut().push(call);
        Box::pin(async {
            crate::context::ForeignCallCompletion::invoked(
                crate::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::String("done".into()))
                        .expect("single status"),
                )
                .map_err(|error| error.into_runtime_error()),
            )
        })
    }
}

fn recording_context() -> (crate::context::RuntimeContext, Rc<RecordingForeign>) {
    let recorder = Rc::new(RecordingForeign::default());
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        crate::context::RuntimeServicePorts::default().with_foreign(recorder.clone()),
    );
    (runtime, recorder)
}

#[test]
fn terminate_forwards_original_environment_to_foreign_owner() {
    let (runtime, recorder) = recording_context();
    let mut environment = ObjectInstance::new(PYTHON_ENVIRONMENT_CLASS);
    environment.replace_property_unbounded("Version".into(), Value::String("3.12".into()));
    let original = Value::Object(environment);
    let result = futures::executor::block_on(terminate_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![original.clone()]),
    ));
    assert!(result.is_ok(), "{result:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].adapter, "python");
    assert_eq!(calls[0].symbol, "terminate");
    assert!(calls[0].arguments.iter().eq([original].iter()));
}

#[test]
fn pyenv_status_and_configure_are_distinct_foreign_symbols() {
    let (runtime, recorder) = recording_context();
    let status = futures::executor::block_on(pyenv_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    assert!(status.is_ok(), "{status:?}");
    let pair = vec![
        Value::String("ExecutionMode".into()),
        Value::String("OutOfProcess".into()),
    ];
    let configure = futures::executor::block_on(pyenv_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(pair.clone()),
    ));
    assert!(configure.is_ok(), "{configure:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].symbol, "status");
    assert!(calls[0].arguments.is_empty());
    assert_eq!(calls[1].symbol, "configure");
    assert!(calls[1].arguments.iter().eq(pair.iter()));
}

#[test]
fn run_and_runfile_preserve_named_input_and_output_order_at_foreign_boundary() {
    let (runtime, recorder) = recording_context();
    let code = Value::String("answer = input_value + 2".into());
    let outputs = Value::String("answer".into());
    let name = Value::String("input_value".into());
    let input = Value::Int(runmat_value::IntValue::I64(40));
    let run = futures::executor::block_on(pyrun_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            code.clone(),
            outputs.clone(),
            name.clone(),
            input.clone(),
        ]),
    ));
    assert!(run.is_ok(), "{run:?}");
    let file = Value::String("answer.py".into());
    let runfile = futures::executor::block_on(pyrunfile_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            file.clone(),
            outputs.clone(),
            name.clone(),
            input.clone(),
        ]),
    ));
    assert!(runfile.is_ok(), "{runfile:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].symbol, "pyrun");
    assert_eq!(calls[0].requested_outputs, 1);
    assert!(calls[0]
        .arguments
        .iter()
        .eq([code, outputs.clone(), name.clone(), input.clone()].iter()));
    assert_eq!(calls[1].symbol, "pyrunfile");
    assert!(calls[1]
        .arguments
        .iter()
        .eq([file, outputs, name, input].iter()));
}
