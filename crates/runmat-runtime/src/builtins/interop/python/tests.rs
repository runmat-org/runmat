use super::*;
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
    ) -> crate::context::RuntimeServiceFuture<
        Result<runmat_value::ValueSequence, crate::RuntimeError>,
    > {
        self.calls.borrow_mut().push(call);
        Box::pin(async {
            Ok(
                runmat_value::ValueSequence::single(Value::String("done".into()))
                    .expect("single status"),
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
fn pyargs_constructs_local_bundle_without_foreign_service() {
    let bundle = pyargs_builtin(vec![
        Value::String("count".into()),
        Value::Int(runmat_value::IntValue::I64(7)),
    ])
    .expect("bundle");
    let Value::Object(bundle) = bundle else {
        panic!("expected object")
    };
    assert!(bundle
        .class_name
        .is(runmat_types::standard::PYTHON_ARGUMENTS));
    let Value::Cell(names) = bundle.properties.get("Names").expect("Names") else {
        panic!("names")
    };
    let Value::Cell(values) = bundle.properties.get("Values").expect("Values") else {
        panic!("values")
    };
    assert_eq!(names.rows, 1);
    assert_eq!(names.cols, 1);
    assert_eq!(names.data, vec![Value::String("count".into())]);
    assert_eq!(
        values.data,
        vec![Value::Int(runmat_value::IntValue::I64(7))]
    );
}

#[test]
fn pyargs_rejects_invalid_pairs_before_any_python_service() {
    let odd = pyargs_builtin(vec![Value::String("x".into())])
        .err()
        .expect("odd pair");
    assert_eq!(odd.identifier(), Some("RunMat:Foreign:InvalidCall"));
    let nontext = pyargs_builtin(vec![Value::Num(1.0), Value::Num(2.0)])
        .err()
        .expect("text name");
    assert_eq!(nontext.identifier(), Some("RunMat:Foreign:InvalidCall"));
}

#[test]
fn terminate_forwards_original_environment_to_foreign_owner() {
    let (runtime, recorder) = recording_context();
    let mut environment = ObjectInstance::new(PYTHON_ENVIRONMENT_CLASS);
    environment
        .properties
        .insert("Version".into(), Value::String("3.12".into()));
    let original = Value::Object(environment);
    let result =
        futures::executor::block_on(terminate_builtin(Some(runtime), vec![original.clone()]));
    assert!(result.is_ok(), "{result:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].adapter, "python");
    assert_eq!(calls[0].symbol, "terminate");
    assert_eq!(calls[0].arguments, vec![original]);
}

#[test]
fn pyenv_status_and_configure_are_distinct_foreign_symbols() {
    let (runtime, recorder) = recording_context();
    let status = futures::executor::block_on(pyenv_builtin(Some(runtime.clone()), Vec::new()));
    assert!(status.is_ok(), "{status:?}");
    let pair = vec![
        Value::String("ExecutionMode".into()),
        Value::String("OutOfProcess".into()),
    ];
    let configure = futures::executor::block_on(pyenv_builtin(Some(runtime), pair.clone()));
    assert!(configure.is_ok(), "{configure:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].symbol, "status");
    assert!(calls[0].arguments.is_empty());
    assert_eq!(calls[1].symbol, "configure");
    assert_eq!(calls[1].arguments, pair);
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
        vec![code.clone(), outputs.clone(), name.clone(), input.clone()],
    ));
    assert!(run.is_ok(), "{run:?}");
    let file = Value::String("answer.py".into());
    let runfile = futures::executor::block_on(pyrunfile_builtin(
        Some(runtime),
        vec![file.clone(), outputs.clone(), name.clone(), input.clone()],
    ));
    assert!(runfile.is_ok(), "{runfile:?}");
    let calls = recorder.calls.borrow();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].symbol, "pyrun");
    assert_eq!(calls[0].requested_outputs, 1);
    assert_eq!(
        calls[0].arguments,
        vec![code, outputs.clone(), name.clone(), input.clone()]
    );
    assert_eq!(calls[1].symbol, "pyrunfile");
    assert_eq!(calls[1].arguments, vec![file, outputs, name, input]);
}
