use futures::executor::block_on;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_runtime::{call_builtin_async_with_outputs, RuntimeError};
use runmat_value::{Tensor, Value};
use std::{cell::RefCell, rc::Rc};

struct Recorder(Rc<RefCell<Vec<&'static str>>>);
impl RuntimeAccelerationService for Recorder {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(
            request.form().operation,
            runmat_builtins::BuiltinPreparationOperation::Unary
        );
        assert_eq!(request.admission().input_count, 1);
        assert_eq!(request.admission().signature, 0);
        self.0
            .borrow_mut()
            .push(request.admission().binding.builtin.name);
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::passthrough(
                request.into_operands(),
            ))
        })
    }
}

#[test]
fn admitted_rounding_plain_forms_follow_domain_validation() {
    let records = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Recorder(records.clone()))),
    );
    let input = Value::Tensor(Tensor::new(vec![1.25, -2.75], vec![1, 1, 2]).unwrap());
    for (name, expected) in [
        ("ceil", vec![2.0, -2.0]),
        ("floor", vec![1.0, -3.0]),
        ("fix", vec![1.0, -2.0]),
        ("round", vec![1.0, -3.0]),
    ] {
        for outputs in [0, 1] {
            let before = records.borrow().len();
            let output = block_on(context.scope(call_builtin_async_with_outputs(
                name,
                std::slice::from_ref(&input),
                outputs,
            )))
            .unwrap();
            assert_eq!(records.borrow().len(), before + 1);
            assert_eq!(records.borrow().last(), Some(&name));
            if outputs == 1 {
                let Value::Tensor(output) = output else {
                    panic!("{name}: pages")
                };
                assert_eq!(output.shape, vec![1, 1, 2]);
                assert_eq!(output.as_f64_slice().unwrap(), expected);
            }
        }
        let before = records.borrow().len();
        for args in [
            vec![],
            vec![Value::String("invalid".into())],
            vec![input.clone(), Value::String("bad option".into())],
        ] {
            assert!(
                block_on(context.scope(call_builtin_async_with_outputs(name, &args, 1))).is_err()
            );
        }
        assert!(block_on(context.scope(call_builtin_async_with_outputs(
            name,
            std::slice::from_ref(&input),
            2
        )))
        .is_err());
        assert_eq!(records.borrow().len(), before);
    }
    // Digit controls remain in their parsed domain route, not falsely claimed
    // as the plain unary hook or included in the prepared operand list.
    let before = records.borrow().len();
    block_on(context.scope(call_builtin_async_with_outputs(
        "round",
        &[input, Value::Num(1.0)],
        1,
    )))
    .unwrap();
    assert_eq!(records.borrow().len(), before);
}
