use futures::executor::block_on;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_runtime::RuntimeError;
use runmat_value::{Tensor, Value};
use std::{cell::RefCell, rc::Rc};

struct Recorder(Rc<RefCell<Vec<&'static str>>>);
impl RuntimeAccelerationService for Recorder {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(
            request.form().data_arguments,
            if matches!(
                request.form().operation,
                runmat_builtins::BuiltinPreparationOperation::FloatingRemainder(_)
            ) {
                &[0, 1][..]
            } else {
                &[0][..]
            }
        );
        match request.form().operation {
            runmat_builtins::BuiltinPreparationOperation::Unary => {
                assert_eq!(request.admission().input_count, 1);
                assert_eq!(request.admission().signature, 0);
            }
            runmat_builtins::BuiltinPreparationOperation::RoundDigits => {
                assert_eq!(request.admission().binding.builtin.name, "round");
                assert_eq!(
                    request.admission().signature + 1,
                    request.admission().input_count
                );
                assert!(matches!(
                    request.numerical_domain(),
                    runmat_accelerate_api::NumericalDomainEvidence::RoundDigits { digits: 1, .. }
                ));
                assert_eq!(request.operands().len(), 1);
            }
            runmat_builtins::BuiltinPreparationOperation::FloatingRemainder(kind) => {
                assert_eq!(request.admission().input_count, 2);
                assert_eq!(request.admission().signature, 0);
                assert_eq!(
                    request.numerical_domain(),
                    runmat_accelerate_api::NumericalDomainEvidence::FloatingRemainder(kind)
                );
                assert_eq!(request.operands().len(), 2);
            }
            other => panic!("unexpected rounding operation: {other:?}"),
        }
        self.0
            .borrow_mut()
            .push(request.admission().binding.builtin.name);
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

#[test]
fn admitted_floating_remainder_validates_shape_and_outputs_before_preparation() {
    let records = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Recorder(records.clone()))),
    );
    let left = Value::Tensor(Tensor::new(vec![7.0, -7.0], vec![2, 1]).unwrap());
    let right = Value::Tensor(Tensor::new(vec![3.0, -3.0, 2.0], vec![1, 3]).unwrap());
    for (name, expected) in [
        ("mod", vec![1.0, 2.0, -2.0, -1.0, 1.0, 1.0]),
        ("rem", vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    ] {
        for outputs in [0, 1] {
            let before = records.borrow().len();
            let result = block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &[left.clone(), right.clone()],
                    outputs,
                ),
            ))
            .unwrap();
            assert_eq!(records.borrow().len(), before + 1);
            if outputs == 1 {
                let Value::Tensor(result) = result.value() else {
                    panic!("floating matrix")
                };
                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result.as_f64_slice().unwrap(), expected);
            }
        }
        let before = records.borrow().len();
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[left.clone(), right.clone()],
                2
            )
        ))
        .is_err());
        let wrong = Value::Tensor(Tensor::new(vec![1.0; 3], vec![3, 1]).unwrap());
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[left.clone(), wrong],
                1
            )
        ))
        .is_err());
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[Value::Complex(1.0, 2.0), right.clone()],
                1
            )
        ))
        .is_err());
        assert_eq!(records.borrow().len(), before);
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
            let output = block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    std::slice::from_ref(&input),
                    outputs,
                ),
            ))
            .unwrap();
            assert_eq!(records.borrow().len(), before + 1);
            assert_eq!(records.borrow().last(), Some(&name));
            if outputs == 1 {
                let Value::Tensor(output) = output.value() else {
                    panic!("{name}: pages")
                };
                assert_eq!(output.shape(), &[1, 1, 2]);
                assert_eq!(output.as_f64_slice().unwrap(), expected);
            }
        }
        let before = records.borrow().len();
        for args in [
            vec![],
            vec![Value::String("invalid".into())],
            vec![input.clone(), Value::String("bad option".into())],
        ] {
            assert!(block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context, name, &args, 1
                )
            ))
            .is_err());
        }
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                std::slice::from_ref(&input),
                2
            )
        ))
        .is_err());
        assert_eq!(records.borrow().len(), before);
    }
    for (controls, expected) in [
        (vec![Value::Num(1.0)], vec![1.3, -2.8]),
        (
            vec![Value::Num(1.0), Value::String("decimals".into())],
            vec![1.3, -2.8],
        ),
        (
            vec![Value::Num(1.0), Value::String("significant".into())],
            vec![1.0, -3.0],
        ),
    ] {
        let before = records.borrow().len();
        let mut args = vec![input.clone()];
        args.extend(controls);
        let output = block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(&context, "round", &args, 1),
        ))
        .unwrap();
        assert_eq!(records.borrow().len(), before + 1);
        let Value::Tensor(output) = output.value() else {
            panic!("round pages")
        };
        assert_eq!(output.as_f64_slice().unwrap(), expected);
        assert_eq!(output.shape(), &[1, 1, 2]);
    }
    let before = records.borrow().len();
    for controls in [
        vec![Value::Num(0.0), Value::String("significant".into())],
        vec![Value::Num(1.5)],
        vec![Value::Num(1.0), Value::String("invalid".into())],
    ] {
        let mut args = vec![input.clone()];
        args.extend(controls);
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(&context, "round", &args, 1)
        ))
        .is_err());
    }
    assert_eq!(records.borrow().len(), before);
}
