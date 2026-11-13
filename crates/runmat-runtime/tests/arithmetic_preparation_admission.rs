use futures::executor::block_on;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_runtime::RuntimeError;
use runmat_value::{Tensor, Value};
use std::{cell::Cell, rc::Rc};

struct Recorder(Rc<Cell<usize>>);
impl RuntimeAccelerationService for Recorder {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        self.0.set(self.0.get() + 1);
        assert_eq!(request.admission().input_count, 2);
        assert_eq!(request.admission().signature, 0);
        assert_eq!(request.form().data_arguments, &[0, 1]);
        let expected = if request.form().operation
            == runmat_builtins::BuiltinPreparationOperation::RealPower
        {
            runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                runmat_accelerate_api::PowerResultDomain::Real,
            )
        } else {
            runmat_accelerate_api::NumericalDomainEvidence::Unknown
        };
        assert_eq!(request.numerical_domain(), expected);
        assert!(matches!(
            request.operands(),
            [Value::Tensor(_), Value::Tensor(_)]
        ));
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

fn tensor(data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
}

#[test]
fn actual_dispatch_prepares_admitted_pairs_and_preserves_results() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Recorder(calls.clone()))),
    );
    for (name, expected) in [
        ("plus", [3.0, 6.0]),
        ("minus", [1.0, 2.0]),
        ("times", [2.0, 8.0]),
        ("rdivide", [2.0, 2.0]),
        ("ldivide", [0.5, 0.5]),
        ("power", [2.0, 16.0]),
    ] {
        for outputs in [0, 1] {
            let before = calls.get();
            let result = block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &[tensor(&[2.0, 4.0], &[1, 2]), tensor(&[1.0, 2.0], &[1, 2])],
                    outputs,
                ),
            ))
            .unwrap();
            let Value::Tensor(value) = result.value() else {
                panic!("{name}: expected tensor")
            };
            assert_eq!(value.shape(), &[1, 2]);
            assert_eq!(value.as_f64_slice().unwrap(), expected.as_slice(), "{name}");
            assert_eq!(calls.get(), before + 1, "{name}");
        }
        let before = calls.get();
        for (args, outputs) in [
            (
                vec![
                    tensor(&[1.0, 2.0], &[1, 2]),
                    tensor(&[1.0, 2.0, 3.0], &[1, 3]),
                ],
                1,
            ),
            (
                vec![tensor(&[1.0, 2.0], &[1, 2]), tensor(&[1.0, 2.0], &[1, 2])],
                2,
            ),
            (
                vec![
                    tensor(&[1.0], &[1, 1]),
                    tensor(&[1.0], &[1, 1]),
                    Value::String("bad-option".into()),
                ],
                1,
            ),
        ] {
            assert!(
                block_on(context.scope(
                    runmat_runtime::call_builtin_async_with_outputs_in_context(
                        &context, name, &args, outputs
                    )
                ))
                .is_err(),
                "{name}"
            );
        }
        assert_eq!(
            calls.get(),
            before,
            "{name}: invalid calls must not prepare"
        );
    }
}
