use futures::executor::block_on;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_runtime::RuntimeError;
use runmat_value::{NumericDType, Tensor, Value};
use std::{cell::Cell, rc::Rc};

struct Recorder(Rc<Cell<usize>>);
impl RuntimeAccelerationService for Recorder {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        self.0.set(self.0.get() + 1);
        assert!(request
            .operands()
            .iter()
            .all(|v| matches!(v, Value::Tensor(_))));
        assert_eq!(request.admission().input_count, request.operands().len());
        assert_eq!(
            request.form().data_arguments.len(),
            request.operands().len()
        );
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

fn data(single: bool) -> Value {
    Value::Tensor(if single {
        Tensor::from_f32(vec![-2.0, 4.0], vec![1, 2]).unwrap()
    } else {
        Tensor::new(vec![-2.0, 4.0], vec![1, 2]).unwrap()
    })
}

#[test]
fn components_and_construction_prepare_only_admitted_floating_arrays() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Recorder(calls.clone()))),
    );
    for single in [false, true] {
        for (name, count) in [
            ("conj", 1),
            ("real", 1),
            ("imag", 1),
            ("complex", 1),
            ("complex", 2),
        ] {
            for outputs in [0, 1] {
                let before = calls.get();
                let args = vec![data(single); count];
                let result = block_on(context.scope(
                    runmat_runtime::call_builtin_async_with_outputs_in_context(
                        &context, name, &args, outputs,
                    ),
                ))
                .unwrap();
                let expected_class = if single {
                    NumericDType::F32
                } else {
                    NumericDType::F64
                };
                if name == "complex" {
                    let Value::ComplexTensor(value) = result.value() else {
                        panic!("complex tensor")
                    };
                    assert_eq!(value.shape(), &[1, 2]);
                    assert_eq!(value.numeric_dtype(), expected_class);
                } else {
                    let Value::Tensor(value) = result.value() else {
                        panic!("real tensor")
                    };
                    assert_eq!(value.shape(), &[1, 2]);
                    assert_eq!(value.numeric_dtype(), expected_class);
                    if single {
                        assert_eq!(
                            value.as_f32_slice().unwrap(),
                            if name == "imag" {
                                &[0.0, 0.0]
                            } else {
                                &[-2.0, 4.0]
                            }
                        );
                    } else {
                        assert_eq!(
                            value.as_f64_slice().unwrap(),
                            if name == "imag" {
                                &[0.0, 0.0]
                            } else {
                                &[-2.0, 4.0]
                            }
                        );
                    }
                }
                assert_eq!(calls.get(), before + 1, "{name}");
            }
            let before = calls.get();
            assert!(block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &vec![data(single); count],
                    2
                )
            ))
            .is_err());
            assert_eq!(calls.get(), before);
        }
    }
    let before = calls.get();
    for args in [
        vec![data(false), Value::Complex(1.0, 2.0)],
        vec![
            data(false),
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
        ],
        vec![data(false), data(false), data(false)],
    ] {
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context, "complex", &args, 1
            )
        ))
        .is_err());
    }
    assert_eq!(calls.get(), before);
}
