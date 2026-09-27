//! A single runtime library supplies the real canonical bindings for these tests.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostLogicalTensorView, ThreadProviderGuard,
};
use runmat_runtime::{builtins::common::gpu_helpers, call_builtin_async_with_outputs};
use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, NumericDType, Tensor, Value};
use std::sync::OnceLock;

fn provider() -> &'static InProcessProvider {
    static PROVIDER: OnceLock<InProcessProvider> = OnceLock::new();
    PROVIDER.get_or_init(InProcessProvider::new)
}

fn upload(value: &Value) -> GpuTensorHandle {
    match value {
        Value::Tensor(value) => gpu_helpers::upload_tensor(provider(), value).unwrap(),
        Value::ComplexTensor(value) => {
            gpu_helpers::upload_complex_tensor(provider(), value).unwrap()
        }
        Value::LogicalArray(value) => provider()
            .upload_logical(&HostLogicalTensorView {
                data: &value.data,
                shape: &value.shape,
            })
            .unwrap(),
        _ => panic!("fixture requires an array"),
    }
}

fn mapped(name: &str, inputs: Vec<Value>) -> runmat_runtime::BuiltinResult<Value> {
    let mut args = vec![Value::FunctionHandle(name.into())];
    args.extend(inputs);
    block_on(call_builtin_async_with_outputs("arrayfun", &args, 1))
}

fn host(value: Value) -> Value {
    block_on(runmat_runtime::gather_if_needed_async(&value)).unwrap()
}

fn same_values(actual: &[f64], expected: &[f64], dtype: NumericDType, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let epsilon = if dtype == NumericDType::F32 {
        2e-6
    } else {
        2e-13
    };
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual == expected
                || (actual.is_nan() && expected.is_nan())
                || (actual - expected).abs() <= epsilon * expected.abs().max(1.0),
            "{label}: expected {expected:?}, actual {actual:?}"
        );
    }
}

fn same_result(actual: Value, expected: Value, label: &str) {
    match (host(actual), host(expected)) {
        (Value::Tensor(actual), Value::Tensor(expected)) => {
            assert_eq!(actual.shape, expected.shape, "{label}");
            assert_eq!(actual.numeric_dtype(), expected.numeric_dtype(), "{label}");
            if expected.integer_storage().is_some() {
                assert_eq!(
                    actual.integer_storage(),
                    expected.integer_storage(),
                    "{label}"
                );
            } else {
                same_values(
                    &actual.materialize_f64(),
                    &expected.materialize_f64(),
                    expected.numeric_dtype(),
                    label,
                );
            }
        }
        (Value::ComplexTensor(actual), Value::ComplexTensor(expected)) => {
            assert_eq!(actual.shape, expected.shape, "{label}");
            assert_eq!(actual.numeric_dtype(), expected.numeric_dtype(), "{label}");
            let actual = actual
                .materialize_f64()
                .into_iter()
                .flat_map(|(r, i)| [r, i])
                .collect::<Vec<_>>();
            let values = expected
                .materialize_f64()
                .into_iter()
                .flat_map(|(r, i)| [r, i])
                .collect::<Vec<_>>();
            same_values(&actual, &values, expected.numeric_dtype(), label);
        }
        (actual, expected) => assert_eq!(actual, expected, "{label}"),
    }
}

fn compare(name: &str, inputs: Vec<Value>) {
    let handles = inputs.iter().map(upload).collect::<Vec<_>>();
    let expected = mapped(name, inputs);
    let actual = mapped(
        name,
        handles.iter().cloned().map(Value::GpuTensor).collect(),
    );
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => same_result(actual, expected, name),
        (Err(actual), Err(expected)) => assert_eq!(
            actual.identifier(),
            expected.identifier(),
            "{name}: {actual:?} / {expected:?}"
        ),
        (actual, expected) => {
            panic!("{name}: batch/scalar disposition differs: {actual:?} / {expected:?}")
        }
    }
    for handle in handles {
        handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn canonical_eleven_operation_class_and_value_matrix() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(provider()));
        let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        let inputs = vec![
            Value::Tensor(Tensor::new(vec![0.25, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![0.25, 2.0], vec![1, 2]).unwrap()),
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![-2, 3]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                    vec![1, 2],
                )
                .unwrap(),
            ),
            Value::ComplexTensor(
                ComplexTensor::new(vec![(0.25, 0.5), (2.0, -1.0)], vec![1, 2]).unwrap(),
            ),
            Value::ComplexTensor(
                ComplexTensor::from_f32(vec![(0.25, 0.5), (2.0, -1.0)], vec![1, 2]).unwrap(),
            ),
        ];
        for name in [
            "sin", "cos", "abs", "exp", "log", "sqrt", "plus", "minus", "times", "rdivide",
            "ldivide",
        ] {
            let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
            assert_eq!(bindings.len(), 1, "{name} compiled binding cardinality");
            assert!(
                runmat_runtime::builtins::common::spec::selected_builtin_binding(
                    name,
                    Some(bindings[0].identity),
                    None
                )
                .is_some(),
                "{name} production admission"
            );
            for input in &inputs {
                let mut arguments = vec![input.clone()];
                if matches!(name, "plus" | "minus" | "times" | "rdivide" | "ldivide") {
                    arguments.push(input.clone());
                }
                compare(name, arguments);
            }
            if matches!(name, "plus" | "minus" | "times" | "rdivide" | "ldivide") {
                compare(name, vec![inputs[0].clone(), inputs[1].clone()]);
                compare(name, vec![inputs[5].clone(), inputs[1].clone()]);
                compare(name, vec![inputs[4].clone(), inputs[0].clone()]);
                compare(name, vec![inputs[0].clone(), inputs[4].clone()]);
            }
        }
        compare(
            "sqrt",
            vec![Value::Tensor(
                Tensor::new(vec![-1.0], vec![1, 1, 1]).unwrap(),
            )],
        );
        compare(
            "abs",
            vec![Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![-3]), vec![1, 1, 1]).unwrap(),
            )],
        );
    });
}

#[test]
fn stateful_scoped_service_cannot_replace_an_admitted_batch() {
    use runmat_runtime::{
        builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
        context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
    };
    use std::{cell::Cell, rc::Rc};

    struct Service {
        sin_lookups: Cell<usize>,
    }
    fn replacement(_: &[Value]) -> RuntimeBuiltinFuture {
        Box::pin(async { Ok(Value::Num(99.0)) })
    }
    impl RuntimeBuiltinService for Service {
        fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
            let mut bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
            if name.eq_ignore_ascii_case("sin") {
                let lookups = self.sin_lookups.get();
                self.sin_lookups.set(lookups + 1);
                if lookups > 0 {
                    assert_eq!(bindings.len(), 1);
                    bindings[0].implementation = replacement;
                }
            }
            bindings
        }
    }
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(provider()));
        let service = Rc::new(Service {
            sin_lookups: Cell::new(0),
        });
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
        let source = upload(&Value::Tensor(
            Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap(),
        ));
        let args = [
            Value::FunctionHandle("sin".into()),
            Value::GpuTensor(source.clone()),
        ];
        let first =
            block_on(context.scope(call_builtin_async_with_outputs("arrayfun", &args, 1))).unwrap();
        same_result(
            first,
            Value::Tensor(Tensor::new(vec![0.0, 1.0f64.sin()], vec![1, 2]).unwrap()),
            "admitted sin",
        );
        assert_eq!(
            service.sin_lookups.get(),
            1,
            "execution consumes the admitted binding without another service lookup"
        );
        let second =
            block_on(context.scope(call_builtin_async_with_outputs("arrayfun", &args, 1))).unwrap();
        same_result(
            second,
            Value::Tensor(Tensor::new(vec![99.0, 99.0], vec![1, 2]).unwrap()),
            "next invocation override",
        );
        assert_eq!(
            service.sin_lookups.get(),
            4,
            "next invocation re-admits and then invokes two scalar overrides"
        );
        source.allocation_lease().validate_live().unwrap();
    });
}
