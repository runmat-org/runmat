//! A single runtime library supplies the real canonical bindings for these tests.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostLogicalTensorView, ThreadProviderGuard,
};
use runmat_runtime::{builtins::common::gpu_helpers, call_builtin_async_with_outputs};
use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, NumericDType, Tensor, Value};

#[path = "support/retained_provider.rs"]
mod retained_provider;

#[path = "arrayfun_owned_batch/division_fallback.rs"]
mod division_fallback;
#[path = "arrayfun_owned_batch/lifecycle.rs"]
mod lifecycle;

fn provider() -> &'static InProcessProvider {
    retained_provider::retained_test_provider!(PROVIDER: InProcessProvider = InProcessProvider::new())
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

fn mapped(
    name: &str,
    inputs: Vec<Value>,
) -> Result<runmat_runtime::execution::RootedValue, Box<runmat_runtime::RuntimeError>> {
    let mut args = vec![Value::FunctionHandle(name.into())];
    args.extend(inputs);
    block_on(call_builtin_async_with_outputs("arrayfun", &args, 1)).map_err(Box::new)
}

fn host(value: &Value) -> Value {
    block_on(runmat_runtime::gather_if_needed_async(value)).unwrap()
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

fn same_result(actual: &Value, expected: &Value, label: &str) {
    match (host(actual), host(expected)) {
        (Value::Tensor(actual), Value::Tensor(expected)) => {
            assert_eq!(actual.shape(), expected.shape(), "{label}");
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
            assert_eq!(actual.shape(), expected.shape(), "{label}");
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
        (Ok(actual), Ok(expected)) => same_result(actual.value(), expected.value(), name),
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
    fn replacement(
        _: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>,
    ) -> RuntimeBuiltinFuture {
        runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
            let sequence = runmat_value::ValueSequence::single(Value::Num(99.0))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
            runmat_runtime::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error())
        })
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
        let first = block_on(runmat_runtime::call_builtin_async_with_outputs_in_context(
            &context, "arrayfun", &args, 1,
        ))
        .unwrap();
        same_result(
            first.value(),
            &Value::Tensor(Tensor::new(vec![0.0, 1.0f64.sin()], vec![1, 2]).unwrap()),
            "admitted sin",
        );
        assert_eq!(
            service.sin_lookups.get(),
            1,
            "execution consumes the admitted binding without another service lookup"
        );
        let second = block_on(runmat_runtime::call_builtin_async_with_outputs_in_context(
            &context, "arrayfun", &args, 1,
        ))
        .unwrap();
        same_result(
            second.value(),
            &Value::Tensor(Tensor::new(vec![99.0, 99.0], vec![1, 2]).unwrap()),
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

#[test]
fn empty_mapping_does_not_select_or_execute_the_callback() {
    use runmat_runtime::{
        builtin::RuntimeBuiltinBinding,
        context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
    };
    use std::rc::Rc;
    struct Service;
    impl RuntimeBuiltinService for Service {
        fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
            assert_ne!(
                name, "log",
                "empty mapping must not resolve an executable callback"
            );
            runmat_runtime::builtin::runtime_builtin_bindings_by_name(name)
        }
    }
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(provider()));
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Service)));
        let source = upload(&Value::Tensor(Tensor::new(vec![], vec![0, 3]).unwrap()));
        let args = [
            Value::FunctionHandle("log".into()),
            Value::GpuTensor(source.clone()),
        ];
        let output = block_on(runmat_runtime::call_builtin_async_with_outputs_in_context(
            &context, "arrayfun", &args, 1,
        ))
        .unwrap();
        same_result(
            output.value(),
            &Value::Tensor(Tensor::new(vec![], vec![0, 3]).unwrap()),
            "empty log",
        );
        source.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn rank_zero_owned_singleton_maps_to_the_same_uniform_shape_as_host() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(provider()));
        let source = upload(&Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()));
        let scalar_view = source
            .with_view(source.view().clone().with_shape(vec![]))
            .unwrap();
        assert_eq!(scalar_view.shape(), &[] as &[usize]);
        assert!(scalar_view.shares_allocation_with(&source));
        let expected = mapped(
            "sin",
            vec![Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap())],
        )
        .unwrap();
        let actual = mapped("sin", vec![Value::GpuTensor(scalar_view)]).unwrap();
        same_result(
            actual.value(),
            expected.value(),
            "rank-zero singleton is not an empty mapping",
        );
        source.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn asymmetric_mixed_class_batches_preserve_column_order_and_all_original_intents() {
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
    use std::rc::Rc;
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(provider()));
        // Mixed floating classes need the same canonical operand preparation
        // and placement owners as a native session, not the legacy raw hook.
        let context = runmat_accelerate::native_auto::install_preparation_service(
            RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(RuntimeServicePorts::default().with_placement(
                Rc::new(runmat_accelerate::placement::PlacementSession::default()),
            )),
        );
        for (left_intent, right_intent) in [
            (Automatic, Explicit),
            (Explicit, Automatic),
            (Automatic, Automatic),
        ] {
            let left = upload(&Value::Tensor(
                Tensor::from_f32(vec![2.0, 4.0], vec![2, 1]).unwrap(),
            ))
            .with_provenance(left_intent)
            .unwrap();
            let right = upload(&Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 8.0], vec![1, 3]).unwrap(),
            ))
            .with_provenance(right_intent)
            .unwrap();
            for name in ["plus", "minus", "times", "rdivide", "ldivide"] {
                let output = block_on(runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    "arrayfun",
                    &[
                        Value::FunctionHandle(name.into()),
                        Value::GpuTensor(left.clone()),
                        Value::GpuTensor(right.clone()),
                    ],
                    1,
                ))
                .unwrap();
                let Value::GpuTensor(handle) = output.value() else {
                    panic!("native class-capable owner must preserve resident output for {name}")
                };
                assert_eq!(
                    runmat_accelerate_api::handle_is_explicit(handle),
                    left_intent == Explicit || right_intent == Explicit,
                    "{name}: both original intents participate"
                );
                let mut expected = Vec::new();
                for right in [1.0f64, 2.0, 8.0] {
                    for left in [2.0f64, 4.0] {
                        expected.push(match name {
                            "plus" => left + right,
                            "minus" => left - right,
                            "times" => left * right,
                            "rdivide" => left / right,
                            "ldivide" => right / left,
                            _ => unreachable!(),
                        } as f32);
                    }
                }
                same_result(
                    output.value(),
                    &Value::Tensor(Tensor::from_f32(expected, vec![2, 3]).unwrap()),
                    name,
                );
                left.allocation_lease().validate_live().unwrap();
                right.allocation_lease().validate_live().unwrap();
            }
        }
    });
}
