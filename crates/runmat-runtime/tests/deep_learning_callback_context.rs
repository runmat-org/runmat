use futures::executor::block_on;
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation},
    call_builtin_async, call_builtin_async_with_outputs_in_context,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use runmat_value::{CellArray, StructArray, Tensor, Value, ValueSequence};
use std::{cell::RefCell, rc::Rc};

#[path = "deep_learning_callback_context/lifecycle.rs"]
mod lifecycle;

thread_local! {
    static CALLBACKS: RefCell<Vec<(runmat_runtime::context::RuntimeContextLocalId, Option<usize>)>> =
        const { RefCell::new(Vec::new()) };
}

fn selected_callback(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let runtime = invocation
        .context
        .expect("callback retains the invoking runtime");
    let outputs = invocation.requested_outputs;
    let first = invocation
        .args
        .get(0)
        .cloned()
        .map(runmat_runtime::execution::RootedValue::new)
        .transpose();
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        let first = first.map_err(|error| error.into_runtime_error())?;
        CALLBACKS.with(|calls| calls.borrow_mut().push((runtime.local_identity(), outputs)));
        if let Some(Value::Tensor(shape)) = first.as_ref().map(|owner| owner.value()) {
            let dimensions: Vec<usize> = shape
                .host_buffer()
                .materialize_f64()
                .into_iter()
                .map(|size| size as usize)
                .collect();
            let count = dimensions.iter().product();
            let sequence = ValueSequence::single(Value::Tensor(
                Tensor::new(vec![0.25; count], dimensions).expect("fixture initializer shape"),
            ))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
            return runmat_runtime::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error());
        }
        let sequence = match outputs {
            Some(2) => ValueSequence::comma_separated(vec![Value::Num(23.0), Value::Num(29.0)])
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
            _ => ValueSequence::single(Value::Num(23.0))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
        }?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

struct SelectedCallback;

impl RuntimeBuiltinService for SelectedCallback {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        let mut bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        if name.eq_ignore_ascii_case("deal") || name.eq_ignore_ascii_case("plus") {
            for binding in &mut bindings {
                binding.implementation = selected_callback;
            }
        }
        bindings
    }
}

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(SelectedCallback)))
}

fn training_layers(regression_output: bool) -> runmat_runtime::execution::RootedValue {
    let input = block_on(call_builtin_async("featureInputLayer", &[Value::Num(1.0)]))
        .expect("feature input");
    let dense = block_on(call_builtin_async(
        "fullyConnectedLayer",
        &[
            Value::Num(1.0),
            Value::String("WeightsInitializer".into()),
            Value::FunctionHandle("plus".into()),
        ],
    ))
    .expect("dense layer with callback initializer");
    let mut layers = runmat_runtime::execution::RootedValuesBuilder::with_capacity(3);
    layers.push_owned(input);
    layers.push_owned(dense);
    if regression_output {
        layers
            .push_owned(block_on(call_builtin_async("regressionLayer", &[])).expect("regression"));
    }
    let layers = layers.finish().expect("rooted layer batch");
    let count = layers.values().len();
    layers
        .with_unbounded_values(|layers| {
            runmat_runtime::execution::RootedValue::new(Value::Cell(
                CellArray::new(layers, count, 1).expect("layer column"),
            ))
            .expect("rooted layer column")
        })
        .expect("fixture layer batch is unbounded")
}

fn training_options() -> runmat_runtime::execution::RootedValue {
    block_on(call_builtin_async(
        "trainingOptions",
        &[
            Value::String("adam".into()),
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(2.0),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ],
    ))
    .expect("training options")
}

#[test]
fn dlfeval_callback_uses_invoking_runtime_and_structural_output_count() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlfeval",
        &[Value::FunctionHandle("deal".into()), Value::Num(1.0)],
        2,
    ))
    .expect("selected callback");
    assert_eq!(
        output,
        Value::output_list(vec![Value::Num(23.0), Value::Num(29.0)])
    );
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(2))]
        );
    });
}

#[test]
fn dlfeval_registered_binding_uses_invocation_count_not_ambient_count() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("dlfeval");
    let [binding] = bindings.as_slice() else {
        panic!("dlfeval must have one canonical binding");
    };
    let args = [Value::FunctionHandle("deal".into()), Value::Num(1.0)];
    let _ambient = runmat_runtime::output_count::push_output_count(Some(1));
    let output = block_on((binding.implementation)(RuntimeBuiltinInvocation::new(
        &args,
        Some(runtime.clone()),
        Some(2),
    )))
    .expect("registered callback");
    assert_eq!(output.as_slice(), [Value::Num(23.0), Value::Num(29.0)]);
    assert_eq!(
        output.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(2))]
        );
    });
}

#[test]
fn dlupdate_leaf_callback_uses_invoking_runtime_once() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlupdate",
        &[Value::FunctionHandle("plus".into()), Value::Num(1.0)],
        1,
    ))
    .expect("selected leaf callback");
    assert_eq!(output, Value::Num(23.0));
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(1))]
        );
    });
}

#[test]
fn dlupdate_nested_containers_reuse_one_scoped_runtime() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let mut first = runmat_value::UnboundedStructValueBuilder::new();
    first.insert("Data", Value::Num(1.0));
    let mut second = runmat_value::UnboundedStructValueBuilder::new();
    second.insert("Data", Value::Num(2.0));
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert(
        "Array",
        Value::StructArray(
            StructArray::new(vec![first.finish(), second.finish()], vec![1, 2]).unwrap(),
        ),
    );
    structure.insert("Scalar", Value::Num(3.0));
    let tree = Value::Cell(CellArray::new(vec![Value::Struct(structure.finish())], 1, 1).unwrap());
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlupdate",
        &[Value::FunctionHandle("plus".into()), tree],
        1,
    ))
    .expect("nested callback tree");
    let Value::Cell(cell) = output.value() else {
        panic!("expected cell tree");
    };
    let Value::Struct(structure) = &cell.data()[0] else {
        panic!("expected nested structure");
    };
    assert_eq!(structure.fields.get("Scalar"), Some(&Value::Num(23.0)));
    let Some(Value::StructArray(array)) = structure.fields.get("Array") else {
        panic!("expected structure array");
    };
    assert_eq!(
        array.field_values("Data"),
        Some(&[Value::Num(23.0), Value::Num(23.0)][..])
    );
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(1)); 3]
        );
    });
}

#[test]
fn unpolled_dlfeval_releases_its_runtime_and_callback_service() {
    let service = Rc::new(SelectedCallback);
    let weak_service = Rc::downgrade(&service);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("dlfeval");
    let [binding] = bindings.as_slice() else {
        panic!("dlfeval must have one canonical binding");
    };
    let args = [Value::FunctionHandle("deal".into()), Value::Num(1.0)];
    let future = (binding.implementation)(RuntimeBuiltinInvocation::new(
        &args,
        Some(runtime.clone()),
        Some(1),
    ));
    drop(runtime);
    drop(service);
    assert!(
        weak_service.upgrade().is_some(),
        "the future owns its runtime"
    );
    drop(future);
    assert!(
        weak_service.upgrade().is_none(),
        "dropping the future releases its runtime"
    );
}

#[test]
fn dlnetwork_initializer_uses_selected_runtime_callback_once() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let layers = training_layers(false);
    let output = block_on(layers.with_owned_async(|layers| async {
        call_builtin_async_with_outputs_in_context(&runtime, "dlnetwork", &[layers], 1).await
    }))
    .expect("initialized network");
    assert!(matches!(output.value(), Value::Object(_)));
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(1))]
        );
    });
}

#[test]
fn cancelled_runtime_rejects_both_trainers_before_initializer_effects() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    for name in ["trainNetwork", "trainnet"] {
        CALLBACKS.with(|calls| calls.borrow_mut().clear());
        let source = CancellationSource::new();
        let runtime = runtime().with_cancellation_token(source.token());
        source.cancel(CancellationReason::User);
        let layers = training_layers(name == "trainNetwork");
        let options = training_options();
        let mut owned_args = runmat_runtime::execution::RootedValuesBuilder::with_capacity(5);
        let args = vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
        ];
        for arg in args {
            owned_args.push(arg).expect("root training input");
        }
        owned_args.push_owned(layers);
        if name == "trainnet" {
            owned_args
                .push(Value::String("mse".into()))
                .expect("root loss name");
        }
        owned_args.push_owned(options);
        let args = owned_args.finish().expect("root training arguments");
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        let [binding] = bindings.as_slice() else {
            panic!("{name} must have one canonical binding");
        };
        // This invokes the registered implementation without entering a
        // thread-local runtime scope. Cancellation must come from the
        // invocation, before either callback or built-in RNG initializer.
        let error = block_on(args.with_unbounded_values_async(|args| async move {
            (binding.implementation)(RuntimeBuiltinInvocation::new(&args, Some(runtime), Some(1)))
                .await
        }))
        .and_then(std::convert::identity)
        .expect_err("cancelled training");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Cancelled"),
            "{name}: {error:?}"
        );
        CALLBACKS.with(|calls| assert!(calls.borrow().is_empty(), "{name} ran initializer"));
    }
}
