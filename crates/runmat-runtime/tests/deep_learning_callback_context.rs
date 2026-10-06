use futures::executor::block_on;
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation},
    call_builtin_async, call_builtin_async_with_outputs_in_context,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use runmat_value::{CellArray, StructArray, StructValue, Tensor, Value, ValueSequence};
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
    let first = invocation.args.first().cloned();
    Box::pin(async move {
        CALLBACKS.with(|calls| calls.borrow_mut().push((runtime.local_identity(), outputs)));
        if let Some(Value::Tensor(shape)) = first {
            let dimensions: Vec<usize> = shape
                .host_buffer()
                .materialize_f64()
                .into_iter()
                .map(|size| size as usize)
                .collect();
            let count = dimensions.iter().product();
            return Ok(ValueSequence::single(Value::Tensor(
                Tensor::new(vec![0.25; count], dimensions).expect("fixture initializer shape"),
            ))?);
        }
        match outputs {
            Some(2) => ValueSequence::comma_separated(vec![Value::Num(23.0), Value::Num(29.0)])
                .map_err(Into::into),
            _ => ValueSequence::single(Value::Num(23.0)).map_err(Into::into),
        }
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

fn training_layers(regression_output: bool) -> Value {
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
    let mut layers = vec![input, dense];
    if regression_output {
        layers.push(block_on(call_builtin_async("regressionLayer", &[])).expect("regression"));
    }
    let count = layers.len();
    Value::Cell(CellArray::new(layers, count, 1).expect("layer column"))
}

fn training_options() -> Value {
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
    assert_eq!(
        output,
        Value::OutputList(vec![Value::Num(23.0), Value::Num(29.0)])
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
    let mut first = StructValue::new();
    first.insert("Data", Value::Num(1.0));
    let mut second = StructValue::new();
    second.insert("Data", Value::Num(2.0));
    let mut structure = StructValue::new();
    structure.insert(
        "Array",
        Value::StructArray(StructArray::new(vec![first, second], vec![1, 2]).unwrap()),
    );
    structure.insert("Scalar", Value::Num(3.0));
    let tree = Value::Cell(CellArray::new(vec![Value::Struct(structure)], 1, 1).unwrap());
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlupdate",
        &[Value::FunctionHandle("plus".into()), tree],
        1,
    ))
    .expect("nested callback tree");
    let Value::Cell(cell) = output else {
        panic!("expected cell tree");
    };
    let Value::Struct(structure) = &cell.data[0] else {
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
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlnetwork",
        &[training_layers(false)],
        1,
    ))
    .expect("initialized network");
    assert!(matches!(output, Value::Object(_)));
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
        let mut args = vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            training_layers(name == "trainNetwork"),
        ];
        if name == "trainnet" {
            args.push(Value::String("mse".into()));
        }
        args.push(training_options());
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        let [binding] = bindings.as_slice() else {
            panic!("{name} must have one canonical binding");
        };
        // This invokes the registered implementation without entering a
        // thread-local runtime scope. Cancellation must come from the
        // invocation, before either callback or built-in RNG initializer.
        let error = block_on((binding.implementation)(RuntimeBuiltinInvocation::new(
            &args,
            Some(runtime),
            Some(1),
        )))
        .expect_err("cancelled training");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Cancelled"),
            "{name}: {error:?}"
        );
        CALLBACKS.with(|calls| assert!(calls.borrow().is_empty(), "{name} ran initializer"));
    }
}
