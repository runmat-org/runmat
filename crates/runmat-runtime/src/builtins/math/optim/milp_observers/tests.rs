use super::*;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::rc::Rc;
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn metadata(fval: f64) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("fval", Value::Num(fval));
    Value::Struct(fields.finish())
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(v) => v.materialize_f64(),
        _ => panic!("numeric history"),
    }
}
#[test]
fn solution_events_use_original_logical_base_and_do_not_replay_empty_or_done() {
    let owner = crate::workspace::BaseWorkspaceOwner::new();
    let mut bindings = crate::workspace::WorkspaceDelta::new();
    bindings.bind_global("xIntSol");
    owner.apply(bindings).unwrap();
    let service = Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let ports = runtime.service_ports().clone().with_base_workspace(service);
    let runtime = runtime.with_service_ports(ports);
    for (state, point, fval) in [
        ("init", Vec::new(), 0.0),
        ("iter", vec![1.0, 3.0], 4.0),
        ("iter", Vec::new(), 99.0),
        ("iter", vec![1.0, 2.0], 3.0),
        ("done", vec![9.0, 9.0], 99.0),
    ] {
        let shape = if point.is_empty() {
            vec![0, 0]
        } else {
            vec![2, 1]
        };
        let result = block_on(execute(
            MilpObserverKind::SaveSolutions,
            Some(&runtime),
            args(vec![
                Value::Tensor(Tensor::new(point, shape).unwrap()),
                metadata(fval),
                Value::from(state),
            ]),
        ))
        .unwrap();
        assert!(matches!(result, Value::Bool(false)));
    }
    assert!(
        owner.lookup("xIntSol").unwrap().is_none(),
        "bound global identity is not collapsed into base-local storage"
    );
    let points = runtime
        .session_variable_authority()
        .global_value("xIntSol")
        .unwrap()
        .unwrap();
    assert_eq!(numbers(&points), vec![1.0, 3.0, 1.0, 2.0]);
    let Value::Tensor(points) = points else {
        panic!("point matrix");
    };
    assert_eq!(points.shape(), &[2, 2]);
    assert_eq!(
        numbers(&owner.lookup("fIntSol").unwrap().unwrap()),
        vec![4.0, 3.0]
    );
    let before = owner.revision().unwrap();
    let invalid = block_on(execute(
        MilpObserverKind::SaveSolutions,
        Some(&runtime),
        args(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            metadata(6.0),
            Value::from("iter"),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        invalid.identifier(),
        Some("RunMat:savemilpsolutions:InvalidArgument")
    );
    assert_eq!(owner.revision().unwrap(), before);
    assert_eq!(
        numbers(&owner.lookup("fIntSol").unwrap().unwrap()),
        vec![4.0, 3.0]
    );
}
#[test]
fn predefined_constructor_forms_validate_real_identity_without_accept_ignore() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let prepared = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "optimoptions",
        &[
            Value::from("intlinprog"),
            Value::from("OutputFcn"),
            Value::from("savemilpsolutions"),
            Value::from("PlotFcn"),
            Value::from("optimplotmilp"),
        ],
        1,
    ))
    .unwrap();
    let Value::Struct(options) = prepared.value() else {
        panic!("options");
    };
    assert!(matches!(&options.fields["OutputFcn"],Value::String(v) if v=="savemilpsolutions"));
    assert!(matches!(&options.fields["PlotFcn"],Value::String(v) if v=="optimplotmilp"));
    let error = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "optimoptions",
        &[
            Value::from("intlinprog"),
            Value::from("OutputFcn"),
            Value::from("optimplotmilp"),
        ],
        1,
    ))
    .unwrap_err();
    assert!(error.to_string().contains("observer") || error.to_string().contains("callback"));
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let error = block_on(execute(
        MilpObserverKind::SaveSolutions,
        Some(&runtime),
        args(vec![
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            metadata(0.0),
            Value::from("init"),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
