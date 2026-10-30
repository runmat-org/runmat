use super::*;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::IntegerStorage;
use std::sync::Arc;

use crate::builtins::stats::ml::bayesian_test_provider as provider_fixture;

fn variable(name: &str, range: Vec<f64>, var_type: Option<&str>) -> Value {
    let mut object = ObjectInstance::new("optimizableVariable");
    object.replace_property_unbounded("Name".into(), Value::String(name.into()));
    object.replace_property_unbounded(
        "Range".into(),
        Value::Tensor(Tensor::new(range, vec![1, 2]).unwrap()),
    );
    object.replace_property_unbounded(
        "Type".into(),
        Value::String(var_type.unwrap_or("real").into()),
    );
    object.replace_property_unbounded("Transform".into(), Value::String("none".into()));
    object.replace_property_unbounded("Optimize".into(), Value::Bool(true));
    Value::Object(object)
}

fn variable_with_range(name: &str, range: Value, var_type: Option<&str>) -> Value {
    let mut object = ObjectInstance::new("optimizableVariable");
    object.replace_property_unbounded("Name".into(), Value::String(name.into()));
    object.replace_property_unbounded("Range".into(), range);
    object.replace_property_unbounded(
        "Type".into(),
        Value::String(var_type.unwrap_or("real").into()),
    );
    object.replace_property_unbounded("Transform".into(), Value::String("none".into()));
    object.replace_property_unbounded("Optimize".into(), Value::Bool(true));
    Value::Object(object)
}

#[test]
fn output_count_and_bad_variable_identity_are_rejected_before_callbacks() {
    let error = block_on(bayesopt_in_runtime(
        None,
        Some(2),
        Value::Num(0.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("multiple results are unsupported before callback admission");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);

    let mut unrelated = ObjectInstance::new(runmat_types::standard::LINEAR_MODEL);
    unrelated.replace_property_unbounded("Name".into(), Value::String("x".into()));
    let error = parse_variables(Value::Object(unrelated)).expect_err("wrong object class");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn later_invalid_host_control_precedes_earlier_resident_control_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let arguments = vec![
        Value::FunctionHandle("objective".into()),
        variable("x", vec![0.0, 1.0], None),
        Value::String("MaxObjectiveEvaluations".into()),
        Value::GpuTensor(resident.handle.clone()),
        Value::String("UseParallel".into()),
        Value::Bool(true),
    ];
    let error = block_on(crate::call_builtin_async("bayesopt", &arguments))
        .expect_err("host-invalid control must win before resident control download");
    assert_eq!(error.identifier(), ERROR_UNSUPPORTED.identifier);
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed resident operand remains live");
}

#[test]
fn numeric_resident_acquisition_is_rejected_before_other_downloads() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let handle = || Value::GpuTensor(resident.handle.clone());
    let error = block_on(bayesopt_builtin(
        Value::FunctionHandle("objective".into()),
        variable("x", vec![0.0, 1.0], None),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxObjectiveEvaluations".into()),
            handle(),
            Value::String("AcquisitionFunctionName".into()),
            handle(),
        ]),
    ))
    .expect_err("numeric resident acquisition name is invalid before downloads");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert!(error.message().contains("AcquisitionFunctionName"));
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed resident control remains live");
}

#[test]
fn failed_variable_download_keeps_nested_cause_and_borrowed_allocation() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let error = block_on(bayesopt_builtin(
        Value::FunctionHandle("objective".into()),
        Value::GpuTensor(resident.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("fixture download failure");
    assert_eq!(error.identifier(), ERROR_FLOW.identifier);
    let source = error.source.as_ref().expect("retained gather error");
    let mut found = source
        .to_string()
        .contains("Bayesian fixture download rejected");
    let mut cause = source.source();
    while let Some(next) = cause {
        found |= next
            .to_string()
            .contains("Bayesian fixture download rejected");
        cause = next.source();
    }
    assert!(
        found,
        "original provider cause must remain in the source chain"
    );
    assert_eq!(provider.downloads(), 1);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("failed gather must not retire caller allocation");
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn option_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, vec![1, 1]);
        assert!(option_bool(&value, "Verbose").unwrap());
        assert_eq!(
            option_usize(&value, "MaxObjectiveEvaluations", 1, 2).unwrap(),
            1
        );
        assert_eq!(option_unit_f64(&value, "ExplorationRatio").unwrap(), 1.0);
    }
}

#[test]
fn cancelled_invocation_stops_before_search_or_callback() {
    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(bayesopt_in_runtime(
        Some(&runtime),
        None,
        Value::Num(0.0),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("cancelled call must not invoke objective");
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}

fn categorical_variable(name: &str, categories: Vec<&str>) -> Value {
    let mut object = ObjectInstance::new("optimizableVariable");
    object.replace_property_unbounded("Name".into(), Value::String(name.into()));
    object.replace_property_unbounded(
        "Range".into(),
        Value::StringArray(
            StringArray::new(
                categories.into_iter().map(str::to_string).collect(),
                vec![1, 2],
            )
            .unwrap(),
        ),
    );
    object.replace_property_unbounded("Type".into(), Value::String("categorical".into()));
    object.replace_property_unbounded("Transform".into(), Value::String("none".into()));
    object.replace_property_unbounded("Optimize".into(), Value::Bool(true));
    Value::Object(object)
}

fn variable_cell(values: Vec<Value>) -> Value {
    Value::Cell(CellArray::new(values, 1, 2).unwrap())
}

fn table_scalar_arg(args: &[Value], name: &str) -> f64 {
    let Value::Object(table) = &args[0] else {
        panic!("expected table");
    };
    let Value::Struct(vars) = table.properties().get("__table_variables").unwrap() else {
        panic!("expected table variables");
    };
    let Value::Tensor(tensor) = vars.fields.get(name).unwrap() else {
        panic!("expected numeric column");
    };
    tensor.materialize_f64()[0]
}

fn table_string_arg(args: &[Value], name: &str) -> String {
    let Value::Object(table) = &args[0] else {
        panic!("expected table");
    };
    let Value::Struct(vars) = table.properties().get("__table_variables").unwrap() else {
        panic!("expected table variables");
    };
    let Value::StringArray(array) = vars.fields.get(name).unwrap() else {
        panic!("expected string column");
    };
    array.data()[0]
        .as_str()
        .expect("fixture requires a present string parameter")
        .to_owned()
}

#[test]
fn bayesopt_minimizes_quadratic_over_optimizable_variables() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let objective = crate::user_functions::test_support::exact_local_binding(&runtime, 1);
    let entry = runtime.enter();
    let _bound =
        crate::user_functions::test_support::install_bound_forwarders(vec![objective.clone()]);
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 1);
            assert_eq!(requested_outputs, 1);
            let x = table_scalar_arg(args, "x");
            let y = table_scalar_arg(args, "y");
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    (x - 0.25).powi(2) + (y + 0.5).powi(2),
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let result = block_on(runtime.scope(bayesopt_builtin(
        crate::user_functions::test_support::exact_local_value("objective", objective),
        variable_cell(vec![
            variable("x", vec![-1.0, 1.0], None),
            variable("y", vec![-1.0, 1.0], None),
        ]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxObjectiveEvaluations".into()),
            Value::Num(24.0),
            Value::String("NumSeedPoints".into()),
            Value::Num(6.0),
        ]),
    )))
    .unwrap();
    let Value::Object(object) = result else {
        panic!("expected BayesianOptimization object");
    };
    assert!(object.class_name().is(RESULT_CLASS));
    let Some(Value::Num(minimum)) = object.properties().get("MinObjective") else {
        panic!("expected minimum");
    };
    assert!(*minimum < 0.08, "minimum was {minimum}");
    assert!(matches!(
        object.properties().get("XTrace"),
        Some(Value::Object(table)) if table.class_name().is(runmat_types::standard::TABLE)
    ));
}

#[test]
fn bayesopt_rejects_parallel_option_explicitly() {
    let err = block_on(bayesopt_builtin(
        Value::FunctionHandle("objective".into()),
        variable("x", vec![0.0, 1.0], None),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("UseParallel".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:bayesopt:Unsupported"));
}

#[test]
fn bayesopt_honors_x_constraint_function() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let objective = crate::user_functions::test_support::exact_local_binding(&runtime, 1);
    let constraint = crate::user_functions::test_support::exact_local_binding(&runtime, 2);
    let entry = runtime.enter();
    let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
        objective.clone(),
        constraint.clone(),
    ]);
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert!(matches!(function, 1 | 2));
            let x = table_scalar_arg(args, "x");
            Box::pin(async move {
                if function == 2 {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Bool(
                        x >= 0.0,
                    ))
                } else {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        (x - 0.4).powi(2),
                    ))
                }
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let result = block_on(runtime.scope(bayesopt_builtin(
        crate::user_functions::test_support::exact_local_value("objective", objective),
        variable("x", vec![-1.0, 1.0], None),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxObjectiveEvaluations".into()),
            Value::Num(12.0),
            Value::String("XConstraintFcn".into()),
            crate::user_functions::test_support::exact_local_value("constraint", constraint),
        ]),
    )))
    .unwrap();
    let Value::Object(object) = result else {
        panic!("expected result object");
    };
    let Some(Value::Num(count)) = object.properties().get("NumObjectiveEvaluations") else {
        panic!("expected count");
    };
    assert_eq!(*count, 12.0);
}

#[test]
fn bayesopt_supports_mixed_variables_and_objective_constraint_struct() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let objective = crate::user_functions::test_support::exact_local_binding(&runtime, 3);
    let entry = runtime.enter();
    let _bound =
        crate::user_functions::test_support::install_bound_forwarders(vec![objective.clone()]);
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 3);
            let n = table_scalar_arg(args, "n");
            let family = table_string_arg(args, "family");
            Box::pin(async move {
                let mut out = runmat_value::UnboundedStructValueBuilder::new();
                let family_penalty = if family == "rbf" { 0.0 } else { 1.0 };
                out.insert("Objective", Value::Num((n - 2.0).powi(2) + family_penalty));
                out.insert("ConstraintViolations", Value::Num(1.5 - n));
                let out = out.finish();
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Struct(
                    out,
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let result = block_on(runtime.scope(bayesopt_builtin(
        crate::user_functions::test_support::exact_local_value("objective", objective),
        variable_cell(vec![
            variable("n", vec![1.0, 3.0], Some("integer")),
            categorical_variable("family", vec!["linear", "rbf"]),
        ]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxObjectiveEvaluations".into()),
            Value::Num(18.0),
            Value::String("AcquisitionFunctionName".into()),
            Value::String("expected-improvement".into()),
        ]),
    )))
    .unwrap();
    let Value::Object(object) = result else {
        panic!("expected result object");
    };
    let Some(Value::Num(minimum)) = object.properties().get("MinObjective") else {
        panic!("expected minimum");
    };
    assert!(*minimum <= 0.25, "minimum was {minimum}");
    assert!(matches!(
        object.properties().get("FeasibilityTrace"),
        Some(Value::LogicalArray(_))
    ));
}

#[test]
fn bayesopt_reads_typed_integer_storage_exactly() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let objective = crate::user_functions::test_support::exact_local_binding(&runtime, 10);
    let constraint = crate::user_functions::test_support::exact_local_binding(&runtime, 11);
    let entry = runtime.enter();
    let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
        objective.clone(),
        constraint.clone(),
    ]);
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert!(matches!(function, 10 | 11));
            let x = table_scalar_arg(args, "x");
            Box::pin(async move {
                if function == 11 {
                    return crate::user_functions::test_support::rooted_single_value_sequence(
                        poisoned_int_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
                    );
                }
                let mut out = runmat_value::UnboundedStructValueBuilder::new();
                out.insert("Objective", Value::Num((x - 1.0).powi(2)));
                out.insert(
                    "ConstraintViolations",
                    poisoned_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
                );
                let out = out.finish();
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Struct(
                    out,
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let result = block_on(runtime.scope(bayesopt_builtin(
        crate::user_functions::test_support::exact_local_value("objective", objective),
        variable_with_range(
            "x",
            poisoned_int_tensor(IntegerStorage::I16(vec![-1, 2]), vec![1, 2]),
            None,
        ),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxObjectiveEvaluations".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![6]), vec![1, 1]),
            Value::String("NumSeedPoints".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            Value::String("ExplorationRatio".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::String("IsObjectiveDeterministic".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::String("XConstraintFcn".into()),
            crate::user_functions::test_support::exact_local_value("constraint", constraint),
        ]),
    )))
    .unwrap();
    let Value::Object(object) = result else {
        panic!("expected BayesianOptimization object");
    };
    let Some(Value::Num(count)) = object.properties().get("NumObjectiveEvaluations") else {
        panic!("expected count");
    };
    assert_eq!(*count, 6.0);
    let Some(Value::Num(minimum)) = object.properties().get("MinObjective") else {
        panic!("expected minimum");
    };
    assert!(*minimum <= 0.25, "minimum was {minimum}");
}
