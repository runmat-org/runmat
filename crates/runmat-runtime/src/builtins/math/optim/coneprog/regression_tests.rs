use super::tests::{empty, run, soc_unit_disk, tensor};
use super::*;
use futures::executor::block_on;
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(t) => t.materialize_f64(),
        _ => panic!("numeric result"),
    }
}
fn number(value: &Value) -> f64 {
    values(value)[0]
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("structure")
    };
    &value.fields[name]
}
fn descriptor(a: Value, b: Value, d: Value, gamma: f64) -> Value {
    block_on(secondordercone_builtin(a, b, d, Value::Num(gamma))).unwrap()
}

#[test]
fn reported_unit_ball_has_actual_optimum_and_numeric_soc_dual() {
    let cone = descriptor(
        tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 1, 2),
        -1.0,
    );
    let output = run(vec![tensor(vec![-1.0, -1.0], 1, 2), cone], 5);
    assert_eq!(number(&output[2]), 1.0);
    let x = values(&output[0]);
    for value in &x {
        assert!((value - 0.5_f64.sqrt()).abs() < 1e-6);
    }
    assert!((number(&output[1]) + 2.0_f64.sqrt()).abs() < 1e-6);
    let Value::Tensor(dual) = field(&output[4], "soc") else {
        panic!("numeric SOC column, not legacy cell")
    };
    assert_eq!(dual.shape(), &[1, 1]);
    let dual = dual.materialize_f64()[0];
    assert!((dual - 2.0_f64.sqrt()).abs() < 1e-6);
    for value in x {
        assert!((-1.0 + dual * value).abs() < 1e-6);
    }
    assert!(number(field(&output[3], "dualitygap")) < 1e-6);
}

#[test]
fn affine_centers_rhs_and_multiple_descriptor_containers_preserve_signs_order() {
    // |x-1|<=0.5*x+1 => 0<=x<=4, with multiplier2 at max x.
    let cone = descriptor(Value::Num(1.0), Value::Num(1.0), Value::Num(0.5), -1.0);
    let output = run(vec![Value::Num(-1.0), cone], 5);
    assert_eq!(number(&output[2]), 1.0);
    assert!((number(&output[0]) - 4.0).abs() < 1e-6);
    assert!((number(field(&output[4], "soc")) - 2.0).abs() < 1e-6);
    let first = descriptor(Value::Num(1.0), Value::Num(0.0), Value::Num(0.0), -1.0);
    let second = descriptor(Value::Num(1.0), Value::Num(0.5), Value::Num(0.0), -1.0);
    let Value::Struct(a) = first.clone() else {
        unreachable!()
    };
    let Value::Struct(b) = second.clone() else {
        unreachable!()
    };
    let containers = [
        Value::Cell(
            runmat_value::CellArray::new(vec![first.clone(), second.clone()], 2, 1).unwrap(),
        ),
        Value::StructArray(runmat_value::StructArray::new(vec![a, b], vec![2, 1]).unwrap()),
        Value::output_list(vec![first, second]),
    ];
    for container in containers {
        let output = run(vec![Value::Num(1.0), container], 5);
        assert_eq!(number(&output[2]), 1.0);
        assert!((number(&output[0]) + 0.5).abs() < 1e-6);
        let dual = values(field(&output[4], "soc"));
        assert_eq!(dual.len(), 2);
        assert!(dual[0].abs() < 1e-6 && (dual[1] - 1.0).abs() < 1e-6);
    }
    // A zero-radius cone has no strict interior but a genuine attained optimum.
    let apex = descriptor(
        tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 2, 1),
        0.0,
    );
    let output = run(vec![tensor(vec![-1.0, 0.0], 2, 1), apex], 5);
    assert_eq!(number(&output[2]), 1.0);
    assert!(values(&output[0]).iter().all(|value| value.abs() < 1e-6));
}

#[test]
fn sparse_constructor_linear_equalities_bounds_and_real_iteration_display_work() {
    let a = Value::SparseTensor(
        runmat_value::SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
            .unwrap(),
    );
    let cone = descriptor(
        a,
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 2, 1),
        -1.0,
    );
    assert!(
        matches!(field(&cone,"A"),Value::SparseTensor(value) if value.numeric_dtype()==Some(runmat_value::NumericDType::F64))
    );
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in [
        ("MaxIterations", Value::Num(200.0)),
        ("TolFun", Value::Num(1e-8)),
        ("TolX", Value::Num(1e-8)),
        ("Algorithm", Value::from("interior-point-convex")),
        ("Display", Value::from("iter")),
    ] {
        options.insert(name, value);
    }
    let a = Value::SparseTensor(
        runmat_value::SparseTensor::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap(),
    );
    let eq = Value::SparseTensor(
        runmat_value::SparseTensor::new(1, 2, vec![0, 0, 1], vec![0], vec![1.0]).unwrap(),
    );
    let capture = crate::console::begin_capture().unwrap();
    let output = run(
        vec![
            tensor(vec![-1.0, 0.0], 2, 1),
            cone,
            a,
            Value::Num(0.75),
            eq,
            Value::Num(0.0),
            tensor(vec![0.0, 0.0], 2, 1),
            tensor(vec![1.0, 0.0], 2, 1),
            Value::Struct(options.finish()),
        ],
        5,
    );
    let text = capture.finish().unwrap();
    assert!(
        text.contains("Primal objective")
            && text.contains("optimization phase")
            && text.contains("Iterations:")
    );
    assert_eq!(number(&output[2]), 1.0);
    let x = values(&output[0]);
    assert!((x[0] - 0.75).abs() < 1e-6 && x[1].abs() < 1e-6);
    assert!((number(field(&output[4], "ineqlin")) - 1.0).abs() < 1e-6);
    assert!(number(field(&output[4], "soc")).abs() < 1e-6);
}

#[test]
fn terminal_status_limits_and_unsupported_options_are_not_fabricated() {
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::CONEPROG_CATALOG_ENTRY.identity, None);
    let mut precise = runmat_value::UnboundedStructValueBuilder::new();
    precise.insert("TolFun", Value::Num(1e-14));
    precise.insert("TolX", Value::Num(1e-14));
    let precise = block_on(options::Controls::parse(
        Some(precise.finish()),
        None,
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(
        precise.tol, 1e-14,
        "requested precision must not be silently relaxed"
    );
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut option = runmat_value::UnboundedStructValueBuilder::new();
        option.insert("TolFun", Value::Num(invalid));
        assert!(block_on(options::Controls::parse(
            Some(option.finish()),
            None,
            &mut cooperation
        ))
        .is_err());
    }
    let impossible = descriptor(
        tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 2, 1),
        1.0,
    );
    let output = run(vec![tensor(vec![-1.0, -1.0], 2, 1), impossible], 5);
    assert_eq!(number(&output[2]), -2.0);
    assert!(
        values(&output[0]).is_empty()
            && values(&output[1]).is_empty()
            && values(&output[4]).is_empty()
    );
    let cone = descriptor(
        tensor(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 3),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0, 1.0], 3, 1),
        0.0,
    );
    let output = run(vec![tensor(vec![0.0, 0.0, -1.0], 3, 1), cone], 5);
    assert_eq!(number(&output[2]), -3.0);
    assert!(values(&output[0]).is_empty() && values(&output[4]).is_empty());
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("MaxIterations", Value::Num(0.0));
    let output = run(
        vec![
            tensor(vec![-1.0, -1.0], 2, 1),
            soc_unit_disk(),
            Value::Struct(options.finish()),
        ],
        5,
    );
    assert_eq!(number(&output[2]), 0.0);
    assert_eq!(number(field(&output[3], "iterations")), 0.0);
    assert!(values(&output[4]).is_empty());
    for (name, value) in [
        ("MaxFunEvals", Value::Num(1.0)),
        ("OutputFcn", Value::Num(1.0)),
        ("Algorithm", Value::from("bfgs")),
        ("Unexpected", Value::Num(1.0)),
    ] {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(name, value);
        let error = block_on(coneprog_builtin(vec![
            Value::Num(1.0),
            empty(),
            Value::Struct(options.finish()),
        ]))
        .unwrap_err();
        assert!(error.message().contains("unsupported"), "{name}: {error:?}");
    }
}

#[test]
fn finite_policy_refuses_both_public_entries_before_argument_or_workspace_work() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let before = ledger.snapshot();
    let error = block_on(secondordercone_builtin_registered(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    let error = block_on(coneprog_builtin_registered(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    let after = ledger.snapshot();
    assert_eq!(before.live_claimed_bytes, after.live_claimed_bytes);
    assert_eq!(before.live_reserved_bytes, after.live_reserved_bytes);
}
