use super::*;
use futures::executor::block_on;

fn options(args: Vec<Value>) -> crate::execution::RootedValue {
    let args = crate::execution::RootedValues::new(args).unwrap();
    block_on(optimoptions_builtin(None, args.into())).unwrap()
}

#[test]
fn typed_constructor_preserves_observers_and_gradient_and_drives_actual_lp_iteration_limit() {
    let observers = Value::Cell(
        runmat_value::CellArray::new(
            vec![
                Value::FunctionHandle("observer_one".into()),
                Value::FunctionHandle("observer_two".into()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let built = options(vec![
        Value::from("fminunc"),
        Value::from("MaxIterations"),
        Value::Num(7.0),
        Value::from("OutputFcn"),
        observers.clone(),
    ]);
    let Value::Struct(ref fields) = built.value() else {
        panic!("options")
    };
    assert_eq!(fields.fields["MaxIter"], Value::Num(7.0));
    assert_eq!(fields.fields["OutputFcn"], observers);
    let built = options(vec![
        Value::from("fsolve"),
        Value::from("SpecifyObjectiveGradient"),
        Value::Bool(true),
    ]);
    let Value::Struct(ref fields) = built.value() else {
        panic!("options")
    };
    assert_eq!(fields.fields["SpecifyObjectiveGradient"], Value::Bool(true));

    let built = options(vec![
        Value::from("linprog"),
        Value::from("MaxIterations"),
        Value::Num(0.0),
        Value::from("Display"),
        Value::from("off"),
    ]);
    let empty = || Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    let result = block_on(crate::call_builtin_async_with_outputs(
        "linprog",
        &[
            Value::Num(1.0),
            Value::Num(-1.0),
            Value::Num(-1.0),
            empty(),
            empty(),
            empty(),
            empty(),
            built.value().clone(),
        ],
        4,
    ))
    .unwrap();
    let Value::OutputList(outputs) = result.value() else {
        panic!("outputs")
    };
    assert_eq!(outputs[2], Value::Num(0.0));
    let Value::Struct(ref info) = outputs[3] else {
        panic!("info")
    };
    assert_eq!(info.fields["iterations"], Value::Num(0.0));
    let Value::Tensor(point) = &outputs[0] else {
        panic!("decision column")
    };
    assert_eq!(point.shape(), [1, 1]);
    let x = tensor::tensor_value_f64(point, 0);
    assert!(x.is_finite());
    // The original-row incumbent gate uses normalized feasibility TOL=1e-8:
    // row coefficient scale 1 + |b| 1 + sum|a*x| |x|, not exact feasibility.
    assert!((1.0 - x).max(0.0) / (2.0 + x.abs()) <= 1e-8);
    assert_eq!(
        outputs[1],
        Value::Num(x),
        "fval must match the original objective f=1"
    );
}

#[test]
fn unsupported_algorithms_are_not_accepted_as_bfgs_or_enumeration_aliases() {
    for (solver, name, value) in [
        ("linprog", "Algorithm", Value::from("dual-simplex-highs")),
        (
            "linprog",
            "OutputFcn",
            Value::FunctionHandle("observer".into()),
        ),
    ] {
        let args = crate::execution::RootedValues::new(vec![
            Value::from(solver),
            Value::from(name),
            value,
        ])
        .unwrap();
        assert!(block_on(optimoptions_builtin(None, args.into())).is_err());
    }
}

#[test]
fn trust_region_controls_keep_semantic_tolerances_and_supplied_origins() {
    let built = options(vec![
        Value::from("fminunc"),
        Value::from("Algorithm"),
        Value::from("trust-region"),
        Value::from("SpecifyObjectiveGradient"),
        Value::Bool(true),
        Value::from("TolFun"),
        Value::Num(0.0),
        Value::from("OptimalityTolerance"),
        Value::Num(2e-7),
    ]);
    let Value::Struct(fields) = built.value() else {
        panic!("options")
    };
    assert_eq!(fields.fields["FunctionTolerance"], Value::Num(0.0));
    assert_eq!(fields.fields["OptimalityTolerance"], Value::Num(2e-7));
    assert_eq!(fields.fields["TolPCG"], Value::Num(0.1));
    assert_eq!(fields.fields["PrecondBandWidth"], Value::Num(0.0));
    assert_eq!(fields.fields["SubproblemAlgorithm"], Value::from("cg"));
    assert!(matches!(&fields.fields["MaxFunEvals"], Value::Tensor(t) if t.is_empty()));
    assert!(!fields.fields.contains_key("TolFun"));
    let overridden = options(vec![
        built.value().clone(),
        Value::from("MaxFunctionEvaluations"),
        Value::Num(40000.0),
    ]);
    let rebuilt = options(vec![overridden.value().clone()]);
    let Value::Struct(fields) = rebuilt.value() else {
        panic!("options")
    };
    assert_eq!(
        fields.fields["MaxFunEvals"],
        Value::Num(40000.0),
        "an imported value equal to the QN default remains supplied"
    );
    let conflicting = crate::execution::RootedValues::new(vec![
        Value::from("fminunc"),
        Value::from("TolFun"),
        Value::Num(1e-7),
        Value::from("OptimalityTolerance"),
        Value::Num(2e-7),
    ])
    .unwrap();
    assert!(block_on(optimoptions_builtin(None, conflicting.into())).is_err());
    let inactive = options(vec![
        Value::from("fminunc"),
        Value::from("Algorithm"),
        Value::from("quasi-newton"),
        Value::from("FunctionTolerance"),
        Value::Num(3e-7),
    ]);
    let Value::Struct(fields) = inactive.value() else {
        panic!("options")
    };
    assert_eq!(fields.fields["FunctionTolerance"], Value::Num(3e-7));
}
