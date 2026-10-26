use super::*;
use futures::executor::block_on;
fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(data, shape).unwrap()
}
fn number(value: &Value) -> f64 {
    let Value::Num(value) = value else {
        panic!("scalar")
    };
    *value
}
fn field(value: &Value, name: &str) -> f64 {
    let Value::Struct(value) = value else {
        panic!("stats")
    };
    number(&value.fields[name])
}
#[test]
fn one_sample_probability_has_independent_cauchy_oracle_and_unshifted_confidence_interval() {
    let x = tensor(vec![1.0, 2.0], vec![2, 1]);
    let outputs = block_on(engine::evaluate(
        &x,
        None,
        options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert!(
        (number(&outputs[1]) - 2.0 * (1.0f64 / 3.0).atan() / std::f64::consts::PI).abs() < 1e-12
    );
    assert_eq!(field(&outputs[3], "df"), 1.0);
    assert!((field(&outputs[3], "tstat") - 3.0).abs() < 1e-12);
    let reference = tensor(vec![1.5], vec![1, 1]);
    let outputs = block_on(engine::evaluate(
        &x,
        Some(&reference),
        options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert_eq!(number(&outputs[1]), 1.0);
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert!((ci.materialize_f64().iter().sum::<f64>() / 2.0 - 1.5).abs() < 1e-12);
    let x = tensor(vec![1e308, -1e308], vec![2, 1]);
    let reference = tensor(vec![1e308], vec![1, 1]);
    let outputs = block_on(engine::evaluate(
        &x,
        Some(&reference),
        options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert!((field(&outputs[3], "tstat") + 1.0).abs() < 1e-12);
    assert!((number(&outputs[1]) - 0.5).abs() < 1e-12);
    let paired = tensor(vec![-1e308, 1e308], vec![2, 1]);
    let outputs = block_on(engine::evaluate(
        &x,
        Some(&paired),
        options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert_eq!(field(&outputs[3], "tstat"), 0.0);
    assert_eq!(number(&outputs[1]), 1.0);
    assert_eq!(field(&outputs[3], "df"), 1.0);
    assert_eq!(field(&outputs[3], "sd"), f64::INFINITY);
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert_eq!(ci.materialize_f64(), [f64::NEG_INFINITY, f64::INFINITY]);
}

#[test]
fn positive_dimensions_above_old_cap_work_and_unrepresentable_metadata_refuses() {
    let args = crate::builtins::common::test_support::builtin_arguments(vec![
        Value::Num(3.0),
        Value::from("Dim"),
        Value::Num(1025.0),
    ]);
    let options = block_on(options::Options::parse(args.view(), 1, None)).unwrap();
    let outputs = block_on(engine::evaluate(
        &tensor(vec![3.0], vec![1, 1]),
        None,
        options,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert_eq!(ci.shape().len(), 1025);
    assert_eq!(ci.shape()[1024], 2);
    let error = block_on(engine::evaluate(
        &tensor(vec![3.0], vec![1, 1]),
        None,
        options::Options {
            dim: Some(usize::MAX - 1),
            ..Default::default()
        },
        &mut Cooperation::new(None),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ttest:AllocationFailed"));
    let args = crate::builtins::common::test_support::builtin_arguments(vec![
        Value::Num(3.0),
        Value::from("Dim"),
        Value::Num(2f64.powi(usize::BITS as i32)),
    ]);
    assert!(block_on(options::Options::parse(args.view(), 1, None)).is_err());
}

#[test]
fn one_sided_quantiles_reflect_above_half_and_keep_small_tail_precision() {
    let x = tensor(vec![1.0, 2.0], vec![2, 1]);
    for (tail, bound, expected) in [
        (runmat_builtins::TtestTail::Right, 0, 2.0),
        (runmat_builtins::TtestTail::Left, 1, 1.0),
    ] {
        let options = options::Options {
            alpha: 0.75,
            tail,
            ..Default::default()
        };
        let outputs = block_on(engine::evaluate(
            &x,
            None,
            options,
            &mut Cooperation::new(None),
        ))
        .unwrap();
        let Value::Tensor(ci) = &outputs[2] else {
            panic!("ci")
        };
        assert!((ci.materialize_f64()[bound] - expected).abs() < 1e-10);
    }
    let options = options::Options {
        alpha: 1e-20,
        tail: runmat_builtins::TtestTail::Right,
        ..Default::default()
    };
    let outputs = block_on(engine::evaluate(
        &x,
        None,
        options,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert!(ci.materialize_f64()[0].is_finite());
}
#[test]
fn paired_nan_omission_dimension_geometry_and_one_sided_limits() {
    let x = tensor(vec![2.0, 4.0, f64::NAN, 8.0, 10.0], vec![5, 1]);
    let y = tensor(vec![1.0, 2.0, 3.0, 4.0, f64::NAN], vec![5, 1]);
    let options = options::Options {
        tail: runmat_builtins::TtestTail::Right,
        ..Default::default()
    };
    let outputs = block_on(engine::evaluate(
        &x,
        Some(&y),
        options,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert_eq!(field(&outputs[3], "df"), 2.0);
    assert!((field(&outputs[3], "sd") - (7.0f64 / 3.0).sqrt()).abs() < 1e-12);
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert_eq!(ci.materialize_f64()[1], f64::INFINITY);
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let options = options::Options {
        dim: Some(1),
        ..Default::default()
    };
    let outputs = block_on(engine::evaluate(
        &x,
        None,
        options,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    let Value::Tensor(p) = &outputs[1] else {
        panic!("array")
    };
    assert_eq!(p.shape(), [2, 1]);
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert_eq!(ci.shape(), [2, 2]);
    assert!(block_on(engine::evaluate(
        &x,
        Some(&tensor(vec![1.0, 2.0], vec![1, 2])),
        options::Options::default(),
        &mut Cooperation::new(None)
    ))
    .is_err());
}
#[test]
fn floating_classes_invalid_options_and_undefined_singleton_are_not_silently_coerced() {
    let x = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let args = crate::builtins::common::test_support::builtin_arguments(vec![x]);
    let output = block_on(evaluate(None, args)).unwrap();
    assert!(matches!(output, Value::OutputList(_)));
    for value in [
        Value::Bool(true),
        Value::Int(runmat_value::IntValue::I64(3)),
    ] {
        assert!(inferential::real_sample(NAME, &value).is_err());
    }
    for options in [
        vec![Value::from("VarType"), Value::from("equal")],
        vec![Value::from("Alpha"), Value::Num(1.0)],
        vec![Value::from("Dim"), Value::Num(0.0)],
        vec![Value::from("Tail"), Value::from("larger")],
    ] {
        let mut args = vec![Value::Num(1.0)];
        args.extend(options);
        assert!(block_on(evaluate(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }
    let outputs = block_on(engine::evaluate(
        &tensor(vec![1.0], vec![1, 1]),
        None,
        options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert!(number(&outputs[1]).is_nan());
    assert_eq!(outputs[0], Value::Bool(false));
}
