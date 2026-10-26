use super::*;
use futures::executor::block_on;
fn groups(y: Vec<f64>, shape: Vec<usize>, group: Option<Value>) -> groups::Groups {
    block_on(groups::collect(
        &Tensor::new(y, shape).unwrap(),
        group.as_ref(),
        &mut Cooperation::new(None),
    ))
    .unwrap()
}
#[test]
fn balanced_f_and_table_geometry_have_independent_beta_integral_oracle() {
    let data = groups(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], None);
    let outputs = block_on(engine::evaluate(&data, &mut Cooperation::new(None))).unwrap();
    let Value::Cell(table) = &outputs[1] else {
        panic!("table")
    };
    assert_eq!((table.rows, table.cols), (4, 6));
    assert!(matches!(table.data()[6+4],Value::Num(value) if (value-13.5).abs()<1e-12));
    assert_eq!(table.data()[6 + 2], Value::Num(1.0));
    assert_eq!(table.data()[12 + 2], Value::Num(4.0));
    let s = (13.5f64 / 17.5).sqrt();
    let p = 1.0 - 1.5 * s + 0.5 * s.powi(3);
    assert!(matches!(outputs[0],Value::Num(actual) if (actual-p).abs()<1e-12));
    let Value::Struct(stats) = &outputs[2] else {
        panic!("stats")
    };
    assert_eq!(stats.fields["s"], Value::Num(1.0));
}

#[test]
fn f_two_one_tail_survives_ratio_overflow_normalization_underflow_and_displayed_infinity() {
    // k=3, N=4: dfg=2,dfe=1. SSE=2*w^2, SSB=2*s^2,
    // hence p=sqrt(SSE/(SSE+SSB))=w/hypot(w,s).
    for (w, s) in [
        (1.0, 2.0f64.sqrt() * 1e154),
        (1.0, 1e200),
        (1.0, 1e300),
        (1e-150, 1e150),
    ] {
        let data = groups::Groups {
            names: vec!["A".into(), "B".into(), "C".into()],
            values: vec![vec![-w, w], vec![-s], vec![s]],
        };
        let outputs = block_on(engine::evaluate(&data, &mut Cooperation::new(None))).unwrap();
        let Value::Num(actual) = outputs[0] else {
            panic!("p")
        };
        let expected = w / w.hypot(s);
        assert!(
            actual > 0.0 && (actual / expected - 1.0).abs() < 1e-12,
            "{w} {s}: {actual} != {expected}"
        );
        let Value::Cell(table) = &outputs[1] else {
            panic!("table")
        };
        let Value::Num(f) = table.data()[6 + 4] else {
            panic!("F")
        };
        let scaled = (s / w) / 2.0f64.sqrt();
        let expected_f = scaled * scaled;
        if expected_f.is_finite() {
            assert!(f.is_finite() && (f / expected_f - 1.0).abs() < 1e-12);
        } else {
            assert!(f.is_infinite());
        }
        let Value::Struct(stats) = &outputs[2] else {
            panic!("stats")
        };
        assert!(matches!(stats.fields["s"],Value::Num(sd) if sd>0.0 && sd.is_finite()));
    }
    for (values, expected) in [
        (vec![vec![1.0, 1.0], vec![2.0], vec![3.0]], 0.0),
        (vec![vec![1.0, 1.0], vec![1.0], vec![1.0]], f64::NAN),
    ] {
        let data = groups::Groups {
            names: vec!["A".into(), "B".into(), "C".into()],
            values,
        };
        let outputs = block_on(engine::evaluate(&data, &mut Cooperation::new(None))).unwrap();
        assert!(
            matches!(outputs[0],Value::Num(actual) if actual==expected || actual.is_nan()&&expected.is_nan())
        );
    }
}
#[test]
fn duplicate_column_labels_merge_and_missing_observations_are_omitted() {
    let labels = Value::Cell(
        CellArray::new(
            vec![Value::from("A"), Value::from("B"), Value::from("A")],
            1,
            3,
        )
        .unwrap(),
    );
    let data = groups(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Some(labels));
    assert_eq!(data.names, ["A", "B"]);
    assert_eq!(data.values[0], [1.0, 2.0, 5.0, 6.0]);
    let outputs = block_on(engine::evaluate(&data, &mut Cooperation::new(None))).unwrap();
    assert_eq!(outputs[0], Value::Num(1.0));
    let labels =
        Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0, f64::NAN], vec![5, 1]).unwrap());
    let data = groups(
        vec![1.0, f64::NAN, 3.0, 5.0, 100.0],
        vec![5, 1],
        Some(labels),
    );
    assert_eq!(data.values, [vec![1.0], vec![3.0, 5.0]]);
    let labels = Value::CharArray(
        CharArray::from_unicode_scalars(vec!['A', 'a', 'B', 'b', 'A', 'a'], 3, 2).unwrap(),
    );
    let data = groups(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Some(labels));
    assert_eq!(data.names, ["Aa", "Bb"]);
    assert_eq!(data.values, [vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0]]);
}
#[test]
fn public_off_form_and_unsupported_shapes_classes_or_display_reject() {
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let args = vec![
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
        Value::from("off"),
    ];
    assert!(block_on(evaluate(
        None,
        crate::builtins::common::test_support::builtin_arguments(args)
    ))
    .is_ok());
    // Only real numeric [] is the default sentinel. Two empty character
    // rows are two missing labels; an empty cell has the wrong label count.
    for group in [
        Value::CharArray(CharArray::from_code_units(vec![], 2, 0).unwrap()),
        Value::Cell(CellArray::new(vec![], 0, 0).unwrap()),
    ] {
        let args = vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            group,
            Value::from("off"),
        ];
        assert!(block_on(evaluate(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }
    for args in [
        vec![Value::Bool(true)],
        vec![Value::Num(1.0)],
        vec![Value::Num(1.0), Value::Num(1.0), Value::from("none")],
    ] {
        assert!(block_on(evaluate(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }
}
