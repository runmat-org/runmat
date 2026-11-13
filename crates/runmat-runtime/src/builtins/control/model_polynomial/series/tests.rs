use super::*;
use runmat_value::StringArray;

#[test]
fn wide_integer_scalar_gain_uses_model_arithmetic_rounding() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let gain = futures::executor::block_on(SeriesModel::from_value(Value::Int(
        runmat_value::IntValue::U64(9_007_199_254_740_993),
    )))
    .expect("wide integer gain follows tf arithmetic conversion");
    let SeriesModel::Gain(gain) = gain else {
        panic!("scalar integer should remain a gain");
    };
    assert_eq!(gain.re, 9_007_199_254_740_992.0);
    assert_eq!(gain.im, 0.0);
}

#[test]
fn named_channels_accept_string_and_cell_property_vectors() {
    let mut object = runmat_value::UnboundedObjectBuilder::new("ss");
    object.insert(
        "InputName".into(),
        Value::StringArray(StringArray::new_2d(vec!["alpha".into(), "beta".into()], 1, 2).unwrap()),
    );
    object.insert(
        "OutputName".into(),
        Value::Cell(
            CellArray::new(vec![Value::CharArray(CharArray::new_row("gamma"))], 1, 1).unwrap(),
        ),
    );
    let object = object.finish();
    assert_eq!(
        names(&object, "InputName", 2).unwrap(),
        vec!["alpha".to_owned(), "beta".to_owned()]
    );
    assert_eq!(
        names(&object, "OutputName", 1).unwrap(),
        vec!["gamma".to_owned()]
    );
}

#[test]
fn scalar_gain_inherits_discrete_model_sample_time() {
    let system = SeriesModel::Ss(RealStateSpace {
        a: DMatrix::zeros(0, 0),
        b: DMatrix::zeros(0, 1),
        c: DMatrix::zeros(1, 0),
        d: DMatrix::from_element(1, 1, 1.0),
        sample_time: 0.25,
        input_names: vec![String::new()],
        output_names: vec![String::new()],
    });
    let (first, _) = SeriesModel::materialize_gains(
        SeriesModel::Gain(num_complex::Complex64::new(3.0, 0.0)),
        system,
    )
    .unwrap();
    let SeriesModel::Tf(gain) = first else {
        panic!("scalar gain should become tf")
    };
    assert_eq!(gain.sample_time(), 0.25);
    assert_eq!(gain.variable.as_str(), "z");
}

#[test]
fn scalar_gain_scales_every_mimo_state_space_channel() {
    let system = RealStateSpace {
        a: DMatrix::from_element(1, 1, -1.0),
        b: DMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
        c: DMatrix::from_column_slice(2, 1, &[3.0, 4.0]),
        d: DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        sample_time: 0.0,
        input_names: vec!["u1".into(), "u2".into()],
        output_names: vec!["y1".into(), "y2".into()],
    };
    let upstream = system
        .scale_gain(num_complex::Complex64::new(2.0, 0.0), true)
        .unwrap();
    assert_eq!(upstream.b[(0, 0)], 2.0);
    assert_eq!(upstream.b[(0, 1)], 4.0);
    assert_eq!(upstream.c[(1, 0)], 4.0);
    assert_eq!(upstream.d[(1, 1)], 16.0);
    let downstream = upstream
        .scale_gain(num_complex::Complex64::new(0.5, 0.0), false)
        .unwrap();
    assert_eq!(downstream.c[(1, 0)], 2.0);
    assert_eq!(downstream.d[(1, 1)], 8.0);
}

#[test]
fn selected_mimo_channels_preserve_cascade_order_and_dimensions() {
    let upstream = RealStateSpace {
        a: DMatrix::from_element(1, 1, -1.0),
        b: DMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
        c: DMatrix::from_column_slice(2, 1, &[3.0, 4.0]),
        d: DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 2.0, 0.0]),
        sample_time: 0.0,
        input_names: vec!["a".into(), "b".into()],
        output_names: vec!["x".into(), "link".into()],
    };
    let downstream = RealStateSpace {
        a: DMatrix::from_element(1, 1, -2.0),
        b: DMatrix::from_row_slice(1, 2, &[5.0, 6.0]),
        c: DMatrix::from_element(1, 1, 7.0),
        d: DMatrix::from_row_slice(1, 2, &[8.0, 9.0]),
        sample_time: 0.0,
        input_names: vec!["link".into(), "other".into()],
        output_names: vec!["result".into()],
    };
    let result = upstream.compose(&downstream, Some((&[1], &[0]))).unwrap();
    assert_eq!(result.a.shape(), (2, 2));
    assert_eq!(result.b.shape(), (2, 2));
    assert_eq!(result.c.shape(), (1, 2));
    assert_eq!(result.d.shape(), (1, 2));
    assert_eq!(result.a[(1, 0)], 20.0);
    assert_eq!(result.d[(0, 0)], 16.0);
    assert_eq!(result.d[(0, 1)], 0.0);
    assert_eq!(result.input_names, vec!["a".to_owned(), "b".to_owned()]);
    assert_eq!(result.output_names, vec!["result".to_owned()]);
}
