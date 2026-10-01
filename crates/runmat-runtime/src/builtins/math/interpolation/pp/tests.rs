use super::*;
use runmat_value::IntegerStorage;

fn scalar_series(y: &[f64]) -> NumericSeries {
    NumericSeries {
        x: (1..=y.len()).map(|v| v as f64).collect(),
        y: y.to_vec(),
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::Last,
    }
}

fn mirrorless_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("typed integer tensor")
}

#[test]
fn query_points_read_typed_integer_storage_length_and_values() {
    let query_tensor = mirrorless_integer_tensor(IntegerStorage::U16(vec![1, 3]), vec![1, 2]);

    let query = futures::executor::block_on(query_points(Value::Tensor(query_tensor), "interp1"))
        .expect("query points");

    assert_eq!(query.shape, vec![1, 2]);
    assert_eq!(query.values, vec![1.0, 3.0]);
}

#[test]
fn implicit_series_reads_typed_integer_storage_length_and_values() {
    let y_tensor =
        mirrorless_integer_tensor(IntegerStorage::I16(vec![1, 4, 9, 10, 40, 90]), vec![3, 2]);

    let series = futures::executor::block_on(implicit_series_from_values(
        Value::Tensor(y_tensor),
        "interp1",
    ))
    .expect("implicit series");

    assert_eq!(series.x, vec![1.0, 2.0, 3.0]);
    assert_eq!(series.y, vec![1.0, 4.0, 9.0, 10.0, 40.0, 90.0]);
    assert_eq!(series.series, 2);
    assert_eq!(series.component_shape, vec![2]);
}

#[test]
fn pp_from_value_reads_typed_integer_coefficients_without_mirror() {
    let breaks = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("breaks");
    let coefs = mirrorless_integer_tensor(IntegerStorage::I16(vec![0, 0, 2, 1]), vec![1, 4]);
    let pieces = mirrorless_integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]);
    let order = mirrorless_integer_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]);
    let dim = mirrorless_integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]);

    let mut st = StructValue::new();
    st.insert("form", Value::String("pp".to_string()));
    st.insert("breaks", Value::Tensor(breaks));
    st.insert("coefs", Value::Tensor(coefs));
    st.insert("pieces", Value::Tensor(pieces));
    st.insert("order", Value::Tensor(order));
    st.insert("dim", Value::Tensor(dim));

    let pp =
        futures::executor::block_on(pp_from_value(Value::Struct(st), "ppval")).expect("pp struct");

    assert_eq!(pp.breaks, vec![1.0, 2.0]);
    assert_eq!(pp.coefs, vec![0.0, 0.0, 2.0, 1.0]);
    assert_eq!(pp.pieces, 1);
    assert_eq!(pp.order, 4);
    assert_eq!(pp.dim, 1);
    assert!(pp.component_shape.is_empty());
}

#[test]
fn pp_metadata_rejects_lossy_integer_dimensions_before_evaluation() {
    let series = scalar_series(&[1.0, 4.0, 9.0]);
    let Value::Struct(mut pp) =
        pp_to_value(build_spline_pp(&series, "spline").unwrap(), "spline").unwrap()
    else {
        panic!("spline creates a pp struct");
    };
    pp.insert(
        "dim",
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
    );
    let error = futures::executor::block_on(pp_from_value(Value::Struct(pp), "ppval"))
        .expect_err("lossy metadata must reject");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn pchip_preserves_monotone_samples() {
    let series = scalar_series(&[0.0, 1.0, 1.5, 1.75]);
    let pp = build_pchip_pp(&series, "pchip").expect("pchip");
    let query = QueryPoints {
        values: vec![1.25, 1.5, 2.5, 3.5],
        shape: vec![1, 4],
    };
    let value = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "pchip",
    )
    .expect("ppval");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert!(tensor
        .materialize_f64()
        .windows(2)
        .all(|pair| pair[1] >= pair[0]));
}

#[test]
fn spline_matches_quadratic_for_three_points() {
    let series = scalar_series(&[1.0, 4.0, 9.0]);
    let pp = build_spline_pp(&series, "spline").expect("spline");
    let query = QueryPoints {
        values: vec![1.5, 2.5],
        shape: vec![1, 2],
    };
    let value = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "spline",
    )
    .expect("ppval");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert!((tensor.materialize_f64()[0] - 2.25).abs() < 1e-10);
    assert!((tensor.materialize_f64()[1] - 6.25).abs() < 1e-10);
}

#[test]
fn not_a_knot_spline_recovers_cubic_on_uneven_samples() {
    let x = [0.0, 0.5, 2.0, 3.0, 5.0];
    let cubic = |point: f64| point * point * point - 2.0 * point + 1.0;
    let series = NumericSeries {
        x: x.to_vec(),
        y: x.into_iter().map(cubic).collect(),
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::Last,
    };
    let pp = build_spline_pp(&series, "spline").unwrap();
    let query = QueryPoints {
        values: vec![0.25, 1.25, 4.0],
        shape: vec![1, 3],
    };
    let Value::Tensor(output) = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "spline",
    )
    .unwrap() else {
        panic!("three queries must return a tensor");
    };
    for (&actual, &point) in output.materialize_f64().iter().zip(&query.values) {
        assert!((actual - cubic(point)).abs() < 1e-10);
    }
}

#[test]
fn not_a_knot_spline_has_no_absolute_spacing_cutoff() {
    let series = NumericSeries {
        x: vec![0.0, 1e-16, 2e-16, 3e-16],
        y: vec![0.0, 1.0, 2.0, 3.0],
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::Last,
    };
    let pp = build_spline_pp(&series, "spline").unwrap();
    let value = eval_pp_scalar(&pp, 0, 1.5e-16, &Extrapolation::Extrapolate);
    assert!((value - 1.5).abs() < 1e-8);
}

#[test]
fn ppval_multiseries_places_component_dimension_before_queries() {
    let series = NumericSeries {
        x: vec![1.0, 2.0, 3.0],
        y: vec![1.0, 4.0, 9.0, 100.0, 400.0, 900.0],
        series: 2,
        component_shape: vec![2],
        sample_axis: SampleAxis::Last,
    };
    let pp = build_spline_pp(&series, "spline").expect("spline");
    let query = QueryPoints {
        values: vec![1.5, 2.5],
        shape: vec![1, 2],
    };
    let value = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "spline",
    )
    .expect("ppval");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.materialize_f64().len(), 4);
    assert!((tensor.materialize_f64()[0] - 2.25).abs() < 1e-10);
    assert!((tensor.materialize_f64()[1] - 225.0).abs() < 1e-10);
    assert!((tensor.materialize_f64()[2] - 6.25).abs() < 1e-10);
    assert!((tensor.materialize_f64()[3] - 625.0).abs() < 1e-10);
}

#[test]
fn generated_vector_valued_pp_coefficients_are_piece_major() {
    let series = NumericSeries {
        x: vec![0.0, 1.0, 2.0],
        y: vec![10.0, 11.0, 12.0, 20.0, 22.0, 24.0],
        series: 2,
        component_shape: vec![2],
        sample_axis: SampleAxis::Last,
    };
    let pp = build_pchip_pp(&series, "pchip").unwrap();
    assert_eq!(pp.pieces, 2);
    assert_eq!(pp.dim, 2);
    let rows = pp.pieces * pp.dim;
    assert_eq!(&pp.coefs[3 * rows..4 * rows], &[10.0, 20.0, 11.0, 22.0]);
}

#[test]
fn final_sample_axis_roundtrips_nonsquare_component_and_query_dimensions() {
    let samples = (0..4)
        .flat_map(|sample| (0..6).map(move |component| (component * 100 + sample) as f64))
        .collect();
    let series = futures::executor::block_on(series_from_values_on_axis(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap()),
        Value::Tensor(Tensor::new(samples, vec![2, 3, 4]).unwrap()),
        SampleAxis::Last,
        "pchip",
    ))
    .unwrap();
    assert_eq!(series.series, 6);
    assert_eq!(series.component_shape, vec![2, 3]);
    assert_eq!(&series.y[..4], &[0.0, 1.0, 2.0, 3.0]);

    let pp = build_pchip_pp(&series, "pchip").unwrap();
    let public = pp_to_value(pp, "pchip").unwrap();
    let parsed = futures::executor::block_on(pp_from_value(public, "ppval")).unwrap();
    assert_eq!(parsed.component_shape, vec![2, 3]);
    let query = QueryPoints {
        values: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2],
    };
    let Value::Tensor(result) = evaluate_pp(
        &parsed,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "ppval",
    )
    .unwrap() else {
        panic!("array-valued pp evaluation must be a tensor");
    };
    assert_eq!(result.shape, vec![2, 3, 2, 2]);
    let values = result.materialize_f64();
    for sample in 0..4 {
        for component in 0..6 {
            assert_eq!(
                values[component + sample * 6],
                (component * 100 + sample) as f64
            );
        }
    }
}

#[test]
fn legacy_first_axis_samples_remain_admitted_when_last_axis_does_not_match() {
    let series = futures::executor::block_on(series_from_values_on_axis(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 4.0, 9.0, 10.0, 40.0, 90.0], vec![3, 2]).unwrap()),
        SampleAxis::Last,
        "pchip",
    ))
    .unwrap();
    assert_eq!(series.sample_axis, SampleAxis::First);
    assert_eq!(&series.y[..3], &[1.0, 4.0, 9.0]);
    let pp = build_pchip_pp(&series, "pchip").unwrap();
    let query = QueryPoints {
        values: vec![1.0, 3.0],
        shape: vec![1, 2],
    };
    let Value::Tensor(result) = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::QueryLeading,
        "pchip",
    )
    .unwrap() else {
        panic!("two-series evaluation must be a tensor");
    };
    assert_eq!(result.shape, vec![1, 2, 2]);
    assert_eq!(result.materialize_f64(), vec![1.0, 9.0, 10.0, 90.0]);
}

#[test]
fn square_sample_matrix_uses_documented_last_axis_for_pchip_and_spline() {
    let series = futures::executor::block_on(series_from_values_on_axis(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        Value::Tensor(
            Tensor::new(
                vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0],
                vec![3, 3],
            )
            .unwrap(),
        ),
        SampleAxis::Last,
        "pchip",
    ))
    .unwrap();
    assert_eq!(series.sample_axis, SampleAxis::Last);
    assert_eq!(series.component_shape, vec![3]);
    assert_eq!(&series.y[..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&series.y[3..6], &[10.0, 20.0, 30.0]);
}
