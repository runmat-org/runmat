use super::*;
use runmat_value::IntegerStorage;

fn build_spline_pp(
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    futures::executor::block_on(super::build_spline_pp(series, name, &mut ()))
}

#[test]
fn long_spline_coefficient_work_suspends_and_retires_on_cancel() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct PendingTurn(Rc<Cell<bool>>);
    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    let source = CancellationSource::new();
    let retired = Rc::new(Cell::new(false));
    let retired_turn = retired.clone();
    let mut work = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || Box::pin(PendingTurn(retired_turn.clone())),
    ));
    let x: Vec<f64> = (0..8192).map(|index| index as f64).collect();
    let y: Vec<f64> = x.iter().map(|value| value * value).collect();
    let mut future = Box::pin(super::coefficients::spline_coefs_for_series(
        &x, &y, None, &mut work,
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(
        future.as_mut().poll(&mut context).is_pending(),
        "coefficient work must yield before completion"
    );
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
        panic!("cancelled pending coefficient work must fail")
    };
    assert_eq!(error.identifier(), Some("RunMat:parallel:Cancelled"));
    assert!(error.source.as_deref().is_some_and(|source| {
        matches!(
            source.downcast_ref::<ExecutionServiceError>(),
            Some(ExecutionServiceError::Cancelled)
        )
    }));
    drop(future);
    assert!(retired.get(), "pending service turn must be retired");
}

fn scalar_series(y: &[f64]) -> NumericSeries {
    NumericSeries {
        x: (1..=y.len()).map(|v| v as f64).collect(),
        y: y.to_vec(),
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::Last,
    }
}

#[test]
fn makima_modified_weights_have_independent_local_oracle() {
    let series = NumericSeries {
        x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        y: vec![0.0, 0.0, 0.0, 1.0, 0.0],
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::First,
    };
    let pp = build_makima_pp(&series, "interp1").expect("makima coefficients");
    // On [2,3], the modified weights give endpoint slopes 0 and 5/11.
    // Hermite evaluation at the midpoint is 1/2 - (5/11)/8 = 39/88.
    assert!((eval_pp_scalar(&pp, 0, 2.5, &Extrapolation::Extrapolate) - 39.0 / 88.0).abs() < 1e-12);
    assert_eq!(
        eval_pp_scalar(&pp, 0, 0.5, &Extrapolation::Extrapolate),
        0.0
    );
    assert_eq!(
        eval_pp_scalar(&pp, 0, 1.5, &Extrapolation::Extrapolate),
        0.0
    );
    for (index, expected) in series.y.iter().enumerate() {
        assert!(
            (eval_pp_scalar(&pp, 0, series.x[index], &Extrapolation::Extrapolate) - expected).abs()
                < 1e-12
        );
    }
    let two = NumericSeries {
        x: vec![2.0, 4.0],
        y: vec![3.0, 7.0],
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::First,
    };
    let two_pp = build_makima_pp(&two, "interp1").expect("two-point linear makima");
    assert_eq!(
        eval_pp_scalar(&two_pp, 0, 3.0, &Extrapolation::Extrapolate),
        5.0
    );
    assert_eq!(
        eval_pp_scalar(&two_pp, 0, 5.0, &Extrapolation::Extrapolate),
        9.0
    );
}

#[test]
fn makima_large_finite_linear_samples_keep_finite_slopes() {
    let scale = 1.0e200;
    let series = scalar_series(&[0.0, scale, 2.0 * scale, 3.0 * scale, 4.0 * scale]);
    let pp = build_makima_pp(&series, "interp1").expect("finite large-amplitude coefficients");
    let value = eval_pp_scalar(&pp, 0, 2.5, &Extrapolation::Extrapolate);
    assert!(value.is_finite());
    assert!(((value / scale) - 1.5).abs() < 1.0e-12);

    let steep = NumericSeries {
        x: vec![0.0, 0.1, 0.2, 0.3, 0.4],
        y: vec![0.0, 1.0e307, 2.0e307, 3.0e307, 4.0e307],
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::First,
    };
    let steep_pp = build_makima_pp(&steep, "interp1").expect("near-limit finite coefficients");
    let steep_value = eval_pp_scalar(&steep_pp, 0, 0.15, &Extrapolation::Extrapolate);
    assert!(steep_value.is_finite());
    assert!(((steep_value / 1.0e307) - 1.5).abs() < 1.0e-12);
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn query_points_preserve_rank_zero_and_rank_one_canonical_shapes() {
    let scalar = futures::executor::block_on(query_points(
        Value::Tensor(Tensor::new(vec![1.5], Vec::new()).unwrap()),
        "interp1",
    ))
    .expect("rank-zero query");
    assert_eq!(scalar.shape, vec![1, 1]);
    assert_eq!(scalar.values, vec![1.5]);

    let vector = futures::executor::block_on(query_points(
        Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2]).unwrap()),
        "interp1",
    ))
    .expect("rank-one query");
    assert_eq!(vector.shape, vec![2]);
    assert_eq!(vector.values, vec![1.5, 2.5]);

    let pp = build_spline_pp(&scalar_series(&[1.0, 4.0, 9.0]), "spline").unwrap();
    let scalar_value = evaluate_pp(
        &pp,
        &scalar,
        &Extrapolation::Extrapolate,
        PpOutputLayout::QueryLeading,
        "ppval",
    )
    .expect("rank-zero pp query");
    assert_eq!(scalar_value, Value::Num(2.25));

    let Value::Tensor(vector_value) = evaluate_pp(
        &pp,
        &vector,
        &Extrapolation::Extrapolate,
        PpOutputLayout::QueryLeading,
        "ppval",
    )
    .expect("rank-one pp query") else {
        panic!("expected rank-one pp output tensor");
    };
    assert_eq!(vector_value.shape(), &[2]);
    assert_eq!(vector_value.materialize_f64(), vec![2.25, 6.25]);
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

    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert("form", Value::String("pp".to_string().into()));
    st.insert("breaks", Value::Tensor(breaks));
    st.insert("coefs", Value::Tensor(coefs));
    st.insert("pieces", Value::Tensor(pieces));
    st.insert("order", Value::Tensor(order));
    st.insert("dim", Value::Tensor(dim));

    let pp = futures::executor::block_on(pp_from_value(Value::Struct(st.finish()), "ppval"))
        .expect("pp struct");

    assert_eq!(pp.breaks, vec![1.0, 2.0]);
    assert_eq!(pp.coefs, vec![0.0, 0.0, 2.0, 1.0]);
    assert_eq!(pp.pieces, 1);
    assert_eq!(pp.order, 4);
    assert_eq!(pp.dim, 1);
    assert!(pp.component_shape.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pp_from_value_admits_rank_zero_and_rank_one_dimension_metadata_only() {
    let public = pp_to_value(
        build_spline_pp(&scalar_series(&[1.0, 4.0, 9.0]), "spline").unwrap(),
        "spline",
    )
    .unwrap();
    for (values, shape, expected_components) in [
        (vec![1.0], vec![], vec![]),
        (vec![1.0, 1.0], vec![2], vec![1, 1]),
    ] {
        let Value::Struct(mut fields) = public.clone() else {
            panic!("spline creates a pp struct");
        };
        crate::builtins::common::test_support::replace_struct_field_unbounded(
            &mut fields,
            "dim",
            Value::Tensor(Tensor::new(values, shape).unwrap()),
        );
        let parsed = futures::executor::block_on(pp_from_value(Value::Struct(fields), "ppval"))
            .expect("vector dimension metadata");
        assert_eq!(parsed.component_shape, expected_components);
        assert_eq!(parsed.dim, 1);
    }

    let Value::Struct(mut fields) = public else {
        panic!("spline creates a pp struct");
    };
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut fields,
        "dim",
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap()),
    );
    let error = futures::executor::block_on(pp_from_value(Value::Struct(fields), "ppval"))
        .expect_err("matrix dimension metadata must reject");
    assert!(error.message().contains("malformed pp dimensions"));
}

#[test]
fn pp_metadata_rejects_lossy_integer_dimensions_before_evaluation() {
    let series = scalar_series(&[1.0, 4.0, 9.0]);
    let Value::Struct(mut pp) =
        pp_to_value(build_spline_pp(&series, "spline").unwrap(), "spline").unwrap()
    else {
        panic!("spline creates a pp struct");
    };
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut pp,
        "dim",
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
    );
    let error = futures::executor::block_on(pp_from_value(Value::Struct(pp), "ppval"))
        .expect_err("lossy metadata must reject");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn external_pp_requires_piece_major_coefficient_matrix_geometry() {
    let series = NumericSeries {
        x: vec![0.0, 1.0],
        y: vec![10.0, 11.0, 20.0, 22.0],
        series: 2,
        component_shape: vec![2],
        sample_axis: SampleAxis::Last,
    };
    let public = pp_to_value(build_pchip_pp(&series, "pchip").unwrap(), "pchip").unwrap();
    let Value::Struct(ref fields) = public else {
        panic!("generated pp must be a struct");
    };
    let Some(Value::Tensor(coefs)) = fields.fields.get("coefs") else {
        panic!("generated coefficients must be a tensor");
    };
    assert_eq!(coefs.shape(), &[2, 4]);

    let parsed = futures::executor::block_on(pp_from_value(public.clone(), "ppval"))
        .expect("piece-major vector-valued pp");
    let query = QueryPoints {
        values: vec![0.0, 1.0],
        shape: vec![1, 2],
    };
    let Value::Tensor(result) = evaluate_pp(
        &parsed,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        "ppval",
    )
    .unwrap() else {
        panic!("two components and two queries produce a tensor");
    };
    assert_eq!(result.materialize_f64(), vec![10.0, 20.0, 11.0, 22.0]);

    for shape in [vec![4, 2], vec![1, 8]] {
        let Value::Struct(mut malformed) = public.clone() else {
            unreachable!();
        };
        crate::builtins::common::test_support::replace_struct_field_unbounded(
            &mut malformed,
            "coefs",
            Value::Tensor(Tensor::new(coefs.materialize_f64(), shape).unwrap()),
        );
        let error = futures::executor::block_on(pp_from_value(Value::Struct(malformed), "ppval"))
            .expect_err("same-count noncanonical coefficient geometry must reject");
        assert!(error.message().contains("2-by-4 matrix"));
    }
}

#[test]
fn external_pp_rejects_inexact_integer_breaks_and_coefficients() {
    let public = pp_to_value(
        build_spline_pp(&scalar_series(&[1.0, 4.0, 9.0]), "spline").unwrap(),
        "spline",
    )
    .unwrap();
    let inexact = (1_u64 << 53) + 1;
    for field in ["breaks", "coefs"] {
        let Value::Struct(mut malformed) = public.clone() else {
            unreachable!();
        };
        let (storage, shape) = if field == "breaks" {
            (IntegerStorage::U64(vec![1, 2, inexact]), vec![1, 3])
        } else {
            (
                IntegerStorage::U64(vec![0, 0, 0, 0, 0, 0, 0, inexact]),
                vec![2, 4],
            )
        };
        crate::builtins::common::test_support::replace_struct_field_unbounded(
            &mut malformed,
            field,
            Value::Tensor(Tensor::new_integer(storage, shape).unwrap()),
        );
        let error = futures::executor::block_on(pp_from_value(Value::Struct(malformed), "ppval"))
            .expect_err("integer payload must not round during external PP admission");
        assert!(error.message().contains("exactly representable"));
    }

    let Value::Struct(mut exact) = public else {
        unreachable!();
    };
    let exact_wide = 1_u64 << 54;
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut exact,
        "coefs",
        Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![0, 0, 0, 0, 0, 0, 0, exact_wide]),
                vec![2, 4],
            )
            .unwrap(),
        ),
    );
    let parsed = futures::executor::block_on(pp_from_value(Value::Struct(exact), "ppval"))
        .expect("representable integers above flintmax stay admitted");
    assert_eq!(parsed.coefs[7], exact_wide as f64);
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
    assert_eq!(tensor.shape(), &[2, 2]);
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
    assert_eq!(result.shape(), &[2, 3, 2, 2]);
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
    assert_eq!(result.shape(), &[1, 2, 2]);
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
