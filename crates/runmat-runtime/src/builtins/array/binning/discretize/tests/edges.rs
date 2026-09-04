use super::*;
use runmat_value::{IntValue, IntegerStorage, Tensor};

#[test]
fn explicit_edges_compare_wide_integers_exactly() {
    let base = 9_007_199_254_740_992_u64;
    let output = call(
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![base + 1]), vec![1, 1]).unwrap(),
        ),
        Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1, base + 2]),
                vec![1, 3],
            )
            .unwrap(),
        ),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(tensor(output).materialize_f64(), vec![2.0]);
}

#[test]
fn computed_edges_honor_requested_output_count() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let output = call(
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Value::Int(IntValue::U8(2)),
        Vec::new(),
    )
    .unwrap();
    let Value::OutputList(outputs) = output else {
        panic!("expected two outputs");
    };
    assert_eq!(tensor(outputs[0].clone()).materialize_f64(), vec![1.0, 2.0]);
    assert_eq!(
        tensor(outputs[1].clone()).materialize_f64(),
        vec![0.0, 0.5, 1.0]
    );
}

#[test]
fn explicit_edges_reject_a_second_output() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(
        Value::Num(0.5),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:discretize:InvalidInput"));
}

#[test]
fn outer_infinities_are_binned_for_both_edge_conventions() {
    let input = Value::Tensor(
        Tensor::new(
            vec![f64::NEG_INFINITY, -1.0, 1.0, f64::INFINITY, f64::NAN],
            vec![1, 5],
        )
        .unwrap(),
    );
    let edges = Value::Tensor(
        Tensor::new(vec![f64::NEG_INFINITY, 0.0, f64::INFINITY], vec![1, 3]).unwrap(),
    );
    for rest in [
        Vec::new(),
        vec![Value::from("IncludedEdge"), Value::from("right")],
    ] {
        let values = tensor(call(input.clone(), edges.clone(), rest).unwrap()).materialize_f64();
        assert_eq!(values[..4], [1.0, 1.0, 2.0, 2.0]);
        assert!(values[4].is_nan());
    }
}

#[test]
fn repeated_edges_leave_the_empty_interval_unassigned() {
    let input = Value::Tensor(Tensor::new(vec![0.5, 1.0, 1.5], vec![1, 3]).unwrap());
    let edges = Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 2.0], vec![1, 4]).unwrap());
    let left = tensor(call(input.clone(), edges.clone(), Vec::new()).unwrap()).materialize_f64();
    let right = tensor(
        call(
            input,
            edges,
            vec![Value::from("IncludedEdge"), Value::from("right")],
        )
        .unwrap(),
    )
    .materialize_f64();
    assert_eq!(left, vec![1.0, 3.0, 3.0]);
    assert_eq!(right, vec![1.0, 1.0, 3.0]);
}

#[test]
fn invalid_edges_and_options_have_identity_specific_errors() {
    let decreasing = call(
        Value::Num(0.5),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        decreasing.identifier(),
        Some("RunMat:discretize:InvalidInput")
    );

    let option = call(
        Value::Num(0.5),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        vec![Value::from("UnknownOption"), Value::from("left")],
    )
    .unwrap_err();
    assert_eq!(option.identifier(), Some("RunMat:discretize:InvalidInput"));
}
