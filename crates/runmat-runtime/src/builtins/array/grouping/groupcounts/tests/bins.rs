use super::*;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn numeric_edges_include_empty_bins() {
    let input =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![0, 2]), vec![2, 1]).unwrap());
    let edges = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I64(vec![0, 1, 2, 3]), vec![4, 1]).unwrap(),
    );
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(
        call(
            input,
            vec![edges, Value::from("IncludeEmptyGroups"), Value::Bool(true)],
        )
        .unwrap(),
    );
    assert!(
        matches!(&outputs[0], Value::Tensor(value) if value.materialize_f64() == [1.0, 0.0, 1.0])
    );
    assert!(matches!(&outputs[1], Value::StringArray(value) if value.data.len() == 3));
}

#[test]
fn included_edge_uses_exact_wide_integer_boundaries() {
    let base = 9_007_199_254_740_992_u64;
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![base + 1]), vec![1, 1]).unwrap(),
    );
    let edges = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            vec![3, 1],
        )
        .unwrap(),
    );
    for (edge, expected_label) in [("left", 1_usize), ("right", 0_usize)] {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(
            call(
                input.clone(),
                vec![
                    edges.clone(),
                    Value::from("IncludedEdge"),
                    Value::from(edge),
                ],
            )
            .unwrap(),
        );
        let Value::StringArray(labels) = &outputs[1] else {
            panic!("expected interval labels");
        };
        assert!(labels.data[0].contains(&(base + expected_label as u64).to_string()));
    }
}
