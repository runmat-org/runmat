use super::*;
use runmat_value::{LogicalArray, Tensor};

#[test]
fn like_is_mode_gated_and_uses_prototype_shape_or_representation() {
    {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(vec![Value::from("like"), Value::Num(1.0)]).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CellLikeExtension")
        );
    }
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
    let cells = output(
        vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::LogicalArray(logical),
        ],
        &[2, 2],
    );
    assert!(cells
        .data
        .iter()
        .all(|value| matches!(value, Value::LogicalArray(array) if array.shape == [0, 0])));

    let prototype = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    output(vec![Value::from("like"), Value::Tensor(prototype)], &[2, 1]);
}

#[test]
fn like_is_case_insensitive_and_cell_prototypes_create_empty_cells() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    let prototype = crate::make_cell_with_shape(Vec::new(), vec![0, 0]).unwrap();
    let cells = output(
        vec![Value::Num(2.0), Value::from("LIKE"), prototype],
        &[2, 2],
    );
    assert!(cells.data.iter().all(
        |value| matches!(value, Value::Cell(inner) if inner.shape == [0, 0] && inner.data.is_empty())
    ));
}

#[test]
fn malformed_like_forms_reject() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_eq!(
        run(vec![Value::from("like")]).unwrap_err().identifier(),
        Some("RunMat:cell:InvalidInput")
    );
    assert_eq!(
        run(vec![
            Value::from("like"),
            Value::Num(1.0),
            Value::from("like"),
            Value::Num(2.0)
        ])
        .unwrap_err()
        .identifier(),
        Some("RunMat:cell:InvalidInput")
    );
}
