use super::*;
use runmat_value::{CellArray, CharArray, StringArray};

#[test]
fn classify_group_geometry_accepts_row_and_column_label_vectors() {
    for (rows, cols) in [(1, 4), (4, 1)] {
        let groups = [
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![0, 0, 1, 1], vec![rows, cols]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["left".into(), "left".into(), "right".into(), "right".into()],
                    vec![rows, cols],
                )
                .unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::from("left"),
                        Value::from("left"),
                        Value::from("right"),
                        Value::from("right"),
                    ],
                    rows,
                    cols,
                )
                .unwrap(),
            ),
        ];
        for group in groups {
            assert_eq!(header::group_count(&group).unwrap(), Some(4));
        }
    }
}

#[test]
fn classify_quadratic_preserves_text_labels() {
    with_outputs(5, || {
        let groups = Value::StringArray(
            StringArray::new(
                vec!["low".into(), "low".into(), "high".into(), "high".into()],
                vec![4, 1],
            )
            .unwrap(),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            vec![Value::String("quadratic".into())],
        );
        let Value::StringArray(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.data(), vec!["low", "high"]);
        let Value::StructArray(coeff) = &values[4] else {
            panic!("coeff");
        };
        assert_eq!(coeff.shape(), &[2, 2]);
        assert!(coeff
            .get_linear(1)
            .expect("pairwise coefficient")
            .fields
            .contains_key("quadratic"));
    });
}

#[test]
fn classify_empirical_prior_and_missing_labels() {
    with_outputs(3, || {
        let values = classify(
            tensor(vec![0.1, 10.0], vec![2, 1]),
            tensor(vec![0.0, 0.2, 10.0, 10.2, 99.0], vec![5, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0, f64::NAN], vec![5, 1]),
            vec![
                Value::String("diagLinear".into()),
                Value::String("empirical".into()),
            ],
        );
        let Value::Tensor(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.as_f64_slice(), Some(&[1.0, 2.0][..]));
        let Value::Tensor(posterior) = &values[2] else {
            panic!("posterior");
        };
        assert_eq!(posterior.shape(), &[2, 2]);
    });
}

#[test]
fn classify_preserves_char_matrix_labels() {
    with_outputs(1, || {
        let groups = Value::CharArray(
            CharArray::from_unicode_scalars(vec!['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], 4, 2)
                .unwrap(),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            Vec::new(),
        );
        let Value::CharArray(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.rows, 2);
        assert_eq!(labels.cols, 2);
        assert_eq!(labels.data(), runmat_value::encode_utf16_code_units("aabb"));
    });
}

#[test]
fn classify_accepts_cellstr_labels_and_struct_priors() {
    with_outputs(2, || {
        let groups = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("left")),
                    Value::CharArray(CharArray::new_row("left")),
                    Value::CharArray(CharArray::new_row("right")),
                    Value::CharArray(CharArray::new_row("right")),
                ],
                4,
                1,
            )
            .unwrap(),
        );
        let mut prior = runmat_value::UnboundedStructValueBuilder::new();
        prior.insert(
            "group",
            Value::StringArray(
                StringArray::new(vec!["left".into(), "right".into()], vec![2, 1]).unwrap(),
            ),
        );
        prior.insert(
            "prob",
            Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap()),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            vec![
                Value::String("linear".into()),
                Value::Struct(prior.finish()),
            ],
        );
        let Value::Cell(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.rows, 2);
        assert!(matches!(values[1], Value::Num(err) if err <= EPS));
    });
}

#[test]
fn classify_rejects_bad_dimensions() {
    let err = block_on(classify_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("same number of columns"));
}
