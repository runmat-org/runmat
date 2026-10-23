use crate::indexing::object::{assign_scalar_indices, assign_with_plan};
use crate::indexing::plan::build_assignment_plan;
use crate::indexing::selectors::SliceSelector;
use runmat_value::{ObjectArray, ObjectInstance, Tensor, Value};

fn object(id: usize) -> Value {
    let mut value = ObjectInstance::new("AssignmentFixture");
    value.replace_property_unbounded("Id".into(), Value::Num(id as f64));
    Value::Object(value)
}

fn ids(value: Value, expected_shape: &[usize]) -> Vec<f64> {
    let Value::ObjectArray(array) = value else {
        panic!("expected object array")
    };
    assert_eq!(array.shape(), expected_shape);
    array
        .data()
        .iter()
        .map(|value| {
            let Value::Object(value) = value else {
                panic!("expected value object")
            };
            let Value::Num(id) = value.properties()["Id"] else {
                panic!("expected identifier")
            };
            id
        })
        .collect()
}

#[test]
fn scalar_and_planned_assignment_share_growth_and_deletion() {
    for (shape, indices) in [(vec![1, 2], vec![3]), (vec![2, 1], vec![3, 1])] {
        let len = shape.iter().product::<usize>();
        let base = Value::ObjectArray(
            ObjectArray::new(
                "AssignmentFixture",
                (1..=len).map(object).collect(),
                shape.clone(),
            )
            .unwrap(),
        );
        let selectors = indices
            .iter()
            .copied()
            .map(SliceSelector::Scalar)
            .collect::<Vec<_>>();
        let plan = build_assignment_plan(&selectors, indices.len(), &shape).unwrap();
        assert_eq!(
            assign_scalar_indices(base.clone(), &indices, object(9), false).unwrap(),
            assign_with_plan(base, &plan, object(9), false).unwrap()
        );
    }

    let base = Value::ObjectArray(
        ObjectArray::row("AssignmentFixture", (1..=3).map(object).collect()).unwrap(),
    );
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let result = assign_scalar_indices(base.clone(), &[2], empty.clone(), true).unwrap();
    assert_eq!(ids(result, &[1, 2]), [1.0, 3.0]);
    let error = assign_scalar_indices(base.clone(), &[999, 1], empty.clone(), true).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:UnsupportedDeletion"));
    let error = assign_scalar_indices(base, &[1, 1], empty, true).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:UnsupportedDeletion"));
    let error = assign_scalar_indices(object(1), &[0], object(2), false).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));

    let matrix = Value::ObjectArray(
        ObjectArray::new(
            "AssignmentFixture",
            (1..=4).map(object).collect(),
            vec![2, 2],
        )
        .unwrap(),
    );
    let error = assign_scalar_indices(matrix, &[5], object(9), false).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:ObjectArrayDefaultConstructionRequired")
    );

    // A single linear write beyond a genuine N-D array extends the final axis,
    // not a flattened row. Slot 6 remains uncovered and needs a constructor.
    let shape = [1, 2, 2];
    let base = Value::ObjectArray(
        ObjectArray::new(
            "AssignmentFixture",
            (1..=4).map(object).collect(),
            shape.to_vec(),
        )
        .unwrap(),
    );
    let plan = build_assignment_plan(&[SliceSelector::Scalar(5)], 1, &shape).unwrap();
    assert_eq!(plan.base_shape, [1, 2, 3]);
    for result in [
        assign_scalar_indices(base.clone(), &[5], object(9), false),
        assign_with_plan(base, &plan, object(9), false),
    ] {
        assert_eq!(
            result.unwrap_err().identifier(),
            Some("RunMat:ObjectArrayDefaultConstructionRequired")
        );
    }
}

#[test]
fn planned_growth_preserves_row_and_column_orientation() {
    for shape in [vec![1, 2], vec![2, 1]] {
        let array = ObjectArray::new(
            "AssignmentFixture",
            vec![object(1), object(2)],
            shape.clone(),
        )
        .unwrap();
        let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &shape).unwrap();
        let result = assign_with_plan(Value::ObjectArray(array), &plan, object(3), false).unwrap();
        let expected = if shape[0] == 1 { [1, 3] } else { [3, 1] };
        assert_eq!(ids(result, &expected), [1.0, 2.0, 3.0]);
    }
}

#[test]
fn dimensional_growth_remaps_existing_column_major_coordinates() {
    let array = ObjectArray::new(
        "AssignmentFixture",
        (1..=4).map(object).collect(),
        vec![2, 2],
    )
    .unwrap();
    let plan = build_assignment_plan(
        &[
            SliceSelector::Scalar(3),
            SliceSelector::Indices(vec![1, 2].into()),
        ],
        2,
        &[2, 2],
    )
    .unwrap();
    let rhs = ObjectArray::row("AssignmentFixture", vec![object(5), object(6)]).unwrap();
    let result = assign_with_plan(
        Value::ObjectArray(array),
        &plan,
        Value::ObjectArray(rhs),
        false,
    )
    .unwrap();
    assert_eq!(ids(result, &[3, 2]), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

#[test]
fn linear_growth_preserves_element_order_when_plan_collapses_rank() {
    let column = ObjectArray::new(
        "AssignmentFixture",
        (1..=2).map(object).collect(),
        vec![2, 1, 1],
    )
    .unwrap();
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[2, 1, 1]).unwrap();
    let result = assign_with_plan(Value::ObjectArray(column), &plan, object(3), false).unwrap();
    assert_eq!(ids(result, &[3, 1]), [1.0, 2.0, 3.0]);

    // Cover the entire new final-axis slab, so growth needs no invented objects.
    let shape = [1, 2, 2];
    let array = ObjectArray::new(
        "AssignmentFixture",
        (1..=4).map(object).collect(),
        shape.to_vec(),
    )
    .unwrap();
    let plan =
        build_assignment_plan(&[SliceSelector::Indices(vec![5, 6].into())], 1, &shape).unwrap();
    assert_eq!(plan.base_shape, [1, 2, 3]);
    let rhs = ObjectArray::row("AssignmentFixture", vec![object(5), object(6)]).unwrap();
    let result = assign_with_plan(
        Value::ObjectArray(array),
        &plan,
        Value::ObjectArray(rhs),
        false,
    )
    .unwrap();
    assert_eq!(ids(result, &[1, 2, 3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dimensional_growth_uses_collapsed_trailing_dimensions() {
    let array = ObjectArray::new(
        "AssignmentFixture",
        (1..=8).map(object).collect(),
        vec![2, 2, 2],
    )
    .unwrap();
    let plan = build_assignment_plan(
        &[SliceSelector::Scalar(3), SliceSelector::Colon],
        2,
        &[2, 2, 2],
    )
    .unwrap();
    let rhs = ObjectArray::row("AssignmentFixture", (9..=12).map(object).collect()).unwrap();
    let result = assign_with_plan(
        Value::ObjectArray(array.clone()),
        &plan,
        Value::ObjectArray(rhs),
        false,
    )
    .unwrap();
    assert_eq!(
        ids(result, &[3, 4]),
        [1.0, 2.0, 9.0, 3.0, 4.0, 10.0, 5.0, 6.0, 11.0, 7.0, 8.0, 12.0]
    );

    let partial = build_assignment_plan(
        &[SliceSelector::Scalar(3), SliceSelector::Scalar(1)],
        2,
        &[2, 2, 2],
    )
    .unwrap();
    let error =
        assign_with_plan(Value::ObjectArray(array), &partial, object(9), false).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:ObjectArrayDefaultConstructionRequired")
    );
}

#[test]
fn duplicate_targets_use_last_replacement_without_requiring_gap_construction() {
    let plan =
        build_assignment_plan(&[SliceSelector::Indices(vec![2, 2].into())], 1, &[1, 1]).unwrap();
    let rhs = ObjectArray::row("AssignmentFixture", vec![object(2), object(3)]).unwrap();
    let result = assign_with_plan(object(1), &plan, Value::ObjectArray(rhs), false).unwrap();
    assert_eq!(ids(result, &[1, 2]), [1.0, 3.0]);
}

#[test]
fn uncovered_new_slots_require_construction_and_class_drift_is_rejected() {
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[1, 1]).unwrap();
    let error = assign_with_plan(object(1), &plan, object(3), false).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:ObjectArrayDefaultConstructionRequired")
    );
    let error = assign_with_plan(
        object(1),
        &plan,
        Value::Object(ObjectInstance::new("OtherFixture")),
        false,
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ObjectArrayClassMismatch"));
}
