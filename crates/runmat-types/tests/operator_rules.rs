use runmat_types::{
    broadcast_shape, infer_binary, infer_range, infer_reshape, infer_unary, permute_shape,
    range_shape, reduction_shape, repmat_shape, DimensionFact, NumericClass, NumericDomain,
    NumericFact, OperatorKind, RangeStepFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn shaped(dims: &[usize]) -> ShapeFact {
    ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>())
}

#[test]
fn broadcast_computes_output_instead_of_borrowing_an_operand() {
    let left = numeric(shaped(&[1, 3]));
    let right = numeric(shaped(&[2, 1]));
    let inferred = infer_binary(OperatorKind::ElementwiseMultiply, &left, &right);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.fact.shape, shaped(&[2, 3]));

    let scalar = numeric(ShapeFact::Scalar);
    assert_eq!(
        infer_binary(OperatorKind::Add, &scalar, &right).fact.shape,
        right.shape
    );
}

#[test]
fn broadcast_reports_each_concrete_incompatibility() {
    for left in 0..=4 {
        for right in 0..=4 {
            let result = broadcast_shape(&shaped(&[left, 3]), &shaped(&[right, 3]));
            let compatible = left == right || left == 1 || right == 1;
            assert_eq!(result.is_ok(), compatible, "left={left}, right={right}");
        }
    }
}

#[test]
fn matrix_rules_handle_scalar_scaling_and_dimension_errors() {
    let left = numeric(shaped(&[2, 3]));
    let right = numeric(shaped(&[3, 4]));
    assert_eq!(
        infer_binary(OperatorKind::MatrixMultiply, &left, &right)
            .fact
            .shape,
        shaped(&[2, 4])
    );
    let scalar = numeric(ShapeFact::Scalar);
    assert_eq!(
        infer_binary(OperatorKind::MatrixMultiply, &scalar, &right)
            .fact
            .shape,
        right.shape
    );
    let invalid = infer_binary(
        OperatorKind::MatrixMultiply,
        &left,
        &numeric(shaped(&[5, 4])),
    );
    assert_eq!(invalid.diagnostics[0].code, "RM-TYPE-MATMUL");
    assert_eq!(invalid.fact.shape, ShapeFact::Unknown);

    let pagewise = infer_binary(
        OperatorKind::MatrixMultiply,
        &numeric(shaped(&[2, 3, 4])),
        &numeric(shaped(&[3, 2, 4])),
    );
    assert_eq!(pagewise.diagnostics[0].code, "RM-TYPE-MATRIX-RANK");
}

#[test]
fn comparisons_and_transpose_preserve_shape_but_change_kind() {
    let matrix = numeric(shaped(&[2, 5]));
    let compared = infer_binary(OperatorKind::Greater, &matrix, &numeric(ShapeFact::Scalar));
    assert_eq!(compared.fact.kind, ValueKindFact::Logical);
    assert_eq!(compared.fact.shape, matrix.shape);
    let transposed = infer_unary(OperatorKind::Transpose, &matrix);
    assert_eq!(transposed.fact.shape, shaped(&[5, 2]));
}

#[test]
fn range_and_shape_transforms_are_dimension_aware() {
    assert_eq!(range_shape(Some(1.0), None, Some(5.0)), shaped(&[1, 5]));
    assert_eq!(range_shape(Some(5.0), None, Some(1.0)), shaped(&[1, 0]));
    assert_eq!(
        reduction_shape(&shaped(&[2, 3, 4]), Some(2)),
        shaped(&[2, 1, 4])
    );
    assert_eq!(
        repmat_shape(
            &shaped(&[2, 3]),
            &[DimensionFact::Known(2), DimensionFact::Known(4)]
        ),
        shaped(&[4, 12])
    );
    assert_eq!(
        permute_shape(&shaped(&[2, 3, 4]), &[3, 1, 2]).unwrap(),
        shaped(&[4, 2, 3])
    );
    assert_eq!(
        permute_shape(&shaped(&[2, 3]), &[2, 1, 3]).unwrap(),
        shaped(&[3, 2, 1])
    );
    assert!(permute_shape(&shaped(&[2, 3]), &[1, 1]).is_err());

    assert_eq!(
        infer_reshape(
            &numeric(shaped(&[2, 6])),
            vec![DimensionFact::Unknown, DimensionFact::Known(3)],
        )
        .fact
        .shape,
        shaped(&[4, 3])
    );
}

#[test]
fn explicit_unknown_and_zero_range_steps_are_not_treated_as_implicit_one() {
    assert_eq!(
        infer_range(Some(1.0), RangeStepFact::Unknown, Some(5.0))
            .fact
            .shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let zero = infer_range(Some(1.0), RangeStepFact::Known(0.0), Some(5.0));
    assert_eq!(zero.diagnostics[0].code, "RM-TYPE-RANGE-STEP");
    assert_eq!(zero.fact.shape, ShapeFact::Unknown);
}

#[test]
fn proven_invalid_operator_categories_fail_but_unknowns_remain_conservative() {
    let structure = ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact {
        fields: Default::default(),
        fields_complete: true,
    }));
    assert_eq!(
        infer_binary(OperatorKind::Add, &structure, &numeric(ShapeFact::Scalar)).diagnostics[0]
            .code,
        "RM-TYPE-BINARY-OPERAND"
    );
    let dynamic = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    assert!(
        infer_binary(OperatorKind::Add, &dynamic, &numeric(ShapeFact::Scalar))
            .diagnostics
            .is_empty()
    );
    let nonscalar = numeric(shaped(&[1, 2]));
    assert_eq!(
        infer_binary(
            OperatorKind::ShortCircuitAnd,
            &nonscalar,
            &numeric(ShapeFact::Scalar)
        )
        .diagnostics[0]
            .code,
        "RM-TYPE-SHORT-CIRCUIT-SCALAR"
    );
}

#[test]
fn storage_inference_is_symmetric_and_widens_mixed_sparse_operations() {
    let dense = numeric(shaped(&[2, 2]));
    let mut sparse = dense.clone();
    sparse.storage = StorageFact::Sparse;
    assert_eq!(
        infer_binary(OperatorKind::Add, &sparse, &dense)
            .fact
            .storage,
        StorageFact::Unknown
    );
    assert_eq!(
        infer_binary(OperatorKind::Add, &dense, &sparse)
            .fact
            .storage,
        StorageFact::Unknown
    );
    assert_eq!(
        infer_binary(OperatorKind::Add, &sparse, &sparse)
            .fact
            .storage,
        StorageFact::Sparse
    );
}
