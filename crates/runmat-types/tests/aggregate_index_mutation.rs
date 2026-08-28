use runmat_types::{
    infer_cell_aggregate, infer_concatenate, infer_index, infer_index_mutation, infer_member_read,
    infer_member_write, infer_mutation, infer_struct, infer_tensor_aggregate,
    AssignmentCreationPolicy, AssignmentShapePolicy, IndexKind, IndexResultContext,
    IndexSelectorFact, MemberName, MutationContract, NumericClass, NumericDomain, NumericFact,
    PlaceMutationKind, ShapeFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }));
    fact.shape = shape;
    fact
}

#[test]
fn cells_preserve_positions_summary_and_brace_semantics() {
    let first = numeric(NumericClass::Double, ShapeFact::Scalar);
    let second = numeric(NumericClass::UInt64, ShapeFact::Scalar);
    let cell = infer_cell_aggregate(&[vec![first.clone(), second.clone()]]).fact;
    let indexed = infer_index(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::KnownOneBasedIndex(2)],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(indexed.fact, second);
    let expanded = infer_index(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::Colon],
        IndexResultContext::FunctionArgumentExpansion,
    );
    let ValueKindFact::OutputList(outputs) = expanded.fact.kind else {
        panic!("expected output list");
    };
    assert_eq!(outputs.outputs, vec![first, second]);
}

#[test]
fn empty_tensor_and_cell_literals_keep_distinct_container_kinds() {
    let tensor = infer_tensor_aggregate(&[]).fact;
    assert_eq!(
        tensor.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(tensor.shape, ShapeFact::from(vec![Some(0), Some(0)]));
    let cell = infer_cell_aggregate(&[]).fact;
    assert!(matches!(cell.kind, ValueKindFact::Cell(_)));
    assert_eq!(cell.shape, ShapeFact::from(vec![Some(0), Some(0)]));
}

#[test]
fn cell_positions_and_expansions_follow_matlab_column_major_order() {
    let one = numeric(NumericClass::Double, ShapeFact::Scalar);
    let two = numeric(NumericClass::UInt8, ShapeFact::Scalar);
    let three = numeric(NumericClass::UInt16, ShapeFact::Scalar);
    let four = numeric(NumericClass::UInt32, ShapeFact::Scalar);
    let cell = infer_cell_aggregate(&[
        vec![one.clone(), three.clone()],
        vec![two.clone(), four.clone()],
    ])
    .fact;
    assert_eq!(
        infer_index(
            &cell,
            IndexKind::Brace,
            &[IndexSelectorFact::KnownOneBasedIndex(2)],
            IndexResultContext::ReadSingle,
        )
        .fact,
        two
    );
    assert_eq!(
        infer_index(
            &cell,
            IndexKind::Brace,
            &[
                IndexSelectorFact::KnownOneBasedIndex(1),
                IndexSelectorFact::KnownOneBasedIndex(2),
            ],
            IndexResultContext::ReadSingle,
        )
        .fact,
        three
    );
    let expanded = infer_index(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::Colon],
        IndexResultContext::ReadCommaList,
    );
    let ValueKindFact::OutputList(expanded) = expanded.fact.kind else {
        panic!("expected output list");
    };
    assert_eq!(expanded.outputs, vec![one, two, three, four]);

    let selector = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let subset = infer_index(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::Numeric(selector.clone())],
        IndexResultContext::ReadCommaList,
    );
    let ValueKindFact::OutputList(subset) = subset.fact.kind else {
        panic!("expected output list");
    };
    assert_eq!(subset.outputs.len(), 2);
    assert!(!subset.variadic);

    let parenthesized = infer_index(
        &cell,
        IndexKind::Paren,
        &[IndexSelectorFact::Numeric(selector)],
        IndexResultContext::ReadSingle,
    );
    let ValueKindFact::Cell(parenthesized) = parenthesized.fact.kind else {
        panic!("expected cell container");
    };
    assert!(!parenthesized.elements_complete);
    assert!(parenthesized.elements.is_empty());
}

#[test]
fn indexing_preserves_subscript_dimensions_and_reports_proven_bounds() {
    let matrix = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let columns = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let slice = infer_index(
        &matrix,
        IndexKind::Paren,
        &[
            IndexSelectorFact::Colon,
            IndexSelectorFact::Numeric(columns),
        ],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(slice.fact.shape, ShapeFact::from(vec![Some(2), Some(2)]));
    assert_eq!(
        infer_index(
            &matrix,
            IndexKind::Paren,
            &[IndexSelectorFact::Colon],
            IndexResultContext::ReadSingle,
        )
        .fact
        .shape,
        ShapeFact::from(vec![Some(6), Some(1)])
    );
    let out_of_bounds = infer_index(
        &matrix,
        IndexKind::Paren,
        &[IndexSelectorFact::KnownOneBasedIndex(7)],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(out_of_bounds.diagnostics[0].code, "RM-TYPE-INDEX-BOUNDS");
}

#[test]
fn structs_and_concatenation_preserve_payload_and_shape() {
    let structure_fact = infer_struct(BTreeMap::from([(
        "count".into(),
        numeric(NumericClass::UInt64, ShapeFact::Scalar),
    )]))
    .fact;
    let ValueKindFact::Struct(structure) = &structure_fact.kind else {
        panic!("expected struct");
    };
    assert!(structure.fields_complete);
    assert!(structure.fields.contains_key("count"));
    assert_eq!(
        infer_member_read(&structure_fact, &MemberName("count".into()))
            .fact
            .kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real
        })
    );
    let updated = infer_member_write(
        &structure_fact,
        &MemberName("label".into()),
        &ValueFact::scalar(ValueKindFact::String),
        true,
    );
    let ValueKindFact::Struct(updated) = updated.fact.kind else {
        panic!("expected struct");
    };
    assert!(updated.fields.contains_key("label"));
    assert_eq!(
        infer_member_read(
            &numeric(NumericClass::Double, ShapeFact::Scalar),
            &MemberName("field".into())
        )
        .diagnostics[0]
            .code,
        "RM-TYPE-MEMBER-READ"
    );

    let left = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let right = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(4)]),
    );
    let concatenated = infer_concatenate(2, &[left, right]);
    assert_eq!(
        concatenated.fact.shape,
        ShapeFact::from(vec![Some(2), Some(7)])
    );
}

#[test]
fn indexed_mutation_tracks_proven_growth_and_cell_payload_updates() {
    let base = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let contract = MutationContract {
        kind: PlaceMutationKind::IndexedAssign,
        creation: AssignmentCreationPolicy::CreateArrayByIndex,
        shape: AssignmentShapePolicy::MatlabCompatible,
    };
    let grown = infer_index_mutation(
        &base,
        &[
            IndexSelectorFact::KnownOneBasedIndex(4),
            IndexSelectorFact::KnownOneBasedIndex(2),
        ],
        &numeric(NumericClass::Double, ShapeFact::Scalar),
        contract,
    );
    assert_eq!(grown.fact.shape, ShapeFact::from(vec![Some(4), Some(3)]));

    let first = numeric(NumericClass::Double, ShapeFact::Scalar);
    let cell = infer_cell_aggregate(&[vec![first]]).fact;
    let assigned = ValueFact::scalar(ValueKindFact::String);
    let updated = infer_index_mutation(
        &cell,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        &assigned,
        MutationContract {
            kind: PlaceMutationKind::CellAssign,
            ..contract
        },
    );
    let ValueKindFact::Cell(updated) = updated.fact.kind else {
        panic!("expected cell");
    };
    assert_eq!(updated.elements, vec![assigned]);

    let created = infer_index_mutation(
        &ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        &[IndexSelectorFact::KnownOneBasedIndex(2)],
        &ValueFact::scalar(ValueKindFact::String),
        MutationContract {
            kind: PlaceMutationKind::CellAssign,
            ..contract
        },
    );
    let ValueKindFact::Cell(created) = created.fact.kind else {
        panic!("expected created cell");
    };
    assert!(created.elements_complete);
    assert_eq!(created.elements.len(), 2);
    assert_eq!(
        created.elements[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert_eq!(created.elements[1].kind, ValueKindFact::String);
    assert_eq!(
        infer_index(
            &ValueFact::proven(
                ValueKindFact::Cell(created),
                ShapeFact::from(vec![Some(1), Some(2)]),
                runmat_types::StorageFact::Dense,
            ),
            IndexKind::Brace,
            &[IndexSelectorFact::KnownOneBasedIndex(2)],
            IndexResultContext::ReadSingle,
        )
        .fact
        .kind,
        ValueKindFact::String
    );

    let linear_growth = infer_index_mutation(
        &base,
        &[IndexSelectorFact::KnownOneBasedIndex(7)],
        &numeric(NumericClass::Double, ShapeFact::Scalar),
        contract,
    );
    assert_eq!(
        linear_growth.fact.shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
}

#[test]
fn indexed_creation_infers_new_array_category_and_cell_assignment_mode() {
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let assigned = numeric(NumericClass::UInt16, ShapeFact::Scalar);
    let created = infer_index_mutation(
        &unknown,
        &[IndexSelectorFact::KnownOneBasedIndex(3)],
        &assigned,
        MutationContract {
            kind: PlaceMutationKind::IndexedAssign,
            creation: AssignmentCreationPolicy::CreateArrayByIndex,
            shape: AssignmentShapePolicy::MatlabCompatible,
        },
    );
    assert_eq!(created.fact.kind, assigned.kind);
    assert_eq!(created.fact.shape, ShapeFact::from(vec![Some(1), Some(3)]));

    let cell = infer_index_mutation(
        &unknown,
        &[IndexSelectorFact::KnownOneBasedIndex(2)],
        &ValueFact::scalar(ValueKindFact::String),
        MutationContract {
            kind: PlaceMutationKind::CellAssign,
            creation: AssignmentCreationPolicy::CreateArrayByIndex,
            shape: AssignmentShapePolicy::MatlabCompatible,
        },
    );
    assert!(matches!(cell.fact.kind, ValueKindFact::Cell(_)));
    assert_eq!(cell.fact.shape, ShapeFact::from(vec![Some(1), Some(2)]));

    let invalid = infer_index_mutation(
        &cell.fact,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        &ValueFact::scalar(ValueKindFact::String),
        MutationContract {
            kind: PlaceMutationKind::IndexedAssign,
            creation: AssignmentCreationPolicy::CreateArrayByIndex,
            shape: AssignmentShapePolicy::MatlabCompatible,
        },
    );
    assert_eq!(invalid.diagnostics[0].code, "RM-TYPE-CELL-PAREN-ASSIGN");
}

#[test]
fn mutation_policies_reject_invalid_creation_and_shape() {
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    let vector = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(3)]),
    );
    let exact = MutationContract {
        kind: PlaceMutationKind::IndexedAssign,
        creation: AssignmentCreationPolicy::ExistingOnly,
        shape: AssignmentShapePolicy::Exact,
    };
    assert_eq!(
        infer_mutation(None, &scalar, exact).diagnostics[0].code,
        "RM-TYPE-MUTATION-CREATION"
    );
    assert_eq!(
        infer_mutation(Some(&vector), &scalar, exact).diagnostics[0].code,
        "RM-TYPE-MUTATION-SHAPE"
    );
    let scalar_expansion = MutationContract {
        shape: AssignmentShapePolicy::ScalarExpansion,
        ..exact
    };
    assert!(infer_mutation(Some(&vector), &scalar, scalar_expansion)
        .diagnostics
        .is_empty());
    let explicit_scalar = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(1), Some(1)]),
    );
    assert!(infer_mutation(Some(&scalar), &explicit_scalar, exact)
        .diagnostics
        .is_empty());
}
