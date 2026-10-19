use runmat_types::{
    infer_cell_aggregate, infer_concatenate, infer_index, infer_index_mutation,
    infer_index_sequence, infer_member_read, infer_member_sequence, infer_member_write,
    infer_mutation, infer_struct, infer_tensor_aggregate, AssignmentCreationPolicy,
    AssignmentShapePolicy, DynamicReason, IndexKind, IndexResultContext, IndexSelectorFact,
    MemberName, MutationContract, NumericClass, NumericDomain, NumericFact, PlaceMutationKind,
    SequenceUse, ShapeFact, ValueFact, ValueKindFact,
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
fn sequence_inference_keeps_zero_one_and_many_outputs_outside_value_facts() {
    let empty = infer_cell_aggregate(&[]).fact;
    let empty_sequence = infer_index_sequence(
        &empty,
        IndexKind::Brace,
        &[IndexSelectorFact::Colon],
        IndexResultContext::ReadCommaList,
    );
    assert!(empty_sequence.sequence.outputs.is_empty());
    assert!(!empty_sequence.sequence.variadic);

    let first = numeric(NumericClass::Double, ShapeFact::Scalar);
    let second = numeric(NumericClass::UInt64, ShapeFact::Scalar);
    let cell = infer_cell_aggregate(&[vec![first.clone(), second.clone()]]).fact;
    let one = infer_index_sequence(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        IndexResultContext::ReadCommaList,
    );
    assert_eq!(one.sequence.outputs, vec![first.clone()]);
    assert!(!one.sequence.variadic);

    let many = infer_index_sequence(
        &cell,
        IndexKind::Brace,
        &[IndexSelectorFact::Colon],
        IndexResultContext::ReadCommaList,
    );
    assert_eq!(many.sequence.outputs, vec![first, second]);
    assert!(!many.sequence.variadic);

    let struct_array = infer_struct(BTreeMap::from([(
        "field".to_owned(),
        numeric(NumericClass::UInt8, ShapeFact::Scalar),
    )]))
    .fact;
    let member = infer_member_sequence(&struct_array, &MemberName("field".to_owned()));
    assert_eq!(member.sequence.outputs.len(), 1);
    assert!(!matches!(
        member.sequence.outputs[0].kind,
        ValueKindFact::OutputList(_)
    ));
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
fn unknown_brace_receivers_remain_dynamic_without_hiding_proven_errors() {
    let unknown = ValueFact::unknown(DynamicReason::ExternalData);
    let single = infer_index(
        &unknown,
        IndexKind::Brace,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        IndexResultContext::ReadSingle,
    );
    assert!(single.diagnostics.is_empty());
    assert_eq!(single.fact, unknown);

    let sequence = infer_index_sequence(
        &unknown,
        IndexKind::Brace,
        &[IndexSelectorFact::Colon],
        IndexResultContext::ReadCommaList,
    );
    assert!(sequence.diagnostics.is_empty());
    assert!(sequence.sequence.variadic);
    assert!(sequence.sequence.outputs.is_empty());

    let proven_noncell = infer_index(
        &numeric(NumericClass::Double, ShapeFact::Scalar),
        IndexKind::Brace,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(proven_noncell.diagnostics.len(), 1);
    assert_eq!(proven_noncell.diagnostics[0].code, "RM-TYPE-BRACE-INDEX");

    let invalid_selector = infer_index(
        &unknown,
        IndexKind::Brace,
        &[IndexSelectorFact::KnownOneBasedIndex(0)],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(invalid_selector.diagnostics.len(), 1);
    assert_eq!(invalid_selector.diagnostics[0].code, "RM-TYPE-INDEX-BOUNDS");
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

    let nd = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(1), Some(3)]),
    );
    let nd_slice = infer_index(
        &nd,
        IndexKind::Paren,
        &[
            IndexSelectorFact::Colon,
            IndexSelectorFact::Colon,
            IndexSelectorFact::Colon,
        ],
        IndexResultContext::ReadSingle,
    );
    assert!(nd_slice.diagnostics.is_empty());
    assert_eq!(nd_slice.fact.shape, nd.shape);
    let collapsed_read = infer_index(
        &nd,
        IndexKind::Paren,
        &[
            IndexSelectorFact::KnownOneBasedIndex(2),
            IndexSelectorFact::KnownOneBasedIndex(3),
        ],
        IndexResultContext::ReadSingle,
    );
    assert!(collapsed_read.diagnostics.is_empty());
    let bad_collapsed_read = infer_index(
        &nd,
        IndexKind::Paren,
        &[
            IndexSelectorFact::KnownOneBasedIndex(2),
            IndexSelectorFact::KnownOneBasedIndex(4),
        ],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(
        bad_collapsed_read.diagnostics[0].code,
        "RM-TYPE-INDEX-BOUNDS"
    );
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
        infer_member_read(
            &structure_fact,
            &MemberName("count".into()),
            SequenceUse::RequireSingle,
        )
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
            &MemberName("field".into()),
            SequenceUse::RequireSingle,
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
        &infer_cell_aggregate(&[]).fact,
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
fn indexed_mutation_invalidates_only_unproved_growth_and_physical_facts() {
    let mut base = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    base.storage = runmat_types::StorageFact::Dense;
    let assigned = numeric(NumericClass::Double, ShapeFact::Scalar);
    let contract = MutationContract {
        kind: PlaceMutationKind::IndexedAssign,
        creation: AssignmentCreationPolicy::CreateArrayByIndex,
        shape: AssignmentShapePolicy::MatlabCompatible,
    };
    let unknown = IndexSelectorFact::Numeric(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let uncertain =
        infer_index_mutation(&base, std::slice::from_ref(&unknown), &assigned, contract).fact;
    assert_eq!(uncertain.shape, ShapeFact::Unknown);
    assert_eq!(uncertain.kind, base.kind);
    assert_eq!(uncertain.storage, runmat_types::StorageFact::Dense);
    assert_eq!(uncertain.layout, runmat_types::LayoutFact::Unknown);
    assert_eq!(uncertain.view, runmat_types::ViewFact::Unknown);
    assert_eq!(uncertain.alias, runmat_types::AliasFact::Unknown);

    for selector in [
        IndexSelectorFact::KnownOneBasedIndex(2),
        IndexSelectorFact::Colon,
        IndexSelectorFact::Logical(ValueFact::scalar(ValueKindFact::Logical)),
    ] {
        let bounded = infer_index_mutation(&base, &[selector], &assigned, contract).fact;
        assert_eq!(bounded.shape, base.shape);
        assert_eq!(bounded.layout, base.layout);
    }
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    let scalar_write = infer_index_mutation(
        &scalar,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        &assigned,
        contract,
    )
    .fact;
    assert_eq!(scalar_write.shape, ShapeFact::Scalar);
    assert_eq!(scalar_write.storage, runmat_types::StorageFact::Dense);
    let dynamic_rhs = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    assert_eq!(
        infer_index_mutation(
            &scalar,
            &[IndexSelectorFact::KnownOneBasedIndex(1)],
            &dynamic_rhs,
            contract,
        )
        .fact
        .storage,
        runmat_types::StorageFact::Dense,
        "unknown RHS domain cannot restore the primitive scalar carrier"
    );

    let mut shared_view = base.clone();
    shared_view.view = runmat_types::ViewFact::ReadOnlyView;
    shared_view.alias = runmat_types::AliasFact::Shared;
    shared_view.layout = runmat_types::LayoutFact::Strided;
    shared_view.contiguity = runmat_types::ContiguityFact::NonContiguous;
    let detached = infer_index_mutation(
        &shared_view,
        &[IndexSelectorFact::KnownOneBasedIndex(2)],
        &assigned,
        contract,
    )
    .fact;
    assert_eq!(detached.shape, shared_view.shape);
    assert_eq!(detached.layout, runmat_types::LayoutFact::Unknown);
    assert_eq!(detached.contiguity, runmat_types::ContiguityFact::Unknown);
    assert_eq!(detached.view, runmat_types::ViewFact::Unknown);
    assert_eq!(detached.alias, runmat_types::AliasFact::Unknown);
    let grown = infer_index_mutation(
        &base,
        &[IndexSelectorFact::KnownOneBasedIndex(3)],
        &assigned,
        contract,
    )
    .fact;
    assert_eq!(grown.shape, ShapeFact::from(vec![Some(1), Some(3)]));
    assert_eq!(grown.layout, runmat_types::LayoutFact::Unknown);
    let mut complex_rhs = assigned.clone();
    complex_rhs.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let promoted_without_growth = infer_index_mutation(
        &base,
        &[IndexSelectorFact::KnownOneBasedIndex(2)],
        &complex_rhs,
        contract,
    )
    .fact;
    assert_eq!(promoted_without_growth.shape, base.shape);
    assert_eq!(
        promoted_without_growth.numeric().unwrap().domain,
        NumericDomain::Complex
    );
    assert_eq!(
        promoted_without_growth.view,
        runmat_types::ViewFact::Unknown
    );
    assert_eq!(
        infer_index_mutation(
            &base,
            &[IndexSelectorFact::End { offset: 1 }],
            &assigned,
            contract,
        )
        .fact
        .shape,
        grown.shape
    );

    let empty = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(0), Some(0)]),
    );
    assert_eq!(
        infer_index_mutation(
            &empty,
            &[IndexSelectorFact::Logical(ValueFact::scalar(
                ValueKindFact::Logical
            ))],
            &assigned,
            contract,
        )
        .fact
        .shape,
        ShapeFact::Unknown,
        "a scalar true logical selector can address position one of an empty array"
    );
    assert_eq!(
        infer_index_mutation(
            &base,
            &[unknown, IndexSelectorFact::KnownOneBasedIndex(1)],
            &assigned,
            contract,
        )
        .fact
        .shape,
        ShapeFact::Unknown,
        "an unknown first subscript may extend the leading dimension"
    );
}

#[test]
fn indexed_mutation_follows_collapsed_nd_growth_and_complex_promotion() {
    let mut base = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(2), Some(2)]),
    );
    base.storage = runmat_types::StorageFact::Dense;
    let real = numeric(NumericClass::Double, ShapeFact::Scalar);
    let mut complex = real.clone();
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let contract = MutationContract {
        kind: PlaceMutationKind::IndexedAssign,
        creation: AssignmentCreationPolicy::CreateArrayByIndex,
        shape: AssignmentShapePolicy::MatlabCompatible,
    };
    let retained_nd = infer_index_mutation(
        &base,
        &[
            IndexSelectorFact::KnownOneBasedIndex(1),
            IndexSelectorFact::KnownOneBasedIndex(1),
            IndexSelectorFact::KnownOneBasedIndex(2),
        ],
        &real,
        contract,
    )
    .fact;
    assert_eq!(retained_nd.shape, base.shape);
    let collapsed = infer_index_mutation(
        &base,
        &[
            IndexSelectorFact::KnownOneBasedIndex(2),
            IndexSelectorFact::KnownOneBasedIndex(5),
        ],
        &real,
        contract,
    )
    .fact;
    assert_eq!(collapsed.shape, ShapeFact::from(vec![Some(2), Some(5)]));
    let linear = infer_index_mutation(
        &base,
        &[IndexSelectorFact::KnownOneBasedIndex(9)],
        &real,
        contract,
    )
    .fact;
    assert_eq!(
        linear.shape,
        ShapeFact::from(vec![Some(2), Some(2), Some(3)])
    );
    let leading_singleton_nd = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(2), Some(3)]),
    );
    let grown_leading_singleton = infer_index_mutation(
        &leading_singleton_nd,
        &[IndexSelectorFact::KnownOneBasedIndex(7)],
        &real,
        contract,
    )
    .fact;
    assert_eq!(
        grown_leading_singleton.shape,
        ShapeFact::from(vec![Some(1), Some(2), Some(4)])
    );
    let unchanged = infer_index_mutation(
        &base,
        &[
            IndexSelectorFact::KnownOneBasedIndex(2),
            IndexSelectorFact::KnownOneBasedIndex(4),
        ],
        &real,
        contract,
    )
    .fact;
    assert_eq!(unchanged.shape, base.shape);

    let promoted = infer_index_mutation(
        &base,
        &[IndexSelectorFact::KnownOneBasedIndex(9)],
        &complex,
        contract,
    )
    .fact;
    assert_eq!(promoted.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(promoted.numeric().unwrap().domain, NumericDomain::Complex);
    let unknown_shape_base = numeric(NumericClass::Single, ShapeFact::Unknown);
    let promoted_unknown_shape = infer_index_mutation(
        &unknown_shape_base,
        &[IndexSelectorFact::KnownOneBasedIndex(1)],
        &complex,
        contract,
    )
    .fact;
    assert_eq!(
        promoted_unknown_shape.numeric().unwrap().domain,
        NumericDomain::Complex,
        "a positive literal selector is nonempty even if the old extent is unknown"
    );
    let unknown = IndexSelectorFact::Numeric(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let maybe_empty = infer_index_mutation(&base, &[unknown], &complex, contract).fact;
    assert_eq!(maybe_empty.kind, ValueKindFact::Unknown);
    assert_eq!(maybe_empty.shape, ShapeFact::Unknown);
}

#[test]
fn indexed_creation_requires_a_known_base_for_class_and_cell_facts() {
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
    assert_eq!(created.fact.kind, ValueKindFact::Unknown);
    assert_eq!(created.fact.shape, ShapeFact::Unknown);

    let mut empty_double = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(0), Some(0)]),
    );
    empty_double.storage = runmat_types::StorageFact::Dense;
    let from_empty_double = infer_index_mutation(
        &empty_double,
        &[IndexSelectorFact::KnownOneBasedIndex(3)],
        &assigned,
        MutationContract {
            kind: PlaceMutationKind::IndexedAssign,
            creation: AssignmentCreationPolicy::CreateArrayByIndex,
            shape: AssignmentShapePolicy::MatlabCompatible,
        },
    )
    .fact;
    assert_eq!(from_empty_double.kind, empty_double.kind);
    assert_eq!(
        from_empty_double.shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );

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
    assert_eq!(cell.fact.kind, ValueKindFact::Unknown);
    assert_eq!(cell.fact.shape, ShapeFact::Unknown);

    let invalid = infer_index_mutation(
        &infer_cell_aggregate(&[]).fact,
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
