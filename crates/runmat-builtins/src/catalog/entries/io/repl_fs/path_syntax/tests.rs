use runmat_types::{
    CellFact, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> runmat_types::CallRequest {
    runmat_types::CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn all_identities_are_complete_catalog_authorities_with_examples() {
    for name in ["fileparts", "filesep", "fullfile", "pathsep"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        assert_eq!(
            entry.contract.maturity,
            crate::BuiltinContractMaturity::Complete
        );
        assert_eq!(
            entry.documentation.authority,
            crate::BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
    }
}

#[test]
fn separators_are_character_scalars_and_reject_inputs() {
    for name in ["filesep", "pathsep"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        let result =
            crate::infer_catalog_call(entry, &request(Vec::new(), RequestedOutputCount::One));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
        let invalid = crate::infer_catalog_call(
            entry,
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                RequestedOutputCount::One,
            ),
        );
        assert!(!invalid.diagnostics.is_empty());
    }
}

#[test]
fn fullfile_and_fileparts_preserve_container_shape() {
    let strings = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let fullfile = crate::builtin_catalog_entry_by_name("fullfile").expect("fullfile");
    let joined = crate::infer_catalog_call(
        fullfile,
        &request(
            vec![ValueFact::scalar(ValueKindFact::Character), strings.clone()],
            RequestedOutputCount::One,
        ),
    );
    assert_eq!(joined.outputs[0].kind, ValueKindFact::String);
    assert_eq!(joined.outputs[0].shape, strings.shape);

    let cells = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(super::facts::character_row()),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let fileparts = crate::builtin_catalog_entry_by_name("fileparts").expect("fileparts");
    let split = crate::infer_catalog_call(
        fileparts,
        &request(vec![cells.clone()], RequestedOutputCount::Exactly(3)),
    );
    assert_eq!(split.outputs.len(), 3);
    assert_eq!(split.outputs[0].shape, cells.shape);
    assert!(matches!(split.outputs[0].kind, ValueKindFact::Cell(_)));
}

#[test]
fn fullfile_diagnoses_known_container_shape_mismatch() {
    let shaped = |rows, cols| {
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(rows), Some(cols)]),
            StorageFact::Dense,
        )
    };
    let entry = crate::builtin_catalog_entry_by_name("fullfile").expect("fullfile");
    let result = crate::infer_catalog_call(
        entry,
        &request(vec![shaped(1, 2), shaped(2, 1)], RequestedOutputCount::One),
    );
    assert!(!result.diagnostics.is_empty());
}

#[test]
fn fullfile_numeric_extension_is_limited_to_real_dense_rows() {
    let numeric = |domain, shape, storage| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain,
            }),
            shape,
            storage,
        )
    };
    let entry = crate::builtin_catalog_entry_by_name("fullfile").expect("fullfile");
    let valid = crate::infer_catalog_call(
        entry,
        &request(
            vec![numeric(
                NumericDomain::Real,
                ShapeFact::from(vec![Some(1), Some(3)]),
                StorageFact::Dense,
            )],
            RequestedOutputCount::One,
        ),
    );
    assert!(valid.diagnostics.is_empty());

    for invalid in [
        numeric(
            NumericDomain::Complex,
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        ),
        numeric(
            NumericDomain::Real,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        ),
        numeric(
            NumericDomain::Real,
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Sparse,
        ),
    ] {
        let inference =
            crate::infer_catalog_call(entry, &request(vec![invalid], RequestedOutputCount::One));
        assert!(!inference.diagnostics.is_empty());
    }
}

#[test]
fn fullfile_does_not_invent_a_text_representation_for_unknown_inputs() {
    let entry = crate::builtin_catalog_entry_by_name("fullfile").expect("fullfile");
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let dynamic = crate::infer_catalog_call(
        entry,
        &request(
            vec![ValueFact::scalar(ValueKindFact::Character), unknown.clone()],
            RequestedOutputCount::One,
        ),
    );
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));

    let with_string = crate::infer_catalog_call(
        entry,
        &request(
            vec![ValueFact::scalar(ValueKindFact::String), unknown],
            RequestedOutputCount::One,
        ),
    );
    assert!(matches!(with_string.outputs[0].kind, ValueKindFact::String));
    assert_eq!(with_string.outputs[0].shape, ShapeFact::Unknown);
}
