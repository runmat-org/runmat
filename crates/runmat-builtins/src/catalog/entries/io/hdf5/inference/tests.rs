use super::*;
use crate::catalog::entries::io::hdf5::{
    h5info::H5INFO_CATALOG_ENTRY, h5read::H5READ_CATALOG_ENTRY, h5write::H5WRITE_CATALOG_ENTRY,
};

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn real(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    let literals = vec![LiteralValue::Unknown; arguments.len()];
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn vector_shape_does_not_guess_external_dataset_geometry_or_class() {
    let vector = real(
        NumericClass::UInt64,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let inferred = infer(
        &request(vec![text(), text(), vector.clone(), vector], 1),
        &H5READ_CATALOG_ENTRY,
        Hdf5Form::Read,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn invalid_selection_and_sparse_complex_write_are_definite_errors() {
    let invalid = infer(
        &request(
            vec![
                text(),
                text(),
                ValueFact::scalar(ValueKindFact::Logical),
                real(NumericClass::Double, ShapeFact::Scalar),
            ],
            1,
        ),
        &H5READ_CATALOG_ENTRY,
        Hdf5Form::Read,
    );
    assert!(!invalid.diagnostics.is_empty());

    let mut sparse = real(NumericClass::Int64, ShapeFact::from(vec![Some(2), Some(2)]));
    sparse.storage = StorageFact::Sparse;
    let sparse_write = infer(
        &request(vec![text(), text(), sparse], 0),
        &H5WRITE_CATALOG_ENTRY,
        Hdf5Form::Write,
    );
    assert!(!sparse_write.diagnostics.is_empty());
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let complex_write = infer(
        &request(vec![text(), text(), complex], 0),
        &H5WRITE_CATALOG_ENTRY,
        Hdf5Form::Write,
    );
    assert!(!complex_write.diagnostics.is_empty());
}

#[test]
fn character_text_geometry_rejects_only_proven_invalid_pages_or_rows() {
    let character = |shape| ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
    for shape in [
        ShapeFact::from(vec![Some(2), Some(2)]),
        ShapeFact::from(vec![Some(1), Some(1), Some(2)]),
    ] {
        let inferred = infer(
            &request(vec![character(shape), text()], 1),
            &H5READ_CATALOG_ENTRY,
            Hdf5Form::Read,
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-HDF5-TEXT"));
    }
    for shape in [
        ShapeFact::from(vec![Some(1), Some(3), Some(1)]),
        ShapeFact::from(vec![Some(1), Some(3), None]),
    ] {
        let inferred = infer(
            &request(vec![character(shape), text()], 1),
            &H5READ_CATALOG_ENTRY,
            Hdf5Form::Read,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
    }
}

#[test]
fn unknown_write_data_stays_conservative_and_root_metadata_has_fields() {
    let write = infer(
        &request(
            vec![
                text(),
                text(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            0,
        ),
        &H5WRITE_CATALOG_ENTRY,
        Hdf5Form::Write,
    );
    assert!(write.diagnostics.is_empty(), "{:?}", write.diagnostics);
    assert!(write.outputs.is_empty());
    assert!(write.effects.0.contains(&EffectKind::FilesystemWrite));

    let root = infer(
        &request(vec![text()], 1),
        &H5INFO_CATALOG_ENTRY,
        Hdf5Form::Info,
    );
    let ValueKindFact::Struct(fields) = &root.outputs[0].kind else {
        panic!("root metadata struct")
    };
    assert!(fields.fields_complete);
    assert!(fields.fields.contains_key("Filename"));
    assert!(fields.fields.contains_key("Groups"));

    let selected = infer(
        &request(vec![text(), text()], 1),
        &H5INFO_CATALOG_ENTRY,
        Hdf5Form::Info,
    );
    let ValueKindFact::Struct(fields) = &selected.outputs[0].kind else {
        panic!("selected metadata struct")
    };
    assert!(!fields.fields_complete);
    assert!(fields.fields.contains_key("Name"));
    assert!(!fields.fields.contains_key("Filename"));
}
