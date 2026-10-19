use super::{HORZCAT_CATALOG_ENTRY, VERTCAT_CATALOG_ENTRY};
use crate::{infer_catalog_call, BuiltinCatalogEntry};
use runmat_types::{
    CallInference, CallRequest, CellFact, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric(class: NumericClass, rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn infer(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> CallInference {
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn horizontal_and_vertical_proven_shapes_preserve_old_resolver_assertions() {
    let horizontal = infer(
        &HORZCAT_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 2, 3),
        ],
    );
    assert!(horizontal.diagnostics.is_empty());
    assert_eq!(
        horizontal.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    assert!(matches!(
        &horizontal.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));

    let vertical = infer(
        &VERTCAT_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, 1, 3),
            numeric(NumericClass::Double, 2, 3),
        ],
    );
    assert!(vertical.diagnostics.is_empty());
    assert_eq!(
        vertical.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
}

#[test]
fn mixed_numeric_classes_keep_proven_shape_without_proving_class() {
    let result = infer(
        &HORZCAT_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Single, 2, 3),
        ],
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    assert!(matches!(&result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn cell_summary_survives_but_old_linear_positions_do_not() {
    let element = ValueFact::scalar(ValueKindFact::Logical);
    let cell = |columns| {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element.clone()),
                elements: vec![element.clone(); columns],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(columns)]),
            StorageFact::Dense,
        )
    };
    let result = infer(&HORZCAT_CATALOG_ENTRY, vec![cell(1), cell(2)]);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected a cell result");
    };
    assert!(matches!(&cell.element.kind, ValueKindFact::Logical));
    assert!(!cell.elements_complete);
    assert!(cell.elements.is_empty());
}

#[test]
fn mismatched_dimensions_diagnose_but_neutral_empty_and_device_remain_dynamic() {
    let mismatch = infer(
        &HORZCAT_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 3, 1),
        ],
    );
    assert!(mismatch
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CONCAT"));

    let neutral = infer(
        &HORZCAT_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, 0, 0),
            numeric(NumericClass::Double, 2, 1),
        ],
    );
    assert_eq!(neutral.outputs[0].shape, ShapeFact::Unknown);

    let mut device = numeric(NumericClass::Double, 2, 1);
    device.residency = ResidencyFact::Device {
        provider: Some("exact".into()),
    };
    let device_result = infer(
        &HORZCAT_CATALOG_ENTRY,
        vec![device, numeric(NumericClass::Double, 2, 3)],
    );
    assert_eq!(device_result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(device_result.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn canonical_documentation_retains_legacy_reference_inventory() {
    // Immutable test inputs conserved from 52863699e3's retired authoring sidecars:
    // horzcat SHA-256 4a9ef9103a0a6cde4291c120511f25e57a5b635b76f6fd587c92e9eda6937718;
    // vertcat SHA-256 bea6bf5075a99f98e640194cca8b3ab4461f9d7d0797e542af0b6df5538f8237.
    // Catalog documentation remains the sole production authoring source.
    for (reference, entry) in [
        (
            include_str!("tests/fixtures/horzcat_52863699e3.json"),
            &HORZCAT_CATALOG_ENTRY,
        ),
        (
            include_str!("tests/fixtures/vertcat_52863699e3.json"),
            &VERTCAT_CATALOG_ENTRY,
        ),
    ] {
        let legacy: serde_json::Value =
            serde_json::from_str(reference).expect("legacy concatenation reference JSON");
        let documentation = &entry.documentation;
        assert_eq!(documentation.summary, legacy["summary"].as_str().unwrap());
        assert_eq!(
            documentation.description,
            legacy["description"].as_str().unwrap()
        );
        assert_eq!(
            documentation.examples.len(),
            legacy["examples"].as_array().unwrap().len()
        );
        assert_eq!(
            documentation.faqs.len(),
            legacy["faqs"].as_array().unwrap().len()
        );
        assert_eq!(
            documentation.links.len(),
            legacy["links"].as_array().unwrap().len() + 1
        );
        assert_eq!(
            documentation.keywords.len(),
            legacy["keywords"].as_array().unwrap().len()
        );
        for (old, canonical) in legacy["examples"]
            .as_array()
            .unwrap()
            .iter()
            .zip(documentation.examples)
        {
            assert_eq!(canonical.title, old["description"].as_str().unwrap());
            assert_eq!(canonical.program, old["input"].as_str().unwrap());
            assert_eq!(canonical.display_output, old["output"].as_str());
        }
        for (old, canonical) in legacy["links"]
            .as_array()
            .unwrap()
            .iter()
            .zip(documentation.links)
        {
            assert_eq!(canonical.label, old["label"].as_str().unwrap());
        }
        for (index, behavior) in legacy["behaviors"].as_array().unwrap().iter().enumerate() {
            let mut expected = behavior.as_str().unwrap().to_owned();
            if entry.identity.name == "vertcat" && index == 2 {
                expected = expected.replace("row-building", "column-building");
            }
            // The old horzcat text rejected mixed host/device inputs, but cat
            // has an exact gather/restore path for compatible numeric values.
            if entry.identity.name == "horzcat" && index == 3 {
                assert!(
                    documentation
                        .sections
                        .iter()
                        .flat_map(|section| section.paragraphs.iter())
                        .all(|paragraph| !paragraph
                            .contains("all-empty input produces the canonical"))
                );
                assert!(documentation
                    .sections
                    .iter()
                    .flat_map(|section| section.paragraphs.iter())
                    .any(|paragraph| paragraph.contains("nonzero all-empty operands")));
            } else if entry.identity.name == "horzcat" && index == 5 {
                assert!(documentation
                    .sections
                    .iter()
                    .flat_map(|section| section.paragraphs.iter())
                    .any(|paragraph| paragraph.contains("mixed `gpuArray` and host")));
            } else {
                assert!(documentation
                    .sections
                    .iter()
                    .flat_map(|section| section.paragraphs.iter())
                    .any(|paragraph| *paragraph == expected.as_str()));
            }
        }
        // The old GPU prose guaranteed re-upload even when upload failed.
        // Preserve the hook/fallback guidance, but state that outcome honestly.
        assert_eq!(legacy["gpu_behavior"].as_array().unwrap().len(), 1);
        assert!(documentation
            .sections
            .iter()
            .flat_map(|section| section.paragraphs.iter())
            .any(|paragraph| paragraph.contains("attempts to upload the result")));
    }
}
