use super::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn gca_canonical_inference_preserves_handle_shapes_and_literal_sensitive_modes() {
    let numeric_kind = scalar().kind;
    let dense_scalar = ValueFact::proven(
        numeric_kind.clone(),
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        },
        StorageFact::Dense,
    );
    for arguments in [vec![], vec![scalar()], vec![dense_scalar]] {
        let inferred = crate::catalog::inference::infer_catalog_call(
            &GCA_CATALOG_ENTRY,
            &request(arguments, 1),
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, numeric_kind);
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    }
    let string = ValueFact::scalar(ValueKindFact::String);
    let unknown_mode = crate::catalog::inference::infer_catalog_call(
        &GCA_CATALOG_ENTRY,
        &request(vec![string.clone()], 1),
    );
    assert!(unknown_mode.diagnostics.is_empty());
    assert_eq!(unknown_mode.outputs[0].kind, ValueKindFact::Unknown);
    let mut known_mode = request(vec![string], 1);
    known_mode.literals = LiteralContext::new(vec![LiteralValue::String("struct".into())]);
    let structure = crate::catalog::inference::infer_catalog_call(&GCA_CATALOG_ENTRY, &known_mode);
    assert!(structure.diagnostics.is_empty());
    assert!(matches!(
        structure.outputs[0].kind,
        ValueKindFact::Struct(_)
    ));
    let vector = ValueFact::proven(
        numeric_kind,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
        },
        StorageFact::Dense,
    );
    for arguments in [vec![scalar(), scalar()], vec![vector]] {
        let inferred = crate::catalog::inference::infer_catalog_call(
            &GCA_CATALOG_ENTRY,
            &request(arguments, 1),
        );
        assert!(!inferred.diagnostics.is_empty());
    }
}

#[test]
fn every_axes_control_keeps_effects_when_results_are_discarded_and_rejects_two_outputs() {
    let cases: &[(&BuiltinCatalogEntry, usize)] = &[
        (&CAXIS_CATALOG_ENTRY, 0),
        (&CLIM_CATALOG_ENTRY, 0),
        (&COLORORDER_CATALOG_ENTRY, 0),
        (&GCA_CATALOG_ENTRY, 0),
        (&SUBPLOT_CATALOG_ENTRY, 3),
        (&XLIM_CATALOG_ENTRY, 0),
        (&XSCALE_CATALOG_ENTRY, 0),
        (&XTICKANGLE_CATALOG_ENTRY, 0),
        (&XTICKFORMAT_CATALOG_ENTRY, 0),
        (&XTICKLABELS_CATALOG_ENTRY, 0),
        (&XTICKS_CATALOG_ENTRY, 0),
        (&YLIM_CATALOG_ENTRY, 0),
        (&YSCALE_CATALOG_ENTRY, 0),
        (&YTICKANGLE_CATALOG_ENTRY, 0),
        (&YTICKFORMAT_CATALOG_ENTRY, 0),
        (&YTICKLABELS_CATALOG_ENTRY, 0),
        (&YTICKS_CATALOG_ENTRY, 0),
        (&ZLIM_CATALOG_ENTRY, 0),
    ];
    assert_eq!(cases.len(), 18);
    for &(entry, inputs) in cases {
        let arguments = vec![scalar(); inputs];
        for output_count in [0, 1, 2] {
            let inferred = crate::catalog::inference::infer_catalog_call(
                entry,
                &request(arguments.clone(), output_count),
            );
            assert_eq!(
                !inferred.diagnostics.is_empty(),
                output_count > 1,
                "{} with {output_count} outputs",
                entry.identity.name,
            );
            assert!(inferred.effects.0.contains(&EffectKind::UserInterface));
            assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
            if output_count == 0 {
                assert!(inferred.outputs.is_empty());
            }
        }
    }
}

#[test]
fn host_controls_do_not_claim_known_device_inputs() {
    // Exact native receiver classes are facts, not numeric aliases. Unknown
    // object identities remain deferred to the actual selected receiver owner.
    for entry in [
        &CAXIS_CATALOG_ENTRY,
        &CLIM_CATALOG_ENTRY,
        &XLIM_CATALOG_ENTRY,
        &YLIM_CATALOG_ENTRY,
        &ZLIM_CATALOG_ENTRY,
        &XSCALE_CATALOG_ENTRY,
        &YSCALE_CATALOG_ENTRY,
        &XTICKS_CATALOG_ENTRY,
        &YTICKS_CATALOG_ENTRY,
        &XTICKLABELS_CATALOG_ENTRY,
        &YTICKLABELS_CATALOG_ENTRY,
        &XTICKANGLE_CATALOG_ENTRY,
        &YTICKANGLE_CATALOG_ENTRY,
        &XTICKFORMAT_CATALOG_ENTRY,
        &YTICKFORMAT_CATALOG_ENTRY,
        &GCA_CATALOG_ENTRY,
        &COLORORDER_CATALOG_ENTRY,
    ] {
        for class in [standard::AXES, standard::FIGURE] {
            let target = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(class.into()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(true),
            }));
            let expected_invalid = match entry.identity.name {
                "gca" => class != standard::FIGURE,
                "colororder" => false,
                _ => class != standard::AXES,
            };
            let inferred =
                crate::catalog::inference::infer_catalog_call(entry, &request(vec![target], 1));
            assert_eq!(
                !inferred.diagnostics.is_empty(),
                expected_invalid,
                "{} native {class:?}",
                entry.identity.name
            );
        }
    }
    // The actual style parsers flatten CharArray controls but do not accept a
    // StringArray as scalar mode text. Do not infer modes from payload alone.
    for entry in [
        &XLIM_CATALOG_ENTRY,
        &XSCALE_CATALOG_ENTRY,
        &XTICKFORMAT_CATALOG_ENTRY,
        &GCA_CATALOG_ENTRY,
    ] {
        for (kind, shape, invalid) in [
            (
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(4), Some(1)]),
                false,
            ),
            (
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(1)]),
                true,
            ),
        ] {
            let value = ValueFact::proven(kind, shape, StorageFact::Dense);
            let inferred =
                crate::catalog::inference::infer_catalog_call(entry, &request(vec![value], 1));
            assert_eq!(
                !inferred.diagnostics.is_empty(),
                invalid,
                "{} text carrier",
                entry.identity.name
            );
        }
    }
    let mut device = scalar();
    device.residency = ResidencyFact::Device { provider: None };
    for entry in [
        &CAXIS_CATALOG_ENTRY,
        &CLIM_CATALOG_ENTRY,
        &COLORORDER_CATALOG_ENTRY,
        &GCA_CATALOG_ENTRY,
        &XLIM_CATALOG_ENTRY,
        &XSCALE_CATALOG_ENTRY,
        &XTICKANGLE_CATALOG_ENTRY,
        &XTICKFORMAT_CATALOG_ENTRY,
        &XTICKLABELS_CATALOG_ENTRY,
        &XTICKS_CATALOG_ENTRY,
        &YLIM_CATALOG_ENTRY,
        &YSCALE_CATALOG_ENTRY,
        &YTICKANGLE_CATALOG_ENTRY,
        &YTICKFORMAT_CATALOG_ENTRY,
        &YTICKLABELS_CATALOG_ENTRY,
        &YTICKS_CATALOG_ENTRY,
        &ZLIM_CATALOG_ENTRY,
    ] {
        let inferred =
            crate::catalog::inference::infer_catalog_call(entry, &request(vec![device.clone()], 1));
        assert!(
            !inferred.diagnostics.is_empty(),
            "{} admitted a known device control",
            entry.identity.name,
        );
    }
    let inferred = crate::catalog::inference::infer_catalog_call(
        &SUBPLOT_CATALOG_ENTRY,
        &request(vec![device, scalar(), scalar()], 1),
    );
    assert!(!inferred.diagnostics.is_empty());

    let row = ValueFact::proven(
        scalar().kind,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    for entry in [
        &XSCALE_CATALOG_ENTRY,
        &YSCALE_CATALOG_ENTRY,
        &XTICKLABELS_CATALOG_ENTRY,
        &YTICKLABELS_CATALOG_ENTRY,
        &XTICKS_CATALOG_ENTRY,
        &YTICKS_CATALOG_ENTRY,
    ] {
        let inferred = crate::catalog::inference::infer_catalog_call(
            entry,
            &request(
                vec![row.clone(), ValueFact::unknown(DynamicReason::RuntimeValue)],
                1,
            ),
        );
        assert!(
            !inferred.diagnostics.is_empty(),
            "{} admitted a vector target",
            entry.identity.name
        );
    }
    // Angle/format setters deliberately support arrays of existing axes.
    for entry in [
        &XTICKANGLE_CATALOG_ENTRY,
        &YTICKANGLE_CATALOG_ENTRY,
        &XTICKFORMAT_CATALOG_ENTRY,
        &YTICKFORMAT_CATALOG_ENTRY,
    ] {
        let inferred = crate::catalog::inference::infer_catalog_call(
            entry,
            &request(
                vec![row.clone(), ValueFact::unknown(DynamicReason::RuntimeValue)],
                1,
            ),
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{} rejected an axes array setter",
            entry.identity.name
        );
    }
    for entry in [&XTICKS_CATALOG_ENTRY, &YTICKS_CATALOG_ENTRY] {
        let inferred =
            crate::catalog::inference::infer_catalog_call(entry, &request(vec![row.clone()], 1));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, scalar().kind);
        assert_eq!(inferred.outputs[0].shape, row.shape);
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    }
    for entry in [&XTICKLABELS_CATALOG_ENTRY, &YTICKLABELS_CATALOG_ENTRY] {
        let query = crate::catalog::inference::infer_catalog_call(entry, &request(vec![], 1));
        assert!(matches!(query.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(query.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
        assert_eq!(query.outputs[0].residency, ResidencyFact::Host);
        let mut mode = request(vec![ValueFact::scalar(ValueKindFact::String)], 1);
        mode.literals = LiteralContext::new(vec![LiteralValue::String("mode".into())]);
        let inferred = crate::catalog::inference::infer_catalog_call(entry, &mode);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::String);
        mode.arguments[0] = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        );
        let inferred = crate::catalog::inference::infer_catalog_call(entry, &mode);
        assert!(
            matches!(inferred.outputs[0].kind, ValueKindFact::Cell(_)),
            "singleton StringArray content is labels, not a mode"
        );
        mode.arguments[0].storage = StorageFact::Unknown;
        let inferred = crate::catalog::inference::infer_catalog_call(entry, &mode);
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Unknown,
            "unknown carrier may be scalar text or StringArray labels"
        );
    }
}
