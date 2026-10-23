use super::*;
use crate::*;
use runmat_types::*;
fn call(class: StaticClassIdentity, outputs: usize) -> CallRequest {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    CallRequest {
        arguments: vec![object, ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn writer_classes_arity_and_filesystem_effects_are_exact() {
    for (entry, own, other) in [
        (
            &WRITETABLE_CATALOG_ENTRY,
            standard::TABLE,
            standard::TIMETABLE,
        ),
        (
            &WRITETIMETABLE_CATALOG_ENTRY,
            standard::TIMETABLE,
            standard::TABLE,
        ),
    ] {
        let result = infer_catalog_call(entry, &call(own, 0));
        assert!(result.diagnostics.is_empty());
        assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
        assert!(result.effects.0.contains(&EffectKind::EnvironmentRead));
        assert_eq!(
            entry.contract.capabilities,
            &[CapabilityRequirement::Filesystem]
        );
        assert!(!infer_catalog_call(entry, &call(other, 0))
            .diagnostics
            .is_empty());
        assert!(!infer_catalog_call(entry, &call(own, 2))
            .diagnostics
            .is_empty());
        let result = infer_catalog_call(entry, &call(own, 1));
        assert_eq!(result.outputs.len(), 1);
        assert!(matches!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
    }
}
