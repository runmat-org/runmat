use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn call(arguments: Vec<ValueFact>) -> CallInference {
    crate::infer_catalog_call(
        &DLARRAY_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn numeric(class: NumericClass, shape: ShapeFact, residency: ResidencyFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    value.residency = residency;
    value
}

#[test]
fn data_geometry_and_residency_are_carried_by_the_object_fact() {
    let shape = ShapeFact::from(vec![Some(3), Some(4), Some(2)]);
    let data = numeric(
        NumericClass::Single,
        shape.clone(),
        ResidencyFact::Device { provider: None },
    );
    let result = call(vec![data.clone(), ValueFact::scalar(ValueKindFact::String)]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, shape);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected object")
    };
    assert_eq!(
        object.runtime_class,
        Some(runmat_types::standard::DL_ARRAY.owned())
    );
    assert_eq!(object.properties[crate::DLARRAY_DATA_PROPERTY], data);
    assert!(!object.properties_complete);
    assert_eq!(
        DLARRAY_CATALOG_ENTRY.placement.preparation,
        BuiltinOperandPreparation::RuntimeOwned
    );
    assert_eq!(
        DLARRAY_CATALOG_ENTRY.placement.residency,
        BuiltinResidencyPolicy::Host
    );
    assert_eq!(
        DLARRAY_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
}

#[test]
fn known_non_numeric_and_integer_data_are_diagnosed_without_rejecting_unknown() {
    for data in [
        ValueFact::scalar(ValueKindFact::String),
        numeric(NumericClass::UInt64, ShapeFact::Scalar, ResidencyFact::Host),
    ] {
        assert!(!call(vec![data]).diagnostics.is_empty());
    }
    let unknown = call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(DLARRAY_CATALOG_ENTRY.integer_capabilities.len(), 1);
    assert_eq!(
        DLARRAY_CATALOG_ENTRY.descriptor.errors[0].identifier,
        Some("RunMat:deepLearning:InvalidInput")
    );
}

#[test]
fn ambient_autodiff_tape_effect_and_resident_unsupported_error_are_declared() {
    assert_eq!(DLARRAY_CATALOG_ENTRY.contract.purity, BuiltinPurity::Impure);
    for effect in [
        runmat_types::EffectKind::EnvironmentRead,
        runmat_types::EffectKind::EnvironmentWrite,
        runmat_types::EffectKind::MayThrow,
    ] {
        assert!(DLARRAY_CATALOG_ENTRY.contract.effects.contains(&effect));
    }
    assert!(DLARRAY_DESCRIPTOR
        .errors
        .iter()
        .any(|error| { error.identifier == Some("RunMat:deepLearning:Unsupported") }));
}

#[test]
fn format_must_be_host_text_even_when_data_is_resident() {
    let data = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
        ResidencyFact::Device { provider: None },
    );
    let mut resident_format = ValueFact::scalar(ValueKindFact::String);
    resident_format.residency = ResidencyFact::Device { provider: None };
    assert!(call(vec![data.clone(), resident_format])
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLARRAY-FORMAT"));
    assert!(call(vec![data, ValueFact::scalar(ValueKindFact::String)])
        .diagnostics
        .is_empty());
}
