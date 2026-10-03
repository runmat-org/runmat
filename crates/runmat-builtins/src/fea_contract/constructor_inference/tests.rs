use super::*;
use crate::fea_contract::identities::{FEA_MODEL_CLASS, FEA_STUDY_CLASS};
use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    descriptors::FEA_MODEL_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::{BuiltinCatalogProvenance, BuiltinDocumentation};
use runmat_types::{
    CellFact, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn entry() -> BuiltinCatalogEntry {
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.model",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_MODEL_DESCRIPTOR,
        inference: |request, entry| infer_constructor(request, entry, FeaConstructorKind::Model),
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    })
}

#[test]
fn path_capability_is_declared_without_changing_pure_constructor_admission() {
    let pure = entry();
    assert!(pure.contract.capabilities.is_empty());
    let path = catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.load",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_MODEL_DESCRIPTOR,
        inference: |request, entry| {
            infer_constructor(request, entry, FeaConstructorKind::LoadDocument)
        },
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: true,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    });
    assert_eq!(
        path.contract.capabilities,
        &[runmat_types::CapabilityRequirement::Filesystem]
    );
}

#[test]
fn model_requires_literal_profile_and_host_geometry_class() {
    assert_eq!(
        entry().contract.compatibility,
        crate::BuiltinCompatibility::RunMat
    );
    let text = ValueFact::scalar(ValueKindFact::String);
    let geometry = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("geometry.Asset").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let missing = infer_constructor(
        &request(
            vec![text.clone(), geometry.clone()],
            vec![LiteralValue::Unknown; 2],
        ),
        &entry(),
        FeaConstructorKind::Model,
    );
    assert!(missing
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
    let valid = infer_constructor(
        &request(
            vec![text.clone(), geometry, text.clone(), text],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("P_rofile".into()),
                LiteralValue::String("linear_static_structural".into()),
            ],
        ),
        &entry(),
        FeaConstructorKind::Model,
    );
    assert!(valid.diagnostics.is_empty());
    assert_eq!(valid.outputs[0].residency, ResidencyFact::Host);
    assert!(
        matches!(&valid.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_MODEL_CLASS)))
    );
}

#[test]
fn document_class_is_dynamic_and_text_arrays_are_rejected() {
    let mut text_array = ValueFact::scalar(ValueKindFact::String);
    text_array.shape = ShapeFact::from(vec![Some(2), Some(1)]);
    let loaded = infer_constructor(
        &request(vec![text_array], vec![LiteralValue::Unknown]),
        &entry(),
        FeaConstructorKind::LoadDocument,
    );
    assert!(loaded
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-TEXT"));
    assert!(matches!(
        loaded.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));

    // A partially typed [1, N] value could be a valid character row.
    let mut possible_char_row = ValueFact::unknown(DynamicReason::RuntimeValue);
    possible_char_row.shape = ShapeFact::from(vec![Some(1), Some(7)]);
    let uncertain = infer_constructor(
        &request(vec![possible_char_row], vec![LiteralValue::Unknown]),
        &entry(),
        FeaConstructorKind::LoadDocument,
    );
    assert!(!uncertain
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-TEXT"));
    let mut possible_ranked_char = ValueFact::scalar(ValueKindFact::Character);
    possible_ranked_char.shape = ShapeFact::Ranked { rank: 3 };
    assert!(host_text(&possible_ranked_char));

    // fea.study(path) currently delegates to the same loader and may
    // return a sweep. The path spelling cannot prove Study class.
    let path_form = infer_constructor(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown],
        ),
        &entry(),
        FeaConstructorKind::Study,
    );
    assert!(matches!(
        path_form.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
    assert!(
        matches!(&path_form.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.is_none())
    );
}

#[test]
fn study_constructor_requires_profile_and_step_has_exact_arity() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let geometry = object_result(FeaConstructorKind::Model, false);
    let mut geometry = geometry;
    if let ValueKindFact::Object(object) = &mut geometry.kind {
        object.runtime_class = Some(StaticClassIdentity::new("geometry.Asset").owned());
    }
    let study = infer_constructor(
        &request(vec![text.clone(), geometry], vec![LiteralValue::Unknown; 2]),
        &entry(),
        FeaConstructorKind::Study,
    );
    assert!(study
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
    let step = infer_constructor(
        &request(
            vec![text.clone(), text.clone(), text],
            vec![LiteralValue::Unknown; 3],
        ),
        &entry(),
        FeaConstructorKind::Step,
    );
    assert!(step
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-ARITY"));
}

#[test]
fn sweep_rejects_known_non_study_cell_and_numeric_failfast() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let number = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }));
    let studies = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(number.clone()),
            elements: vec![number.clone()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer_constructor(
        &request(
            vec![text.clone(), studies, text, number],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("FAIL_FAST".into()),
                LiteralValue::Unknown,
            ],
        ),
        &entry(),
        FeaConstructorKind::Sweep,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-STUDIES"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-FAILFAST"));
}

#[test]
fn author_study_requires_summary_and_profile_before_result_class() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let geometry = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("geometry.Asset").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let numeric_summary = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }));
    let inferred = infer_constructor(
        &request(
            vec![text, geometry, numeric_summary],
            vec![LiteralValue::Unknown; 3],
        ),
        &entry(),
        FeaConstructorKind::AuthorStudy,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-AUTHORING-SUMMARY"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_STUDY_CLASS)))
    );
}
