use super::*;
use crate::fea_contract::identities::{FEA_MODEL_CLASS, FEA_STUDY_CLASS};
use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    descriptors::FEA_MODEL_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::{BuiltinCatalogProvenance, BuiltinDocumentation};
use runmat_types::{
    CellFact, DynamicReason, LiteralContext, LiteralValue, ObjectFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
fn study_and_material_assignment_require_the_current_typed_operands() {
    let object = |class| {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new(class).owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )
    };
    let arguments = vec![
        ValueFact::scalar(ValueKindFact::String),
        object("geometry.Asset"),
        object("fea.Model"),
    ];
    let typed = infer_constructor(
        &request(arguments, vec![LiteralValue::Unknown; 3]),
        &crate::FEA_STUDY_CATALOG_ENTRY,
        FeaConstructorKind::Study,
    );
    assert!(typed.diagnostics.is_empty(), "{:?}", typed.diagnostics);
    let path = infer_constructor(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown],
        ),
        &crate::FEA_STUDY_CATALOG_ENTRY,
        FeaConstructorKind::Study,
    );
    assert!(path
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-ARITY"));
    let assigned = infer_constructor(
        &request(
            vec![object("fea.PropertyRegionTarget"), object("fea.Material")],
            vec![LiteralValue::Unknown; 2],
        ),
        &crate::FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY,
        FeaConstructorKind::MaterialAssignment,
    );
    assert!(
        assigned.diagnostics.is_empty(),
        "{:?}",
        assigned.diagnostics
    );
    let text = infer_constructor(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String); 2],
            vec![LiteralValue::Unknown; 2],
        ),
        &crate::FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY,
        FeaConstructorKind::MaterialAssignment,
    );
    assert_eq!(
        text.diagnostics
            .iter()
            .filter(|d| d.code == "RM-CATALOG-FEA-COMPONENT")
            .count(),
        2
    );
}

#[test]
fn pure_model_and_mesh_persisting_study_declare_their_actual_effects() {
    assert!(crate::FEA_MODEL_CATALOG_ENTRY
        .contract
        .capabilities
        .is_empty());
    assert!(crate::FEA_STUDY_CATALOG_ENTRY
        .contract
        .effects
        .contains(&runmat_types::EffectKind::FilesystemWrite));
}

#[test]
fn model_requires_explicit_sequence_or_steps_and_host_geometry_class() {
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
        .any(|d| d.code == "RM-CATALOG-FEA-SEQUENCE"));
    let valid = infer_constructor(
        &request(
            vec![
                text.clone(),
                geometry.clone(),
                text.clone(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("Steps".into()),
                LiteralValue::Unknown,
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

    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let steps = |shape| {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(unknown.clone()),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape,
            StorageFact::Dense,
        )
    };
    let empty = steps(ShapeFact::from(vec![Some(0), Some(0)]));
    let nonempty = steps(ShapeFact::from(vec![Some(1), Some(1)]));
    let unknown_shape = steps(ShapeFact::Unknown);
    let options = |tail: Vec<(Option<&str>, ValueFact)>| {
        let mut arguments = vec![text.clone(), geometry.clone()];
        let mut literals = vec![LiteralValue::Unknown; 2];
        for (name, value) in tail {
            arguments.extend([text.clone(), value]);
            literals.extend([
                name.map(|name| LiteralValue::String(name.into()))
                    .unwrap_or(LiteralValue::Unknown),
                LiteralValue::Unknown,
            ]);
        }
        infer_constructor(
            &request(arguments, literals),
            &entry(),
            FeaConstructorKind::Model,
        )
    };
    for (label, tail, rejects) in [
        (
            "sequence and empty steps",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), empty.clone()),
            ],
            false,
        ),
        (
            "sequence and unknown steps",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), unknown.clone()),
            ],
            false,
        ),
        (
            "unknown cell shape",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), unknown_shape),
            ],
            false,
        ),
        (
            "nonempty conflict",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), nonempty.clone()),
            ],
            true,
        ),
        (
            "empty last wins",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), nonempty.clone()),
                (Some("Steps"), empty.clone()),
            ],
            false,
        ),
        (
            "nonempty last wins",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), empty.clone()),
                (Some("Steps"), nonempty.clone()),
            ],
            true,
        ),
        (
            "later unknown name",
            vec![
                (Some("Sequence"), unknown.clone()),
                (Some("Steps"), nonempty.clone()),
                (None, empty.clone()),
            ],
            false,
        ),
        ("empty without sequence", vec![(Some("Steps"), empty)], true),
        (
            "unknown without sequence",
            vec![(Some("Steps"), unknown)],
            false,
        ),
        (
            "nonempty without sequence",
            vec![(Some("Steps"), nonempty)],
            false,
        ),
    ] {
        let result = options(tail);
        assert_eq!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEA-SEQUENCE"),
            rejects,
            "{label}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn structural_text_identifiers_reject_arrays_and_keep_possible_character_rows() {
    let mut text_array = ValueFact::scalar(ValueKindFact::String);
    text_array.shape = ShapeFact::from(vec![Some(2), Some(1)]);
    assert!(!host_text(&text_array));
    let mut row = ValueFact::unknown(DynamicReason::RuntimeValue);
    row.shape = ShapeFact::from(vec![Some(1), Some(7)]);
    assert!(host_text(&row));
}

#[test]
fn study_constructor_requires_an_explicit_third_model_operand() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let mut geometry = object_result(FeaConstructorKind::Model);
    if let ValueKindFact::Object(object) = &mut geometry.kind {
        object.runtime_class = Some(StaticClassIdentity::new("geometry.Asset").owned());
    }
    let study = infer_constructor(
        &request(vec![text, geometry], vec![LiteralValue::Unknown; 2]),
        &crate::FEA_STUDY_CATALOG_ENTRY,
        FeaConstructorKind::Study,
    );
    assert!(study
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
fn study_rejects_numeric_replacement_for_the_typed_model() {
    let mut geometry = object_result(FeaConstructorKind::Model);
    if let ValueKindFact::Object(object) = &mut geometry.kind {
        object.runtime_class = Some(StaticClassIdentity::new("geometry.Asset").owned());
    }
    let numeric_model = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }));
    let inferred = infer_constructor(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                geometry,
                numeric_model,
            ],
            vec![LiteralValue::Unknown; 3],
        ),
        &crate::FEA_STUDY_CATALOG_ENTRY,
        FeaConstructorKind::Study,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-COMPONENT"));
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_STUDY_CLASS)))
    );
}
