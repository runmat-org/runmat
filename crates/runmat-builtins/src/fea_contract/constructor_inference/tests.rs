use super::*;
use crate::fea_contract::identities::{FEA_SWEEP_CLASS, GEOMETRY_ASSET_CLASS};
use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    descriptors::FEA_MODEL_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::{BuiltinCatalogProvenance, BuiltinDocumentation};
use runmat_types::{
    CellFact, DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, ValueFact, ValueKindFact,
};

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

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn literal(text: &str) -> LiteralValue {
    LiteralValue::String(text.into())
}

fn codes(inference: &CallInference) -> Vec<&str> {
    inference
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect()
}

fn output_class(inference: &CallInference) -> Option<String> {
    match &inference.outputs[0].kind {
        ValueKindFact::Object(object) => object.runtime_class.as_ref().map(ToString::to_string),
        _ => None,
    }
}

fn infer(
    kind: FeaConstructorKind,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    infer_constructor(&request(arguments, literals), &entry(), kind)
}

#[test]
fn artifact_writing_constructor_declares_write_without_suspension() {
    let writer = catalog_entry::artifact_writing_entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.study",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_MODEL_DESCRIPTOR,
        inference: |request, entry| infer_constructor(request, entry, FeaConstructorKind::Study),
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    });
    assert!(writer
        .contract
        .effects
        .contains(&runmat_types::EffectKind::FilesystemWrite));
    assert!(!writer
        .contract
        .effects
        .contains(&runmat_types::EffectKind::MaySuspend));
    assert!(entry().contract.capabilities.is_empty());
}

#[test]
fn model_requires_geometry_and_exactly_one_procedure_source() {
    let geometry = object_fact(GEOMETRY_ASSET_CLASS);
    let steps = infer(
        FeaConstructorKind::Model,
        vec![
            text(),
            geometry.clone(),
            text(),
            object_fact(FEA_MODEL_CLASS),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Steps"),
            LiteralValue::Unknown,
        ],
    );
    assert!(steps.diagnostics.is_empty(), "{:?}", steps.diagnostics);
    assert_eq!(output_class(&steps).as_deref(), Some("fea.Model"));

    let missing = infer(
        FeaConstructorKind::Model,
        vec![text(), geometry.clone()],
        vec![LiteralValue::Unknown; 2],
    );
    assert!(codes(&missing).contains(&"RM-CATALOG-FEA-SEQUENCE"));

    let both = infer(
        FeaConstructorKind::Model,
        vec![text(), geometry.clone(), text(), text(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Steps"),
            LiteralValue::Unknown,
            literal("Sequence"),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&both).contains(&"RM-CATALOG-FEA-SEQUENCE"));

    let wrong = infer(
        FeaConstructorKind::Model,
        vec![text(), text(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Template"),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&wrong).contains(&"RM-CATALOG-FEA-GEOMETRY"));
    assert!(codes(&wrong).contains(&"RM-CATALOG-FEA-OPTION"));

    let odd = infer(
        FeaConstructorKind::Model,
        vec![text(), geometry, text()],
        vec![LiteralValue::Unknown; 3],
    );
    assert!(codes(&odd).contains(&"RM-CATALOG-FEA-ARITY"));
}

#[test]
fn study_requires_model_and_typed_options() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let valid = infer(
        FeaConstructorKind::Study,
        vec![
            text(),
            unknown.clone(),
            unknown.clone(),
            text(),
            unknown.clone(),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Mesh"),
            LiteralValue::Unknown,
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    assert_eq!(output_class(&valid).as_deref(), Some("fea.Study"));

    let invalid = infer(
        FeaConstructorKind::Study,
        vec![
            text(),
            object_fact(GEOMETRY_ASSET_CLASS),
            object_fact(GEOMETRY_ASSET_CLASS),
            text(),
            object_fact(FEA_MODEL_CLASS),
            text(),
            text(),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Execution"),
            LiteralValue::Unknown,
            literal("Profile"),
            LiteralValue::Unknown,
        ],
    );
    let codes = codes(&invalid);
    assert!(codes.contains(&"RM-CATALOG-FEA-MODEL"));
    assert!(codes.contains(&"RM-CATALOG-FEA-OPTION-VALUE"));
    assert!(codes.contains(&"RM-CATALOG-FEA-OPTION"));
}

#[test]
fn sweep_accepts_study_cells_and_logical_fail_fast() {
    let studies = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(object_fact(FEA_STUDY_CLASS)),
            elements: vec![object_fact(FEA_STUDY_CLASS)],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        runmat_types::StorageFact::Opaque,
    );
    let valid = infer(
        FeaConstructorKind::Sweep,
        vec![
            text(),
            studies,
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("Fail_Fast"),
            LiteralValue::Unknown,
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    assert_eq!(output_class(&valid).as_deref(), Some("fea.Sweep"));

    let invalid = infer(
        FeaConstructorKind::Sweep,
        vec![text(), object_fact(FEA_SWEEP_CLASS), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            literal("FailFast"),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-STUDIES"));
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-OPTION-VALUE"));
}

#[test]
fn material_requires_a_capability_and_host_values() {
    let empty = infer(
        FeaConstructorKind::Material,
        vec![text()],
        vec![LiteralValue::Unknown],
    );
    assert!(codes(&empty).contains(&"RM-CATALOG-FEA-MATERIAL-EMPTY"));
    assert_eq!(output_class(&empty).as_deref(), Some("fea.Material"));
    let mut resident = ValueFact::scalar(ValueKindFact::Unknown);
    resident.residency = ResidencyFact::Device { provider: None };
    let invalid = infer(
        FeaConstructorKind::Material,
        vec![ValueFact::scalar(ValueKindFact::Logical), text(), resident],
        vec![
            LiteralValue::Unknown,
            literal("YoungsModulusPa"),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-TEXT"));
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-RESIDENT"));
}
