use super::*;
use crate::fea_contract::identities::{
    FEA_OUTPUT_FIELD_CLASS, FEA_STRUCTURAL_FIELD_CLASS, FEA_STUDY_CLASS,
};
use crate::fea_contract::{
    descriptors::FEA_RESULTS_DESCRIPTOR,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
};
use crate::{
    BuiltinAcceleratorPolicy, BuiltinCatalogProvenance, BuiltinDistributedPolicy,
    BuiltinDocumentation,
};
use runmat_types::{
    DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
};

fn entry() -> BuiltinCatalogEntry {
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.results",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_RESULTS_DESCRIPTOR,
        inference: |request, entry| infer_result(request, entry, FeaResultKind::Results),
        integer_capabilities: &[],
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    })
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn infer(
    kind: FeaResultKind,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    infer_result(&request(arguments, literals), &entry(), kind)
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

#[test]
fn results_accepts_run_references_and_typed_selectors() {
    let valid = infer(
        FeaResultKind::Results,
        vec![
            object_fact(FEA_STEP_RUN_RESULT_CLASS),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Include-Field-Values".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    assert_eq!(output_class(&valid).as_deref(), Some("fea.Results"));

    let invalid = infer(
        FeaResultKind::Results,
        vec![
            object_fact(FEA_RESULTS_CLASS),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("IncludeElectromagneticResults".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-RESULTS-OPTIONS"));
    assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-OPTION"));

    let arity = infer(FeaResultKind::Results, vec![], vec![]);
    assert!(codes(&arity).contains(&"RM-CATALOG-FEA-ARITY"));

    let dynamic = infer(
        FeaResultKind::Results,
        vec![unknown()],
        vec![LiteralValue::Unknown],
    );
    assert!(dynamic.diagnostics.is_empty(), "{:?}", dynamic.diagnostics);
}

#[test]
fn field_requires_typed_selector_and_frequency_sample() {
    let valid = infer(
        FeaResultKind::Field,
        vec![
            object_fact(FEA_RESULTS_CLASS),
            object_fact(FEA_STRUCTURAL_FIELD_CLASS),
            object_fact(FEA_FREQUENCY_CLASS),
        ],
        vec![LiteralValue::Unknown; 3],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    assert_eq!(output_class(&valid).as_deref(), Some("fea.Field"));

    let text_selector = infer(
        FeaResultKind::Field,
        vec![text(), text()],
        vec![LiteralValue::Unknown, LiteralValue::String("stress".into())],
    );
    assert!(codes(&text_selector).contains(&"RM-CATALOG-FEA-FIELD-SELECTOR"));

    let wrong_sample = infer(
        FeaResultKind::Field,
        vec![text(), object_fact(FEA_OUTPUT_FIELD_CLASS), text()],
        vec![LiteralValue::Unknown; 3],
    );
    assert!(codes(&wrong_sample).contains(&"RM-CATALOG-FEA-SAMPLE"));

    let arity = infer(
        FeaResultKind::Field,
        vec![text()],
        vec![LiteralValue::Unknown],
    );
    assert!(codes(&arity).contains(&"RM-CATALOG-FEA-ARITY"));

    let dynamic = infer(
        FeaResultKind::Field,
        vec![unknown(), unknown()],
        vec![LiteralValue::Unknown; 2],
    );
    assert!(dynamic.diagnostics.is_empty(), "{:?}", dynamic.diagnostics);
}

#[test]
fn field_page_step_result_and_compare_validate_shapes() {
    let page = infer(
        FeaResultKind::FieldPage,
        vec![object_fact(FEA_FIELD_CLASS), unknown(), unknown()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
            LiteralValue::Unknown,
        ],
    );
    assert!(codes(&page).contains(&"RM-CATALOG-FEA-PAGE-BOUND"));
    assert_eq!(output_class(&page).as_deref(), Some("fea.FieldPage"));
    let wrong_field = infer(
        FeaResultKind::FieldPage,
        vec![object_fact(FEA_RESULTS_CLASS)],
        vec![LiteralValue::Unknown],
    );
    assert!(codes(&wrong_field).contains(&"RM-CATALOG-FEA-FIELD"));

    let step = infer(
        FeaResultKind::StepResult,
        vec![object_fact(FEA_RUN_RESULT_CLASS), text()],
        vec![LiteralValue::Unknown, LiteralValue::String("static".into())],
    );
    assert!(step.diagnostics.is_empty(), "{:?}", step.diagnostics);
    assert_eq!(output_class(&step).as_deref(), Some("fea.StepRunResult"));
    let step_text = infer(
        FeaResultKind::StepResult,
        vec![text(), text()],
        vec![LiteralValue::Unknown; 2],
    );
    assert!(codes(&step_text).contains(&"RM-CATALOG-FEA-RUN"));

    let compare = infer(
        FeaResultKind::Compare,
        vec![text(), ValueFact::scalar(ValueKindFact::Logical)],
        vec![LiteralValue::Unknown; 2],
    );
    assert!(codes(&compare).contains(&"RM-CATALOG-FEA-RUN-ID"));
    assert_eq!(output_class(&compare).as_deref(), Some("fea.Compare"));
}

#[test]
fn plot_and_mesh_overloads_peel_display_options() {
    let plot = infer(
        FeaResultKind::Plot,
        vec![object_fact(FEA_FIELD_CLASS), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Overlay".into()),
            LiteralValue::String("cad".into()),
        ],
    );
    assert!(plot.diagnostics.is_empty(), "{:?}", plot.diagnostics);
    let study_context = infer(
        FeaResultKind::Plot,
        vec![object_fact(FEA_STUDY_CLASS), text()],
        vec![LiteralValue::Unknown, LiteralValue::String("run".into())],
    );
    assert!(codes(&study_context).contains(&"RM-CATALOG-FEA-PLOT-CONTEXT"));
    assert!(codes(&study_context).contains(&"RM-CATALOG-FEA-ARITY"));
    let mesh = infer(
        FeaResultKind::Mesh,
        vec![
            object_fact(FEA_RUN_RESULT_CLASS),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Edges".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(mesh.diagnostics.is_empty(), "{:?}", mesh.diagnostics);
}
