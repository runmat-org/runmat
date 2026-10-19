use super::*;
use crate::fea_contract::{
    integer::{FEA_FIELD_DESCRIPTOR, FEA_PLOT_DESCRIPTOR, FEA_RESULTS_DESCRIPTOR},
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
        name: "fea.field",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_FIELD_DESCRIPTOR,
        inference: |request, entry| infer_result(request, entry, FeaResultKind::Field),
        integer_capabilities: &[],
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    })
}

#[test]
fn result_references_do_not_accept_numeric_or_resident_values() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!possible_run_reference(&number, true));
    let mut unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(possible_run_reference(&unknown, true));
    unknown.residency = ResidencyFact::Device { provider: None };
    assert!(!possible_run_reference(&unknown, true));

    let inferred = infer_result(
        &CallRequest {
            arguments: vec![number, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &entry(),
        FeaResultKind::Field,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-RUN"));
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_FIELD_CLASS)))
    );
}

#[test]
fn result_index_selector_requires_exact_class_and_vector_geometry() {
    let mut selector = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    }));
    assert!(!possible_index_vector(&selector));
    selector.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    });
    selector.shape = ShapeFact::from(vec![Some(2), Some(2)]);
    assert!(!possible_index_vector(&selector));
    selector.shape = ShapeFact::from(vec![Some(1), Some(4)]);
    assert!(possible_index_vector(&selector));
}

#[test]
fn result_aliases_share_typed_selector_and_duplicate_diagnostic() {
    let results_entry = operation_entry::entry(OperationEntryDefinition {
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
    });
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer_result(
        &CallRequest {
            arguments: vec![text; 5],
            literals: LiteralContext::new(vec![
                LiteralValue::String("run".into()),
                LiteralValue::String("FIELDS".into()),
                LiteralValue::String("stress".into()),
                LiteralValue::String("Include-Fields".into()),
                LiteralValue::String("strain".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &results_entry,
        FeaResultKind::Results,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEA-DUPLICATE"));
    assert!(!inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEA-SELECTOR"));
}

#[test]
fn plot_rejects_known_numeric_field_and_invalid_option_value() {
    let plot_entry = operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.plot",
        documentation: BuiltinDocumentation::EMPTY,
        descriptor: &FEA_PLOT_DESCRIPTOR,
        inference: |request, entry| infer_result(request, entry, FeaResultKind::Plot),
        integer_capabilities: &[],
        integer_audit: None,
        effects: OperationEffectProfile::Figure,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
    let run = object_result(FEA_RUN_RESULT_CLASS);
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let invalid_field = infer_result(
        &CallRequest {
            arguments: vec![run.clone(), numeric.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &plot_entry,
        FeaResultKind::Plot,
    );
    assert!(invalid_field
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PLOT-FIELD"));
    let invalid_option = infer_result(
        &CallRequest {
            arguments: vec![run, ValueFact::scalar(ValueKindFact::String), numeric],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Field_Id".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        },
        &plot_entry,
        FeaResultKind::Plot,
    );
    assert!(invalid_option
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PLOT-OPTION"));

    let dynamic_name = infer_result(
        &CallRequest {
            arguments: vec![
                object_result(FEA_RUN_RESULT_CLASS),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &plot_entry,
        FeaResultKind::Plot,
    );
    assert!(!dynamic_name
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEA-PLOT-CONTEXT" || d.code == "RM-CATALOG-FEA-PLOT-FIELD"));

    for context in [FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS] {
        let invalid_prefix = infer_result(
            &CallRequest {
                arguments: vec![
                    object_result(context),
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })),
                    ValueFact::scalar(ValueKindFact::String),
                    ValueFact::scalar(ValueKindFact::String),
                ],
                literals: LiteralContext::new(vec![LiteralValue::Unknown; 4]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
            &plot_entry,
            FeaResultKind::Plot,
        );
        assert!(invalid_prefix.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "RM-CATALOG-FEA-PLOT-FIELD" && diagnostic.argument == Some(1)
        }));
    }
}
