//! Typed static facts for FEA evidence operations. Runtime remains the final
//! authority for study content, artifact availability, and solver validity.

use super::constructor_inference::{
    dynamic_object_fact, host_text, known_class, object_fact, possible_exact_scalar,
    possible_object,
};
use super::identities::{
    FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS, FEA_SWEEP_CLASS, FEA_SWEEP_PLAN_CLASS,
    FEA_SWEEP_RUN_RESULT_CLASS, FEA_SWEEP_VALIDATION_CLASS, FEA_TRENDS_CLASS, FEA_VALIDATION_CLASS,
};
use super::options::FeaTrendsOption;
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, StaticClassIdentity, ValueFact};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaEvidenceKind {
    Validate,
    Plan,
    Run,
    Trends,
}

impl FeaEvidenceKind {
    /// Result classes for a `fea.Study` and a `fea.Sweep` input respectively.
    const fn result_classes(self) -> (StaticClassIdentity, StaticClassIdentity) {
        match self {
            Self::Validate => (FEA_VALIDATION_CLASS, FEA_SWEEP_VALIDATION_CLASS),
            Self::Plan => (FEA_PLAN_CLASS, FEA_SWEEP_PLAN_CLASS),
            Self::Run => (FEA_RUN_RESULT_CLASS, FEA_SWEEP_RUN_RESULT_CLASS),
            Self::Trends => (FEA_TRENDS_CLASS, FEA_TRENDS_CLASS),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Validate => "fea.validate",
            Self::Plan => "fea.plan",
            Self::Run => "fea.run",
            Self::Trends => "fea.trends",
        }
    }
}

fn study_result(kind: FeaEvidenceKind, input: Option<&ValueFact>) -> ValueFact {
    let (study, sweep) = kind.result_classes();
    match input {
        Some(input) if known_class(input, FEA_STUDY_CLASS) => object_fact(study),
        Some(input) if known_class(input, FEA_SWEEP_CLASS) => object_fact(sweep),
        _ => dynamic_object_fact(),
    }
}

pub fn infer_evidence(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaEvidenceKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if kind == FeaEvidenceKind::Trends {
        if !request.arguments.len().is_multiple_of(2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-ARITY",
                "fea.trends requires complete Name, Value pairs",
                request.arguments.len(),
            ));
        }
        let mut window_seen = false;
        for index in (0..request.arguments.len()).step_by(2) {
            if !host_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-OPTION",
                    "fea.trends option names must be host text scalars",
                    index,
                ));
            }
            if let Some(name) = request.literals.literal_string_at(index) {
                if FeaTrendsOption::decode(&name) != Some(FeaTrendsOption::WindowSize) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        "WindowSize is the only fea.trends option",
                        index,
                    ));
                } else if std::mem::replace(&mut window_seen, true) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-DUPLICATE",
                        "duplicate fea.trends option WindowSize",
                        index,
                    ));
                }
            }
            if let Some(value) = request.arguments.get(index + 1) {
                if !possible_exact_scalar(value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-WINDOW",
                        "WindowSize must be a host real numeric scalar",
                        index + 1,
                    ));
                }
                if request
                    .literals
                    .numeric_at(index + 1)
                    .is_some_and(|value| !value.is_finite() || value < 1.0 || value.fract() != 0.0)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-WINDOW",
                        "WindowSize must be a positive exact integer",
                        index + 1,
                    ));
                }
            }
        }
        return finish_fixed(entry, request, object_fact(FEA_TRENDS_CLASS), diagnostics);
    }
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            format!(
                "{} requires exactly one fea.Study or fea.Sweep",
                kind.name()
            ),
            request.arguments.len(),
        ));
    }
    let input = request.arguments.first();
    if input.is_some_and(|input| !possible_object(input, &[FEA_STUDY_CLASS, FEA_SWEEP_CLASS])) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-DOCUMENT",
            format!(
                "{} input must be a host fea.Study or fea.Sweep object; paths are not accepted",
                kind.name()
            ),
            0,
        ));
    }
    finish_fixed(entry, request, study_result(kind, input), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::{
        descriptors::{FEA_TRENDS_DESCRIPTOR, FEA_VALIDATE_DESCRIPTOR},
        operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    };
    use crate::{
        BuiltinAcceleratorPolicy, BuiltinCatalogProvenance, BuiltinDistributedPolicy,
        BuiltinDocumentation,
    };
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ShapeFact, ValueKindFact,
    };

    fn entry(kind: FeaEvidenceKind) -> BuiltinCatalogEntry {
        operation_entry::entry(OperationEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: kind.name(),
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: if kind == FeaEvidenceKind::Trends {
                &FEA_TRENDS_DESCRIPTOR
            } else {
                &FEA_VALIDATE_DESCRIPTOR
            },
            inference: |request, entry| infer_evidence(request, entry, FeaEvidenceKind::Validate),
            integer_capabilities: &[],
            integer_audit: None,
            effects: OperationEffectProfile::SynchronousEvidenceWrite,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            distributed: BuiltinDistributedPolicy::Unsupported,
        })
    }

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            literals: LiteralContext::new(literals),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    fn codes(inference: &CallInference) -> Vec<&str> {
        inference
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect()
    }

    fn runtime_class(fact: &ValueFact) -> Option<String> {
        match &fact.kind {
            ValueKindFact::Object(object) => {
                object.runtime_class.as_ref().map(|class| class.to_string())
            }
            _ => None,
        }
    }

    #[test]
    fn study_and_sweep_inputs_select_their_result_classes() {
        for (kind, study, sweep) in [
            (
                FeaEvidenceKind::Validate,
                "fea.Validation",
                "fea.SweepValidation",
            ),
            (FeaEvidenceKind::Plan, "fea.Plan", "fea.SweepPlan"),
            (FeaEvidenceKind::Run, "fea.RunResult", "fea.SweepRunResult"),
        ] {
            let entry = entry(kind);
            let inferred = infer_evidence(
                &request(
                    vec![object_fact(FEA_STUDY_CLASS)],
                    vec![LiteralValue::Unknown],
                ),
                &entry,
                kind,
            );
            assert!(
                inferred.diagnostics.is_empty(),
                "{:?}",
                inferred.diagnostics
            );
            assert_eq!(runtime_class(&inferred.outputs[0]).as_deref(), Some(study));
            let inferred = infer_evidence(
                &request(
                    vec![object_fact(FEA_SWEEP_CLASS)],
                    vec![LiteralValue::Unknown],
                ),
                &entry,
                kind,
            );
            assert_eq!(runtime_class(&inferred.outputs[0]).as_deref(), Some(sweep));
        }
    }

    #[test]
    fn paths_wrong_arity_and_unknown_inputs_are_classified() {
        let entry = entry(FeaEvidenceKind::Validate);
        let path = infer_evidence(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::String("cube.fea".into())],
            ),
            &entry,
            FeaEvidenceKind::Validate,
        );
        assert!(codes(&path).contains(&"RM-CATALOG-FEA-DOCUMENT"));
        let arity = infer_evidence(&request(vec![], vec![]), &entry, FeaEvidenceKind::Validate);
        assert!(codes(&arity).contains(&"RM-CATALOG-FEA-ARITY"));
        let unknown = infer_evidence(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
            FeaEvidenceKind::Validate,
        );
        assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
        assert_eq!(runtime_class(&unknown.outputs[0]), None);
    }

    #[test]
    fn trends_window_size_is_normalized_and_validated() {
        let entry = entry(FeaEvidenceKind::Trends);
        let uint = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        let valid = infer_evidence(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String), uint.clone()],
                vec![
                    LiteralValue::String("Window-Size".into()),
                    LiteralValue::Unknown,
                ],
            ),
            &entry,
            FeaEvidenceKind::Trends,
        );
        assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
        assert_eq!(
            runtime_class(&valid.outputs[0]).as_deref(),
            Some("fea.Trends")
        );
        let mut matrix = uint;
        matrix.shape = ShapeFact::from(vec![Some(2), Some(1)]);
        let invalid = infer_evidence(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String), matrix],
                vec![LiteralValue::String("Limit".into()), LiteralValue::Unknown],
            ),
            &entry,
            FeaEvidenceKind::Trends,
        );
        assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-OPTION"));
        assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-WINDOW"));
        let odd = infer_evidence(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
            FeaEvidenceKind::Trends,
        );
        assert!(codes(&odd).contains(&"RM-CATALOG-FEA-ARITY"));
    }
}
