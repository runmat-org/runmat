//! Typed static facts for FEA evidence operations. Runtime remains the final
//! authority for document content, artifact availability, and solver validity.

use super::constructor_inference::host_text;
use super::identities::{
    FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS, FEA_SWEEP_CLASS, FEA_TRENDS_CLASS,
    FEA_VALIDATION_CLASS,
};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, ObjectFact, ResidencyFact, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaEvidenceKind {
    Validate,
    Plan,
    Run,
    Trends,
}

impl FeaEvidenceKind {
    const fn result_class(self) -> StaticClassIdentity {
        match self {
            Self::Validate => FEA_VALIDATION_CLASS,
            Self::Plan => FEA_PLAN_CLASS,
            Self::Run => FEA_RUN_RESULT_CLASS,
            Self::Trends => FEA_TRENDS_CLASS,
        }
    }
}

fn result_object(kind: FeaEvidenceKind) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(kind.result_class().owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn possible_document(value: &ValueFact) -> bool {
    if matches!(value.residency, ResidencyFact::Device { .. }) {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            !matches!(value.shape.element_count(), Some(count) if count != 1)
                && object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(FEA_STUDY_CLASS) || class.is(FEA_SWEEP_CLASS))
        }
        _ => host_text(value),
    }
}

fn possible_positive_window(value: &ValueFact) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
        && !matches!(value.shape.element_count(), Some(count) if count != 1)
        && match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(number) => {
                number.domain == NumericDomain::Real && number.class != NumericClass::Single
            }
            _ => false,
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
                "fea.trends requires complete name/value pairs",
                request.arguments.len(),
            ));
        }
        for index in (0..request.arguments.len()).step_by(2) {
            if !host_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-OPTION",
                    "fea.trends option names must be host text scalars",
                    index,
                ));
            }
            if let Some(name) = request.literals.literal_string_at(index) {
                if !name.eq_ignore_ascii_case("windowsize") {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        "WindowSize is the only fea.trends option",
                        index,
                    ));
                }
            }
            if let Some(value) = request.arguments.get(index + 1) {
                if !possible_positive_window(value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-WINDOW",
                        "WindowSize must be a host real numeric scalar",
                        index + 1,
                    ));
                }
                if request
                    .literals
                    .numeric_at(index + 1)
                    .is_some_and(|value| !value.is_finite() || value <= 0.0 || value.fract() != 0.0)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-WINDOW",
                        "WindowSize must be a positive exact integer",
                        index + 1,
                    ));
                }
            }
        }
    } else {
        if request.arguments.len() != 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-ARITY",
                "FEA validation, planning, and solving require exactly one document, study, or sweep",
                request.arguments.len(),
            ));
        }
        if let Some(input) = request.arguments.first() {
            if !possible_document(input) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-DOCUMENT",
                    "FEA operation input must be a host path, fea.Study, or fea.Sweep",
                    0,
                ));
            }
        }
    }
    finish_fixed(entry, request, result_object(kind), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::{
        descriptors::FEA_VALIDATE_DESCRIPTOR,
        operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    };
    use crate::{
        BuiltinAcceleratorPolicy, BuiltinCatalogProvenance, BuiltinDistributedPolicy,
        BuiltinDocumentation,
    };
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericFact, OutputSelection,
        RequestedOutputCount,
    };

    fn entry() -> BuiltinCatalogEntry {
        operation_entry::entry(OperationEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.validate",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &FEA_VALIDATE_DESCRIPTOR,
            inference: |request, entry| infer_evidence(request, entry, FeaEvidenceKind::Validate),
            integer_capabilities: &[],
            integer_audit: None,
            effects: OperationEffectProfile::EvidenceWrite,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            distributed: BuiltinDistributedPolicy::Unsupported,
        })
    }

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    #[test]
    fn document_and_window_shapes_are_conservative() {
        let mut possible_char = ValueFact::unknown(DynamicReason::RuntimeValue);
        possible_char.shape = ShapeFact::from(vec![Some(1), Some(10)]);
        assert!(possible_document(&possible_char));
        let mut non_scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        non_scalar.shape = ShapeFact::from(vec![Some(2), Some(1)]);
        assert!(!possible_positive_window(&non_scalar));
        let invalid = infer_evidence(
            &request(vec![non_scalar]),
            &entry(),
            FeaEvidenceKind::Validate,
        );
        assert!(invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEA-DOCUMENT"));
        assert!(
            matches!(&invalid.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_VALIDATION_CLASS)))
        );
    }
}
