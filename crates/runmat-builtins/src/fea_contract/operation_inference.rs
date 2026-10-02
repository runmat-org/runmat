//! Typed static facts for FEA evidence operations. Runtime remains the final
//! authority for document content, artifact availability, and solver validity.

use super::constructor_inference::host_text;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericDomain, ObjectFact, ResidencyFact, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaEvidenceKind {
    Validate,
    Plan,
    Trends,
}

impl FeaEvidenceKind {
    const fn result_class(self) -> &'static str {
        match self {
            Self::Validate => "fea.Validation",
            Self::Plan => "fea.Plan",
            Self::Trends => "fea.Trends",
        }
    }
}

fn result_object(kind: FeaEvidenceKind) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new(kind.result_class()).owned()),
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
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
            class.is(StaticClassIdentity::new("fea.Study"))
                || class.is(StaticClassIdentity::new("fea.Sweep"))
        }),
        _ => host_text(value),
    }
}

fn possible_positive_window(value: &ValueFact) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
        && !matches!(value.shape.element_count(), Some(count) if count != 1)
        && match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
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
        if request.arguments.len() % 2 != 0 {
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
            }
        }
    } else {
        if request.arguments.len() != 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-ARITY",
                "FEA validation and planning require exactly one document, study, or sweep",
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
    use runmat_types::{DynamicReason, NumericClass, NumericFact};

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
    }
}
