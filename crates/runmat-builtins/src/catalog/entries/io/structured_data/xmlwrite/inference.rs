use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if !matches!(count, 1 | 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-XMLWRITE-ARITY",
            "xmlwrite accepts a document or filename and document",
            0,
        ));
    }
    if count == 2 {
        super::super::inference::input(request, 0, &mut diagnostics);
    }
    let document_index = if count == 2 { 1 } else { 0 };
    if let Some(value) = request.arguments.get(document_index) {
        if !document_possible(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-XMLWRITE-DOCUMENT",
                "XML text, compatible object, or scalar struct required",
                document_index,
            ));
        }
    }
    let output = if count == 2 {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    } else {
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        )
    };
    let mut result = super::super::inference::finish(request, entry, output, diagnostics);
    if count == 1 {
        result.effects.0.remove(&EffectKind::EnvironmentRead);
        result.effects.0.remove(&EffectKind::FilesystemWrite);
        result
            .capabilities
            .0
            .remove(&CapabilityRequirement::Filesystem);
    }
    result
}

fn document_possible(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String => {
            super::super::inference::scalar_text(value)
        }
        ValueKindFact::Struct(_) => !value.shape.is_definitely_nonscalar(),
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
            class == &super::super::XML_DOCUMENT_CLASS.owned()
                || class == &super::super::XML_ELEMENT_CLASS.owned()
        }),
        _ => false,
    }
}
