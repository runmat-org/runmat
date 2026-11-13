use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, text_comparison_result,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STARTSWITH-ARITY",
            "startsWith requires subject and pattern",
            request.arguments.len(),
        ));
    }
    if request.arguments.len() > 4 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STARTSWITH-ARITY",
            "startsWith accepts at most one IgnoreCase option",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request.arguments.len() == 4
        && request
            .literals
            .literal_string_at(2)
            .is_some_and(|name| name != "ignorecase")
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STARTSWITH-OPTION",
            "startsWith expects the IgnoreCase option name",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        let pattern_object = index == 1
            && matches!(&argument.kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_none_or(|class| class == &runmat_types::standard::PATTERN.owned()));
        let invalid = !pattern_object
            && (invalid_text_comparison_container(argument)
                || matches!(
                    &argument.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
                ));
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-STARTSWITH-TEXT",
                "startsWith requires host text arguments",
                index,
            ));
        }
    }
    let output = match (request.arguments.first(), request.arguments.get(1)) {
        (Some(subject), Some(pattern)) => {
            if matches!(&pattern.kind, ValueKindFact::Object(_)) {
                logical_output(subject_shape(subject))
            } else {
                let (mut result, shape_error) = text_comparison_result(subject, pattern);
                if let Some(error) = shape_error {
                    diagnostics.push(error);
                }
                if matches!(
                    &subject.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
                ) || matches!(
                    &pattern.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
                ) {
                    result = dynamic_logical();
                }
                result
            }
        }
        _ => dynamic_logical(),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn subject_shape(subject: &ValueFact) -> ShapeFact {
    match &subject.kind {
        ValueKindFact::String | ValueKindFact::Cell(_) => subject.shape.clone(),
        ValueKindFact::Character => subject
            .shape
            .known_dims()
            .map_or(ShapeFact::Unknown, |dims| {
                ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
            }),
        _ => ShapeFact::Unknown,
    }
}

fn logical_output(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Logical, shape, storage);
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}
fn dynamic_logical() -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Logical;
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn known_wrong_option_is_diagnosed() {
        let entry = &super::super::STARTSWITH_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::new(vec![
                runmat_types::LiteralValue::Unknown,
                runmat_types::LiteralValue::Unknown,
                runmat_types::LiteralValue::String("unknown".into()),
                runmat_types::LiteralValue::Bool(true),
            ]),
            outputs: OutputSelection::default(),
        };
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STARTSWITH-OPTION"));
    }

    #[test]
    fn known_text_and_unknown_shapes_are_distinct() {
        let entry = &super::super::STARTSWITH_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let known = infer(&request, entry);
        assert!(known.diagnostics.is_empty());
        assert_eq!(
            known.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
        let mut unknown = request;
        unknown.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
        assert_eq!(infer(&unknown, entry).outputs[0].shape, ShapeFact::Unknown);
    }
    #[test]
    fn invalid_text_is_diagnosed() {
        let entry = &super::super::STARTSWITH_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STARTSWITH-TEXT"));
    }
}
