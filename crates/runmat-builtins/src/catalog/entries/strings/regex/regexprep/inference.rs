use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REGEXPREP-ARITY",
            "regexprep requires subject, pattern and replacement",
            request.arguments.len(),
        ));
    }
    for (index, argument) in request.arguments.iter().take(3).enumerate() {
        if !matches!(
            &argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REGEXPREP-TEXT",
                "regexprep requires host text in subject, pattern and replacement roles",
                index,
            ));
        }
    }
    let output = match request.arguments.first() {
        Some(subject) => match &subject.kind {
            ValueKindFact::String | ValueKindFact::Cell(_) => {
                let mut output = subject.clone();
                output.alias = runmat_types::AliasFact::Unique;
                output.residency = runmat_types::ResidencyFact::Host;
                output
            }
            ValueKindFact::Character => {
                let shape = subject
                    .shape
                    .known_dims()
                    .map_or(ShapeFact::Unknown, |dims| {
                        ShapeFact::from(vec![dims.first().copied().flatten(), None])
                    });
                let mut output =
                    ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
                output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
                output
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn string_array_shape_survives_and_char_width_does_not() {
        let entry = &super::super::REGEXPREP_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(3)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            request.arguments[0].shape
        );
        request.arguments[0] = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(8)]),
            StorageFact::Dense,
        );
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}
