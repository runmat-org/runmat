use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRJOIN-ARITY",
            "strjoin requires one or two inputs",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        let legacy_numeric_delimiter = index == 1
            && matches!(
                request.arguments.first().map(|v| &v.kind),
                Some(ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character)
            )
            && matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
            && input.shape.element_count().is_none_or(|n| n == 1);
        if !legacy_numeric_delimiter
            && !matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRJOIN-TEXT",
                "strjoin accepts text only",
                index,
            ));
        }
        if input
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().skip(2).any(|d| d.is_some_and(|d| d != 1)))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRJOIN-RANK",
                "strjoin does not accept non-singleton dimensions beyond two",
                index,
            ));
        }
    }
    if let Some(input) = request.arguments.first() {
        let standard_only =
            request.arguments.len() == 1 || matches!(input.kind, ValueKindFact::Cell(_));
        let known_matrix = input.shape.known_dims().is_some_and(|dims| {
            !dims.contains(&Some(0)) && dims.iter().filter(|d| d.is_some_and(|n| n > 1)).count() > 1
        });
        if standard_only && (known_matrix || input.kind == ValueKindFact::Character) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRJOIN-VECTOR",
                "standard strjoin requires a string or cellstr vector",
                0,
            ));
        }
        if let ValueKindFact::Cell(cell) = &input.kind {
            if cell.elements.iter().any(|element| {
                !matches!(
                    element.kind,
                    ValueKindFact::Unknown | ValueKindFact::Character
                )
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRJOIN-CELLSTR",
                    "input cell elements must be character rows",
                    0,
                ));
            }
        }
    }
    let output = match request.arguments.first().map(|input| &input.kind) {
        Some(ValueKindFact::Cell(_)) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        Some(ValueKindFact::String) if request.arguments.len() == 1 => {
            ValueFact::scalar(ValueKindFact::String)
        }
        Some(ValueKindFact::String | ValueKindFact::Character) => {
            // Policy facts do not encode extension enablement. Explicit scalar
            // delimiters may produce either scalar or row-wise string output.
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::Unknown,
                StorageFact::Unknown,
            )
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_delimiter_does_not_assume_compatibility_mode() {
        let input = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(3), Some(1)]),
            StorageFact::Dense,
        );
        let mut request = CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, &super::super::STRJOIN_CATALOG_ENTRY).outputs[0].shape,
            ShapeFact::Scalar
        );
        request
            .arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        assert_eq!(
            infer(&request, &super::super::STRJOIN_CATALOG_ENTRY).outputs[0].shape,
            ShapeFact::Unknown
        );
    }
    #[test]
    fn unknown_input_remains_unknown_and_numeric_input_rejects() {
        for input in [
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Logical),
        ] {
            let invalid = input.kind == ValueKindFact::Logical;
            let request = CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            };
            let result = infer(&request, &super::super::STRJOIN_CATALOG_ENTRY);
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
            assert_eq!(!result.diagnostics.is_empty(), invalid);
        }
    }
}
