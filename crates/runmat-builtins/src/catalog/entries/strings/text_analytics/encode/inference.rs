use super::super::scalar_text_control::known_invalid_scalar_text_shape;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count < 2 || !count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ENCODE-ARITY",
            "encode requires a bag model, documents or words, and optional name-value pairs",
            count.min(2),
        ));
    }
    if let Some(bag) = request.arguments.first() {
        match &bag.kind {
            ValueKindFact::Object(object)
                if [
                    runmat_types::standard::BAG_OF_WORDS.owned(),
                    runmat_types::standard::BAG_OF_NGRAMS.owned(),
                ]
                .iter()
                .any(|class| object.runtime_class.as_ref() == Some(class)) => {}
            ValueKindFact::Object(object) if object.runtime_class.is_none() => {}
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-BAG",
                "encode requires bagOfWords or bagOfNgrams",
                0,
            )),
        }
    }
    if let Some(input) = request.arguments.get(1) {
        match &input.kind {
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref()
                    == Some(&runmat_types::standard::TOKENIZED_DOCUMENT.owned()) => {}
            ValueKindFact::Object(object) if object.runtime_class.is_none() => {}
            ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-INPUT",
                "encode requires tokenizedDocument or row text words",
                1,
            )),
        }
        if matches!(
            &input.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_)
        ) && input
            .shape
            .known_dims()
            .and_then(|dimensions| dimensions.first().copied().flatten())
            .is_some_and(|rows| rows > 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-ROW-WORDS",
                "non-tokenized input must be a row word vector",
                1,
            ));
        }
    }
    let mut force_cell = Some(false);
    for index in (2..count).step_by(2) {
        if request.arguments.get(index).is_some_and(|value| {
            !matches!(
                &value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-OPTION-TEXT",
                "option name must be scalar text",
                index,
            ));
        }
        if request
            .arguments
            .get(index)
            .is_some_and(known_invalid_scalar_text_shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-OPTION-TEXT",
                "option name must be scalar text",
                index,
            ));
        }
        match request.literals.literal_string_at(index) {
            Some(name) if name.eq_ignore_ascii_case("DocumentsIn") => {
                if request
                    .arguments
                    .get(index + 1)
                    .is_some_and(known_invalid_scalar_text_shape)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ENCODE-DOCUMENTS-IN",
                        "DocumentsIn must be scalar text",
                        index + 1,
                    ));
                }
                if request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|value| {
                        !value.eq_ignore_ascii_case("rows")
                            && !value.eq_ignore_ascii_case("columns")
                    })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ENCODE-DOCUMENTS-IN",
                        "DocumentsIn must be rows or columns",
                        index + 1,
                    ));
                }
            }
            Some(name) if name.eq_ignore_ascii_case("ForceCellOutput") => {
                force_cell = request.literals.literal_bool_at(index + 1);
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-CATALOG-ENCODE-OPTION",
                "encode accepts DocumentsIn and ForceCellOutput",
                index,
            )),
            None => force_cell = None,
        }
    }
    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Sparse,
    );
    let output = if !diagnostics.is_empty() || force_cell.is_none() {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else if force_cell == Some(true) {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(sparse.clone()),
                elements: vec![sparse],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        )
    } else {
        sparse
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};

    #[test]
    fn bag_and_input_then_option_pairs_have_even_arity() {
        for (count, valid) in [
            (0, false),
            (1, false),
            (2, true),
            (3, false),
            (4, true),
            (5, false),
            (6, true),
        ] {
            let result = infer(
                &CallRequest {
                    arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); count],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::default(),
                },
                &super::super::ENCODE_CATALOG_ENTRY,
            );
            assert_eq!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "RM-CATALOG-ENCODE-ARITY"),
                !valid,
                "argument count {count}",
            );
        }
    }
}
