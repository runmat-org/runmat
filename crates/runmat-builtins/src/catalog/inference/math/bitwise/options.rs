//! Literal/control facts; runtime admission remains in the bitwise engine.
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, InferenceDiagnostic, IntegerClass, NumericDomain, StorageFact, ValueKindFact,
};

pub(super) fn assumed_type(
    request: &CallRequest,
    index: usize,
    data: &[usize],
) -> Vec<InferenceDiagnostic> {
    let Some(argument) = request.arguments.get(index) else {
        return vec![];
    };
    let invalid_text_shape = match argument.kind {
        ValueKindFact::String => argument
            .shape
            .element_count()
            .is_some_and(|count| count != 1),
        ValueKindFact::Character => argument.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .copied()
                .flatten()
                .is_some_and(|rows| rows != 1)
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|extent| extent != 1))
        }),
        _ => false,
    };
    if invalid_text_shape {
        return vec![argument_error(
            "RM-CATALOG-BITWISE-ASSUMED-TYPE",
            "assumed type must be a character row or scalar string",
            index,
        )];
    }
    if let Some(text) = request.literals.literal_string_at(index) {
        let Some(class) = IntegerClass::from_class_name(&text) else {
            return vec![argument_error(
                "RM-CATALOG-BITWISE-ASSUMED-TYPE",
                "assumed type must name a native integer class",
                index,
            )];
        };
        if data.iter().any(|position| {
            request
                .arguments
                .get(*position)
                .and_then(|value| value.numeric())
                .and_then(|numeric| numeric.class.integer_class())
                .is_some_and(|native| native != class)
        }) {
            return vec![argument_error(
                "RM-CATALOG-BITWISE-ASSUMED-TYPE",
                "assumed type must match native integer input",
                index,
            )];
        }
    } else if !matches!(
        argument.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) {
        return vec![argument_error(
            "RM-CATALOG-BITWISE-ASSUMED-TYPE",
            "assumed type must be text",
            index,
        )];
    }
    vec![]
}

pub(super) fn numeric_control(request: &CallRequest, index: usize) -> Vec<InferenceDiagnostic> {
    let Some(value) = request.arguments.get(index) else {
        return vec![];
    };
    let supported = matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
        || matches!(value.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real);
    if !supported || value.storage == StorageFact::Sparse {
        vec![argument_error(
            "RM-CATALOG-BITWISE-CONTROL",
            "bit controls require dense real numeric or logical values",
            index,
        )]
    } else {
        vec![]
    }
}
