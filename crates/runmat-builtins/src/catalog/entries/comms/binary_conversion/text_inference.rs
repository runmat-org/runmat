//! Static facts for host text conversion. Geometry never stands in for text contents.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TextConversion {
    Dec2Bin,
    Dec2Hex,
    Bin2Dec,
    Hex2Dec,
}

fn known_invalid_cell_text(fact: &ValueFact) -> bool {
    match fact.kind {
        ValueKindFact::String => fact.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Character => fact
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .is_some_and(|rows| rows != 1),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests;

impl TextConversion {
    fn name(self) -> &'static str {
        match self {
            Self::Dec2Bin => "dec2bin",
            Self::Dec2Hex => "dec2hex",
            Self::Bin2Dec => "bin2dec",
            Self::Hex2Dec => "hex2dec",
        }
    }

    fn numeric_to_text(self) -> bool {
        matches!(self, Self::Dec2Bin | Self::Dec2Hex)
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    conversion: TextConversion,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    let numeric_to_text = conversion.numeric_to_text();
    let expected = if numeric_to_text { "one or two" } else { "one" };
    if request.arguments.len() != 1 && !(numeric_to_text && request.arguments.len() == 2) {
        diagnostics.push(argument_error(
            "RM.TEXTCONVERSION.ARITY",
            format!("{} expects {expected} input(s)", conversion.name()),
            0,
        ));
    }
    if let Some(input) = input {
        let admitted = if numeric_to_text {
            matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Character
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
        } else {
            matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            )
        };
        if !admitted {
            diagnostics.push(argument_error(
                "RM.TEXTCONVERSION.INPUT",
                format!("{} input has an unsupported class", conversion.name()),
                0,
            ));
        }
        if !numeric_to_text {
            if let ValueKindFact::Cell(cell) = &input.kind {
                if known_invalid_cell_text(&cell.element)
                    || cell.elements.iter().any(known_invalid_cell_text)
                {
                    diagnostics.push(argument_error(
                        "RM.TEXTCONVERSION.CELL",
                        "cell elements must be text",
                        0,
                    ));
                }
            }
        }
    }
    if numeric_to_text {
        if let Some(width) = request.literals.literal_args.get(1) {
            if runmat_types::LiteralContext::numeric_from_literal(width)
                .is_some_and(|value| !value.is_finite() || value < 0.0 || value.fract() != 0.0)
            {
                diagnostics.push(argument_error(
                    "RM.TEXTCONVERSION.WIDTH",
                    "minDigits must be a nonnegative integer scalar",
                    1,
                ));
            }
        }
        if let Some(control) = request.arguments.get(1) {
            if !matches!(
                control.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || control.shape.is_definitely_nonscalar()
            {
                diagnostics.push(argument_error(
                    "RM.TEXTCONVERSION.WIDTH",
                    "minDigits must be a numeric or logical scalar",
                    1,
                ));
            }
        }
    }
    let output = if numeric_to_text {
        let rows = input.and_then(|fact| fact.shape.element_count());
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![rows, None]),
            StorageFact::Dense,
        )
    } else {
        let shape = match input.map(|fact| &fact.kind) {
            Some(ValueKindFact::String) => {
                input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone())
            }
            Some(ValueKindFact::Character) => {
                let rows = input
                    .and_then(|fact| fact.shape.known_dims())
                    .and_then(|dims| dims.first().copied().flatten());
                ShapeFact::from(vec![rows, Some(1)])
            }
            Some(ValueKindFact::Cell(_)) => ShapeFact::from(vec![
                input.and_then(|fact| fact.shape.element_count()),
                Some(1),
            ]),
            _ => ShapeFact::Unknown,
        };
        let shape = if shape.element_count() == Some(1) {
            ShapeFact::Scalar
        } else {
            shape
        };
        let storage = if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        )
    };
    let mut result = finish_fixed(entry, request, output, diagnostics);
    if !result.diagnostics.is_empty() && input.is_none() {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result
}
