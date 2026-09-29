use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-HEAD-ARITY",
            "head requires an input and an optional positive row count",
            0,
        ));
    }
    if request.arguments.len() == 2 {
        let count = &request.arguments[1];
        if !matches!(
            count.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || (count.shape.element_count().is_some_and(|n| n != 1))
        {
            diagnostics.push(argument_error(
                "RM-HEAD-COUNT",
                "row count must be a positive numeric scalar",
                1,
            ));
        }
        if let Some(value) = request.literals.numeric_at(1) {
            if !value.is_finite()
                || value < 1.0
                || value.fract() != 0.0
                || value >= usize::MAX as f64
            {
                diagnostics.push(argument_error(
                    "RM-HEAD-COUNT",
                    "row count must be a positive integer",
                    1,
                ));
            }
        }
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(input) = request.arguments.first() {
        output = input.clone();
        match &input.kind {
            ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Cell(_) => {
                if !input.is_scalar() {
                    output.shape = selected_shape(
                        &input.shape,
                        if request.arguments.len() == 1 {
                            Some(8)
                        } else {
                            count_literal(request)
                        },
                    );
                }
                output.alias = AliasFact::Unique;
                output.view = ViewFact::Materialized;
            }
            ValueKindFact::Object(_) => {
                // A table/timetable is scalar object storage; its row schema is
                // runtime metadata. Other object classes may have array extents.
                output.shape = if matches!(&input.kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class == &standard::TABLE.owned() || class == &standard::TIMETABLE.owned()))
                {
                    ShapeFact::Scalar
                } else {
                    ShapeFact::Unknown
                };
                output.alias = AliasFact::Unique;
            }
            _ => {
                output = ValueFact::unknown(DynamicReason::RuntimeValue);
            }
        }
        output.residency = input.residency.clone();
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn count_literal(request: &CallRequest) -> Option<usize> {
    match request.literals.literal_args.get(1) {
        Some(LiteralValue::Integer { text, .. }) => text.parse().ok(),
        _ => request.literals.numeric_at(1).and_then(|number| {
            (number.is_finite()
                && number >= 1.0
                && number.fract() == 0.0
                && number < usize::MAX as f64)
                .then_some(number as usize)
        }),
    }
}

fn selected_shape(input: &ShapeFact, count: Option<usize>) -> ShapeFact {
    let Some(mut dims) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    if let Some(first) = dims.first_mut() {
        *first = match (*first, count) {
            (Some(rows), Some(n)) => Some(rows.min(n)),
            _ => None,
        };
    }
    ShapeFact::from(dims)
}
