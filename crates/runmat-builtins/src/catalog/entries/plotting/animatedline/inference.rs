use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    // A leading scalar numeric value might be an existing axes handle. Its value
    // cannot be classified from its numeric shape; leave that decision to state.
    if let Some(first) = request.arguments.first() {
        if first.residency == ResidencyFact::Host
            && !matches!(
                first.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(_)
                    | ValueKindFact::String
                    | ValueKindFact::Character
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ANIMATEDLINE-INPUT",
                "expected coordinates, an axes handle, or property names",
                0,
            ));
        }
        // Text-first syntax has no leading axes/coordinate ambiguity.
        if matches!(first.kind, ValueKindFact::String | ValueKindFact::Character) {
            if !request.arguments.len().is_multiple_of(2) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ANIMATEDLINE-PAIRS",
                    "name-value arguments must come in pairs",
                    request.arguments.len() - 1,
                ));
            }
            for index in (0..request.arguments.len().saturating_sub(1)).step_by(2) {
                let Some(name) = request.literals.literal_string_at(index) else {
                    continue;
                };
                let input = &request.arguments[index + 1];
                match name.trim().to_ascii_lowercase().as_str() {
                    "xdata" | "ydata" | "zdata" => {
                        if input.kind == ValueKindFact::Logical
                            && input.residency == ResidencyFact::Host
                        {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-ANIMATEDLINE-COORDINATE",
                                "coordinates must not be logical data",
                                index + 1,
                            ));
                        }
                        // Runtime uses its first two extents, admitting higher
                        // dimensions when either of these extents is singleton.
                        if input.shape.known_dims().is_some_and(|dims| {
                            dims.len() >= 2
                                && dims[0].is_some_and(|n| n != 1)
                                && dims[1].is_some_and(|n| n != 1)
                        }) {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-ANIMATEDLINE-VECTOR",
                                "coordinate data must have a singleton first or second extent",
                                index + 1,
                            ));
                        }
                    }
                    "maximumnumpoints" => {
                        if input.shape.element_count().is_some_and(|n| n != 1)
                            || request.literals.numeric_at(index + 1).is_some_and(|n| {
                                n <= 0.0 || n.is_nan() || (n.is_finite() && n.fract() != 0.0)
                            })
                        {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-ANIMATEDLINE-MAXIMUM",
                                "MaximumNumPoints must be a positive integer or positive infinity",
                                index + 1,
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
