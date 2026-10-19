use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ResidencyFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 5 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCATTERPLOT-ARITY",
            "scatterplot expects samples and at most four controls",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid = matches!(
            &input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single | NumericClass::Double,
                    ..
                })
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        );
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SCATTERPLOT-SAMPLES",
                "x must be real or complex floating samples, or real typed integers admitted by RunMat mode",
                0,
            ));
        }
    }
    for (index, minimum) in [(1, 1.0), (2, 0.0)] {
        if let Some(control) = request.arguments.get(index) {
            if control
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
                || matches!(control.residency, ResidencyFact::Remote { .. })
                || !matches!(
                    control.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Real,
                            ..
                        })
                )
                || request.literals.numeric_at(index).is_some_and(|number| {
                    !number.is_finite() || number.fract() != 0.0 || number < minimum
                })
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SCATTERPLOT-CONTROL",
                    "n and offset must be scalar whole values in range",
                    index,
                ));
            }
        }
    }
    if let (Some(n), Some(offset)) = (
        request.literals.numeric_at(1),
        request.literals.numeric_at(2),
    ) {
        if n.is_finite() && offset.is_finite() && offset >= n {
            diagnostics.push(argument_error(
                "RM-CATALOG-SCATTERPLOT-OFFSET",
                "offset must be less than n",
                2,
            ));
        }
    }
    if let Some(marker) = request.arguments.get(3) {
        if !matches!(
            marker.kind,
            ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SCATTERPLOT-MARKER",
                "marker must be a LineSpec text scalar",
                3,
            ));
        }
    }
    if let Some(figure) = request.arguments.get(4) {
        if figure.shape.element_count().is_some_and(|count| count != 1)
            || !matches!(
                figure.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real
                    })
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SCATTERPLOT-FIGURE",
                "scatfig must be a scalar double Figure handle",
                4,
            ));
        }
    }
    // The rendering state is local and cannot be inferred from a numeric
    // figure value. The returned scatter handle is always a host double.
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
