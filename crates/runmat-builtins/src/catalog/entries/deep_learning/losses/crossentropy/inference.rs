use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric_input_possible(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_none_or(|class| class == &runmat_types::standard::DL_ARRAY.owned()),
        _ => false,
    }
}

fn option_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "classificationmode"
            | "targetcategories"
            | "reduction"
            | "normalizationfactor"
            | "dataformat"
            | "weightsformat"
            | "weights"
            | "mask"
    )
}

fn reduction_role(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<bool> {
    if request.arguments.len() <= 2 {
        return Some(false);
    }
    let start = match request.literals.literal_string_at(2) {
        Some(name) if option_name(&name) => 2,
        Some(_) => 3,
        None => match &request.arguments[2].kind {
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Object(_) => 3,
            _ => return None,
        },
    };
    if !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-DL-CROSSENTROPY-OPTIONS",
            "crossentropy name-value options must be paired",
            request.arguments.len().saturating_sub(1),
        ));
        return None;
    }
    let mut reduction = Some(false);
    for index in (start..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            reduction = None;
            continue;
        };
        if !option_name(&name) {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-OPTION",
                "unsupported crossentropy option",
                index,
            ));
            continue;
        }
        if name.eq_ignore_ascii_case("reduction") {
            reduction = request
                .literals
                .literal_string_at(index + 1)
                .and_then(|value| match value.to_ascii_lowercase().as_str() {
                    "none" => Some(true),
                    "sum" => Some(false),
                    _ => {
                        diagnostics.push(argument_error(
                            "RM-DL-CROSSENTROPY-REDUCTION",
                            "Reduction must be 'sum' or 'none'",
                            index + 1,
                        ));
                        None
                    }
                });
        }
    }
    reduction
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-DL-CROSSENTROPY-ARITY",
            "crossentropy requires predictions and targets",
            request.arguments.len(),
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        if !numeric_input_possible(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-INPUT",
                "predictions and targets must be real numeric arrays or dlarray objects",
                index,
            ));
        }
        if request.arguments[index].shape.element_count() == Some(0) {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-EMPTY",
                "predictions and targets must not be empty",
                index,
            ));
        }
    }
    if let [predictions, targets, ..] = request.arguments.as_slice() {
        if predictions
            .shape
            .element_count()
            .zip(targets.shape.element_count())
            .is_some_and(|(left, right)| left != right)
        {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-SIZE",
                "predictions and targets must have the same number of elements",
                1,
            ));
        }
    }

    // Reduction and dlarray wrapping are runtime-dependent unless the named
    // reduction can be read literally. Never infer resident output merely
    // because one input happens to be resident: unsupported provider forms
    // gather and return a host value.
    let reduction_none = reduction_role(request, &mut diagnostics);
    let mut result = match (request.arguments.first(), reduction_none) {
        (Some(predictions), Some(unreduced))
            if matches!(&predictions.kind, ValueKindFact::Numeric(_)) =>
        {
            let class = if unreduced {
                predictions
                    .numeric()
                    .filter(|number| number.class == NumericClass::Single)
                    .map_or(NumericClass::Double, |number| number.class)
            } else {
                NumericClass::Double
            };
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                if unreduced {
                    predictions.shape.clone()
                } else {
                    ShapeFact::Scalar
                },
                StorageFact::Dense,
            )
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if reduction_none == Some(true)
        && request
            .arguments
            .first()
            .is_some_and(|value| value.residency != ResidencyFact::Host)
    {
        result.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, result, diagnostics)
}
