use super::{
    CrossentropyClassificationMode, CrossentropyNormalizationMode, CrossentropyOptionKey,
    CrossentropyReductionMode,
};
use crate::DeepLearningAxisLayout;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric_input_possible(value: &ValueFact) -> bool {
    if value.storage == StorageFact::Sparse {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        ValueKindFact::Object(object) => {
            object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(runmat_types::standard::DL_ARRAY))
                && match object.properties.get(crate::DLARRAY_DATA_PROPERTY) {
                    Some(data) => numeric_input_possible(data),
                    None => !object.properties_complete,
                }
        }
        _ => false,
    }
}

fn shapes_proven_different(left: &ShapeFact, right: &ShapeFact) -> bool {
    match (left.known_dims(), right.known_dims()) {
        (Some(left), Some(right)) => {
            left.len() != right.len()
                || left.iter().zip(right.iter()).any(|(left, right)| {
                    left.as_ref()
                        .zip(right.as_ref())
                        .is_some_and(|(a, b)| a != b)
                })
        }
        _ => false,
    }
}

fn check_literal_option(
    request: &CallRequest,
    key: CrossentropyOptionKey,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = request.literals.literal_string_at(index) else {
        return;
    };
    let valid = match key {
        CrossentropyOptionKey::ClassificationMode | CrossentropyOptionKey::TargetCategories => {
            CrossentropyClassificationMode::from_name(&value).is_some()
        }
        CrossentropyOptionKey::Reduction => return,
        CrossentropyOptionKey::NormalizationFactor => {
            CrossentropyNormalizationMode::from_name(&value).is_some()
        }
        CrossentropyOptionKey::DataFormat => {
            if value.is_empty() {
                true
            } else if let Some(rank) = request.arguments.first().and_then(|input| {
                let shape = match &input.kind {
                    ValueKindFact::Object(object) => object
                        .properties
                        .get(crate::DLARRAY_DATA_PROPERTY)
                        .map(|data| &data.shape),
                    ValueKindFact::Numeric(_) => Some(&input.shape),
                    _ => None,
                }?;
                shape.known_dims().map(|dims| dims.len())
            }) {
                DeepLearningAxisLayout::parse(&value.to_ascii_uppercase(), rank).is_ok()
            } else {
                return;
            }
        }
        CrossentropyOptionKey::WeightsFormat => {
            let positional = request.arguments.get(2).is_some_and(|value| {
                matches!(
                    &value.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical
                )
            });
            let named = (2..request.arguments.len())
                .step_by(2)
                .filter_map(|index| request.literals.literal_string_at(index))
                .any(|name| {
                    CrossentropyOptionKey::from_name(&name) == Some(CrossentropyOptionKey::Weights)
                });
            if !positional && !named {
                return;
            }
            // The weight shape can be positional or named, and may be dynamic.
            // Runtime validates its rank after selecting the actual weight value.
            value.is_empty()
                || DeepLearningAxisLayout::parse(&value.to_ascii_uppercase(), value.chars().count())
                    .is_ok()
        }
        CrossentropyOptionKey::Weights | CrossentropyOptionKey::Mask => return,
    };
    if !valid {
        diagnostics.push(argument_error(
            "RM-DL-CROSSENTROPY-OPTION-VALUE",
            "invalid crossentropy option value or dimension format",
            index,
        ));
    }
}

fn reduction_role(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<bool> {
    if request.arguments.len() <= 2 {
        return Some(false);
    }
    let start = match request.literals.literal_string_at(2) {
        Some(name) if CrossentropyOptionKey::from_name(&name).is_some() => 2,
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
        let Some(key) = CrossentropyOptionKey::from_name(&name) else {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-OPTION",
                "unsupported crossentropy option",
                index,
            ));
            continue;
        };
        check_literal_option(request, key, index + 1, diagnostics);
        if key == CrossentropyOptionKey::Reduction {
            reduction = request
                .literals
                .literal_string_at(index + 1)
                .and_then(|value| match CrossentropyReductionMode::from_name(&value) {
                    Some(CrossentropyReductionMode::None) => Some(true),
                    Some(CrossentropyReductionMode::Sum) => Some(false),
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
        if shapes_proven_different(&predictions.shape, &targets.shape) {
            diagnostics.push(argument_error(
                "RM-DL-CROSSENTROPY-SHAPE",
                "predictions and targets must have the same shape",
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
