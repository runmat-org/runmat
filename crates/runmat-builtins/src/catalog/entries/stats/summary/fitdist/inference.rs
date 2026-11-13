use super::super::fitted_distribution_model::{FittedDistributionFamily, FittedFitOption};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DimensionFact, NumericDomain, ObjectFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    if arguments.len() < 2 || !(arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITDIST-ARITY",
            "fitdist requires observations, a family, and paired options",
            arguments.len().saturating_sub(1),
        ));
    }
    if let Some(data) = arguments.first() {
        if !real_or_logical(data) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITDIST-CLASS",
                "observations must be a real numeric or logical vector",
                0,
            ));
        }
        if known_nonvector(&data.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITDIST-SHAPE",
                "observations must be a vector",
                0,
            ));
        }
    }
    if let Some(family) = arguments.get(1) {
        if !matches!(
            family.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITDIST-FAMILY",
                "distribution family must be text",
                1,
            ));
        }
        if request
            .literals
            .literal_string_at(1)
            .is_some_and(|keyword| FittedDistributionFamily::from_keyword(&keyword).is_none())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITDIST-FAMILY",
                "unsupported fitted distribution family",
                1,
            ));
        }
    }
    let mut frequency_seen = false;
    for index in (2..arguments.len()).step_by(2) {
        if !matches!(
            arguments[index].kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITDIST-OPTION",
                "option names must be text",
                index,
            ));
        }
        if let Some(keyword) = request.literals.literal_string_at(index) {
            if FittedFitOption::from_keyword(&keyword) != Some(FittedFitOption::Frequency) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FITDIST-OPTION",
                    "only Frequency is supported",
                    index,
                ));
            } else if frequency_seen {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FITDIST-OPTION",
                    "Frequency may be specified only once",
                    index,
                ));
            } else {
                frequency_seen = true;
            }
        }
        if let Some(weight) = arguments.get(index + 1) {
            if !real_or_logical(weight) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FITDIST-FREQUENCY",
                    "Frequency must be a dense real numeric or logical vector",
                    index + 1,
                ));
            }
            if known_nonvector(&weight.shape) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FITDIST-FREQUENCY",
                    "Frequency must be a vector",
                    index + 1,
                ));
            }
            if let Some(data) = arguments.first() {
                if data
                    .shape
                    .element_count()
                    .zip(weight.shape.element_count())
                    .is_some_and(|(left, right)| left != right)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FITDIST-FREQUENCY",
                        "Frequency requires one count per observation",
                        index + 1,
                    ));
                }
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::PROBABILITY_DISTRIBUTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn real_or_logical(value: &ValueFact) -> bool {
    value.storage != StorageFact::Sparse
        && matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(runmat_types::NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        )
}

fn known_nonvector(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(size) if *size > 1)).count() > 1)
}
