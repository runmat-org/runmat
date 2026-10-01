use super::{LassoglmDistribution, LassoglmOption};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 || request.arguments.len().saturating_sub(3) % 2 != 0 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LASSOGLM-ARITY",
            "lassoglm requires X, Y, distr, and optional name-value pairs",
            request.arguments.len(),
        ));
    }
    let predictors = request.arguments.first().and_then(|x| {
        if !matches!(
            x.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LASSOGLM-X",
                "X must be a real numeric predictor matrix",
                0,
            ));
        }
        match &x.shape {
            ShapeFact::Shaped { dims } if dims.len() == 2 => dims.get(1).and_then(known),
            ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LASSOGLM-X-RANK",
                    "X must be two-dimensional",
                    0,
                ));
                None
            }
            _ => None,
        }
    });
    if let Some(y) = request.arguments.get(1) {
        if !matches!(
            y.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LASSOGLM-Y",
                "Y must be real numeric",
                1,
            ));
        }
    }
    if let Some(name) = request.literals.literal_string_at(2) {
        if LassoglmDistribution::from_public_name(name).is_none() {
            diagnostics.push(argument_error(
                "RM-CATALOG-LASSOGLM-DISTR",
                "distribution must be normal, binomial, or poisson",
                2,
            ));
        }
    }
    let mut lambdas = None;
    let mut unknown_option_name = false;
    for index in (3..request.arguments.len().saturating_sub(1)).step_by(2) {
        if let Some(name) = request.literals.literal_string_at(index) {
            match LassoglmOption::from_public_name(name) {
                Some(LassoglmOption::Lambda) => {
                    // Runtime sorts and de-duplicates multi-element Lambda vectors.
                    lambdas = request
                        .arguments
                        .get(index + 1)
                        .and_then(|value| (value.shape.element_count() == Some(1)).then_some(1));
                }
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-LASSOGLM-OPTION",
                    "unsupported lassoglm option",
                    index,
                )),
            }
        } else {
            // A later runtime option can replace the statically visible Lambda.
            unknown_option_name = true;
        }
    }
    if unknown_option_name {
        lambdas = None;
    }
    let coefficients = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![predictors, lambdas]),
        StorageFact::Dense,
    );
    let coefficients = if diagnostics.is_empty() {
        coefficients
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let info = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(BTreeMap::new(), false)),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut contract = CallContract::fixed(vec![coefficients, info]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn known(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}
