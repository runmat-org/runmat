use super::LassoOption;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 || !request.arguments.len().saturating_sub(2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LASSO-ARITY",
            "lasso requires X, y, and optional name-value pairs",
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
                "RM-CATALOG-LASSO-X",
                "X must be a real numeric predictor matrix",
                0,
            ));
        }
        match &x.shape {
            ShapeFact::Shaped { dims }
                if dims
                    .iter()
                    .skip(2)
                    .any(|dim| matches!(dim, DimensionFact::Known(value) if *value != 1)) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LASSO-X-RANK",
                    "X must be a matrix with only singleton trailing dimensions",
                    0,
                ));
                None
            }
            ShapeFact::Shaped { dims }
                if dims
                    .iter()
                    .skip(2)
                    .all(|dim| matches!(dim, DimensionFact::Known(1))) =>
            {
                if dims.len() < 2 {
                    Some(1)
                } else {
                    dims.get(1).and_then(known)
                }
            }
            ShapeFact::Ranked { rank: 0 | 1 } => Some(1),
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
                "RM-CATALOG-LASSO-Y",
                "y must be a real numeric response vector",
                1,
            ));
        }
    }
    let mut lambdas = None;
    let mut width_depends_on_runtime_option = false;
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        if let Some(name) = request.literals.literal_string_at(index) {
            match LassoOption::from_public_name(&name) {
                Some(LassoOption::Lambda) => {
                    // Multi-element Lambda vectors are sorted and de-duplicated at runtime.
                    lambdas = request
                        .arguments
                        .get(index + 1)
                        .and_then(|value| (value.shape.element_count() == Some(1)).then_some(1));
                }
                Some(LassoOption::DfMax) => width_depends_on_runtime_option = true,
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-LASSO-OPTION",
                    "unsupported lasso option",
                    index,
                )),
            }
        } else {
            // An unknown name can select either Lambda or DFmax at execution.
            width_depends_on_runtime_option = true;
        }
    }
    if width_depends_on_runtime_option {
        lambdas = None;
    }
    // Without a literal Lambda value, the path length depends on runtime controls.
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
