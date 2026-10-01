//! Static facts shared by the fitted-distribution evaluators.

use super::fitted_distribution_model::FittedDistributionFamily;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, standard, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EvaluationForm {
    Density,
    Cumulative,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: EvaluationForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DISTRIBUTION-ARITY",
            "distribution and evaluation points are required",
            arguments.len().saturating_sub(1),
        ));
        return finish_fixed(entry, request, output, diagnostics);
    }
    let mut numeric_end = arguments.len();
    if form == EvaluationForm::Cumulative && arguments.len() > 2 {
        let last = arguments.len() - 1;
        match request.literals.literal_string_at(last) {
            Some(text) if text.eq_ignore_ascii_case("upper") => numeric_end -= 1,
            None if matches!(
                arguments[last].kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) =>
            {
                // A nonliteral trailing text value may be the `upper` control.
                // Its runtime contents, not its scalar shape, select the form.
                return finish_fixed(entry, request, output, diagnostics);
            }
            _ => {}
        }
    }
    let fitted = match &arguments[0].kind {
        ValueKindFact::Object(object) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| !class.is(standard::PROBABILITY_DISTRIBUTION))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DISTRIBUTION-OBJECT",
                    "expected a fitted ProbabilityDistribution object",
                    0,
                ));
            }
            Some(true)
        }
        ValueKindFact::String | ValueKindFact::Character => Some(false),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISTRIBUTION-FAMILY",
                "expected a fitted object or distribution name",
                0,
            ));
            None
        }
    };
    match fitted {
        Some(true) if numeric_end != 2 => diagnostics.push(argument_error(
            "RM-CATALOG-DISTRIBUTION-ARITY",
            "fitted-object form accepts one evaluation array",
            numeric_end.saturating_sub(1),
        )),
        Some(false) => {
            if let Some(keyword) = request.literals.literal_string_at(0) {
                match FittedDistributionFamily::from_keyword(&keyword) {
                    Some(family) if !accepts_named_count(family, numeric_end.saturating_sub(2)) => {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-DISTRIBUTION-ARITY",
                            "incorrect named-family parameter count",
                            numeric_end.saturating_sub(1),
                        ))
                    }
                    None => diagnostics.push(argument_error(
                        "RM-CATALOG-DISTRIBUTION-FAMILY",
                        "unsupported distribution family",
                        0,
                    )),
                    _ => {}
                }
            }
            if numeric_end > 4 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DISTRIBUTION-ARITY",
                    "too many named-family parameters",
                    numeric_end - 1,
                ));
            }
        }
        _ => {}
    }
    let mut shape = ShapeFact::Scalar;
    let mut any_single = false;
    let mut unknown_class = fitted.is_none();
    let mut all_host = true;
    for (index, argument) in arguments[1..numeric_end].iter().enumerate() {
        match &argument.kind {
            ValueKindFact::Numeric(fact) if fact.domain == NumericDomain::Real => {
                any_single |= fact.class == NumericClass::Single;
            }
            ValueKindFact::Logical => {}
            ValueKindFact::Unknown => unknown_class = true,
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-DISTRIBUTION-CLASS",
                "evaluation inputs must be real numeric values",
                index + 1,
            )),
        }
        if argument.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISTRIBUTION-CLASS",
                "sparse evaluation inputs are unsupported",
                index + 1,
            ));
        }
        all_host &= argument.residency == ResidencyFact::Host;
        shape = match broadcast_shape(&shape, &argument.shape) {
            Ok(shape) => shape,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DISTRIBUTION-SHAPE",
                    "evaluation inputs cannot be singleton-broadcast",
                    index + 1,
                ));
                ShapeFact::Unknown
            }
        };
    }
    let class = if any_single {
        Some(NumericClass::Single)
    } else if unknown_class {
        None
    } else {
        Some(NumericClass::Double)
    };
    if let Some(class) = class {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        });
    }
    output.shape = shape;
    output.storage = match (class, output.shape.element_count(), all_host) {
        (Some(NumericClass::Double), Some(1), true) => StorageFact::Scalar,
        (Some(_), Some(_), true) => StorageFact::Dense,
        _ => StorageFact::Unknown,
    };
    output.residency = if all_host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn accepts_named_count(family: FittedDistributionFamily, count: usize) -> bool {
    match family {
        FittedDistributionFamily::Normal | FittedDistributionFamily::Lognormal => count <= 2,
        FittedDistributionFamily::Exponential | FittedDistributionFamily::Poisson => count == 1,
        FittedDistributionFamily::Gamma | FittedDistributionFamily::Weibull => count == 2,
    }
}
