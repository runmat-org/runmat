use super::super::scalar_facts;
use super::{IntegralCallbackMode, IntegralOptionName};
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, LiteralContext,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn scalar_bound(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(fact) = request.arguments.get(index) else {
        return;
    };
    let admissible = matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) && fact.storage != StorageFact::Sparse
        && !fact.shape.element_count().is_some_and(|count| count != 1);
    let finite_literal = request
        .literals
        .literal_args
        .get(index)
        .and_then(runmat_types::LiteralContext::numeric_from_literal)
        .is_none_or(f64::is_finite);
    if !admissible || !finite_literal {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-BOUND",
            "integral requires finite real scalar bounds",
            index,
        ));
    }
}

fn controls(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<IntegralCallbackMode> {
    let rest = request.arguments.len().saturating_sub(3);
    let mut known_abs_tol = Some(IntegralOptionName::DEFAULT_ABS_TOL);
    let mut known_rel_tol = Some(IntegralOptionName::DEFAULT_REL_TOL);
    let mut known_callback_mode = Some(IntegralCallbackMode::Vectorized);
    if rest == 1 {
        let option = &request.arguments[3];
        if !matches!(
            option.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) || option.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGRAL-OPTIONS",
                "integral expects a scalar options struct or name/value pairs",
                3,
            ));
        }
        return None;
    }
    if !rest.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OPTIONS",
            "integral requires complete name/value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for index in (3..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        let valid_kind = matches!(
            name.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        );
        let valid_shape = match name.kind {
            ValueKindFact::Character => !matches!(
                &name.shape,
                ShapeFact::Shaped { dims }
                    if matches!(dims.first(), Some(runmat_types::DimensionFact::Known(rows)) if *rows != 1)
            ),
            ValueKindFact::String => !name.shape.element_count().is_some_and(|count| count != 1),
            _ => true,
        };
        if !valid_kind || !valid_shape {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGRAL-OPTION-NAME",
                "integral option names must be scalar text",
                index,
            ));
        }
        if let Some(text) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            if let Some(option) = IntegralOptionName::parse(text) {
                let literal = request.literals.literal_args.get(index + 1);
                let value = request.arguments.get(index + 1);
                known_option(
                    option,
                    literal,
                    value,
                    index + 1,
                    diagnostics,
                    &mut known_abs_tol,
                    &mut known_rel_tol,
                    &mut known_callback_mode,
                );
            } else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTEGRAL-OPTION-NAME",
                    "unsupported integral option",
                    index,
                ));
            }
        } else {
            // A runtime option name could overwrite either tolerance.
            known_abs_tol = None;
            known_rel_tol = None;
            known_callback_mode = None;
        }
    }
    if known_abs_tol == Some(0.0) && known_rel_tol == Some(0.0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-TOLERANCES",
            "AbsTol and RelTol cannot both be zero",
            3,
        ));
    }
    if known_abs_tol.is_some_and(|value| value < 0.0)
        || known_rel_tol.is_some_and(|value| value < 0.0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-TOLERANCES",
            "integral tolerances must be nonnegative",
            3,
        ));
    }
    known_callback_mode
}

fn known_option(
    option: IntegralOptionName,
    literal: Option<&LiteralValue>,
    value: Option<&ValueFact>,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    known_abs_tol: &mut Option<f64>,
    known_rel_tol: &mut Option<f64>,
    known_callback_mode: &mut Option<IntegralCallbackMode>,
) {
    let numeric = literal.and_then(|literal| match literal {
        LiteralValue::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        other => LiteralContext::numeric_from_literal(other),
    });
    let known_bad_kind = value.is_some_and(|fact| {
        !matches!(
            fact.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) || fact.storage == StorageFact::Sparse
            || fact.shape.element_count().is_some_and(|count| count != 1)
    });
    let bad = match option {
        IntegralOptionName::AbsTol | IntegralOptionName::RelTol => {
            if option == IntegralOptionName::AbsTol {
                *known_abs_tol = numeric;
            } else {
                *known_rel_tol = numeric;
            }
            numeric.is_some_and(|number| !number.is_finite()) || known_bad_kind
        }
        IntegralOptionName::MaxFunEvals | IntegralOptionName::MaxIntervalCount => {
            numeric.is_some_and(|number| {
                !number.is_finite()
                    || number.fract() != 0.0
                    || number < 5.0
                    || number > usize::MAX as f64
                    || (usize::BITS == 64 && number == usize::MAX as f64)
            }) || known_bad_kind
        }
        IntegralOptionName::ArrayValued => {
            let flag = match literal {
                Some(LiteralValue::Bool(flag)) => Some(*flag),
                _ => numeric.map(|number| number != 0.0),
            };
            flag == Some(true)
                || numeric.is_some_and(|number| number != 0.0 && number != 1.0)
                || known_bad_kind
        }
        IntegralOptionName::Vectorized => {
            let valid = numeric.is_none_or(|number| number == 0.0 || number == 1.0);
            *known_callback_mode =
                numeric.map(|number| IntegralCallbackMode::from_vectorized(number != 0.0));
            !valid || known_bad_kind
        }
    };
    if bad {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OPTION-VALUE",
            "invalid integral option value",
            index,
        ));
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-ARITY",
            "integral requires a callback and two bounds",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OUTPUTS",
            "integral returns one numerical output",
            1,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    scalar_bound(request, 1, &mut diagnostics);
    scalar_bound(request, 2, &mut diagnostics);
    let callback_mode = controls(request, &mut diagnostics);
    let callback_argument = match callback_mode {
        Some(IntegralCallbackMode::Vectorized) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Shaped {
                dims: vec![
                    runmat_types::DimensionFact::Known(1),
                    runmat_types::DimensionFact::Unknown,
                ],
            },
            StorageFact::Dense,
        ),
        Some(IntegralCallbackMode::Scalar) => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let callback = scalar_facts::callback_invocation(request, callback_argument);
    if let Some(callback) = &callback {
        match callback_mode {
            Some(IntegralCallbackMode::Vectorized) => scalar_facts::real_row_batch_callback_result(
                callback,
                &mut diagnostics,
                "RM-CATALOG-INTEGRAL-CALLBACK-RESULT",
            ),
            Some(IntegralCallbackMode::Scalar) => scalar_facts::real_scalar_callback_result(
                callback,
                &mut diagnostics,
                "RM-CATALOG-INTEGRAL-CALLBACK-RESULT",
            ),
            None => {}
        }
        diagnostics.extend(callback.diagnostics.iter().cloned());
    }
    let value = if diagnostics.is_empty() {
        scalar_facts::host_scalar(NumericClass::Double)
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![value]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) = callback {
        contract.effects.0.extend(callback.effects.0);
        contract.capabilities.0.extend(callback.capabilities.0);
    }
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.extend(diagnostics);
    inference
}
