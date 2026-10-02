mod path_domain;

use super::super::scalar_facts;
use super::{IntegralCallbackMode, IntegralOptionName};
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use path_domain::result_domain;
pub(super) use path_domain::{path_domain, PathDomain};
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
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) && fact.storage != StorageFact::Sparse
        && fact.shape.element_count().is_none_or(|count| count == 1);
    let admitted_literal = request
        .literals
        .literal_args
        .get(index)
        .and_then(runmat_types::LiteralContext::numeric_from_literal)
        .is_none_or(|value| !value.is_nan());
    if !admissible || !admitted_literal {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-BOUND",
            "integral requires numeric scalar bounds that are not NaN",
            index,
        ));
    }
}

struct ControlFacts {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    callback_mode: Option<IntegralCallbackMode>,
    array_valued: Option<bool>,
}

fn controls(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> (Option<IntegralCallbackMode>, Option<bool>) {
    let rest = request.arguments.len().saturating_sub(3);
    let mut facts = ControlFacts {
        abs_tol: Some(IntegralOptionName::DEFAULT_ABS_TOL),
        rel_tol: Some(IntegralOptionName::DEFAULT_REL_TOL),
        callback_mode: Some(IntegralCallbackMode::Vectorized),
        array_valued: Some(false),
    };
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
        return (None, None);
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
            ValueKindFact::String => name.shape.element_count().is_none_or(|count| count == 1),
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
            if let Some(option) = IntegralOptionName::parse(&text) {
                let literal = request.literals.literal_args.get(index + 1);
                let value = request.arguments.get(index + 1);
                known_option(option, literal, value, index + 1, diagnostics, &mut facts);
            } else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTEGRAL-OPTION-NAME",
                    "unsupported integral option",
                    index,
                ));
            }
        } else {
            // A runtime option name could overwrite either tolerance.
            facts.abs_tol = None;
            facts.rel_tol = None;
            facts.callback_mode = None;
            facts.array_valued = None;
        }
    }
    if facts.abs_tol == Some(0.0) && facts.rel_tol == Some(0.0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-TOLERANCES",
            "AbsTol and RelTol cannot both be zero",
            3,
        ));
    }
    if facts.abs_tol.is_some_and(|value| value < 0.0)
        || facts.rel_tol.is_some_and(|value| value < 0.0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-TOLERANCES",
            "integral tolerances must be nonnegative",
            3,
        ));
    }
    (facts.callback_mode, facts.array_valued)
}

fn known_option(
    option: IntegralOptionName,
    literal: Option<&LiteralValue>,
    value: Option<&ValueFact>,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    facts: &mut ControlFacts,
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
        IntegralOptionName::AbsTol
        | IntegralOptionName::RelTol
        | IntegralOptionName::AbsoluteTolerance
        | IntegralOptionName::RelativeTolerance => {
            if matches!(
                option,
                IntegralOptionName::AbsTol | IntegralOptionName::AbsoluteTolerance
            ) {
                facts.abs_tol = numeric;
            } else {
                facts.rel_tol = numeric;
            }
            numeric.is_some_and(|number| !number.is_finite()) || known_bad_kind
        }
        IntegralOptionName::MaxFunEvals => {
            numeric.is_some_and(|number| {
                !number.is_finite()
                    || number.fract() != 0.0
                    || number < 5.0
                    || number > usize::MAX as f64
                    || (usize::BITS == 64 && number == usize::MAX as f64)
            }) || known_bad_kind
        }
        IntegralOptionName::MaxIntervalCount => {
            numeric.is_some_and(|number| {
                !number.is_finite()
                    || number.fract() != 0.0
                    || number < 1.0
                    || number > usize::MAX as f64
                    || (usize::BITS == 64 && number == usize::MAX as f64)
            }) || known_bad_kind
        }
        IntegralOptionName::ArrayValued => {
            let flag = match literal {
                Some(LiteralValue::Bool(flag)) => Some(*flag),
                _ => numeric.map(|number| number != 0.0),
            };
            facts.array_valued = flag;
            numeric.is_some_and(|number| number != 0.0 && number != 1.0) || known_bad_kind
        }
        IntegralOptionName::Vectorized => {
            let valid = numeric.is_none_or(|number| number == 0.0 || number == 1.0);
            facts.callback_mode =
                numeric.map(|number| IntegralCallbackMode::from_vectorized(number != 0.0));
            !valid || known_bad_kind
        }
        IntegralOptionName::Waypoints => value.is_some_and(|fact| {
            let numeric = matches!(
                fact.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            );
            let nonvector = match &fact.shape {
                ShapeFact::Shaped { dims } => {
                    let rows = dims.first();
                    let cols = dims.get(1);
                    matches!((rows, cols), (Some(runmat_types::DimensionFact::Known(r)), Some(runmat_types::DimensionFact::Known(c))) if *r > 1 && *c > 1)
                        || dims.iter().skip(2).any(|dim| matches!(dim, runmat_types::DimensionFact::Known(axis) if *axis != 1))
                }
                _ => false,
            };
            !numeric || fact.storage == StorageFact::Sparse || nonvector
        }),
    };
    if bad {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OPTION-VALUE",
            "invalid integral option value",
            index,
        ));
    }
}

fn definitely_invalid_row(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Unknown => false,
        ShapeFact::Scalar => true,
        ShapeFact::Ranked { rank } => *rank != 2,
        ShapeFact::Shaped { dims } => {
            dims.len() != 2
                || matches!(dims.first(), Some(runmat_types::DimensionFact::Known(rows)) if *rows != 1)
        }
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
    let path_domain = path_domain(request);
    let (callback_mode, array_valued) = controls(request, &mut diagnostics);
    let effective_mode = if array_valued == Some(true) {
        Some(IntegralCallbackMode::Scalar)
    } else if array_valued == Some(false) {
        callback_mode
    } else {
        None
    };
    let callback_argument = match (effective_mode, path_domain) {
        (Some(IntegralCallbackMode::Vectorized), PathDomain::Real | PathDomain::Complex) => {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: if path_domain == PathDomain::Complex {
                        NumericDomain::Complex
                    } else {
                        NumericDomain::Real
                    },
                }),
                ShapeFact::Shaped {
                    dims: vec![
                        runmat_types::DimensionFact::Known(1),
                        runmat_types::DimensionFact::Unknown,
                    ],
                },
                StorageFact::Dense,
            )
        }
        (Some(IntegralCallbackMode::Scalar), PathDomain::Real | PathDomain::Complex) => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: if path_domain == PathDomain::Complex {
                    NumericDomain::Complex
                } else {
                    NumericDomain::Real
                },
            }))
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let callback = scalar_facts::callback_invocation(request, callback_argument);
    if let Some(callback) = &callback {
        match (array_valued, effective_mode) {
            (Some(true), _) => {
                if let Some(result) = callback.outputs.first() {
                    let numeric = matches!(
                        result.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
                    );
                    if !numeric || result.storage == StorageFact::Sparse {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-INTEGRAL-CALLBACK-RESULT",
                            "array-valued callback must return a numeric array",
                            0,
                        ));
                    }
                }
            }
            (_, Some(IntegralCallbackMode::Vectorized)) => {
                if let Some(result) = callback.outputs.first() {
                    let numeric = matches!(
                        result.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
                    );
                    if !numeric
                        || result.storage == StorageFact::Sparse
                        || definitely_invalid_row(&result.shape)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-INTEGRAL-CALLBACK-RESULT",
                            "vectorized callback must return a numeric row vector",
                            0,
                        ));
                    }
                }
            }
            (_, Some(IntegralCallbackMode::Scalar)) => {
                if let Some(result) = callback.outputs.first() {
                    if !matches!(
                        result.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
                    ) || result.storage == StorageFact::Sparse
                        || result.shape.element_count().is_some_and(|count| count != 1)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-INTEGRAL-CALLBACK-RESULT",
                            "scalar callback must return one numeric value",
                            0,
                        ));
                    }
                }
            }
            (_, None) => {}
        }
        diagnostics.extend(callback.diagnostics.iter().cloned());
    }
    let value = if diagnostics.is_empty() {
        if array_valued == Some(true) {
            callback
                .as_ref()
                .and_then(|result| result.outputs.first())
                .map_or_else(
                    || ValueFact::unknown(DynamicReason::RuntimeValue),
                    |result| {
                        let callback_domain = match result.kind {
                            ValueKindFact::Numeric(NumericFact { domain, .. }) => domain,
                            ValueKindFact::Logical => NumericDomain::Real,
                            _ => return ValueFact::unknown(DynamicReason::UnresolvedCallable),
                        };
                        let Some(domain) = result_domain(callback_domain, path_domain) else {
                            return ValueFact::unknown(DynamicReason::RuntimeValue);
                        };
                        ValueFact::proven(
                            ValueKindFact::Numeric(NumericFact {
                                class: NumericClass::Double,
                                domain,
                            }),
                            result.shape.clone(),
                            StorageFact::Dense,
                        )
                    },
                )
        } else if array_valued == Some(false) {
            let domain = callback
                .as_ref()
                .and_then(|call| call.outputs.first())
                .and_then(|result| match result.kind {
                    ValueKindFact::Numeric(NumericFact { domain, .. }) => Some(domain),
                    ValueKindFact::Logical => Some(NumericDomain::Real),
                    _ => None,
                });
            domain
                .and_then(|domain| result_domain(domain, path_domain))
                .map_or_else(
                    || ValueFact::unknown(DynamicReason::UnresolvedCallable),
                    |domain| {
                        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain,
                        }))
                    },
                )
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
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
