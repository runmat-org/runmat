use super::super::{CrossvalIndMethod, CrossvalIndOption, PartitionCountGrammar};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, EffectKind,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn positive_literal(literal: &LiteralValue) -> Option<Result<usize, ()>> {
    match literal {
        LiteralValue::Unknown => None,
        LiteralValue::Integer { text, .. } => {
            Some(text.parse::<usize>().ok().filter(|n| *n > 0).ok_or(()))
        }
        LiteralValue::Bool(value) => Some(if *value { Ok(1) } else { Err(()) }),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let number = LiteralContext::numeric_from_literal(literal)?;
            Some(
                PartitionCountGrammar::CrossvalInd
                    .positive_from_float(number)
                    .ok_or(()),
            )
        }
        _ => Some(Err(())),
    }
}

fn output(kind: ValueKindFact, n: Option<usize>) -> ValueFact {
    ValueFact::proven(kind, ShapeFact::from(vec![n, Some(1)]), StorageFact::Dense)
}

fn diagnose_numeric_scalar(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let fact = request.arguments.get(index)?;
    if !matches!(
        fact.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) || fact.shape.element_count().is_some_and(|n| n != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-CONTROL",
            "control must be a real numeric scalar",
            index,
        ));
    }
    match request
        .literals
        .literal_args
        .get(index)
        .and_then(positive_literal)
    {
        Some(Ok(value)) => Some(value),
        Some(Err(())) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CROSSVALIND-CONTROL",
                "control must be a positive integer",
                index,
            ));
            None
        }
        None => None,
    }
}

fn diagnose_holdout(
    request: &CallRequest,
    n: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(fact) = request.arguments.get(2) else {
        return;
    };
    if !matches!(
        fact.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) || fact.shape.element_count().is_some_and(|count| count != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-HOLDOUT",
            "HoldOut must be a real numeric scalar",
            2,
        ));
    }
    let valid = match request.literals.literal_args.get(2) {
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<usize>()
            .ok()
            .is_some_and(|count| count > 0 && n.is_none_or(|n| count < n)),
        Some(LiteralValue::Bool(value)) => *value && n.is_none_or(|n| n > 1),
        Some(literal @ (LiteralValue::Number(_) | LiteralValue::Real { .. })) => {
            LiteralContext::numeric_from_literal(literal).is_some_and(|value| {
                value.is_finite()
                    && ((value > 0.0 && value < 1.0 && n.is_none_or(|n| n >= 2))
                        || PartitionCountGrammar::CrossvalInd
                            .positive_from_float(value)
                            .is_some_and(|count| n.is_none_or(|n| count < n)))
            })
        }
        Some(LiteralValue::Unknown) | None => return,
        Some(_) => false,
    };
    if !valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-HOLDOUT",
            "HoldOut must be a fraction in (0,1) or a count in [1,n)",
            2,
        ));
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-ARITY",
            "method and n are required",
            0,
        ));
    }
    let method = request.literals.literal_string_at(0).and_then(|name| {
        let method = CrossvalIndMethod::parse(&name);
        if method.is_none() {
            diagnostics.push(argument_error(
                "RM-CATALOG-CROSSVALIND-METHOD",
                "unsupported method",
                0,
            ));
        }
        method
    });
    let n = diagnose_numeric_scalar(request, 1, &mut diagnostics);
    if n.is_some_and(|n| n > 10_000_000) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-CAPACITY",
            "observation count exceeds supported capacity",
            1,
        ));
    }
    let option_start = match method {
        Some(CrossvalIndMethod::Resubstitution) => Some(2),
        Some(_) => Some(3),
        None => None,
    };
    if let Some(start) = option_start {
        if request.arguments.len() < start || !(request.arguments.len() - start).is_multiple_of(2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CROSSVALIND-ARITY",
                "partition value and paired options are required",
                start,
            ));
        }
        if start == 3 {
            let control = if method == Some(CrossvalIndMethod::HoldOut) {
                diagnose_holdout(request, n, &mut diagnostics);
                None
            } else {
                diagnose_numeric_scalar(request, 2, &mut diagnostics)
            };
            if method == Some(CrossvalIndMethod::KFold)
                && control.is_some_and(|k| k < 2 || n.is_some_and(|n| k > n))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CROSSVALIND-KFOLD",
                    "KFold count must be in [2,n]",
                    2,
                ));
            }
            if method == Some(CrossvalIndMethod::LeaveMOut)
                && control.zip(n).is_some_and(|(m, n)| m >= n)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CROSSVALIND-LEAVEMOUT",
                    "LeaveMOut count must be less than n",
                    2,
                ));
            }
        }
        for index in (start..request.arguments.len()).step_by(2) {
            let Some(name) = request.literals.literal_string_at(index) else {
                continue;
            };
            let option = CrossvalIndOption::parse(&name);
            if option.is_none() || method == Some(CrossvalIndMethod::Resubstitution) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CROSSVALIND-OPTION",
                    "unsupported option for method",
                    index,
                ));
            }
            if option == Some(CrossvalIndOption::Min) {
                diagnose_numeric_scalar(request, index + 1, &mut diagnostics);
            } else if option == Some(CrossvalIndOption::Classes) {
                if let Some(fact) = request.arguments.get(index + 1) {
                    let count = match &fact.kind {
                        ValueKindFact::Numeric(_)
                        | ValueKindFact::Logical
                        | ValueKindFact::String => fact.shape.element_count(),
                        ValueKindFact::Character => fact
                            .shape
                            .known_dims()
                            .and_then(|dims| dims.first().copied().flatten()),
                        ValueKindFact::Unknown => None,
                        _ => {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-CROSSVALIND-CLASSES",
                                "Classes must be numeric, logical, or text labels",
                                index + 1,
                            ));
                            None
                        }
                    };
                    if count.zip(n).is_some_and(|(count, n)| count != n) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-CROSSVALIND-CLASSES",
                            "Classes must have one label per observation",
                            index + 1,
                        ));
                    }
                }
            }
        }
    }
    if method == Some(CrossvalIndMethod::KFold)
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-OUTPUTS",
            "KFold returns at most one output",
            0,
        ));
    } else if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CROSSVALIND-OUTPUTS",
            "partition methods return at most two outputs",
            0,
        ));
    }
    let fact = match method {
        Some(CrossvalIndMethod::KFold) => output(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            n,
        ),
        Some(_) => output(ValueKindFact::Logical, n),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let outputs = if diagnostics.is_empty() {
        vec![fact.clone(), fact]
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 2]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    if method == Some(CrossvalIndMethod::Resubstitution) {
        contract.effects.0.remove(&EffectKind::Randomness);
    }
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
