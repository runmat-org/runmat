use crate::catalog::inference::{argument_error, default_double_scalar, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CallableIdentity, EffectKind, LiteralContext, LiteralValue,
    NumericDomain, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMEIT-ARITY",
            "timeit requires a callback and optional output count",
            request.arguments.len(),
        ));
    }
    if let Some(callback) = request.arguments.first() {
        let admissible = match callback.kind {
            ValueKindFact::Callable(_) | ValueKindFact::Unknown => true,
            ValueKindFact::String => callback
                .shape
                .element_count()
                .is_none_or(|count| count == 1),
            ValueKindFact::Character => callback
                .shape
                .known_dims()
                .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows == 1)),
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CATALOG-TIMEIT-CALLABLE",
                "timeit requires a scalar text handle or callable",
                0,
            ));
        }
        if let Some(LiteralValue::String(text) | LiteralValue::Character(text)) =
            request.literals.literal_args.first()
        {
            let trimmed = text.trim();
            if !trimmed.starts_with('@') || trimmed[1..].trim().is_empty() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TIMEIT-HANDLE-TEXT",
                    "timeit text handle must start with @ and name a function",
                    0,
                ));
            }
        }
    }
    if let Some(count) = request.arguments.get(1) {
        match count.kind {
            ValueKindFact::Numeric(numeric) => {
                if numeric.domain == NumericDomain::Complex
                    || count
                        .shape
                        .element_count()
                        .is_some_and(|elements| elements != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TIMEIT-OUTPUT-COUNT",
                        "timeit output count must be a nonnegative numeric scalar",
                        1,
                    ));
                }
            }
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-TIMEIT-OUTPUT-COUNT",
                "timeit output count must be a nonnegative numeric scalar",
                1,
            )),
        }
        if request
            .literals
            .literal_args
            .get(1)
            .is_some_and(invalid_count_literal)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TIMEIT-OUTPUT-COUNT",
                "timeit output count must be finite, integral, nonnegative, and representable",
                1,
            ));
        }
    }
    let mut inferred = finish_fixed(entry, request, default_double_scalar(), diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Callable(callable),
        ..
    }) = request.arguments.first()
    {
        let known_catalog_callback = matches!(
            &callable.identity,
            Some(CallableIdentity::Builtin(identity))
                if crate::builtin_catalog_entry_by_name(&identity.0).is_some()
        );
        if !known_catalog_callback {
            inferred.effects.0.insert(EffectKind::Unknown);
        }
        let requested = if request.arguments.len() == 1 {
            Some(1)
        } else {
            request.literals.literal_args.get(1).and_then(known_count)
        };
        if let Some(count) = requested {
            let callback = crate::infer_callable_call(
                callable,
                &CallRequest {
                    arguments: Vec::new(),
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
                },
            );
            inferred.effects.0.extend(callback.effects.0);
            inferred.capabilities.0.extend(callback.capabilities.0);
            for mut diagnostic in callback.diagnostics {
                diagnostic.argument = Some(0);
                inferred.diagnostics.push(diagnostic);
            }
        } else {
            inferred.effects.0.insert(EffectKind::Unknown);
            inferred
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
        }
    } else {
        inferred.effects.0.insert(EffectKind::Unknown);
    }
    inferred
}

fn known_count(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { .. } => usize::try_from(literal.integer_value_i128()?).ok(),
        LiteralValue::Number(value)
            if value.is_finite() && *value >= 0.0 && value.fract() == 0.0 =>
        {
            fits_platform_usize(*value).then_some(*value as usize)
        }
        LiteralValue::Real { text, .. } => {
            let value = text.parse::<f64>().ok()?;
            (value.is_finite()
                && value >= 0.0
                && value.fract() == 0.0
                && fits_platform_usize(value))
            .then_some(value as usize)
        }
        _ => None,
    }
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn invalid_count_literal(literal: &LiteralValue) -> bool {
    matches!(
        literal,
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. }
    ) && known_count(literal).is_none()
}
