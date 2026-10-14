use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FREQZ-ARITY",
            "freqz requires b, a, optional n, and optional fs",
            0,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        let value = &request.arguments[index];
        if invalid_coefficients(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FREQZ-COEFFICIENTS",
                "filter coefficients must be a nonempty numeric vector",
                index,
            ));
        }
    }
    let n = if request.arguments.len() <= 2 {
        Some(512)
    } else {
        let control = &request.arguments[2];
        let empty_default = matches!(
            request.literals.literal_args.get(2),
            Some(LiteralValue::Empty)
        );
        if !empty_default && invalid_control(control) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FREQZ-N",
                "n must be a host integer scalar at least two",
                2,
            ));
        }
        match request.literals.literal_args.get(2) {
            Some(LiteralValue::Empty) => Some(512),
            Some(literal) => match count_literal(literal) {
                Some(Some(count)) if count >= 2 => Some(count),
                Some(Some(_)) | None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FREQZ-N",
                        "n must be at least two",
                        2,
                    ));
                    None
                }
                Some(None) => None,
            },
            None => None,
        }
    };
    if let Some(fs) = request.arguments.get(3) {
        if invalid_control(fs)
            || request
                .literals
                .literal_args
                .get(3)
                .is_some_and(invalid_fs_literal)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FREQZ-FS",
                "fs must be a positive finite host scalar",
                3,
            ));
        }
    }
    let class = if request.arguments.iter().take(2).any(|value| {
        matches!(
            &value.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                ..
            })
        )
    }) {
        Some(NumericClass::Single)
    } else if request
        .arguments
        .iter()
        .take(2)
        .all(|value| !matches!(value.kind, ValueKindFact::Unknown))
    {
        Some(NumericClass::Double)
    } else {
        None
    };
    let shape = ShapeFact::from(vec![n, Some(1)]);
    let response = class.map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |class| {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Complex,
                }),
                shape.clone(),
                StorageFact::Dense,
            )
        },
    );
    let frequency = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(if diagnostics.is_empty() {
        vec![response, frequency]
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 2]
    });
    // The runtime pads surplus requested slots; padding is not a response.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(
        DynamicReason::UnsupportedRepresentation,
    )));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
    {
        contract.effects.0.remove(&EffectKind::UserInterface);
        contract.effects.0.remove(&EffectKind::HostCallback);
        contract
            .capabilities
            .0
            .remove(&CapabilityRequirement::UserInterface);
    }
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn invalid_coefficients(value: &ValueFact) -> bool {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        || value.shape.element_count() == Some(0)
        || matches!(value.residency, ResidencyFact::Remote { .. })
    {
        return true;
    }
    matches!(&value.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(size) if *size > 1)).count() > 1)
}

fn invalid_control(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || value.shape.is_definitely_nonscalar()
        || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        || !matches!(
            value.residency,
            ResidencyFact::Host | ResidencyFact::Unknown
        )
}

/// `Some(None)` is genuinely dynamic; `None` is a proven invalid literal.
fn count_literal(value: &LiteralValue) -> Option<Option<usize>> {
    match value {
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => Some(None),
        LiteralValue::Bool(value) => Some(Some(usize::from(*value))),
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some),
        LiteralValue::Number(n) => floating_count(*n).map(Some),
        LiteralValue::Real {
            text,
            class: NumericClass::Single,
        } => text
            .parse::<f32>()
            .ok()
            .and_then(|n| floating_count(f64::from(n)))
            .map(Some),
        LiteralValue::Real { text, .. } => {
            text.parse::<f64>().ok().and_then(floating_count).map(Some)
        }
        _ => None,
    }
}

fn floating_count(value: f64) -> Option<usize> {
    let rounded = value.round();
    (value.is_finite()
        && (rounded - value).abs() <= 1e-9
        && rounded >= 0.0
        && rounded < usize::MAX as f64)
        .then_some(rounded as usize)
}

fn invalid_fs_literal(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => false,
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .is_some_and(|n| n <= 0 || value.integer_is_exact_binary64() == Some(false)),
        LiteralValue::Number(n) => !n.is_finite() || *n <= 0.0,
        LiteralValue::Real {
            text,
            class: NumericClass::Single,
        } => text
            .parse::<f32>()
            .is_ok_and(|n| !n.is_finite() || n <= 0.0),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|n| !n.is_finite() || n <= 0.0),
        LiteralValue::Bool(value) => !value,
        _ => true,
    }
}
