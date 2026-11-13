use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterKind {
    Low,
    High,
    Bandpass,
    Stop,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BUTTER-ARITY",
            "butter requires order, cutoff, and at most two options",
            0,
        ));
    }
    let order = request.arguments.first().and_then(|value| {
        if invalid_order_fact(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BUTTER-ORDER",
                "order must be a positive integer scalar",
                0,
            ));
        }
        request
            .literals
            .literal_args
            .first()
            .and_then(|literal| match order_literal(literal) {
                Ok(value) => value,
                Err(()) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-BUTTER-ORDER",
                        "order must be an integer from 1 through 1024",
                        0,
                    ));
                    None
                }
            })
    });
    let cutoff_count = request.arguments.get(1).and_then(|value| {
        if invalid_cutoff_fact(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BUTTER-CUTOFF",
                "Wn must be a real scalar or two-element vector",
                1,
            ));
        }
        value.shape.element_count()
    });
    if cutoff_count.is_some_and(|count| count != 1 && count != 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BUTTER-CUTOFF",
            "Wn must contain one or two values",
            1,
        ));
    }
    let mut kind = match cutoff_count {
        Some(1) => Some(FilterKind::Low),
        Some(2) => Some(FilterKind::Bandpass),
        _ => None,
    };
    let mut analog = Some(false);
    let mut seen_kind = false;
    let mut seen_analog = false;
    for index in 2..request.arguments.len() {
        let value = &request.arguments[index];
        if !matches!(
            value.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) || (matches!(value.kind, ValueKindFact::String)
            && value.shape.is_definitely_nonscalar())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BUTTER-OPTION",
                "filter options must be scalar text",
                index,
            ));
        }
        match request
            .literals
            .literal_args
            .get(index)
            .and_then(text_literal)
        {
            Some(token) => match decode_option(token) {
                Some(OptionToken::Analog) => {
                    if seen_analog {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-BUTTER-OPTION",
                            "analog option may appear once",
                            index,
                        ));
                    }
                    seen_analog = true;
                    analog = Some(true);
                }
                Some(OptionToken::Kind(parsed)) => {
                    if seen_kind {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-BUTTER-OPTION",
                            "filter type may appear once",
                            index,
                        ));
                    }
                    seen_kind = true;
                    kind = Some(parsed);
                }
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-BUTTER-OPTION",
                    "unrecognized filter option",
                    index,
                )),
            },
            None => {
                kind = None;
                analog = None;
            }
        }
    }
    if let (Some(count), Some(kind)) = (cutoff_count, kind) {
        let expected = if matches!(kind, FilterKind::Low | FilterKind::High) {
            1
        } else {
            2
        };
        if count != expected {
            diagnostics.push(argument_error(
                "RM-CATALOG-BUTTER-CUTOFF",
                "filter type and Wn length are incompatible",
                1,
            ));
        }
    }
    if request
        .literals
        .literal_args
        .get(1)
        .is_some_and(|literal| invalid_cutoff_literal(literal, analog))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-BUTTER-CUTOFF",
            "cutoff must be finite, positive, and digital cutoff less than one",
            1,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 4)
    {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-BUTTER-OUTPUTS",
            "butter returns at most four outputs",
        ));
    }
    let effective_order = order.zip(kind).and_then(|(order, kind)| {
        order.checked_mul(if matches!(kind, FilterKind::Bandpass | FilterKind::Stop) {
            2
        } else {
            1
        })
    });
    let width = effective_order.and_then(|n| n.checked_add(1));
    let coefficient = || {
        numeric(
            ShapeFact::from(vec![Some(1), width]),
            NumericDomain::Real,
            StorageFact::Dense,
        )
    };
    let column = |rows| uncertain_numeric(ShapeFact::from(vec![rows, Some(1)]), StorageFact::Dense);
    let scalar = uncertain_numeric(ShapeFact::Scalar, StorageFact::Unknown);
    let zeros = match (analog, kind) {
        (Some(false), _) => effective_order,
        (Some(true), Some(FilterKind::Low)) => Some(0),
        (Some(true), Some(FilterKind::High | FilterKind::Bandpass)) => order,
        (Some(true), Some(FilterKind::Stop)) => effective_order,
        _ => None,
    };
    let requested = request.outputs.requested.known_count();
    let output_facts = match requested {
        Some(3) => vec![column(zeros), column(effective_order), scalar],
        Some(4) => vec![
            numeric(
                ShapeFact::from(vec![effective_order, effective_order]),
                NumericDomain::Real,
                StorageFact::Dense,
            ),
            numeric(
                ShapeFact::from(vec![effective_order, Some(1)]),
                NumericDomain::Real,
                StorageFact::Dense,
            ),
            numeric(
                ShapeFact::from(vec![Some(1), effective_order]),
                NumericDomain::Real,
                StorageFact::Dense,
            ),
            numeric(ShapeFact::Scalar, NumericDomain::Real, StorageFact::Scalar),
        ],
        Some(0..=2) => vec![coefficient(), coefficient()],
        _ => vec![ValueFact::unknown(DynamicReason::RuntimeValue); 4],
    };
    let mut contract = CallContract::fixed(if diagnostics.is_empty() {
        output_facts
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 4]
    });
    contract.maximum_outputs = Some(4);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = requested.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn numeric(shape: ShapeFact, domain: NumericDomain, storage: StorageFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        }),
        shape,
        storage,
    )
}

fn uncertain_numeric(shape: ShapeFact, storage: StorageFact) -> ValueFact {
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    fact.shape = shape;
    fact.storage = storage;
    fact.residency = ResidencyFact::Host;
    fact
}

fn invalid_order_fact(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || value.shape.is_definitely_nonscalar()
        || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        || matches!(value.residency, ResidencyFact::Remote { .. })
}

fn invalid_cutoff_fact(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        || matches!(&value.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(n) if *n > 1)).count() > 1)
        || matches!(value.residency, ResidencyFact::Remote { .. })
}

fn order_literal(value: &LiteralValue) -> Result<Option<usize>, ()> {
    let number = match value {
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => return Ok(None),
        LiteralValue::Integer { .. } => {
            return value
                .integer_value_i128()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| (1..=1024).contains(n))
                .map(Some)
                .ok_or(());
        }
        LiteralValue::Bool(value) => return if *value { Ok(Some(1)) } else { Err(()) },
        LiteralValue::Number(n) => *n,
        LiteralValue::Real {
            text,
            class: NumericClass::Single,
        } => f64::from(text.parse::<f32>().map_err(|_| ())?),
        LiteralValue::Real { text, .. } => text.parse::<f64>().map_err(|_| ())?,
        _ => return Err(()),
    };
    let rounded = number.round();
    if number.is_finite() && (rounded - number).abs() <= 1e-12 && (1.0..=1024.0).contains(&rounded)
    {
        Ok(Some(rounded as usize))
    } else {
        Err(())
    }
}

enum OptionToken {
    Analog,
    Kind(FilterKind),
}
fn text_literal(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(s) | LiteralValue::Character(s) | LiteralValue::Keyword(s) => Some(s),
        _ => None,
    }
}
fn decode_option(token: &str) -> Option<OptionToken> {
    match token.trim().to_ascii_lowercase().as_str() {
        "s" => Some(OptionToken::Analog),
        "low" | "lowpass" => Some(OptionToken::Kind(FilterKind::Low)),
        "high" | "highpass" => Some(OptionToken::Kind(FilterKind::High)),
        "bandpass" | "pass" => Some(OptionToken::Kind(FilterKind::Bandpass)),
        "stop" | "bandstop" | "bandreject" | "notch" => Some(OptionToken::Kind(FilterKind::Stop)),
        _ => None,
    }
}

fn invalid_cutoff_literal(value: &LiteralValue, analog: Option<bool>) -> bool {
    let values = match value {
        LiteralValue::Vector(values) => values.iter().collect::<Vec<_>>(),
        LiteralValue::Matrix(rows) if rows.len() == 1 => rows[0].iter().collect::<Vec<_>>(),
        LiteralValue::Matrix(rows) if rows.iter().all(|row| row.len() == 1) => {
            rows.iter().map(|row| &row[0]).collect::<Vec<_>>()
        }
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => return false,
        other => vec![other],
    };
    if values.is_empty() || values.len() > 2 {
        return true;
    }
    let mut parsed = Vec::new();
    for value in values {
        let n = match value {
            LiteralValue::Integer { .. } => value.integer_value_i128().map(|n| n as f64),
            LiteralValue::Number(n) => Some(*n),
            LiteralValue::Real {
                text,
                class: NumericClass::Single,
            } => text.parse::<f32>().ok().map(f64::from),
            LiteralValue::Real { text, .. } => text.parse::<f64>().ok(),
            LiteralValue::Bool(value) => Some(f64::from(u8::from(*value))),
            LiteralValue::Unknown | LiteralValue::Symbolic(_) => return false,
            _ => return true,
        };
        let Some(n) = n else {
            return false;
        };
        if !n.is_finite() || n <= 0.0 || (analog == Some(false) && n >= 1.0) {
            return true;
        }
        parsed.push(n);
    }
    parsed.len() == 2 && parsed[0] >= parsed[1]
}
