use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Known(usize),
    Unknown,
    Invalid,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterKind {
    Low,
    High,
    Bandpass,
    Stop,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIR1-ARITY",
            "fir1 requires an order and cutoff",
            0,
        ));
    }
    let order = request.arguments.first().map_or(Order::Unknown, |value| {
        if invalid_numeric_control(value, true) {
            Order::Invalid
        } else {
            request
                .literals
                .literal_args
                .first()
                .map_or(Order::Unknown, order_literal)
        }
    });
    if order == Order::Invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIR1-ORDER",
            "order must be a finite nonnegative integer scalar",
            0,
        ));
    }
    let cutoff_count = request.arguments.get(1).and_then(|value| {
        if invalid_numeric_control(value, false) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIR1-CUTOFF",
                "Wn must be a real scalar or two-element vector",
                1,
            ));
        }
        value.shape.element_count()
    });
    if cutoff_count.is_some_and(|count| count != 1 && count != 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIR1-CUTOFF",
            "Wn must contain one or two values",
            1,
        ));
    }
    if request
        .literals
        .literal_args
        .get(1)
        .is_some_and(invalid_cutoff_literal)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIR1-CUTOFF",
            "Wn must contain increasing normalized frequencies in (0,1)",
            1,
        ));
    }
    let mut kind = cutoff_count.map(|count| {
        if count == 1 {
            FilterKind::Low
        } else {
            FilterKind::Bandpass
        }
    });
    let mut unknown_kind = cutoff_count.is_none();
    let mut window_count = 0usize;
    for (index, value) in request.arguments.iter().enumerate().skip(2) {
        let literal = request.literals.literal_args.get(index);
        if let Some(token) = literal.and_then(text_literal) {
            match decode_option(token) {
                Some(Some(parsed)) => kind = Some(parsed),
                Some(None) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-FIR1-OPTION",
                    "unrecognized filter or scaling option",
                    index,
                )),
            }
        } else if matches!(
            value.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            unknown_kind = true;
        } else if invalid_numeric_control(value, false) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIR1-WINDOW",
                "window must be a real vector",
                index,
            ));
        } else {
            if literal.is_some_and(known_nonreal_window_literal) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIR1-WINDOW",
                    "window values must have zero imaginary component",
                    index,
                ));
            }
            window_count += 1;
            if window_count > 1 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIR1-WINDOW",
                    "only one window vector may be supplied",
                    index,
                ));
            }
        }
    }
    let width = match order {
        Order::Known(order) if !unknown_kind => {
            let extra = usize::from(
                order % 2 == 1 && matches!(kind, Some(FilterKind::High | FilterKind::Stop)),
            );
            order
                .checked_add(1)
                .and_then(|width| width.checked_add(extra))
        }
        Order::Known(order) if order % 2 == 0 => order.checked_add(1),
        _ => None,
    };
    if let Some(expected) = width {
        for (index, value) in request.arguments.iter().enumerate().skip(2) {
            if !matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) {
                continue;
            }
            if value
                .shape
                .element_count()
                .is_some_and(|actual| actual != expected)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIR1-WINDOW",
                    "window length must equal the adjusted order plus one",
                    index,
                ));
            }
        }
    }
    let (output_shape, output_storage) = match width {
        Some(1) => (ShapeFact::Scalar, StorageFact::Scalar),
        Some(width) => (
            ShapeFact::from(vec![Some(1), Some(width)]),
            StorageFact::Dense,
        ),
        None => (ShapeFact::from(vec![Some(1), None]), StorageFact::Unknown),
    };
    let output = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            output_shape,
            output_storage,
        )
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_numeric_control(value: &ValueFact, scalar: bool) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        || (scalar && value.shape.is_definitely_nonscalar())
        || (scalar
            && matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Complex,
                    ..
                })
            ))
        || matches!(value.residency, ResidencyFact::Remote { .. })
}

fn order_literal(value: &LiteralValue) -> Order {
    match value {
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|n| usize::try_from(n).ok())
            .map_or(Order::Invalid, Order::Known),
        LiteralValue::Number(n) => floating_order(*n),
        LiteralValue::Real { text, class } => {
            let number = if *class == NumericClass::Single {
                text.parse::<f32>().ok().map(f64::from)
            } else {
                text.parse::<f64>().ok()
            };
            number.map_or(Order::Invalid, floating_order)
        }
        LiteralValue::Bool(value) => Order::Known(usize::from(*value)),
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => Order::Unknown,
        _ => Order::Invalid,
    }
}

fn floating_order(n: f64) -> Order {
    let rounded = n.round();
    if !n.is_finite() || rounded < 0.0 || (rounded - n).abs() > 1e-9 || rounded >= usize::MAX as f64
    {
        Order::Invalid
    } else {
        Order::Known(rounded as usize)
    }
}

fn text_literal(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn decode_option(value: &str) -> Option<Option<FilterKind>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" | "lowpass" => Some(Some(FilterKind::Low)),
        "high" | "highpass" => Some(Some(FilterKind::High)),
        "bandpass" | "pass" => Some(Some(FilterKind::Bandpass)),
        "stop" | "bandstop" | "bandreject" => Some(Some(FilterKind::Stop)),
        "scale" | "noscale" => Some(None),
        _ => None,
    }
}

fn invalid_cutoff_literal(value: &LiteralValue) -> bool {
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
        let number = match value {
            LiteralValue::Number(n) => Some(*n),
            LiteralValue::Real {
                text,
                class: NumericClass::Single,
            } => text.parse::<f32>().ok().map(f64::from),
            LiteralValue::Real { text, .. } => text.parse::<f64>().ok(),
            LiteralValue::Integer { .. } => value.integer_value_i128().map(|n| n as f64),
            LiteralValue::Complex {
                real,
                imaginary,
                class,
            } => {
                let parsed = if *class == NumericClass::Single {
                    real.parse::<f32>()
                        .ok()
                        .zip(imaginary.parse::<f32>().ok())
                        .map(|(re, im)| (f64::from(re), f64::from(im)))
                } else {
                    real.parse::<f64>().ok().zip(imaginary.parse::<f64>().ok())
                };
                let Some((re, im)) = parsed else {
                    return false;
                };
                if !im.is_finite() || im.abs() > 1.0e-12 {
                    return true;
                }
                Some(re)
            }
            LiteralValue::Bool(value) => Some(f64::from(u8::from(*value))),
            LiteralValue::Unknown | LiteralValue::Symbolic(_) => return false,
            _ => return true,
        };
        let Some(n) = number else {
            return false;
        };
        if !n.is_finite() || n <= 0.0 || n >= 1.0 {
            return true;
        }
        parsed.push(n);
    }
    parsed.len() == 2 && parsed[0] >= parsed[1]
}

fn known_nonreal_window_literal(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Complex {
            imaginary, class, ..
        } => {
            let parsed = if *class == NumericClass::Single {
                imaginary.parse::<f32>().ok().map(f64::from)
            } else {
                imaginary.parse::<f64>().ok()
            };
            parsed.is_some_and(|imaginary| !imaginary.is_finite() || imaginary.abs() > 1.0e-12)
        }
        LiteralValue::Vector(values) => values.iter().any(known_nonreal_window_literal),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(known_nonreal_window_literal),
        _ => false,
    }
}
