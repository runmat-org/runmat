use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Voltage,
    Power,
    Resistance,
    TextUnknown,
    Unknown,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if !(1..=2).contains(&args.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DB-ARITY",
            "db expects a signal and at most one mode or resistance argument",
            args.len().saturating_sub(1),
        ));
    }
    let mode = if args.len() < 2 {
        Mode::Voltage
    } else {
        match &args[1].kind {
            ValueKindFact::String | ValueKindFact::Character => {
                if matches!(args[1].kind, ValueKindFact::String)
                    && args[1].shape.element_count().is_some_and(|n| n != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-DB-MODE",
                        "mode must be scalar text",
                        1,
                    ));
                }
                if matches!(args[1].kind, ValueKindFact::Character)
                    && args[1].shape.known_dims().is_some_and(|dims| {
                        dims.first()
                            .copied()
                            .flatten()
                            .is_some_and(|rows| rows != 1)
                    })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-DB-MODE",
                        "character mode must be a row vector",
                        1,
                    ));
                }
                match request.literals.literal_string_at(1).as_deref() {
                    Some("voltage") => Mode::Voltage,
                    Some("power") => Mode::Power,
                    Some(_) => {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-DB-MODE",
                            "mode must be voltage or power",
                            1,
                        ));
                        Mode::TextUnknown
                    }
                    None => Mode::TextUnknown,
                }
            }
            ValueKindFact::Unknown => Mode::Unknown,
            _ => Mode::Resistance,
        }
    };
    if let Some(y) = args.first() {
        if !matches!(
            y.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DB-INPUT",
                "signal must be numeric or logical",
                0,
            ));
        }
        if mode == Mode::Power
            && y.numeric()
                .is_some_and(|n| n.domain == NumericDomain::Complex)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DB-POWER",
                "power input must be real and nonnegative",
                0,
            ));
        }
    }
    if mode == Mode::Resistance {
        if let Some(r) = args.get(1) {
            if !matches!(
                r.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            ) || r
                .numeric()
                .is_some_and(|n| n.domain == NumericDomain::Complex)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DB-RESISTANCE",
                    "resistance must be positive finite real numeric data",
                    1,
                ));
            }
        }
    }
    let shape = match (args.first(), args.get(1), mode) {
        (Some(y), Some(r), Mode::Resistance) => match broadcast_shape(&y.shape, &r.shape) {
            Ok(shape) => shape,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DB-SIZE",
                    "signal and resistance cannot be broadcast together",
                    1,
                ));
                ShapeFact::Unknown
            }
        },
        (Some(y), _, Mode::Voltage | Mode::Power | Mode::TextUnknown) => y.shape.clone(),
        _ => ShapeFact::Unknown,
    };
    let known_single = args
        .first()
        .is_some_and(|y| y.numeric().is_some_and(|n| n.class == NumericClass::Single))
        || (mode == Mode::Resistance
            && args
                .get(1)
                .is_some_and(|r| r.numeric().is_some_and(|n| n.class == NumericClass::Single)));
    let unknown_class = args
        .first()
        .is_none_or(|y| matches!(y.kind, ValueKindFact::Unknown))
        || (mode == Mode::Resistance
            && args
                .get(1)
                .is_some_and(|r| matches!(r.kind, ValueKindFact::Unknown)))
        || mode == Mode::Unknown;
    let output = if unknown_class && !known_single {
        let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
        output.shape = shape;
        output.residency = runmat_types::ResidencyFact::Host;
        output
    } else {
        let class = if known_single {
            NumericClass::Single
        } else {
            NumericClass::Double
        };
        let storage = if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        )
    };
    finish_fixed(entry, request, output, diagnostics)
}
