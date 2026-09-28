use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn integer_class_if_proven(
    class: NumericClass,
    input: &ValueFact,
    request: &CallRequest,
) -> Option<NumericClass> {
    let limit = match class {
        NumericClass::Int8 => i8::MAX as u128,
        NumericClass::UInt8 => u8::MAX as u128,
        NumericClass::Int16 => i16::MAX as u128,
        NumericClass::UInt16 => u16::MAX as u128,
        NumericClass::Int32 => i32::MAX as u128,
        NumericClass::UInt32 => u32::MAX as u128,
        NumericClass::Int64 | NumericClass::UInt64 => return Some(class),
        NumericClass::Double | NumericClass::Single => return None,
    };
    let dims = input.shape.known_dims()?;
    let columns = if dims.len() == 1 {
        dims[0]?
    } else {
        *dims.get(1)?.as_ref()?
    };
    if columns > 32 {
        return None;
    }
    let base = match request.literals.literal_args.get(1) {
        None
        | Some(LiteralValue::Character(_) | LiteralValue::String(_) | LiteralValue::Keyword(_)) => {
            2_u128
        }
        Some(value) => {
            let value = runmat_types::LiteralContext::numeric_from_literal(value)?;
            if !value.is_finite() || value < 2.0 || value.fract() != 0.0 {
                return None;
            }
            value as u128
        }
    };
    let upper_bound = base.checked_pow(columns as u32)?;
    (upper_bound.saturating_sub(1) <= limit).then_some(class)
}

fn decimal_output(input: Option<&ValueFact>, request: &CallRequest, shape: ShapeFact) -> ValueFact {
    let kind = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
        | Some(ValueKindFact::Logical) => Some(NumericClass::Double),
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })) => Some(NumericClass::Single),
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => input.and_then(|fact| integer_class_if_proven(*class, fact, request)),
        _ => None,
    };
    match kind {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape.clone(),
            if shape.element_count() == Some(1) {
                StorageFact::Scalar
            } else {
                StorageFact::Dense
            },
        ),
        None => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = shape;
            fact.residency = ResidencyFact::Host;
            fact
        }
    }
}

fn digit_rows(input: Option<&ValueFact>) -> ShapeFact {
    let Some(input) = input else {
        return ShapeFact::Unknown;
    };
    match input.shape.known_dims() {
        Some(dims) if dims.len() <= 2 => {
            let rows = if dims.len() == 1 {
                Some(1)
            } else {
                dims.first().copied().flatten()
            };
            ShapeFact::from(vec![rows, Some(1)])
        }
        _ => ShapeFact::Unknown,
    }
}

fn invalid_data(input: &ValueFact) -> bool {
    !matches!(
        input.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}

fn invalid_control_literal(value: &LiteralValue, minimum: f64) -> bool {
    runmat_types::LiteralContext::numeric_from_literal(value)
        .is_some_and(|number| !number.is_finite() || number.fract() != 0.0 || number < minimum)
}

fn invalid_order_literal(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => {
            let normalized: String = text
                .chars()
                .filter(|character| {
                    *character != '-' && *character != '_' && !character.is_whitespace()
                })
                .flat_map(char::to_lowercase)
                .collect();
            !matches!(normalized.as_str(), "rightmsb" | "leftmsb")
        }
        _ => false,
    }
}

fn literal_width(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|width| *width > 0),
        LiteralValue::Number(number) => (number.is_finite()
            && *number >= 1.0
            && number.fract() == 0.0
            && *number < usize::MAX as f64)
            .then_some(*number as usize),
        LiteralValue::Real { text, .. } => text.parse::<f64>().ok().and_then(|number| {
            (number.is_finite()
                && number >= 1.0
                && number.fract() == 0.0
                && number < usize::MAX as f64)
                .then_some(number as usize)
        }),
        _ => None,
    }
}

pub(super) fn infer_bi2de(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-BI2DE-ARITY",
            "bi2de expects B, optional P, and optional digit order",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if invalid_data(input) {
            diagnostics.push(argument_error(
                "RM-BI2DE-DIGITS",
                "B must be real numeric or logical digits",
                0,
            ));
        }
        if input.shape.rank().is_some_and(|rank| rank > 2) {
            diagnostics.push(argument_error(
                "RM-BI2DE-SHAPE",
                "B must be a vector or 2-D matrix",
                0,
            ));
        }
    }
    if request.arguments.len() == 3 {
        if request
            .literals
            .literal_args
            .get(1)
            .is_some_and(|value| invalid_control_literal(value, 2.0))
        {
            diagnostics.push(argument_error(
                "RM-BI2DE-BASE",
                "P must be an integer at least two",
                1,
            ));
        }
        if request
            .literals
            .literal_args
            .get(2)
            .is_some_and(invalid_order_literal)
        {
            diagnostics.push(argument_error(
                "RM-BI2DE-ORDER",
                "digit order must be right-msb or left-msb",
                2,
            ));
        }
    } else if let Some(value) = request.literals.literal_args.get(1) {
        if matches!(
            value,
            LiteralValue::Character(_) | LiteralValue::String(_) | LiteralValue::Keyword(_)
        ) {
            if invalid_order_literal(value) {
                diagnostics.push(argument_error(
                    "RM-BI2DE-ORDER",
                    "digit order must be right-msb or left-msb",
                    1,
                ));
            }
        } else if invalid_control_literal(value, 2.0) {
            diagnostics.push(argument_error(
                "RM-BI2DE-BASE",
                "P must be an integer at least two",
                1,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        decimal_output(
            request.arguments.first(),
            request,
            digit_rows(request.arguments.first()),
        ),
        diagnostics,
    )
}

pub(super) fn infer_de2bi(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-DE2BI-ARITY",
            "de2bi expects D and at most three controls",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if invalid_data(input) {
            diagnostics.push(argument_error(
                "RM-DE2BI-DATA",
                "D must be real numeric or logical data",
                0,
            ));
        }
    }
    let rows = request
        .arguments
        .first()
        .and_then(|input| input.shape.element_count());
    let width = request.literals.literal_args.get(1).and_then(literal_width);
    for (index, value) in request.literals.literal_args.iter().enumerate().skip(1) {
        let min = if index == 1 { 1.0 } else { 2.0 };
        if index < 3 && invalid_control_literal(value, min) {
            diagnostics.push(argument_error(
                "RM-DE2BI-CONTROL",
                "width must be positive and base at least two",
                index,
            ));
        }
        if index == request.arguments.len().saturating_sub(1) && invalid_order_literal(value) {
            diagnostics.push(argument_error(
                "RM-DE2BI-ORDER",
                "digit order must be right-msb or left-msb",
                index,
            ));
        }
    }
    let shape = ShapeFact::from(vec![rows, width]);
    let output = match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: *class,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        ),
        Some(ValueKindFact::Logical) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        ),
        _ => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = shape;
            fact.residency = ResidencyFact::Host;
            fact
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
