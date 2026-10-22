use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operand {
    Datetime,
    Duration,
    Calendar,
    Numeric,
    Logical,
    Unknown,
    Unsupported,
}
fn operand(value: &ValueFact) -> Operand {
    match &value.kind {
        ValueKindFact::Unknown => Operand::Unknown,
        ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
            Some(class) if class.is(standard::DATETIME) => Operand::Datetime,
            Some(class) if class.is(standard::DURATION) => Operand::Duration,
            Some(class) if class.is(standard::CALENDAR_DURATION) => Operand::Calendar,
            None => Operand::Unknown,
            _ => Operand::Unsupported,
        },
        ValueKindFact::Numeric(n)
            if n.domain == NumericDomain::Real
                && !matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) =>
        {
            Operand::Numeric
        }
        ValueKindFact::Logical => Operand::Logical,
        _ => Operand::Unsupported,
    }
}
fn operand_shape(value: &ValueFact) -> ShapeFact {
    let property = match operand(value) {
        Operand::Datetime => Some(DATETIME_SERIAL_PROPERTY),
        Operand::Duration => Some(DURATION_DAYS_PROPERTY),
        Operand::Calendar => Some(CALENDAR_MONTHS_PROPERTY),
        _ => None,
    };
    if let (Some(property), ValueKindFact::Object(object)) = (property, &value.kind) {
        if let Some(payload) = object.properties.get(property) {
            return payload.shape.clone();
        }
    }
    value.shape.clone()
}
fn exact_dimensions(shape: &ShapeFact) -> Option<Vec<usize>> {
    shape.known_dims()?.into_iter().collect()
}
fn output_shape(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> ShapeFact {
    let [lhs, rhs] = request.arguments.as_slice() else {
        return ShapeFact::Unknown;
    };
    let left = operand_shape(lhs);
    let right = operand_shape(rhs);
    let ls = left.element_count() == Some(1);
    let rs = right.element_count() == Some(1);
    if ls && rs {
        return ShapeFact::from(vec![Some(1), Some(1)]);
    }
    if ls {
        return right;
    }
    if rs {
        return left;
    }
    if left == right {
        return left;
    }
    if let (Some(left), Some(right)) = (exact_dimensions(&left), exact_dimensions(&right)) {
        if left != right {
            diagnostics.push(argument_error("RM-DATETIME-OP-SHAPE","operands must be scalar or have exactly matching full shapes; implicit expansion is not implemented",1));
        }
    }
    ShapeFact::Unknown
}
pub(super) fn double(shape: ShapeFact) -> ValueFact {
    if shape.element_count() == Some(1) {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    } else if shape.is_definitely_nonscalar() || shape.element_count().is_some() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    } else {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Unknown,
            StorageFact::Unknown,
        )
    }
}
pub(super) fn datetime(shape: ShapeFact) -> ValueFact {
    let serial = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        StorageFact::Dense,
    );
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DATETIME.owned()),
            properties: BTreeMap::from([(DATETIME_SERIAL_PROPERTY.into(), serial)]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    )
}
fn unknown_host(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
    value.shape = shape;
    value.residency = ResidencyFact::Host;
    value
}
fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: DatetimeBinaryOperation,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DATETIME-OP-ARITY",
            "expected exactly a datetime receiver and RHS",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        if matches!(
            input.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-DATETIME-OP-PLACEMENT",
                "datetime methods require host values; no resident gather is admitted",
                index,
            ));
        }
    }
    if request
        .arguments
        .first()
        .is_some_and(|lhs| !matches!(operand(lhs), Operand::Datetime | Operand::Unknown))
    {
        diagnostics.push(argument_error(
            "RM-DATETIME-OP-RECEIVER",
            "left receiver must be datetime",
            0,
        ));
    }
    let rhs = request
        .arguments
        .get(1)
        .map(operand)
        .unwrap_or(Operand::Unknown);
    let admitted = if operation.is_comparison() {
        matches!(
            rhs,
            Operand::Datetime | Operand::Numeric | Operand::Logical | Operand::Unknown
        )
    } else if operation == DatetimeBinaryOperation::Plus {
        matches!(
            rhs,
            Operand::Duration
                | Operand::Calendar
                | Operand::Numeric
                | Operand::Logical
                | Operand::Unknown
        )
    } else {
        !matches!(rhs, Operand::Unsupported)
    };
    if !admitted {
        diagnostics.push(argument_error(
            "RM-DATETIME-OP-RHS",
            "RHS class is not accepted by this datetime operation",
            1,
        ));
    }
    if request
        .arguments
        .get(1)
        .and_then(ValueFact::numeric)
        .is_some_and(|n| !matches!(n.class, NumericClass::Single | NumericClass::Double))
    {
        if let Some(value) = request
            .literals
            .literal_args
            .get(1)
            .and_then(LiteralValue::integer_value_i128)
        {
            if !(-1_000_000_000..=1_000_000_000).contains(&value) {
                diagnostics.push(argument_error(
                    "RM-DATETIME-OP-INTEGER-RANGE",
                    "integer RHS must be in the existing exact serial-date range",
                    1,
                ));
            }
        }
    }
    let shape = output_shape(request, &mut diagnostics);
    let output = if operation.is_comparison()
        || (operation == DatetimeBinaryOperation::Minus && rhs == Operand::Datetime)
    {
        double(shape)
    } else if operation == DatetimeBinaryOperation::Minus && rhs == Operand::Unknown {
        unknown_host(shape)
    } else {
        datetime(shape)
    };
    finish_fixed(entry, request, output, diagnostics)
}
macro_rules! rule {
    ($name:ident,$operation:ident) => {
        pub(super) fn $name(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            infer(request, entry, DatetimeBinaryOperation::$operation)
        }
    };
}
rule!(infer_eq, Eq);
rule!(infer_ne, Ne);
rule!(infer_lt, Lt);
rule!(infer_le, Le);
rule!(infer_gt, Gt);
rule!(infer_ge, Ge);
rule!(infer_plus, Plus);
rule!(infer_minus, Minus);
