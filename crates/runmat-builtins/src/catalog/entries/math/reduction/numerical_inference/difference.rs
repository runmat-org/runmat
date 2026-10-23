use super::*;

pub(crate) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = if let Some(input) = request.arguments.first() {
        if request.arguments.len() > 3 {
            diagnostics.push(invalid(entry, "expected data, order, and dimension"));
        }
        if input.kind == ValueKindFact::Symbolic {
            symbolic(request, entry, &mut diagnostics)
        } else {
            numerical(input, request, entry, &mut diagnostics)
        }
    } else {
        diagnostics.push(invalid(entry, "input data is required"));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish(request, entry, vec![output], diagnostics)
}

fn numerical(
    input: &ValueFact,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let order = if request.arguments.len() < 2 || empty(literal(request, 1)) {
        Some(1)
    } else {
        literal(request, 1).and_then(|value| {
            if value.integer_value_i128() == Some(0)
                || crate::ResolveContext::numeric_from_literal(value) == Some(0.0)
            {
                Some(0)
            } else {
                value
                    .integer_value_i128()
                    .and_then(|value| usize::try_from(value).ok())
                    .or_else(|| {
                        crate::ResolveContext::numeric_from_literal(value)
                            .and_then(|value| runmat_types::decode_shape_dimension(value).ok())
                    })
            }
        })
    };
    if input.storage == StorageFact::Sparse
        || !matches!(
            input.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::Unknown
        )
    {
        diagnostics.push(invalid_input(
            crate::NumericalReductionKind::Difference,
            "expected dense numeric, logical, or character data",
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    if order == Some(0) {
        let mut identity = input.clone();
        identity.alias = runmat_types::AliasFact::Identity;
        return identity;
    }
    if !dense_samples(input)
        && !matches!(
            input.kind,
            ValueKindFact::Character | ValueKindFact::Unknown
        )
    {
        diagnostics.push(invalid_input(
            crate::NumericalReductionKind::Difference,
            "expected dense numeric, logical, or character data",
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    for position in 1..request.arguments.len().min(3) {
        let control = &request.arguments[position];
        if !empty(literal(request, position))
            && control
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
        {
            diagnostics.push(invalid(
                entry,
                "difference controls must be scalar or empty",
            ));
        }
    }
    if request.arguments.len() > 1
        && !empty(literal(request, 1))
        && literal(request, 1).is_some_and(|value| !matches!(value, LiteralValue::Unknown))
        && order.is_none()
    {
        diagnostics.push(invalid(
            entry,
            "difference order must be a nonnegative integer",
        ));
    }
    let axis = if request.arguments.len() < 3 || empty(literal(request, 2)) {
        default_axis(&input.shape, false)
    } else {
        literal(request, 2).and_then(positive_axis)
    };
    if request.arguments.len() == 3
        && !empty(literal(request, 2))
        && literal(request, 2).is_some_and(|value| !matches!(value, LiteralValue::Unknown))
        && axis.is_none()
    {
        diagnostics.push(invalid(entry, "dimension must be a positive integer"));
    }
    let shape = match (input.shape.known_dims(), axis, order) {
        (Some(mut dimensions), Some(axis), Some(order)) => {
            // Runtime pads trailing singleton metadata before differencing.
            // An unbounded public axis is not materialized by inference.
            if axis > dimensions.len() {
                ShapeFact::Ranked { rank: axis }
            } else {
                dimensions[axis - 1] =
                    dimensions[axis - 1].map(|extent| extent.saturating_sub(order));
                ShapeFact::from(dimensions)
            }
        }
        _ => ShapeFact::Unknown,
    };
    let mut output = input.clone();
    output.shape = shape;
    output.alias = runmat_types::AliasFact::Unique;
    output.storage = StorageFact::Dense;
    if matches!(
        input.kind,
        ValueKindFact::Character | ValueKindFact::Logical
    ) {
        if order.is_none() {
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    }
    if input.residency != ResidencyFact::Host {
        output.residency = ResidencyFact::Unknown;
        output.storage = StorageFact::Unknown;
    }
    output
}

fn empty(value: Option<&LiteralValue>) -> bool {
    matches!(value, Some(LiteralValue::Empty))
        || matches!(value, Some(LiteralValue::Vector(values)) if values.is_empty())
}

fn symbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if request
        .arguments
        .first()
        .is_some_and(|input| input.shape.element_count().is_some_and(|count| count != 1))
    {
        diagnostics.push(invalid(
            entry,
            "symbolic differentiation requires a scalar expression",
        ));
    }
    let variable = |position: usize| match request
        .arguments
        .get(position)
        .map(|argument| &argument.kind)
    {
        Some(ValueKindFact::String | ValueKindFact::Character) => Some(true),
        Some(ValueKindFact::Symbolic | ValueKindFact::Unknown) | None => None,
        _ => Some(false),
    };
    let order_position = match request.arguments.len() {
        2 if variable(1) == Some(false) => Some(1),
        3 if variable(1) == Some(true) => Some(2),
        3 if variable(2) == Some(true) => Some(1),
        3 if variable(1) == Some(false) && variable(2) == Some(false) => {
            diagnostics.push(invalid(
                entry,
                "two symbolic controls require a variable and an optional order",
            ));
            None
        }
        _ => None,
    };
    if let Some(position) = order_position {
        let argument = &request.arguments[position];
        if argument
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
        {
            diagnostics.push(invalid(entry, "symbolic order must be scalar"));
        }
        if let Some(value) =
            literal(request, position).and_then(crate::ResolveContext::numeric_from_literal)
        {
            if !value.is_finite()
                || value < 0.0
                || value > u32::MAX as f64
                || (value.round() - value).abs() > 1e-12
            {
                diagnostics.push(invalid(
                    entry,
                    "symbolic order must be a nonnegative 32-bit integer",
                ));
            }
        }
    }
    ValueFact::scalar(ValueKindFact::Symbolic)
}
