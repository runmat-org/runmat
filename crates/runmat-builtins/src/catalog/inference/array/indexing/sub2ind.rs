use super::super::super::argument_error;
use super::common::{double, finish};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallContract, CallInference, CallRequest, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_sub2ind(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUB2IND-ARITY",
            "sub2ind requires a size vector and subscripts",
            request.arguments.len().min(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|size| !valid_size(size))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUB2IND-SIZE-TYPE",
            "sub2ind size must be a real numeric vector or numeric cell vector",
            0,
        ));
    }
    if invalid_size_literal(request.literals.literal_args.first(), 1)
        || request.arguments.first().and_then(known_size_cell_count) == Some(0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUB2IND-SIZE",
            "sub2ind requires positive dimensions",
            0,
        ));
    }
    if let Some(dims) = request.literals.numeric_vector_at(0) {
        if dims.len() != request.arguments.len().saturating_sub(1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SUB2IND-COUNT",
                "sub2ind needs one subscript per dimension",
                0,
            ));
        }
    } else if request
        .arguments
        .first()
        .and_then(known_size_cell_count)
        .is_some_and(|count| count != request.arguments.len().saturating_sub(1))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUB2IND-COUNT",
            "sub2ind needs one subscript per dimension",
            0,
        ));
    }
    let mut shape = ShapeFact::Scalar;
    for (index, argument) in request.arguments.iter().enumerate().skip(1) {
        if !valid_real_numeric(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SUB2IND-SUBSCRIPT",
                "sub2ind subscripts must be real numeric or logical",
                index,
            ));
        }
        if argument.shape.element_count() != Some(1) {
            let (merged, incompatible) = merge_subscript_shape(&shape, &argument.shape);
            if incompatible {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SUB2IND-SHAPE",
                    "non-scalar subscript arrays must share a shape",
                    index,
                ));
            }
            shape = merged;
        }
    }
    let input = request
        .arguments
        .iter()
        .find(|value| !matches!(value.residency, ResidencyFact::Host));
    finish(
        entry,
        request,
        CallContract::fixed(vec![double(shape, input)]),
        diagnostics,
    )
}

pub(super) fn invalid_size_literal(
    value: Option<&LiteralValue>,
    minimum_dimensions: usize,
) -> bool {
    match value {
        Some(LiteralValue::Vector(values)) => {
            values.len() < minimum_dimensions || values.iter().any(invalid_dimension_literal)
        }
        Some(
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. },
        ) => minimum_dimensions > 1 || value.is_some_and(invalid_dimension_literal),
        _ => false,
    }
}

fn invalid_dimension_literal(value: &LiteralValue) -> bool {
    if let LiteralValue::Integer { .. } = value {
        return value
            .integer_value_i128()
            .is_some_and(|dimension| dimension < 1 || usize::try_from(dimension).is_err());
    }
    LiteralContext::numeric_from_literal(value).is_some_and(|dimension| {
        !dimension.is_finite()
            || dimension < 1.0
            || (dimension.round() - dimension).abs() > f64::EPSILON
            || dimension > usize::MAX.saturating_sub(1) as f64
            || dimension as usize == usize::MAX
            || (dimension as usize) as f64 != dimension
    })
}

fn merge_subscript_shape(left: &ShapeFact, right: &ShapeFact) -> (ShapeFact, bool) {
    let left_non_scalar = left.is_definitely_nonscalar();
    let right_non_scalar = right.is_definitely_nonscalar();
    if !left_non_scalar {
        return (right.clone(), false);
    }
    if !right_non_scalar {
        return (left.clone(), false);
    }
    let (Some(left_dims), Some(right_dims)) = (left.known_dims(), right.known_dims()) else {
        return (left.clone(), false);
    };
    if left_dims.len() != right_dims.len() {
        return (left.clone(), true);
    }
    let mut merged = Vec::with_capacity(left_dims.len());
    let mut incompatible = false;
    for (a, b) in left_dims.into_iter().zip(right_dims) {
        if a.is_some() && b.is_some() && a != b {
            incompatible = true;
        }
        merged.push(a.or(b));
    }
    (ShapeFact::from(merged), incompatible)
}

pub(super) fn valid_real_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    )
}

pub(super) fn valid_size(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            let elements: &[ValueFact] = if cell.elements_complete {
                &cell.elements
            } else {
                std::slice::from_ref(cell.element.as_ref())
            };
            elements.iter().all(valid_cell_dimension)
        }
        _ => false,
    }
}

fn valid_cell_dimension(element: &ValueFact) -> bool {
    if element.shape.is_definitely_nonscalar()
        || !matches!(element.storage, StorageFact::Scalar | StorageFact::Unknown)
    {
        return false;
    }
    match &element.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => false,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Unknown => true,
        _ => false,
    }
}

pub(super) fn known_size_cell_count(value: &ValueFact) -> Option<usize> {
    match &value.kind {
        ValueKindFact::Cell(cell) if cell.elements_complete => Some(cell.elements.len()),
        _ => None,
    }
}
