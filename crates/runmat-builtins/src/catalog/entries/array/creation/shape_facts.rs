//! Structural facts shared by creation identities. Data values never become
//! shape extents merely because a tensor has a known physical shape.
use runmat_types::{CallRequest, ShapeFact};

pub(super) fn dimensions(
    request: &CallRequest,
    first: usize,
    end: usize,
    no_size: ShapeFact,
) -> ShapeFact {
    if end <= first {
        return no_size;
    }
    if end == first + 1 {
        if let Some(dims) = request.literals.numeric_vector_at(first) {
            return normalized(dims);
        }
        if let Some(size) = request
            .literals
            .numeric_dims_from(first)
            .first()
            .copied()
            .flatten()
        {
            return ShapeFact::from(vec![Some(size), Some(size)]);
        }
        // A typed scalar size specifies two unknown extents. A size vector's
        // length specifies rank, not the values of any output extent.
        let Some(argument) = request.arguments.get(first) else {
            return ShapeFact::Unknown;
        };
        if !matches!(
            argument.kind,
            runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
                domain: runmat_types::NumericDomain::Real,
                ..
            })
        ) {
            return ShapeFact::Unknown;
        }
        if argument.is_scalar() {
            return ShapeFact::Ranked { rank: 2 };
        }
        if let Some(shape) = argument.shape.known_dims() {
            if shape.first() == Some(&Some(1)) {
                if let Some(rank) = argument.shape.element_count() {
                    return if rank == 0 {
                        ShapeFact::from(vec![Some(0), Some(0)])
                    } else {
                        ShapeFact::Ranked { rank: rank.max(2) }
                    };
                }
            }
        }
        return ShapeFact::Unknown;
    }
    let mut dimensions = Vec::with_capacity(end - first);
    for index in first..end {
        if let Some(vector) = request.literals.numeric_vector_at(index) {
            // Multiple dimension arguments may include a one-element vector,
            // but a multi-element size vector cannot be mixed with another
            // dimension. Never read subsequent extents out of that vector.
            if vector.len() != 1 {
                return ShapeFact::Unknown;
            }
            dimensions.push(vector[0]);
        } else {
            dimensions.push(scalar_dimension(request, index));
        }
    }
    normalized(dimensions)
}

/// The extent of one source argument, never the first flattened element of a
/// preceding size vector or matrix.
pub(super) fn scalar_dimension(request: &CallRequest, index: usize) -> Option<usize> {
    if let Some(vector) = request.literals.numeric_vector_at(index) {
        return (vector.len() == 1).then(|| vector[0]).flatten();
    }
    request
        .literals
        .numeric_dims_from(index)
        .first()
        .copied()
        .flatten()
}

pub(super) fn normalized(mut dimensions: Vec<Option<usize>>) -> ShapeFact {
    if dimensions.is_empty() {
        return ShapeFact::from(vec![Some(0), Some(0)]);
    }
    if dimensions.len() == 1 {
        dimensions.push(dimensions[0]);
    }
    while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
        dimensions.pop();
    }
    ShapeFact::from(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, LiteralContext, LiteralValue, OutputSelection, ValueFact};

    fn request(literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); literals.len()],
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        }
    }

    #[test]
    fn dimensions_follow_argument_positions_not_prior_vector_contents() {
        let bounds =
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(6.0)]);
        let form = request(vec![
            bounds,
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ]);
        assert_eq!(
            dimensions(&form, 1, 3, ShapeFact::Scalar),
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        let mixed = request(vec![
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]),
            LiteralValue::Number(4.0),
        ]);
        assert_eq!(
            dimensions(&mixed, 0, 2, ShapeFact::Scalar),
            ShapeFact::Unknown
        );
    }
}
