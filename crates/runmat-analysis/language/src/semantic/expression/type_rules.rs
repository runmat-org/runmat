use crate::{FeaType, PhysicalDimension};

pub(super) fn types_compatible(left: &FeaType, right: &FeaType) -> bool {
    left == right
        || matches!((left,right),(FeaType::Quantity(a),FeaType::Quantity(b)) if a.is_compatible_with(*b))
        || matches!(
            (left, right),
            (FeaType::Integer, FeaType::Real) | (FeaType::Real, FeaType::Integer)
        )
        || matches!(left, FeaType::Unknown | FeaType::Error)
        || matches!(right, FeaType::Unknown | FeaType::Error)
}

pub(super) fn numeric_join(left: &FeaType, right: &FeaType) -> FeaType {
    if left == right {
        left.clone()
    } else if let (FeaType::Quantity(left), FeaType::Quantity(right)) = (left, right) {
        if left.is_compatible_with(*right) {
            FeaType::Quantity(
                left.with_semantic_kind(left.kind().or(right.kind()))
                    .expect("compatible quantity types"),
            )
        } else {
            FeaType::Error
        }
    } else if matches!(left, FeaType::Complex) || matches!(right, FeaType::Complex) {
        FeaType::Complex
    } else if left.is_numeric() && right.is_numeric() {
        FeaType::Real
    } else {
        FeaType::Error
    }
}

pub(super) fn multiply_type(left: &FeaType, right: &FeaType) -> Option<FeaType> {
    match (left, right) {
        (FeaType::Quantity(left), FeaType::Quantity(right)) => {
            Some(FeaType::Quantity(left.multiply(*right)?))
        }
        (FeaType::Quantity(dimension), FeaType::Integer | FeaType::Real)
        | (FeaType::Integer | FeaType::Real, FeaType::Quantity(dimension)) => {
            Some(FeaType::Quantity(*dimension))
        }
        _ if left.is_numeric() && right.is_numeric() => Some(numeric_join(left, right)),
        _ => None,
    }
}

pub(super) fn divide_type(left: &FeaType, right: &FeaType) -> Option<FeaType> {
    match (left, right) {
        (FeaType::Quantity(left), FeaType::Quantity(right)) => {
            Some(FeaType::Quantity(left.divide(*right)?))
        }
        (FeaType::Quantity(dimension), FeaType::Integer | FeaType::Real) => {
            Some(FeaType::Quantity(*dimension))
        }
        (FeaType::Integer | FeaType::Real, FeaType::Quantity(dimension)) => {
            Some(FeaType::Quantity(
                runmat_types::physical::DynamicDimension::from(PhysicalDimension::DIMENSIONLESS)
                    .divide(*dimension)?,
            ))
        }
        _ if left.is_numeric() && right.is_numeric() => Some(numeric_join(left, right)),
        _ => None,
    }
}
