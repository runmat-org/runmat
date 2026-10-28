use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};
use runmat_types::physical::DynamicDimension;
use serde::{Deserialize, Serialize};

use super::model_authoring::AuthoredMaterialRef;

/// Closed source-facing input types shared by reusable models and runnable studies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum AuthoredInputType {
    Boolean,
    Integer,
    Real,
    Complex,
    String,
    Quantity(DynamicDimension),
    Material,
    Region(GeometryRegionRole),
}

/// A fully materialized input value in canonical SI units where applicable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum AuthoredInputValue {
    Boolean(bool),
    Integer(#[serde(with = "runmat_types::wire_integer")] i64),
    Real(f64),
    Complex { real: f64, imaginary: f64 },
    String(String),
    Quantity(runmat_types::physical::DimensionedScalar),
    Material(AuthoredMaterialRef),
    Region(GeometryRegionRef),
}

/// One immutable declaration input, including its validation bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredInputDefinition {
    pub input_id: String,
    pub value_type: AuthoredInputType,
    pub default: Option<AuthoredInputValue>,
    pub minimum: Option<AuthoredInputValue>,
    pub maximum: Option<AuthoredInputValue>,
}

impl AuthoredInputDefinition {
    pub fn is_required(&self) -> bool {
        self.default.is_none()
    }

    pub fn validate(&self) -> Result<(), AuthoredInputError> {
        if self.input_id.trim().is_empty() {
            return Err(AuthoredInputError::EmptyId);
        }
        for (label, value) in [
            ("default", self.default.as_ref()),
            ("minimum", self.minimum.as_ref()),
            ("maximum", self.maximum.as_ref()),
        ] {
            if let Some(value) = value {
                value.validate()?;
                if !value.matches(self.value_type) {
                    return Err(AuthoredInputError::WrongValueType(label));
                }
            }
        }
        if (self.minimum.is_some() || self.maximum.is_some()) && !self.value_type.is_ordered() {
            return Err(AuthoredInputError::BoundsOnUnorderedType);
        }
        if let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum) {
            if !minimum.less_than_or_equal(maximum)? {
                return Err(AuthoredInputError::InvalidBounds);
            }
        }
        if let Some(default) = &self.default {
            if let Some(minimum) = &self.minimum {
                if !minimum.less_than_or_equal(default)? {
                    return Err(AuthoredInputError::DefaultOutsideBounds);
                }
            }
            if let Some(maximum) = &self.maximum {
                if !default.less_than_or_equal(maximum)? {
                    return Err(AuthoredInputError::DefaultOutsideBounds);
                }
            }
        }
        Ok(())
    }

    pub fn validate_value(&self, value: &AuthoredInputValue) -> Result<(), AuthoredInputError> {
        value.validate()?;
        if !value.matches(self.value_type) {
            return Err(AuthoredInputError::WrongValueType("binding"));
        }
        if let Some(minimum) = &self.minimum {
            if !minimum.less_than_or_equal(value)? {
                return Err(AuthoredInputError::ValueOutsideBounds);
            }
        }
        if let Some(maximum) = &self.maximum {
            if !value.less_than_or_equal(maximum)? {
                return Err(AuthoredInputError::ValueOutsideBounds);
            }
        }
        Ok(())
    }
}

impl AuthoredInputType {
    pub fn accepts_type(self, actual: Self) -> bool {
        self == actual
            || matches!((self,actual), (Self::Quantity(a),Self::Quantity(b)) if a.is_compatible_with(b))
            || matches!(
                (self, actual),
                (Self::Real, Self::Integer) | (Self::Complex, Self::Integer | Self::Real)
            )
    }
    fn is_ordered(self) -> bool {
        matches!(self, Self::Integer | Self::Real | Self::Quantity(_))
    }
}

impl AuthoredInputValue {
    pub fn value_type(&self) -> AuthoredInputType {
        match self {
            Self::Boolean(_) => AuthoredInputType::Boolean,
            Self::Integer(_) => AuthoredInputType::Integer,
            Self::Real(_) => AuthoredInputType::Real,
            Self::Complex { .. } => AuthoredInputType::Complex,
            Self::String(_) => AuthoredInputType::String,
            Self::Quantity(value) => AuthoredInputType::Quantity(value.descriptor()),
            Self::Material(_) => AuthoredInputType::Material,
            Self::Region(region) => AuthoredInputType::Region(region.role),
        }
    }

    pub fn matches(&self, expected: AuthoredInputType) -> bool {
        expected.accepts_type(self.value_type())
            || matches!(
                (expected, self),
                (AuthoredInputType::Real, Self::Integer(_))
                    | (AuthoredInputType::Complex, Self::Integer(_) | Self::Real(_))
            )
    }

    pub fn validate(&self) -> Result<(), AuthoredInputError> {
        match self {
            Self::Real(value) if !value.is_finite() => Err(AuthoredInputError::NonFiniteValue),
            Self::Quantity(value) if !value.is_finite() => Err(AuthoredInputError::NonFiniteValue),
            Self::Complex { real, imaginary } if !real.is_finite() || !imaginary.is_finite() => {
                Err(AuthoredInputError::NonFiniteValue)
            }
            Self::Material(material)
                if material.module_path.trim().is_empty()
                    || material.material_id.trim().is_empty() =>
            {
                Err(AuthoredInputError::InvalidDeclarationReference)
            }
            Self::Region(region)
                if region.module_path.trim().is_empty()
                    || region.geometry_id.trim().is_empty()
                    || region.region_id.trim().is_empty() =>
            {
                Err(AuthoredInputError::InvalidDeclarationReference)
            }
            _ => Ok(()),
        }
    }

    fn less_than_or_equal(&self, other: &Self) -> Result<bool, AuthoredInputError> {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => Ok(left <= right),
            (Self::Real(left), Self::Real(right)) => Ok(left <= right),
            (Self::Quantity(left), Self::Quantity(right))
                if left.descriptor().is_compatible_with(right.descriptor())
                    && !matches!((left.semantic_kind(), right.semantic_kind()), (Some(a),Some(b)) if a != b) =>
            {
                Ok(left.canonical_value() <= right.canonical_value())
            }
            _ => Err(AuthoredInputError::IncomparableBounds),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredInputError {
    #[error("input id must not be empty")]
    EmptyId,
    #[error("input {0} has the wrong type")]
    WrongValueType(&'static str),
    #[error("only integer, real, and physical-quantity inputs may declare bounds")]
    BoundsOnUnorderedType,
    #[error("input minimum must not exceed its maximum")]
    InvalidBounds,
    #[error("input default lies outside its declared bounds")]
    DefaultOutsideBounds,
    #[error("input binding lies outside its declared bounds")]
    ValueOutsideBounds,
    #[error("input bounds are not comparable")]
    IncomparableBounds,
    #[error("input values must be finite")]
    NonFiniteValue,
    #[error("input declaration reference is incomplete")]
    InvalidDeclarationReference,
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::PhysicalDimension;

    #[test]
    fn quantity_bounds_use_canonical_units_and_reject_bad_defaults() {
        let definition = AuthoredInputDefinition {
            input_id: "load".into(),
            value_type: AuthoredInputType::Quantity((PhysicalDimension::FORCE).into()),
            default: Some(AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_dimension(
                    PhysicalDimension::FORCE,
                    None,
                    6.0,
                )
                .unwrap(),
            )),
            minimum: Some(AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_dimension(
                    PhysicalDimension::FORCE,
                    None,
                    0.0,
                )
                .unwrap(),
            )),
            maximum: Some(AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_dimension(
                    PhysicalDimension::FORCE,
                    None,
                    5.0,
                )
                .unwrap(),
            )),
        };
        assert_eq!(
            definition.validate(),
            Err(AuthoredInputError::DefaultOutsideBounds)
        );
    }
}
