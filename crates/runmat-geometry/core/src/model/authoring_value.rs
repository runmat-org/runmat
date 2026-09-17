use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{GeometryContractError, GeometryFeatureRef, GeometryRegionRef};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryScalarExpression {
    Literal {
        value: f64,
    },
    Parameter {
        name: String,
    },
    Negate {
        value: Box<Self>,
    },
    Binary {
        operator: GeometryScalarOperator,
        left: Box<Self>,
        right: Box<Self>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryScalarOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum GeometryAuthoringValue {
    Scalar(GeometryScalarExpression),
    Vector([GeometryScalarExpression; 3]),
    Matrix(Vec<GeometryScalarExpression>),
    Text(String),
    Boolean(bool),
    Parameter(String),
    Feature(GeometryFeatureRef),
    Region(GeometryRegionRef),
}

impl GeometryScalarExpression {
    pub(crate) fn validate(
        &self,
        parameters: &BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        match self {
            Self::Literal { value } if value.is_finite() => Ok(()),
            Self::Literal { .. } => Err(GeometryContractError::invalid(
                "geometry expression",
                "literal values must be finite",
            )),
            Self::Parameter { name } if parameters.contains(name.as_str()) => Ok(()),
            Self::Parameter { .. } => Err(GeometryContractError::invalid(
                "geometry expression",
                "parameter reference must name a declared geometry input",
            )),
            Self::Negate { value } => value.validate(parameters),
            Self::Binary { left, right, .. } => {
                left.validate(parameters)?;
                right.validate(parameters)
            }
        }
    }

    pub(crate) fn validate_positive_literal(
        &self,
        field: &str,
        parameters: &BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        self.validate(parameters)?;
        if matches!(self, Self::Literal { value } if *value <= 0.0) {
            return Err(GeometryContractError::invalid(
                field,
                "literal value must be positive",
            ));
        }
        Ok(())
    }
}

impl GeometryAuthoringValue {
    pub(crate) fn validate(
        &self,
        parameters: &BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        match self {
            Self::Scalar(value) => value.validate(parameters),
            Self::Vector(values) => validate_expressions(values, parameters),
            Self::Matrix(values) => validate_expressions(values, parameters),
            Self::Text(value) | Self::Parameter(value) if value.trim().is_empty() => Err(
                GeometryContractError::invalid("geometry value", "must not be empty"),
            ),
            Self::Feature(reference) if reference.0.trim().is_empty() => Err(
                GeometryContractError::invalid("geometry feature value", "must not be empty"),
            ),
            Self::Region(reference)
                if reference.module_path.trim().is_empty()
                    || reference.geometry_id.trim().is_empty()
                    || reference.region_id.trim().is_empty() =>
            {
                Err(GeometryContractError::invalid(
                    "geometry region value",
                    "module, geometry, and region identity must not be empty",
                ))
            }
            Self::Text(_) | Self::Boolean(_) | Self::Feature(_) | Self::Region(_) => Ok(()),
            Self::Parameter(value) if parameters.contains(value.as_str()) => Ok(()),
            Self::Parameter(_) => Err(GeometryContractError::invalid(
                "geometry value",
                "parameter reference must name a declared geometry input",
            )),
        }
    }
}

fn validate_expressions(
    values: &[GeometryScalarExpression],
    parameters: &BTreeSet<&str>,
) -> Result<(), GeometryContractError> {
    for value in values {
        value.validate(parameters)?;
    }
    Ok(())
}

impl From<f64> for GeometryScalarExpression {
    fn from(value: f64) -> Self {
        Self::Literal { value }
    }
}
