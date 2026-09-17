use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    procedure_produces_observation_quantity, AuthoredRegionSource, ObservationQuantity,
    ObservationValueShape, ProcedureStep,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredOutput {
    pub output_id: String,
    pub expression: AuthoredOutputExpression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredOutputExpression {
    AnalysisField {
        step_id: String,
        quantity: ObservationQuantity,
    },
    OutputReference {
        output_id: String,
    },
    Projection {
        projection: AuthoredFieldProjection,
        source: Box<AuthoredOutputExpression>,
    },
    SpatialReduction {
        reduction: AuthoredSpatialReduction,
        source: Box<AuthoredOutputExpression>,
        target: AuthoredRegionSource,
        ties: AuthoredExtremaTiePolicy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredFieldProjection {
    Magnitude,
    Component(AuthoredVectorComponent),
    Real,
    Imaginary,
    Phase,
    VonMises,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredVectorComponent {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSpatialReduction {
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredExtremaTiePolicy {
    FirstCanonical,
    AllLocations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredOutputType {
    Field {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
        domain: AuthoredFieldDomain,
    },
    SpatialExtremum {
        value: AuthoredOutputValue,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredOutputValue {
    Physical(ObservationQuantity),
    Dimensionless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredFieldDomain {
    Real,
    Complex,
    Statistical,
}

pub fn validate_authored_outputs(
    outputs: &[AuthoredOutput],
    steps: &[ProcedureStep],
) -> Result<(), AuthoredOutputError> {
    let steps = steps
        .iter()
        .map(|step| (step.step_id.0.as_str(), step))
        .collect::<BTreeMap<_, _>>();
    let mut types = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for output in outputs {
        if output.output_id.trim().is_empty() {
            return Err(AuthoredOutputError::EmptyId);
        }
        if !ids.insert(output.output_id.as_str()) {
            return Err(AuthoredOutputError::DuplicateId(output.output_id.clone()));
        }
        let value_type = output.expression.validate(&steps, &types)?;
        types.insert(output.output_id.as_str(), value_type);
    }
    Ok(())
}

impl AuthoredOutputExpression {
    pub fn value_type(
        &self,
        outputs: &BTreeMap<&str, AuthoredOutputType>,
        steps: &BTreeMap<&str, &ProcedureStep>,
    ) -> Result<AuthoredOutputType, AuthoredOutputError> {
        self.validate(steps, outputs)
    }

    fn validate(
        &self,
        steps: &BTreeMap<&str, &ProcedureStep>,
        outputs: &BTreeMap<&str, AuthoredOutputType>,
    ) -> Result<AuthoredOutputType, AuthoredOutputError> {
        match self {
            Self::AnalysisField { step_id, quantity } => {
                let step = steps
                    .get(step_id.as_str())
                    .ok_or_else(|| AuthoredOutputError::UnknownStep(step_id.clone()))?;
                if !procedure_produces_observation_quantity(step, *quantity) {
                    return Err(AuthoredOutputError::UnsupportedQuantity {
                        step_id: step_id.clone(),
                        quantity: *quantity,
                    });
                }
                Ok(AuthoredOutputType::Field {
                    value: AuthoredOutputValue::Physical(*quantity),
                    shape: quantity.canonical_value_shape(),
                    domain: field_domain(step.procedure),
                })
            }
            Self::OutputReference { output_id } => outputs
                .get(output_id.as_str())
                .cloned()
                .ok_or_else(|| AuthoredOutputError::UnknownOrForwardOutput(output_id.clone())),
            Self::Projection { projection, source } => {
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::ProjectionRequiresField);
                };
                match projection {
                    AuthoredFieldProjection::Magnitude
                        if shape == ObservationValueShape::Vector3 =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value,
                            shape: ObservationValueShape::Scalar,
                            domain: match domain {
                                AuthoredFieldDomain::Complex => AuthoredFieldDomain::Real,
                                domain => domain,
                            },
                        })
                    }
                    AuthoredFieldProjection::Magnitude
                        if shape == ObservationValueShape::Scalar
                            && domain == AuthoredFieldDomain::Complex =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value,
                            shape,
                            domain: AuthoredFieldDomain::Real,
                        })
                    }
                    AuthoredFieldProjection::Magnitude => {
                        Err(AuthoredOutputError::MagnitudeRequiresVector)
                    }
                    AuthoredFieldProjection::Component(_)
                        if shape == ObservationValueShape::Vector3 =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value,
                            shape: ObservationValueShape::Scalar,
                            domain,
                        })
                    }
                    AuthoredFieldProjection::Component(_) => {
                        Err(AuthoredOutputError::ComponentRequiresVector)
                    }
                    AuthoredFieldProjection::Real | AuthoredFieldProjection::Imaginary
                        if domain == AuthoredFieldDomain::Complex =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value,
                            shape,
                            domain: AuthoredFieldDomain::Real,
                        })
                    }
                    AuthoredFieldProjection::Real | AuthoredFieldProjection::Imaginary => {
                        Err(AuthoredOutputError::ComplexProjectionRequiresComplex)
                    }
                    AuthoredFieldProjection::Phase if domain == AuthoredFieldDomain::Complex => {
                        Ok(AuthoredOutputType::Field {
                            value: AuthoredOutputValue::Dimensionless,
                            shape,
                            domain: AuthoredFieldDomain::Real,
                        })
                    }
                    AuthoredFieldProjection::Phase => {
                        Err(AuthoredOutputError::ComplexProjectionRequiresComplex)
                    }
                    AuthoredFieldProjection::VonMises
                        if value == AuthoredOutputValue::Physical(ObservationQuantity::Stress)
                            && shape == ObservationValueShape::SymmetricTensor3
                            && domain == AuthoredFieldDomain::Real =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value: AuthoredOutputValue::Physical(
                                ObservationQuantity::VonMisesStress,
                            ),
                            shape: ObservationValueShape::Scalar,
                            domain,
                        })
                    }
                    AuthoredFieldProjection::VonMises => {
                        Err(AuthoredOutputError::VonMisesRequiresRealStress)
                    }
                }
            }
            Self::SpatialReduction { source, target, .. } => {
                validate_region(target)?;
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::SpatialReductionRequiresField);
                };
                if shape != ObservationValueShape::Scalar {
                    return Err(AuthoredOutputError::SpatialReductionRequiresScalar);
                }
                if domain == AuthoredFieldDomain::Complex {
                    return Err(AuthoredOutputError::SpatialReductionRequiresReal);
                }
                Ok(AuthoredOutputType::SpatialExtremum { value })
            }
        }
    }
}

fn field_domain(procedure: crate::AnalysisProcedure) -> AuthoredFieldDomain {
    match procedure {
        crate::AnalysisProcedure::HarmonicResponse
        | crate::AnalysisProcedure::PrestressedHarmonicResponse => AuthoredFieldDomain::Complex,
        crate::AnalysisProcedure::RandomResponse
        | crate::AnalysisProcedure::ModalResponseSpectrum => AuthoredFieldDomain::Statistical,
        _ => AuthoredFieldDomain::Real,
    }
}

fn validate_region(target: &AuthoredRegionSource) -> Result<(), AuthoredOutputError> {
    match target {
        AuthoredRegionSource::Geometry(region)
            if !region.module_path.trim().is_empty()
                && !region.geometry_id.trim().is_empty()
                && !region.region_id.trim().is_empty() =>
        {
            Ok(())
        }
        AuthoredRegionSource::ModelInput(input) if !input.input_id.trim().is_empty() => Ok(()),
        _ => Err(AuthoredOutputError::InvalidRegion),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredOutputError {
    #[error("output id must not be empty")]
    EmptyId,
    #[error("output `{0}` is duplicated")]
    DuplicateId(String),
    #[error("output references unknown analysis step `{0}`")]
    UnknownStep(String),
    #[error("analysis step `{step_id}` does not produce `{quantity:?}`")]
    UnsupportedQuantity {
        step_id: String,
        quantity: ObservationQuantity,
    },
    #[error("output references unknown or forward output `{0}`")]
    UnknownOrForwardOutput(String),
    #[error("field projection requires a field source")]
    ProjectionRequiresField,
    #[error("magnitude requires a three-component vector field or complex scalar field")]
    MagnitudeRequiresVector,
    #[error("component requires a three-component vector field")]
    ComponentRequiresVector,
    #[error("real, imaginary, and phase projections require a complex field")]
    ComplexProjectionRequiresComplex,
    #[error("von_mises requires a real symmetric stress field")]
    VonMisesRequiresRealStress,
    #[error("spatial reduction requires a field source")]
    SpatialReductionRequiresField,
    #[error("spatial extrema require an explicit scalar projection")]
    SpatialReductionRequiresScalar,
    #[error("spatial extrema require a real or statistical representation")]
    SpatialReductionRequiresReal,
    #[error("output region target is invalid")]
    InvalidRegion,
}
