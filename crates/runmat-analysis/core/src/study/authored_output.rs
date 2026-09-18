use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    procedure_produces_observation_quantity, AuthoredRegionSource, FieldLocation,
    ObservationQuantity, ObservationValueShape, ProcedureStep,
};
use runmat_types::PhysicalDimension;

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
        location: FieldLocation,
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
    Principal(AuthoredPrincipalSelection),
    VonMises,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredPrincipalSelection {
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredVectorComponent {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSpatialReduction {
    Integral,
    Mean,
    RootMeanSquare,
    EuclideanNorm,
    Minimum,
    Maximum,
    Resultant,
    Moment { about: crate::Position },
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
        location: FieldLocation,
    },
    SpatialExtremum {
        value: AuthoredOutputValue,
    },
    Reduced {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredOutputValue {
    Physical(ObservationQuantity),
    Derived(PhysicalDimension),
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
    authored_output_types(outputs, steps).map(|_| ())
}

pub fn authored_output_types(
    outputs: &[AuthoredOutput],
    steps: &[ProcedureStep],
) -> Result<BTreeMap<String, AuthoredOutputType>, AuthoredOutputError> {
    let steps = steps
        .iter()
        .map(|step| (step.step_id.0.as_str(), step))
        .collect::<BTreeMap<_, _>>();
    let mut types = BTreeMap::<String, AuthoredOutputType>::new();
    let mut ids = BTreeSet::new();
    for output in outputs {
        if output.output_id.trim().is_empty() {
            return Err(AuthoredOutputError::EmptyId);
        }
        if !ids.insert(output.output_id.as_str()) {
            return Err(AuthoredOutputError::DuplicateId(output.output_id.clone()));
        }
        let borrowed = types
            .iter()
            .map(|(id, value_type)| (id.as_str(), value_type.clone()))
            .collect::<BTreeMap<_, _>>();
        let value_type = output.expression.validate(&steps, &borrowed)?;
        types.insert(output.output_id.clone(), value_type);
    }
    Ok(types)
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
            Self::AnalysisField {
                step_id,
                quantity,
                location,
            } => {
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
                    location: *location,
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
                    location,
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
                            location,
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
                            location,
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
                            location,
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
                            location,
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
                            location,
                        })
                    }
                    AuthoredFieldProjection::Phase => {
                        Err(AuthoredOutputError::ComplexProjectionRequiresComplex)
                    }
                    AuthoredFieldProjection::Principal(_)
                        if matches!(
                            value,
                            AuthoredOutputValue::Physical(
                                ObservationQuantity::Stress | ObservationQuantity::Strain
                            )
                        ) && shape == ObservationValueShape::SymmetricTensor3
                            && domain == AuthoredFieldDomain::Real =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value,
                            shape: ObservationValueShape::Scalar,
                            domain,
                            location,
                        })
                    }
                    AuthoredFieldProjection::Principal(_) => {
                        Err(AuthoredOutputError::PrincipalRequiresRealTensor)
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
                            location,
                        })
                    }
                    AuthoredFieldProjection::VonMises => {
                        Err(AuthoredOutputError::VonMisesRequiresRealStress)
                    }
                }
            }
            Self::SpatialReduction {
                reduction,
                source,
                target,
                ..
            } => {
                validate_region(target)?;
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                    location: _,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::SpatialReductionRequiresField);
                };
                if domain == AuthoredFieldDomain::Complex {
                    return Err(AuthoredOutputError::SpatialReductionRequiresReal);
                }
                match reduction {
                    AuthoredSpatialReduction::Minimum | AuthoredSpatialReduction::Maximum => {
                        if shape != ObservationValueShape::Scalar {
                            return Err(AuthoredOutputError::SpatialReductionRequiresScalar);
                        }
                        Ok(AuthoredOutputType::SpatialExtremum { value })
                    }
                    AuthoredSpatialReduction::Mean | AuthoredSpatialReduction::RootMeanSquare => {
                        Ok(AuthoredOutputType::Reduced { value, shape })
                    }
                    AuthoredSpatialReduction::EuclideanNorm => Ok(AuthoredOutputType::Reduced {
                        value,
                        shape: ObservationValueShape::Scalar,
                    }),
                    AuthoredSpatialReduction::Integral => Ok(AuthoredOutputType::Reduced {
                        value: AuthoredOutputValue::Derived(
                            output_dimension(value)
                                .multiply(region_measure(target)?)
                                .ok_or(AuthoredOutputError::DimensionOverflow)?,
                        ),
                        shape,
                    }),
                    AuthoredSpatialReduction::Resultant
                    | AuthoredSpatialReduction::Moment { .. } => {
                        let AuthoredOutputValue::Physical(quantity) = value else {
                            return Err(AuthoredOutputError::ResultantRequiresPhysicalField);
                        };
                        if matches!(reduction, AuthoredSpatialReduction::Moment { .. })
                            && quantity == ObservationQuantity::ReactionMoment
                        {
                            return Err(AuthoredOutputError::SecondMomentOfReactionMoment);
                        }
                        let compatible = shape == ObservationValueShape::Vector3
                            || quantity == ObservationQuantity::Stress
                            || matches!(
                                quantity,
                                ObservationQuantity::AcousticPressure
                                    | ObservationQuantity::FluidPressure
                                    | ObservationQuantity::ContactPressure
                            );
                        if !compatible {
                            return Err(AuthoredOutputError::ResultantRequiresVectorOrTraction);
                        }
                        let mut dimension = quantity.physical_dimension();
                        if !matches!(
                            quantity,
                            ObservationQuantity::ReactionForce
                                | ObservationQuantity::ReactionMoment
                        ) {
                            dimension = dimension
                                .multiply(region_measure(target)?)
                                .ok_or(AuthoredOutputError::DimensionOverflow)?;
                        }
                        if matches!(reduction, AuthoredSpatialReduction::Moment { .. }) {
                            dimension = dimension
                                .multiply(PhysicalDimension::LENGTH)
                                .ok_or(AuthoredOutputError::DimensionOverflow)?;
                        }
                        Ok(AuthoredOutputType::Reduced {
                            value: AuthoredOutputValue::Derived(dimension),
                            shape: ObservationValueShape::Vector3,
                        })
                    }
                }
            }
        }
    }
}

fn output_dimension(value: AuthoredOutputValue) -> PhysicalDimension {
    match value {
        AuthoredOutputValue::Physical(quantity) => quantity.physical_dimension(),
        AuthoredOutputValue::Derived(dimension) => dimension,
        AuthoredOutputValue::Dimensionless => PhysicalDimension::DIMENSIONLESS,
    }
}

fn region_measure(target: &AuthoredRegionSource) -> Result<PhysicalDimension, AuthoredOutputError> {
    let AuthoredRegionSource::Geometry(region) = target else {
        return Err(AuthoredOutputError::InvalidRegion);
    };
    Ok(match region.role {
        runmat_geometry_core::GeometryRegionRole::Vertex => {
            return Err(AuthoredOutputError::NonMeasurableRegion)
        }
        runmat_geometry_core::GeometryRegionRole::Edge => PhysicalDimension::LENGTH,
        runmat_geometry_core::GeometryRegionRole::Face => PhysicalDimension::AREA,
        runmat_geometry_core::GeometryRegionRole::Solid
        | runmat_geometry_core::GeometryRegionRole::Cell => PhysicalDimension::VOLUME,
    })
}

pub(super) fn field_domain(procedure: crate::AnalysisProcedure) -> AuthoredFieldDomain {
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
    #[error("principal requires a real symmetric stress or strain field")]
    PrincipalRequiresRealTensor,
    #[error("spatial reduction requires a field source")]
    SpatialReductionRequiresField,
    #[error("spatial extrema require an explicit scalar projection")]
    SpatialReductionRequiresScalar,
    #[error("spatial extrema require a real or statistical representation")]
    SpatialReductionRequiresReal,
    #[error("resultant and moment require a physical field")]
    ResultantRequiresPhysicalField,
    #[error("resultant and moment require a vector field, boundary stress, or boundary pressure")]
    ResultantRequiresVectorOrTraction,
    #[error("moment cannot take a second moment of reaction moments")]
    SecondMomentOfReactionMoment,
    #[error("derived output dimension overflows the supported exponent range")]
    DimensionOverflow,
    #[error("spatial integration requires an edge, face, solid, or cell region")]
    NonMeasurableRegion,
    #[error("output region target is invalid")]
    InvalidRegion,
}
