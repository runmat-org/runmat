use std::collections::{BTreeMap, BTreeSet};

use crate::{
    procedure_produces_observation_quantity, AuthoredRegionSource, ObservationQuantity,
    ObservationValueShape, ProcedureStep,
};
use runmat_types::PhysicalDimension;

mod partition;
mod types;
pub use partition::{authored_output_root, retain_authored_outputs};

pub use types::*;

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
                    domain: field_domain(step.procedure, *quantity),
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
                            value: AuthoredOutputValue::Derived(
                                runmat_types::physical::DynamicDimension::quantity_type(
                                    runmat_types::physical::QuantityKind::ComplexPhase,
                                ),
                            ),
                            shape,
                            domain: AuthoredFieldDomain::Real,
                            location,
                        })
                    }
                    AuthoredFieldProjection::Phase => {
                        Err(AuthoredOutputError::ComplexProjectionRequiresComplex)
                    }
                    AuthoredFieldProjection::Level
                        if domain == AuthoredFieldDomain::Complex
                            && shape == ObservationValueShape::Scalar
                            && value
                                == AuthoredOutputValue::Physical(
                                    ObservationQuantity::AcousticPressure,
                                ) =>
                    {
                        Ok(AuthoredOutputType::Field {
                            value: AuthoredOutputValue::Physical(
                                ObservationQuantity::SoundPressureLevel,
                            ),
                            shape,
                            domain: AuthoredFieldDomain::Real,
                            location,
                        })
                    }
                    AuthoredFieldProjection::Level => {
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
            Self::Recovery {
                source,
                output_location,
                recovery,
            } => {
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                    location,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::RecoveryRequiresField);
                };
                if location != crate::FieldLocation::Element
                    || *output_location != crate::FieldLocation::Node
                    || *recovery != crate::RecoveryPolicy::VolumeWeightedAverage
                    || value != AuthoredOutputValue::Physical(ObservationQuantity::VonMisesStress)
                    || shape != ObservationValueShape::Scalar
                {
                    return Err(AuthoredOutputError::UnsupportedRecovery);
                }
                Ok(AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                    location: *output_location,
                })
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
                    AuthoredSpatialReduction::NormalFlux => {
                        if shape != ObservationValueShape::Vector3 {
                            return Err(AuthoredOutputError::ResultantRequiresVectorOrTraction);
                        }
                        Ok(AuthoredOutputType::Reduced {
                            value: AuthoredOutputValue::Derived(
                                output_dimension(value)?
                                    .multiply(region_measure(target)?.into())
                                    .ok_or(AuthoredOutputError::DimensionOverflow)?,
                            ),
                            shape: ObservationValueShape::Scalar,
                        })
                    }
                    AuthoredSpatialReduction::Integral => Ok(AuthoredOutputType::Reduced {
                        value: AuthoredOutputValue::Derived(
                            output_dimension(value)?
                                .multiply(region_measure(target)?.into())
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
                        let mut dimension = quantity
                            .quantity_descriptor()
                            .ok_or(AuthoredOutputError::UnresolvedQuantityDescriptor)?;
                        if !matches!(
                            quantity,
                            ObservationQuantity::ReactionForce
                                | ObservationQuantity::ReactionMoment
                        ) {
                            dimension = dimension
                                .multiply(region_measure(target)?.into())
                                .ok_or(AuthoredOutputError::DimensionOverflow)?;
                        }
                        if matches!(reduction, AuthoredSpatialReduction::Moment { .. }) {
                            dimension = dimension
                                .multiply(PhysicalDimension::LENGTH.into())
                                .ok_or(AuthoredOutputError::DimensionOverflow)?;
                        }
                        Ok(AuthoredOutputType::Reduced {
                            value: AuthoredOutputValue::Derived(dimension),
                            shape: ObservationValueShape::Vector3,
                        })
                    }
                }
            }
            Self::PortReduction {
                reduction,
                source,
                first,
                second,
            } => {
                for region in [first, second] {
                    validate_region(region)?;
                    if region_measure(region)? != PhysicalDimension::AREA {
                        return Err(AuthoredOutputError::PortReductionRequiresFaceRegions);
                    }
                }
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                    location: _,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::SpatialReductionRequiresField);
                };
                if domain != AuthoredFieldDomain::Real || shape != ObservationValueShape::Scalar {
                    return Err(AuthoredOutputError::PortReductionRequiresRealScalar);
                }
                match reduction {
                    AuthoredPortReduction::MeanDifference => Ok(AuthoredOutputType::Reduced {
                        value,
                        shape: ObservationValueShape::Scalar,
                    }),
                    AuthoredPortReduction::TransmissionLoss => {
                        if value
                            != AuthoredOutputValue::Physical(
                                ObservationQuantity::AcousticPortIntensity,
                            )
                        {
                            return Err(AuthoredOutputError::TransmissionLossRequiresPortIntensity);
                        }
                        Ok(AuthoredOutputType::Reduced {
                            value: AuthoredOutputValue::Physical(
                                ObservationQuantity::TransmissionLoss,
                            ),
                            shape: ObservationValueShape::Scalar,
                        })
                    }
                }
            }
            Self::SpatialSelection { source, target } => {
                validate_region(target)?;
                let output @ AuthoredOutputType::Field { .. } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::SpatialSelectionRequiresField);
                };
                Ok(output)
            }
            Self::Interpolation { source, positions } => {
                if positions.is_empty()
                    || positions.iter().any(|position| {
                        position
                            .canonical_value()
                            .iter()
                            .any(|component| !component.is_finite())
                    })
                {
                    return Err(AuthoredOutputError::InvalidInterpolationPoints);
                }
                let AuthoredOutputType::Field {
                    value,
                    shape,
                    domain,
                    ..
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::InterpolationRequiresField);
                };
                Ok(AuthoredOutputType::Probe {
                    value,
                    shape,
                    domain,
                })
            }
            Self::Sampling {
                source,
                sampling,
                domain,
            } => {
                validate_authored_sampling(sampling, domain)?;
                let sample = source.validate(steps, outputs)?;
                if matches!(sample, AuthoredOutputType::History { .. }) {
                    return Err(AuthoredOutputError::NestedHistory);
                }
                Ok(AuthoredOutputType::History {
                    sample: Box::new(sample),
                    domain: domain.clone(),
                })
            }
            Self::SeriesReduction {
                reduction,
                source,
                domain,
                ..
            } => {
                let AuthoredOutputType::History {
                    sample,
                    domain: sampled_domain,
                } = source.validate(steps, outputs)?
                else {
                    return Err(AuthoredOutputError::SeriesReductionRequiresHistory);
                };
                if sampled_domain != *domain {
                    return Err(AuthoredOutputError::SeriesReductionDomainMismatch);
                }
                let (value, shape) = output_value_shape(&sample)
                    .ok_or(AuthoredOutputError::SeriesReductionRequiresNumericHistory)?;
                match reduction {
                    AuthoredSeriesReduction::Minimum
                    | AuthoredSeriesReduction::Maximum
                    | AuthoredSeriesReduction::AbsoluteMaximum => {
                        if shape != ObservationValueShape::Scalar {
                            return Err(AuthoredOutputError::SeriesExtremumRequiresScalar);
                        }
                        Ok(AuthoredOutputType::SeriesExtremum { value })
                    }
                    AuthoredSeriesReduction::Mean { .. }
                    | AuthoredSeriesReduction::RootMeanSquare { .. } => {
                        Ok(AuthoredOutputType::Reduced { value, shape })
                    }
                    AuthoredSeriesReduction::Variance { .. } => Ok(AuthoredOutputType::Reduced {
                        value: AuthoredOutputValue::Derived(
                            output_dimension(value)?
                                .power(2)
                                .ok_or(AuthoredOutputError::DimensionOverflow)?,
                        ),
                        shape,
                    }),
                    AuthoredSeriesReduction::Envelope => {
                        Ok(AuthoredOutputType::Envelope { value, shape })
                    }
                }
            }
        }
    }
}

fn output_value_shape(
    output: &AuthoredOutputType,
) -> Option<(AuthoredOutputValue, ObservationValueShape)> {
    match output {
        AuthoredOutputType::Field { value, shape, .. }
        | AuthoredOutputType::Reduced { value, shape }
        | AuthoredOutputType::Probe { value, shape, .. }
        | AuthoredOutputType::Envelope { value, shape } => Some((*value, shape.clone())),
        AuthoredOutputType::SpatialExtremum { value }
        | AuthoredOutputType::SeriesExtremum { value } => {
            Some((*value, ObservationValueShape::Scalar))
        }
        AuthoredOutputType::History { .. } => None,
    }
}

fn validate_authored_sampling(
    sampling: &crate::SamplingPolicy,
    domain: &AuthoredSeriesDomain,
) -> Result<(), AuthoredOutputError> {
    let matches = matches!(
        (sampling, domain),
        (
            crate::SamplingPolicy::TimeSeconds(_),
            AuthoredSeriesDomain::Time
        ) | (
            crate::SamplingPolicy::FrequencyHertz(_),
            AuthoredSeriesDomain::Frequency
        ) | (
            crate::SamplingPolicy::LoadFactors(_),
            AuthoredSeriesDomain::LoadFactor
        ) | (
            crate::SamplingPolicy::StudyParameterValues(_),
            AuthoredSeriesDomain::StudyParameter { .. }
        ) | (
            crate::SamplingPolicy::ModeIndices(_),
            AuthoredSeriesDomain::Mode
        )
    );
    matches
        .then_some(())
        .ok_or(AuthoredOutputError::SamplingDomainMismatch)
}

fn output_dimension(
    value: AuthoredOutputValue,
) -> Result<runmat_types::physical::DynamicDimension, AuthoredOutputError> {
    Ok(match value {
        AuthoredOutputValue::Physical(quantity) => quantity
            .quantity_descriptor()
            .ok_or(AuthoredOutputError::UnresolvedQuantityDescriptor)?,
        AuthoredOutputValue::Derived(dimension) => dimension,
        AuthoredOutputValue::Dimensionless => PhysicalDimension::DIMENSIONLESS.into(),
    })
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

pub(super) fn field_domain(
    procedure: crate::AnalysisProcedure,
    quantity: crate::ObservationQuantity,
) -> AuthoredFieldDomain {
    match procedure {
        // Cycle averages (loss density, active intensity) are real quantities of a harmonic
        // analysis.
        crate::AnalysisProcedure::HarmonicResponse
        | crate::AnalysisProcedure::PrestressedHarmonicResponse
            if matches!(
                quantity,
                crate::ObservationQuantity::JoulePowerDensity
                    | crate::ObservationQuantity::AcousticIntensity
                    | crate::ObservationQuantity::AcousticPortIntensity
            ) =>
        {
            AuthoredFieldDomain::Real
        }
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
    #[error("output quantity requires explicit producer or basis dimensions")]
    UnresolvedQuantityDescriptor,
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
    #[error("field recovery requires a field source")]
    RecoveryRequiresField,
    #[error(
        "field recovery currently supports scalar element von Mises stress to nodes using volume_weighted_average"
    )]
    UnsupportedRecovery,
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
    #[error("a port reduction reads two face regions")]
    PortReductionRequiresFaceRegions,
    #[error("a port reduction reads a real scalar field")]
    PortReductionRequiresRealScalar,
    #[error("transmission loss reads the acoustic port intensity of a harmonic analysis")]
    TransmissionLossRequiresPortIntensity,
    #[error("spatial selection requires a field source")]
    SpatialSelectionRequiresField,
    #[error("interpolation requires a field source")]
    InterpolationRequiresField,
    #[error("interpolation requires one or more finite physical positions")]
    InvalidInterpolationPoints,
    #[error("a history cannot be sampled into a nested history")]
    NestedHistory,
    #[error("sampling coordinates do not match their independent domain")]
    SamplingDomainMismatch,
    #[error("series reduction requires a history source")]
    SeriesReductionRequiresHistory,
    #[error("series reduction domain does not match its history")]
    SeriesReductionDomainMismatch,
    #[error("series reduction requires numeric history samples")]
    SeriesReductionRequiresNumericHistory,
    #[error("series extrema require scalar history samples")]
    SeriesExtremumRequiresScalar,
}
