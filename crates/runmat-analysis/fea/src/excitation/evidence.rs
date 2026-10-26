use runmat_analysis_core::{PhysicalQuantityKind, ProjectionRequirement, SignalSampleApproach};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

mod serialization;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialIntegrationRule {
    UniformResultant,
    BoundaryDunavantSeven,
    VolumeKeastFifteen,
    ExplicitConsistentElement,
    ReferencePointGeneralized,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialProjectionEvidence {
    pub excitation_id: String,
    pub signal_id: String,
    pub quantity: PhysicalQuantityKind,
    pub sample: ProjectionSample,
    pub analysis_projection_digest: Digest,
    pub integration_rule: SpatialIntegrationRule,
    pub resolved_entity_count: usize,
    pub resolved_node_count: usize,
    pub reference_point_m: [f64; 3],
    pub balance: SpatialProjectionBalance,
    pub requirements: ProjectionRequirement,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "domain", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectionSample {
    Static,
    TimeSeconds {
        value: f64,
        approach: SignalSampleApproach,
    },
    FrequencyHertz {
        value: f64,
        approach: SignalSampleApproach,
    },
    StudyParameter {
        parameter_id: String,
        canonical_value: f64,
        approach: SignalSampleApproach,
    },
}

impl SpatialProjectionEvidence {
    pub fn resultant_relative_error(&self) -> f64 {
        match self.balance {
            SpatialProjectionBalance::ScalarIntegral { relative_error, .. }
            | SpatialProjectionBalance::VectorResultant {
                resultant_relative_error: relative_error,
                ..
            } => relative_error,
        }
    }

    pub fn moment_relative_error(&self) -> Option<f64> {
        match self.balance {
            SpatialProjectionBalance::ScalarIntegral { .. } => None,
            SpatialProjectionBalance::VectorResultant {
                moment_relative_error,
                ..
            } => Some(moment_relative_error),
        }
    }

    pub fn validate(&self) -> Result<(), SpatialProjectionEvidenceError> {
        if self.excitation_id.trim().is_empty()
            || self.signal_id.trim().is_empty()
            || self.resolved_entity_count == 0
            || (self.resolved_node_count == 0
                && self.integration_rule != SpatialIntegrationRule::ReferencePointGeneralized)
            || self
                .reference_point_m
                .iter()
                .any(|value| !value.is_finite())
            || !self.requirements.relative_tolerance.is_finite()
            || self.requirements.relative_tolerance < 0.0
            || !self.sample.is_valid()
        {
            return Err(SpatialProjectionEvidenceError::InvalidMetadata);
        }
        let expected_accepted = match self.balance {
            SpatialProjectionBalance::ScalarIntegral {
                requested_integral,
                projected_integral,
                normalization_scale,
                relative_error,
            } => {
                if self.requirements.preserve_moment
                    || !scalar_is_finite(requested_integral)
                    || !scalar_is_finite(projected_integral)
                    || !normalization_scale.is_finite()
                    || normalization_scale
                        < requested_integral.real.hypot(requested_integral.imaginary)
                    || normalization_scale
                        < projected_integral.real.hypot(projected_integral.imaginary)
                    || !relative_error.is_finite()
                    || relative_error < 0.0
                {
                    return Err(SpatialProjectionEvidenceError::InvalidBalance);
                }
                let expected_error = if normalization_scale == 0.0 {
                    0.0
                } else {
                    (requested_integral.real - projected_integral.real)
                        .hypot(requested_integral.imaginary - projected_integral.imaginary)
                        / normalization_scale
                };
                if !metric_matches(relative_error, expected_error) {
                    return Err(SpatialProjectionEvidenceError::InvalidBalance);
                }
                !self.requirements.preserve_resultant
                    || relative_error <= self.requirements.relative_tolerance
            }
            SpatialProjectionBalance::VectorResultant {
                requested_resultant,
                projected_resultant,
                requested_moment,
                projected_moment,
                resultant_scale,
                moment_scale,
                resultant_relative_error,
                moment_relative_error,
            } => {
                if !vector_is_finite(requested_resultant)
                    || !vector_is_finite(projected_resultant)
                    || !vector_is_finite(requested_moment)
                    || !vector_is_finite(projected_moment)
                    || !resultant_scale.is_finite()
                    || !moment_scale.is_finite()
                    || resultant_scale < complex_vector_norm(requested_resultant)
                    || resultant_scale < complex_vector_norm(projected_resultant)
                    || moment_scale < complex_vector_norm(requested_moment)
                    || moment_scale < complex_vector_norm(projected_moment)
                    || !resultant_relative_error.is_finite()
                    || !moment_relative_error.is_finite()
                    || resultant_relative_error < 0.0
                    || moment_relative_error < 0.0
                {
                    return Err(SpatialProjectionEvidenceError::InvalidBalance);
                }
                let expected_resultant_error = relative_complex_vector_error(
                    requested_resultant,
                    projected_resultant,
                    resultant_scale,
                );
                let expected_moment_error =
                    relative_complex_vector_error(requested_moment, projected_moment, moment_scale);
                if !metric_matches(resultant_relative_error, expected_resultant_error)
                    || !metric_matches(moment_relative_error, expected_moment_error)
                {
                    return Err(SpatialProjectionEvidenceError::InvalidBalance);
                }
                (!self.requirements.preserve_resultant
                    || resultant_relative_error <= self.requirements.relative_tolerance)
                    && (!self.requirements.preserve_moment
                        || moment_relative_error <= self.requirements.relative_tolerance)
            }
        };
        if self.accepted != expected_accepted {
            return Err(SpatialProjectionEvidenceError::InvalidAcceptance);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialProjectionEvidenceError {
    InvalidMetadata,
    InvalidBalance,
    InvalidAcceptance,
}

impl std::fmt::Display for SpatialProjectionEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid spatial projection evidence: {self:?}")
    }
}

impl std::error::Error for SpatialProjectionEvidenceError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpatialProjectionBalance {
    ScalarIntegral {
        requested_integral: ProjectionScalar,
        projected_integral: ProjectionScalar,
        normalization_scale: f64,
        relative_error: f64,
    },
    VectorResultant {
        requested_resultant: ProjectionVector3,
        projected_resultant: ProjectionVector3,
        requested_moment: ProjectionVector3,
        projected_moment: ProjectionVector3,
        resultant_scale: f64,
        moment_scale: f64,
        resultant_relative_error: f64,
        moment_relative_error: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionScalar {
    pub real: f64,
    pub imaginary: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionVector3 {
    pub real: [f64; 3],
    pub imaginary: [f64; 3],
}

impl ProjectionSample {
    fn is_valid(&self) -> bool {
        match self {
            Self::Static => true,
            Self::TimeSeconds { value, .. } | Self::FrequencyHertz { value, .. } => {
                value.is_finite()
            }
            Self::StudyParameter {
                parameter_id,
                canonical_value,
                ..
            } => !parameter_id.trim().is_empty() && canonical_value.is_finite(),
        }
    }
}

fn scalar_is_finite(value: ProjectionScalar) -> bool {
    value.real.is_finite() && value.imaginary.is_finite()
}

fn vector_is_finite(value: ProjectionVector3) -> bool {
    value
        .real
        .iter()
        .chain(&value.imaginary)
        .all(|value| value.is_finite())
}

fn complex_vector_norm(value: ProjectionVector3) -> f64 {
    value
        .real
        .iter()
        .chain(&value.imaginary)
        .map(|component| component.powi(2))
        .sum::<f64>()
        .sqrt()
}

fn relative_complex_vector_error(
    requested: ProjectionVector3,
    projected: ProjectionVector3,
    scale: f64,
) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    requested
        .real
        .iter()
        .chain(&requested.imaginary)
        .zip(projected.real.iter().chain(&projected.imaginary))
        .map(|(requested, projected)| (requested - projected).powi(2))
        .sum::<f64>()
        .sqrt()
        / scale
}

fn metric_matches(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= f64::EPSILON * 8.0 * actual.abs().max(expected.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_evidence() -> SpatialProjectionEvidence {
        SpatialProjectionEvidence {
            excitation_id: "heat-source".to_string(),
            signal_id: "heat-history".to_string(),
            quantity: PhysicalQuantityKind::HeatGenerationDensity,
            sample: ProjectionSample::TimeSeconds {
                value: 1.0,
                approach: SignalSampleApproach::At,
            },
            analysis_projection_digest: Digest::sha256(b"projection"),
            integration_rule: SpatialIntegrationRule::VolumeKeastFifteen,
            resolved_entity_count: 1,
            resolved_node_count: 4,
            reference_point_m: [0.0; 3],
            balance: SpatialProjectionBalance::ScalarIntegral {
                requested_integral: ProjectionScalar {
                    real: 2.0,
                    imaginary: 0.0,
                },
                projected_integral: ProjectionScalar {
                    real: 2.0,
                    imaginary: 0.0,
                },
                normalization_scale: 2.0,
                relative_error: 0.0,
            },
            requirements: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-12,
            },
            accepted: true,
        }
    }

    #[test]
    fn projection_evidence_rejects_tampered_metadata_balance_and_acceptance() {
        let evidence = scalar_evidence();
        assert_eq!(evidence.validate(), Ok(()));

        let mut invalid_metadata = evidence.clone();
        invalid_metadata.sample = ProjectionSample::TimeSeconds {
            value: f64::NAN,
            approach: SignalSampleApproach::At,
        };
        assert_eq!(
            invalid_metadata.validate(),
            Err(SpatialProjectionEvidenceError::InvalidMetadata)
        );

        let mut invalid_balance = evidence.clone();
        let SpatialProjectionBalance::ScalarIntegral { relative_error, .. } =
            &mut invalid_balance.balance
        else {
            unreachable!()
        };
        *relative_error = 0.5;
        assert_eq!(
            invalid_balance.validate(),
            Err(SpatialProjectionEvidenceError::InvalidBalance)
        );

        let mut invalid_acceptance = evidence;
        invalid_acceptance.accepted = false;
        assert_eq!(
            invalid_acceptance.validate(),
            Err(SpatialProjectionEvidenceError::InvalidAcceptance)
        );
    }
}

// Complex real/imaginary components share the enclosing field's descriptor.
impl runmat_types::physical::QuantityStorage for ProjectionScalar {
    fn all_finite(&self) -> bool {
        self.real.is_finite() && self.imaginary.is_finite()
    }
}
impl runmat_types::physical::QuantityStorage for ProjectionVector3 {
    fn all_finite(&self) -> bool {
        self.real
            .iter()
            .chain(&self.imaginary)
            .all(|value| value.is_finite())
    }
}
