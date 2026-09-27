use runmat_analysis_core::{ErrorContributionKind, RequiredAccuracyEvidence};

use crate::results::accuracy::{AccuracyEvidence, AccuracyEvidenceScope, AccuracyMeasure};

use super::{
    AccuracyEvaluationRequest, AccuracySourceKind, AnalysisExecutionError, AnalysisExecutionResult,
    AnalysisWorkloadIdentity, SolverResultIdentity, StructuralLinearNumericalReceiptArtifact,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralRandomResponseNumericalReceipt,
    ThermalNumericalReceiptArtifact,
};

#[derive(Clone, Debug, PartialEq)]
pub enum AccuracySourceReceipt {
    Thermal(ThermalNumericalReceiptArtifact),
    StructuralLinear(StructuralLinearNumericalReceiptArtifact),
    StructuralModalResponseSpectrum(StructuralModalResponseSpectrumNumericalReceipt),
    StructuralRandomResponse(StructuralRandomResponseNumericalReceipt),
}

impl AccuracySourceReceipt {
    pub fn from_canonical_bytes(
        kind: AccuracySourceKind,
        bytes: &[u8],
    ) -> AnalysisExecutionResult<Self> {
        let receipt = match kind {
            AccuracySourceKind::Thermal => Self::Thermal(
                ThermalNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::StructuralLinear => Self::StructuralLinear(
                StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::StructuralModalResponseSpectrum => {
                Self::StructuralModalResponseSpectrum(
                    StructuralModalResponseSpectrumNumericalReceipt::from_canonical_bytes(bytes)?,
                )
            }
            AccuracySourceKind::StructuralRandomResponse => Self::StructuralRandomResponse(
                StructuralRandomResponseNumericalReceipt::from_canonical_bytes(bytes)?,
            ),
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub const fn kind(&self) -> AccuracySourceKind {
        match self {
            Self::Thermal(_) => AccuracySourceKind::Thermal,
            Self::StructuralLinear(_) => AccuracySourceKind::StructuralLinear,
            Self::StructuralModalResponseSpectrum(_) => {
                AccuracySourceKind::StructuralModalResponseSpectrum
            }
            Self::StructuralRandomResponse(_) => AccuracySourceKind::StructuralRandomResponse,
        }
    }

    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        match self {
            Self::Thermal(receipt) => receipt.source_workload,
            Self::StructuralLinear(receipt) => receipt.source_workload,
            Self::StructuralModalResponseSpectrum(receipt) => receipt.source_workload(),
            Self::StructuralRandomResponse(receipt) => receipt.source_workload(),
        }
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        match self {
            Self::Thermal(receipt) => receipt.source_result(),
            Self::StructuralLinear(receipt) => receipt.source_result(),
            Self::StructuralModalResponseSpectrum(receipt) => receipt.source_result(),
            Self::StructuralRandomResponse(receipt) => receipt.source_result(),
        }
    }

    pub const fn identity(&self) -> super::NumericalReceiptIdentity {
        match self {
            Self::Thermal(receipt) => receipt.identity(),
            Self::StructuralLinear(receipt) => receipt.identity(),
            Self::StructuralModalResponseSpectrum(receipt) => receipt.identity(),
            Self::StructuralRandomResponse(receipt) => receipt.identity(),
        }
    }

    pub fn evidence(
        &self,
        request: &AccuracyEvaluationRequest,
    ) -> AnalysisExecutionResult<Vec<AccuracyEvidence>> {
        if self.kind() != request.source_kind || self.source_result() != request.source_result {
            return Err(AnalysisExecutionError::Identity(
                "accuracy receipt differs from its admitted request",
            ));
        }
        if let Self::Thermal(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "thermal accuracy receipt differs from its requested producer",
                ));
            }
        }
        let mut evidence = Vec::new();
        for criterion in &request.plan.criteria {
            let measured = match self {
                Self::Thermal(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound =>
                {
                    Some((
                        receipt.receipt().maximum_relative_linear_residual,
                        "maximum accepted thermal linear-system residual norm divided by max(load-vector Euclidean norm, 1) over solved samples; measures algebraic equation satisfaction, not spatial or temporal discretization error",
                    ))
                }
                Self::Thermal(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance =>
                {
                    Some((
                        receipt.receipt().maximum_relative_energy_residual,
                        "maximum accepted absolute thermal energy imbalance divided by the largest retained absolute physical storage, boundary-transfer, applied or nodal prescribed-reaction sum before global signed cancellation; exactly zero transfers have zero residual, with no dimensional or absolute-temperature floor; this is equation-based conservation evidence, not a field-accuracy or convergence estimate",
                    ))
                }
                Self::StructuralLinear(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound =>
                {
                    Some((
                        receipt.receipt().relative_equilibrium_residual,
                        "canonical unconstrained equilibrium residual projected into the admissible constraint space",
                    ))
                }
                Self::StructuralModalResponseSpectrum(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::NumericalResult
                        && criterion.contribution == ErrorContributionKind::ModalTruncation
                        && criterion.evidence == RequiredAccuracyEvidence::ConvergenceStudy =>
                {
                    Some((
                        (1.0 - receipt.minimum_achieved_mass_ratio()).max(0.0),
                        "uncaptured directional effective-mass fraction bounds the admitted modal truncation",
                    ))
                }
                Self::StructuralRandomResponse(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::NumericalResult
                        && criterion.contribution == ErrorContributionKind::FrequencySampling
                        && criterion.evidence == RequiredAccuracyEvidence::ConvergenceStudy =>
                {
                    Some((
                        receipt.maximum_estimated_relative_variance_error(),
                        "adaptive interval refinement compares nested quadrature estimates component by component",
                    ))
                }
                _ => None,
            };
            if let Some((measured, assumption)) = measured {
                evidence.push(
                    AccuracyEvidence::new(
                        criterion.target.clone(),
                        criterion.contribution,
                        criterion.evidence,
                        AccuracyMeasure::Relative(measured),
                        vec![assumption.into()],
                        AccuracyEvidenceScope::SolverResult(request.source_result),
                    )
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
                );
            }
        }
        evidence.sort_by_key(AccuracyEvidence::identity);
        evidence.dedup_by_key(|item| item.identity());
        Ok(evidence)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Thermal(receipt) => receipt.validate(),
            Self::StructuralLinear(receipt) => receipt.validate(),
            Self::StructuralModalResponseSpectrum(receipt) => receipt.validate_source(),
            Self::StructuralRandomResponse(receipt) => receipt.validate_source(),
        }
    }
}

#[cfg(test)]
mod thermal_tests;
