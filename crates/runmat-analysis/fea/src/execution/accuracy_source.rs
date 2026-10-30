use runmat_analysis_core::{ErrorContributionKind, RequiredAccuracyEvidence};

use crate::results::accuracy::{AccuracyEvidence, AccuracyEvidenceScope, AccuracyMeasure};

use super::{
    AccuracyEvaluationRequest, AccuracySourceKind, AcousticNumericalReceiptArtifact,
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    ElectricalNumericalReceiptArtifact, FluidNumericalReceiptArtifact,
    MagneticNumericalReceiptArtifact, MaxwellNumericalReceiptArtifact, SolverResultIdentity,
    StructuralLinearNumericalReceiptArtifact, StructuralModalResponseSpectrumNumericalReceipt,
    StructuralRandomResponseNumericalReceipt, ThermalNumericalReceiptArtifact,
};

#[derive(Clone, Debug, PartialEq)]
pub enum AccuracySourceReceipt {
    NonlinearSolid(super::NonlinearSolidNumericalReceiptArtifact),
    FluidSteady(FluidNumericalReceiptArtifact),
    MaxwellHarmonic(MaxwellNumericalReceiptArtifact),
    Magnetostatic(MagneticNumericalReceiptArtifact),
    ElectricalStatic(ElectricalNumericalReceiptArtifact),
    AcousticHarmonic(AcousticNumericalReceiptArtifact),
    AcousticResponse(super::AcousticResponseNumericalReceiptArtifact),
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
            AccuracySourceKind::NonlinearSolid => Self::NonlinearSolid(
                super::NonlinearSolidNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::FluidSteady => {
                Self::FluidSteady(FluidNumericalReceiptArtifact::from_canonical_bytes(bytes)?)
            }
            AccuracySourceKind::MaxwellHarmonic => Self::MaxwellHarmonic(
                MaxwellNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::Magnetostatic => Self::Magnetostatic(
                MagneticNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::ElectricalStatic => Self::ElectricalStatic(
                ElectricalNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::AcousticHarmonic => Self::AcousticHarmonic(
                AcousticNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
            AccuracySourceKind::AcousticResponse => Self::AcousticResponse(
                super::AcousticResponseNumericalReceiptArtifact::from_canonical_bytes(bytes)?,
            ),
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
            Self::NonlinearSolid(_) => AccuracySourceKind::NonlinearSolid,
            Self::FluidSteady(_) => AccuracySourceKind::FluidSteady,
            Self::MaxwellHarmonic(_) => AccuracySourceKind::MaxwellHarmonic,
            Self::Magnetostatic(_) => AccuracySourceKind::Magnetostatic,
            Self::ElectricalStatic(_) => AccuracySourceKind::ElectricalStatic,
            Self::AcousticHarmonic(_) => AccuracySourceKind::AcousticHarmonic,
            Self::AcousticResponse(_) => AccuracySourceKind::AcousticResponse,
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
            Self::NonlinearSolid(receipt) => receipt.source_workload(),
            Self::FluidSteady(receipt) => receipt.source_workload(),
            Self::MaxwellHarmonic(receipt) => receipt.source_workload(),
            Self::Magnetostatic(receipt) => receipt.source_workload(),
            Self::ElectricalStatic(receipt) => receipt.source_workload(),
            Self::AcousticHarmonic(receipt) => receipt.source_workload(),
            Self::AcousticResponse(receipt) => receipt.source_workload(),
            Self::Thermal(receipt) => receipt.source_workload,
            Self::StructuralLinear(receipt) => receipt.source_workload,
            Self::StructuralModalResponseSpectrum(receipt) => receipt.source_workload(),
            Self::StructuralRandomResponse(receipt) => receipt.source_workload(),
        }
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        match self {
            Self::NonlinearSolid(receipt) => receipt.source_result(),
            Self::FluidSteady(receipt) => receipt.source_result(),
            Self::MaxwellHarmonic(receipt) => receipt.source_result(),
            Self::Magnetostatic(receipt) => receipt.source_result(),
            Self::ElectricalStatic(receipt) => receipt.source_result(),
            Self::AcousticHarmonic(receipt) => receipt.source_result(),
            Self::AcousticResponse(receipt) => receipt.source_result(),
            Self::Thermal(receipt) => receipt.source_result(),
            Self::StructuralLinear(receipt) => receipt.source_result(),
            Self::StructuralModalResponseSpectrum(receipt) => receipt.source_result(),
            Self::StructuralRandomResponse(receipt) => receipt.source_result(),
        }
    }

    pub const fn identity(&self) -> super::NumericalReceiptIdentity {
        match self {
            Self::NonlinearSolid(receipt) => receipt.identity(),
            Self::FluidSteady(receipt) => receipt.identity(),
            Self::MaxwellHarmonic(receipt) => receipt.identity(),
            Self::Magnetostatic(receipt) => receipt.identity(),
            Self::ElectricalStatic(receipt) => receipt.identity(),
            Self::AcousticHarmonic(receipt) => receipt.identity(),
            Self::AcousticResponse(receipt) => receipt.identity(),
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
        self.validate()?;
        if self.kind() != request.source_kind || self.source_result() != request.source_result {
            return Err(AnalysisExecutionError::Identity(
                "accuracy receipt differs from its admitted request",
            ));
        }
        if let Self::FluidSteady(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "fluid accuracy original producer differs",
                ));
            }
        }
        if let Self::MaxwellHarmonic(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "Maxwell accuracy receipt differs from original producer",
                ));
            }
        }
        if let Self::Magnetostatic(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "magnetic accuracy receipt differs from requested producer",
                ));
            }
        }
        if let Self::Thermal(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "thermal accuracy receipt differs from its requested producer",
                ));
            }
        }
        if let Self::AcousticHarmonic(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "acoustic accuracy receipt differs from requested producer",
                ));
            }
        }
        if let Self::AcousticResponse(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "acoustic response accuracy receipt differs from requested producer",
                ));
            }
        }
        if let Self::ElectricalStatic(receipt) = self {
            if receipt.receipt().producer != request.producer {
                return Err(AnalysisExecutionError::Identity(
                    "electrical accuracy receipt differs from requested producer",
                ));
            }
        }
        let mut evidence = Vec::new();
        if let Self::NonlinearSolid(receipt) = self {
            if receipt.receipt().producer != request.producer || !receipt.receipt().complete {
                return Err(AnalysisExecutionError::Identity(
                    "nonlinear accuracy requires original producer and complete load schedule",
                ));
            }
        }
        for criterion in &request.plan.criteria {
            let measured = match self {
                Self::NonlinearSolid(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Nonlinear
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().accepted_relative_residual_maximum.max(receipt.receipt().accepted_relative_increment_maximum),
                        "maximum accepted original free-equilibrium relative residual and undamped pending Newton correction normalized by offset-free current/committed displacement; nonlinear closure only, not mesh/load-step accuracy or constitutive qualification"))
                }
                Self::NonlinearSolid(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().maximum_linear_relative_residual,
                        "maximum achieved relative residual of every completed tangent (Newton correction) solve in the original trajectory, including cutback trials, in the solved constraint-reduced equation space; inner linear equation satisfaction only, separate from Newton equilibrium closure and not mesh/load-step accuracy"))
                }
                Self::FluidSteady(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().momentum_relative.max(receipt.receipt().flux_constitutive_relative),
                        "maximum original unrelaxed sparse momentum-action relative residual and owner-face constitutive flux relative defect; exact zero scales use absolute defects; algebraic/constitutive equation satisfaction only, not independent spatial accuracy"))
                }
                Self::FluidSteady(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance => {
                    Some((receipt.receipt().continuity_relative.max(receipt.receipt().maximum_cell_momentum_relative).max(receipt.receipt().global_momentum_relative),
                        "maximum original per-cell owner-oriented signed volume-flux defect normalized by sum of absolute incident fluxes, independent physical face-force/body-force cell momentum defect and global boundary/body momentum defect normalized by sums of absolute physical terms; exact zero scales use absolute defects; conservation closure only, not independent spatial accuracy"))
                }
                Self::MaxwellHarmonic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().relative_physical_residual.max(receipt.receipt().relative_charge_continuity_residual),
                        "maximum original physical weak-equation residual and interior corner-gradient charge-continuity residual; each uses its retained original physical norm with exact zero handling; algebraic satisfaction only, not spatial field accuracy or convergence"))
                }
                Self::MaxwellHarmonic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance => {
                    Some((receipt.receipt().relative_complex_power_error,
                        "complex delivered source power minus ohmic loss and 2omega magnetic-minus-electric stored energy, divided by the larger absolute physical complex term; exact zero scale uses absolute defect, no dimensional floor; equation-based power closure, not independent spatial accuracy"))
                }
                Self::Magnetostatic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().relative_physical_residual.max(receipt.receipt().relative_gauge_residual),
                        "maximum of the physical curl-curl residual Euclidean norm divided by physical load norm and the weak Coulomb-gauge action Euclidean norm divided by scaled saddle right-hand-side norm; zero denominator uses the respective absolute norm; algebraic equation and gauge satisfaction only, not independent field accuracy or spatial convergence"))
                }
                Self::Magnetostatic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance => {
                    Some((receipt.receipt().relative_energy_work_error,
                        "absolute twice-magnetic-energy minus source-work divided by the larger absolute retained term; zero energy and work give zero error with no dimensional floor; weak-equation energy-work closure only, not independent field accuracy or spatial convergence"))
                }
                Self::ElectricalStatic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().linear_relative_residual,
                        "accepted original dimensionless Dirichlet-lifted linear-system residual Euclidean norm divided by its right-hand-side Euclidean norm; exactly zero right-hand side requires zero residual; algebraic equation satisfaction only, not spatial convergence or field accuracy"))
                }
                Self::ElectricalStatic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance => {
                    Some((receipt.receipt().relative_flux_imbalance.max(receipt.receipt().relative_energy_identity_error),
                        "maximum of absolute signed terminal outward-flux sum divided by sum of absolute terminal fluxes, and absolute constitutive-integral minus physical weak-operator energy divided by the larger absolute integral; each denominator is guarded by f64::MIN_POSITIVE; retained equation-based flux and energy closure only, not independent spatial accuracy"))
                }
                Self::AcousticHarmonic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().relative_residual,
                        "accepted reduced real-block Helmholtz residual Euclidean norm divided by the reduced right-hand-side Euclidean norm; zero right-hand-side norm uses the absolute residual norm; algebraic equation satisfaction only, not spatial convergence or field accuracy"))
                }
                Self::AcousticHarmonic(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ConservationBalance => {
                    Some((receipt.receipt().power_relative_error,
                        "absolute mean input-minus-impedance-loss power divided by max(sum of nodal apparent source power magnitudes including prescribed-pressure reactions, absolute impedance loss); zero scale gives zero error with no dimensional floor; equation-based power conservation only, not field accuracy"))
                }
                Self::AcousticResponse(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound => {
                    Some((receipt.receipt().maximum_relative_residual,
                        "cavity modes: largest eigenpair residual |K phi - lambda M phi| / ((|lambda| + s) |M phi|) with shift s; pressure transient: largest effective-operator residual of any accepted Newmark step divided by its right-hand-side norm; algebraic equation satisfaction only, not spatial or temporal discretization error"))
                }
                Self::Thermal(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Nonlinear
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound =>
                {
                    receipt.receipt().maximum_relative_nonlinear_residual.map(|residual| (
                        residual,
                        "maximum accepted original free-coordinate nonlinear thermal equilibrium residual divided by physical power scale; separate from inner linear correction residual, with accepted increments and energy closure required; not spatial or temporal accuracy",
                    ))
                }
                Self::Thermal(receipt)
                    if criterion.target == runmat_analysis_core::AccuracyTarget::GlobalEquilibrium
                        && criterion.contribution == ErrorContributionKind::Algebraic
                        && criterion.evidence == RequiredAccuracyEvidence::ResidualBound =>
                {
                    Some((
                        receipt.receipt().maximum_relative_linear_residual,
                        "maximum accepted shared-solver relative residual in the original prepared equation space over solved samples; normalized by the prepared right-hand-side norm, or initial residual norm for a homogeneous warm solve; a balanced zero equation has zero error; measures algebraic equation satisfaction, not spatial or temporal discretization error",
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
            Self::NonlinearSolid(receipt) => receipt.validate(),
            Self::FluidSteady(receipt) => receipt.validate(),
            Self::MaxwellHarmonic(receipt) => receipt.validate(),
            Self::Magnetostatic(receipt) => receipt.validate(),
            Self::ElectricalStatic(receipt) => receipt.validate(),
            Self::AcousticHarmonic(receipt) => receipt.validate(),
            Self::AcousticResponse(receipt) => receipt.validate(),
            Self::Thermal(receipt) => receipt.validate(),
            Self::StructuralLinear(receipt) => receipt.validate(),
            Self::StructuralModalResponseSpectrum(receipt) => receipt.validate_source(),
            Self::StructuralRandomResponse(receipt) => receipt.validate_source(),
        }
    }
}

#[cfg(test)]
mod thermal_tests;

#[cfg(test)]
mod electrical_tests;

#[cfg(test)]
mod magnetic_tests;

#[cfg(test)]
mod fluid_tests;
