use serde::{Deserialize, Serialize};

/// Physical equations represented by a study step.
///
/// Coupled families are explicit because they carry different interface and state-transfer
/// contracts from their constituent single-family problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicsFamily {
    Structural,
    Thermal,
    Electromagnetic,
    Acoustic,
    Fluid,
    Coupled(CoupledPhysicsFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoupledPhysicsFamily {
    Thermomechanical,
    Electrothermal,
    ConjugateHeatTransfer,
    FluidStructureInteraction,
}

/// Mathematical procedure applied to a physical family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisProcedure {
    Static,
    ModalExtraction,
    PrestressedModalExtraction,
    EigenvalueBuckling,
    TransientResponse,
    HarmonicResponse,
    PrestressedHarmonicResponse,
    ModalResponseSpectrum,
    RandomResponse,
    Nonlinear,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolutionSpace {
    Direct,
    ModalReduced,
}

/// The discrete implementation that actually produced a result.
///
/// This is intentionally more precise than family and procedure. A formulation may only be
/// promoted by changing its maturity after its own verification inventory has passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationImplementation {
    LinearElasticStructural,
    LinearizedStructuralModes,
    ReducedStructuralTransient,
    ReducedStructuralNonlinear,
    ReducedThermalDiffusion,
    ReducedAcousticHelmholtz,
    ReducedElectromagneticPotential,
    ReducedIncompressibleFlow,
    ReducedConjugateHeatTransfer,
    ReducedFluidStructureInteraction,
    ReducedThermomechanical,
    ReducedElectrothermal,
}

/// Exact structural operator assembled for a run.
///
/// These identities deliberately distinguish verified continuum implementations from reduced
/// or proxy operators. They describe realized numerical work, not merely the requested physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralOperatorImplementation {
    Tet4LinearElasticity,
    Tet10LinearElasticity,
    EulerBernoulliBeam3d,
    PairSpringShellProxy,
    SyntheticDiagonalChain,
}

/// Exact structural field-recovery implementation applied after a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralRecoveryImplementation {
    Tet4ConstantStrain,
    BeamSectionResultants,
    PairSpringShellDerivedResultants,
}

/// Assembly evidence carried alongside the operator that was actually produced.
///
/// `recovery` is absent when the implementation has no valid recovery path. In particular, the
/// current Tet10 stiffness implementation must not be mistaken for completed Tet10 recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFormulationRealization {
    pub operator: StructuralOperatorImplementation,
    pub recovery: Option<StructuralRecoveryImplementation>,
    pub maturity: FormulationMaturity,
    pub fallback_policy: ExecutionFallbackPolicy,
}

impl StructuralFormulationRealization {
    pub const fn experimental(
        operator: StructuralOperatorImplementation,
        recovery: Option<StructuralRecoveryImplementation>,
        fallback_policy: ExecutionFallbackPolicy,
    ) -> Self {
        Self {
            operator,
            recovery,
            maturity: FormulationMaturity::Experimental,
            fallback_policy,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.operator == StructuralOperatorImplementation::SyntheticDiagonalChain
            && self.fallback_policy != ExecutionFallbackPolicy::AllowExperimental
        {
            return Err(
                "the synthetic structural operator requires explicit experimental admission",
            );
        }
        if self.maturity == FormulationMaturity::Production {
            return Err(
                "the current structural formulation implementations have not qualified for production maturity",
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationMaturity {
    Production,
    Experimental,
}

/// Whether an implementation may return a result after changing the requested execution path.
///
/// Production formulations never opt into fallback. Experimental formulations may do so, but
/// every fallback must still be recorded in execution provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFallbackPolicy {
    Forbid,
    AllowExperimental,
}

/// Canonical identity persisted with every analysis result.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRunIdentity {
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub formulations: Vec<FormulationImplementation>,
    pub maturity: FormulationMaturity,
    pub fallback_policy: ExecutionFallbackPolicy,
}

impl AnalysisRunIdentity {
    pub fn experimental(
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
        formulations: impl IntoIterator<Item = FormulationImplementation>,
    ) -> Self {
        Self {
            family,
            procedure,
            solution_space,
            formulations: formulations.into_iter().collect(),
            maturity: FormulationMaturity::Experimental,
            fallback_policy: ExecutionFallbackPolicy::AllowExperimental,
        }
    }

    pub fn production(
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
        formulations: impl IntoIterator<Item = FormulationImplementation>,
    ) -> Self {
        Self {
            family,
            procedure,
            solution_space,
            formulations: formulations.into_iter().collect(),
            maturity: FormulationMaturity::Production,
            fallback_policy: ExecutionFallbackPolicy::Forbid,
        }
    }

    pub fn permits_fallback(&self) -> bool {
        self.maturity == FormulationMaturity::Experimental
            && self.fallback_policy == ExecutionFallbackPolicy::AllowExperimental
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.formulations.is_empty() {
            return Err("an analysis run identity must name at least one formulation");
        }
        if self.maturity == FormulationMaturity::Production
            && self.fallback_policy != ExecutionFallbackPolicy::Forbid
        {
            return Err("a production formulation must forbid execution fallback");
        }
        Ok(())
    }

    pub fn operation_name(&self) -> &'static str {
        match (self.family, self.procedure) {
            (PhysicsFamily::Structural, AnalysisProcedure::Static) => "fea.run_linear_static",
            (PhysicsFamily::Structural, AnalysisProcedure::ModalExtraction) => "fea.run_modal",
            (PhysicsFamily::Structural, AnalysisProcedure::TransientResponse) => {
                "fea.run_transient"
            }
            (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear) => "fea.run_nonlinear",
            (
                PhysicsFamily::Thermal,
                AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            ) => "fea.run_thermal",
            (PhysicsFamily::Electromagnetic, AnalysisProcedure::Static) => {
                "fea.run_electromagnetic"
            }
            (PhysicsFamily::Electromagnetic, AnalysisProcedure::HarmonicResponse) => {
                "fea.run_electromagnetic"
            }
            (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse) => "fea.run_acoustic",
            (
                PhysicsFamily::Fluid,
                AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            ) => "fea.run_cfd",
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            ) => "fea.run_cht",
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse,
            ) => "fea.run_fsi",
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Static
                | AnalysisProcedure::TransientResponse
                | AnalysisProcedure::Nonlinear,
            ) => "fea.run_thermo_mechanical",
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal),
                AnalysisProcedure::Static
                | AnalysisProcedure::TransientResponse
                | AnalysisProcedure::Nonlinear,
            ) => "fea.run_electro_thermal",
            _ => "fea.run_unsupported",
        }
    }

    pub fn operation_version(&self) -> String {
        format!("{}/v1", self.operation_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_identity_does_not_collapse_to_transient_structural() {
        let identity = AnalysisRunIdentity::experimental(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal),
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            [FormulationImplementation::ReducedElectrothermal],
        );

        assert_eq!(identity.operation_name(), "fea.run_electro_thermal");
        assert_eq!(identity.operation_version(), "fea.run_electro_thermal/v1");
    }

    #[test]
    fn standalone_thermal_identity_has_thermal_operation_version() {
        let identity = AnalysisRunIdentity::experimental(
            PhysicsFamily::Thermal,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            [FormulationImplementation::ReducedThermalDiffusion],
        );

        assert_eq!(identity.operation_version(), "fea.run_thermal/v1");
        assert!(identity.permits_fallback());
    }

    #[test]
    fn production_identity_forbids_execution_fallback() {
        let identity = AnalysisRunIdentity::production(
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            [FormulationImplementation::LinearElasticStructural],
        );

        assert!(!identity.permits_fallback());
        assert_eq!(identity.validate(), Ok(()));
    }

    #[test]
    fn invalid_production_fallback_policy_is_rejected() {
        let identity = AnalysisRunIdentity {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            formulations: vec![FormulationImplementation::LinearElasticStructural],
            maturity: FormulationMaturity::Production,
            fallback_policy: ExecutionFallbackPolicy::AllowExperimental,
        };

        assert_eq!(
            identity.validate(),
            Err("a production formulation must forbid execution fallback")
        );
    }

    #[test]
    fn exact_structural_realizations_distinguish_operators_and_recovery() {
        let cases = [
            (
                StructuralOperatorImplementation::Tet4LinearElasticity,
                Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
            ),
            (
                StructuralOperatorImplementation::Tet10LinearElasticity,
                None,
            ),
            (
                StructuralOperatorImplementation::EulerBernoulliBeam3d,
                Some(StructuralRecoveryImplementation::BeamSectionResultants),
            ),
            (
                StructuralOperatorImplementation::PairSpringShellProxy,
                Some(StructuralRecoveryImplementation::PairSpringShellDerivedResultants),
            ),
        ];

        for (operator, recovery) in cases {
            let realization = StructuralFormulationRealization::experimental(
                operator,
                recovery,
                ExecutionFallbackPolicy::Forbid,
            );
            assert_eq!(realization.operator, operator);
            assert_eq!(realization.recovery, recovery);
            assert_eq!(realization.maturity, FormulationMaturity::Experimental);
            assert_eq!(realization.validate(), Ok(()));
        }
    }

    #[test]
    fn synthetic_structural_operator_requires_experimental_fallback_admission() {
        let rejected = StructuralFormulationRealization::experimental(
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
            ExecutionFallbackPolicy::Forbid,
        );
        assert_eq!(
            rejected.validate(),
            Err("the synthetic structural operator requires explicit experimental admission")
        );

        let admitted = StructuralFormulationRealization::experimental(
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
            ExecutionFallbackPolicy::AllowExperimental,
        );
        assert_eq!(admitted.validate(), Ok(()));
    }

    #[test]
    fn structural_proxy_cannot_claim_production_maturity() {
        let proxy = StructuralFormulationRealization {
            operator: StructuralOperatorImplementation::PairSpringShellProxy,
            recovery: Some(StructuralRecoveryImplementation::PairSpringShellDerivedResultants),
            maturity: FormulationMaturity::Production,
            fallback_policy: ExecutionFallbackPolicy::Forbid,
        };

        assert_eq!(
            proxy.validate(),
            Err(
                "the current structural formulation implementations have not qualified for production maturity"
            )
        );
    }
}
