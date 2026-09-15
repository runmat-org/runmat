use serde::{Deserialize, Serialize};

/// Qualification requested by the caller at the pre-solve boundary.
///
/// There is intentionally no `Default`: every caller must choose whether experimental numerical
/// work is admissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationRequirement {
    Production,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralOperatorImplementation {
    Tet4LinearElasticity,
    Tet10LinearElasticity,
    EulerBernoulliBeam3d,
    Mitc3TriangularShell,
    SyntheticDiagonalChain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialOperatorImplementation {
    Structural(StructuralOperatorImplementation),
    ReducedThermalDiffusion,
    ThermalGalerkinConduction,
    PrescribedThermalCouplingProfile,
    ReducedAcousticHelmholtz,
    ReducedElectromagneticPotential,
    ReducedEdgeGraphCurlCurl,
    ReducedChannelFlow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalImplementation {
    StructuralStaticLinearSolve,
    StructuralSubspaceIteration,
    StructuralPrestressedSubspaceIteration,
    StructuralEigenvalueBuckling,
    StructuralPrestressedHarmonicSolve,
    StructuralNewmarkAverageAcceleration,
    StructuralScalarResidualIteration,
    ThermalSteadyLinearSolve,
    ThermalBackwardEuler,
    ElectromagneticReducedEdgeGraphHarmonicSolve,
    AcousticDomainGraphHarmonicSolve,
    CfdReducedChannelSteadyIteration,
    CfdReducedChannelTimeStepping,
    CoupledSinglePassProfileTransfer,
    CoupledFixedPointRelaxation,
    CoupledScalarResidualIteration,
    PrescribedThermomechanicalProfileApplication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstitutiveImplementation {
    IsotropicLinearElasticity,
    ReducedScalarStructuralNonlinearity,
    ReducedThermalCoefficientProfile,
    IsotropicThermalConduction,
    ReducedFrequencyMaterialResponse,
    ReducedAcousticMedium,
    ReducedIncompressibleFlowClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouplingImplementation {
    Uncoupled,
    OnePassPrescribedThermomechanicalProfile,
    StaggeredElectrothermalConductanceLine,
    PartitionedConjugateProjection,
    PartitionedFluidStructureRelaxation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralRecoveryImplementation {
    Tet4ConstantStrain,
    Tet10VolumeAverageStrain,
    BeamSectionResultants,
    Mitc3ShellStressResultants,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryImplementation {
    Unavailable,
    Structural(StructuralRecoveryImplementation),
    ThermalSolverMeshGradient,
    ThermalInferredAxisGradient,
    PrescribedThermomechanicalProfileDerivedFields,
    AcousticDomainGraphComplexPressure,
    ElectromagneticPotentialProfile,
    ReducedEdgeGraphFlux,
    CfdReducedChannelProfile,
    CoupledResampledFields,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyImplementation {
    CanonicalSolverMeshTet4,
    CanonicalSolverMeshTet10,
    ExplicitStructuralConnectivityBeam,
    ExplicitStructuralConnectivityTriangularShell,
    InferredLattice,
    GeneratedGraph,
    ImplicitChannel,
    DofAdjacency,
    OrientedEdgeIncidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceImplementation {
    NotApplicable,
    CanonicalVolumeProjection,
    CanonicalBoundaryProjection,
    CanonicalBoundaryPressureIntegration,
    CanonicalEdgeProjection,
    CanonicalPointProjection,
    AcousticHashedTargetGraphProjection,
    BoundaryResultantMomentProjection,
    ReferencePointGeneralizedLoad,
    ExplicitRegionNodalDistribution,
    UniformNodeDistribution,
    PrescribedTemperatureDeltaProfile,
    OrdinalRegionTemperatureDistribution,
    OrdinalDistribution,
    SyntheticProfileFallback,
    UnmappedFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTransferImplementation {
    NotApplicable,
    CanonicalBoundaryProjection,
    CanonicalGroundWithOrdinalInsulationProfile,
    ExplicitRegionConnectivity,
    GlobalCountProfile,
    ImplicitNaturalBoundaryAndGaugeAnchor,
    HeuristicInterfaceTransfer,
    UnmappedFallback,
}

/// Exact realization of one spatial operator and its owned constitutive, recovery, topology, and
/// source-projection algorithms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFormulationComponent {
    pub spatial_operator: SpatialOperatorImplementation,
    pub constitutive: ConstitutiveImplementation,
    pub recovery: RecoveryImplementation,
    pub topologies: Vec<TopologyImplementation>,
    pub sources: Vec<SourceImplementation>,
}

/// Exact numerical realization across every axis that can change a result.
///
/// Coupled procedures carry one component per constituent operator instead of collapsing several
/// discretizations into a synthetic coupled label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFormulationRealization {
    pub components: Vec<ExactFormulationComponent>,
    pub temporals: Vec<TemporalImplementation>,
    pub coupling: CouplingImplementation,
    pub boundary_transfers: Vec<BoundaryTransferImplementation>,
}

/// Assembly evidence carried alongside the structural operator actually produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFormulationRealization {
    pub operator: StructuralOperatorImplementation,
    pub recovery: Option<StructuralRecoveryImplementation>,
}

impl StructuralFormulationRealization {
    pub const fn new(
        operator: StructuralOperatorImplementation,
        recovery: Option<StructuralRecoveryImplementation>,
    ) -> Self {
        Self { operator, recovery }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationAxis {
    SpatialOperator,
    Temporal,
    Constitutive,
    Coupling,
    Recovery,
    Topology,
    Source,
    BoundaryTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationAdmissionRejectionReason {
    UnsupportedFamilyProcedure,
    UnsupportedSolutionSpace,
    UnregisteredRealization,
    IncompatibleAxis(FormulationAxis),
    ProductionQualificationMissing,
    InvalidRealization,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_realization_rejects_unknown_or_extra_wire_values() {
        let unknown = serde_json::json!({
            "components": [{
                "spatial_operator": "invented_operator",
                "constitutive": "isotropic_linear_elasticity",
                "recovery": "unavailable",
                "topologies": ["canonical_solver_mesh_tet4"],
                "sources": ["canonical_boundary_projection"]
            }],
            "temporals": ["structural_static_linear_solve"],
            "coupling": "uncoupled",
            "boundary_transfers": ["canonical_boundary_projection"]
        });
        assert!(serde_json::from_value::<ExactFormulationRealization>(unknown).is_err());

        let extra = serde_json::json!({
            "components": [{
                "spatial_operator": {"structural": "tet4_linear_elasticity"},
                "constitutive": "isotropic_linear_elasticity",
                "recovery": {"structural": "tet4_constant_strain"},
                "topologies": ["canonical_solver_mesh_tet4"],
                "sources": ["canonical_boundary_projection"]
            }],
            "temporals": ["structural_static_linear_solve"],
            "coupling": "uncoupled",
            "boundary_transfers": ["canonical_boundary_projection"],
            "forged_maturity": "production"
        });
        assert!(serde_json::from_value::<ExactFormulationRealization>(extra).is_err());
    }

    #[test]
    fn structural_realization_records_only_the_assembled_algorithm() {
        let realization = StructuralFormulationRealization::new(
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
        );
        assert_eq!(
            realization.operator,
            StructuralOperatorImplementation::SyntheticDiagonalChain
        );
        assert_eq!(realization.recovery, None);
    }
}
