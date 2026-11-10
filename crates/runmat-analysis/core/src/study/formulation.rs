use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralOperatorImplementation {
    Tet4LinearElasticity,
    Tet10LinearElasticity,
    Tet4SmallStrainContinuum,
    Tet10SmallStrainContinuum,
    Tet4TotalLagrangianContinuum,
    Tet10TotalLagrangianContinuum,
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
    IntegratedPrescribedSolidThermoelasticLoad,
    IntegratedSolvedSolidThermoelasticLoad,
    ReducedAcousticHelmholtz,
    AcousticGalerkinHelmholtz,
    ElectricalGalerkinPotentialGradient,
    MagneticGalerkinLowestOrderEdgeCurlCurl,
    MaxwellLowestOrderEdgeCurlCurlComplexMass,
    ReducedChannelFlow,
    FluidCellCenteredConservativeMomentumPressure,
    /// Nodal Galerkin energy over fluid and solid, with conservative Raviart–Thomas advection
    /// of the solved fluid face fluxes and streamline-upwind stabilization in the fluid.
    ConjugateEnergyGalerkinConservativeAdvection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalImplementation {
    StructuralStaticLinearSolve,
    StructuralSubspaceIteration,
    StructuralPrestressedSubspaceIteration,
    StructuralEigenvalueBuckling,
    StructuralHarmonicSolve,
    StructuralPrestressedHarmonicSolve,
    StructuralModalResponseSpectrumCombination,
    StructuralRandomResponseIntegration,
    StructuralNewmarkAverageAcceleration,
    StructuralQuasistaticNewtonEquilibrium,
    ThermalSteadyLinearSolve,
    ThermalBackwardEuler,
    ThermalSteadyDampedNewton,
    ThermalBackwardEulerDampedNewton,
    ElectrostaticPotentialLinearSolve,
    SteadyConductionPotentialLinearSolve,
    MagnetostaticWeakCoulombSaddleSolve,
    MaxwellComplexElectricFieldGeneralSolve,
    AcousticDomainGraphHarmonicSolve,
    AcousticCanonicalHarmonicSolve,
    AcousticCanonicalModalSolve,
    AcousticCanonicalTransientSolve,
    FluidSteadyNonorthogonalPressureCorrection,
    FluidBackwardEulerConservativeMomentumPressure,
    /// Steady finite-volume flow, then one monolithic energy solve across fluid and solid.
    ConjugateSteadyFlowMonolithicEnergy,
    /// Steady finite-volume flow advecting a backward-Euler monolithic energy march.
    ConjugateSteadyFlowBackwardEulerEnergy,
    /// Steady flow on the deformed fluid domain and a small-strain structure, iterated with
    /// Aitken-relaxed interface displacement until both sides agree.
    FluidStructureSteadyAitkenPartitioned,
    /// Moving-mesh backward-Euler flow and average-acceleration Newmark structure, iterated
    /// within every interval with Aitken-relaxed interface displacement.
    FluidStructureTransientAitkenPartitioned,
    PrescribedThermomechanicalProfileApplication,
    SolvedThermalStateApplication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstitutiveImplementation {
    IsotropicLinearElasticity,
    SmallStrainJ2IsotropicHardening,
    GreenLagrangeJ2IsotropicHardening,
    ReducedThermalCoefficientProfile,
    ElementOwnedIsotropicThermalExpansion,
    IsotropicThermalConduction,
    ElementOwnedIsotropicPermittivity,
    ElementOwnedIsotropicElectricalConductivity,
    ElementOwnedIsotropicMagneticPermeability,
    ElementOwnedIsotropicHarmonicElectromagneticLaws,
    ReducedAcousticMedium,
    ElementOwnedLosslessAcousticFluid,
    ReducedIncompressibleFlowClosure,
    CellOwnedIncompressibleNewtonianFluid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouplingImplementation {
    Uncoupled,
    OnePassPrescribedThermomechanicalProfile,
    OneWaySolvedThermoelasticState,
    StaggeredElectrothermalConductanceLine,
    PartitionedFluidStructureRelaxation,
    /// Flow advects heat; fluid and solid energy share one system through declared interfaces.
    FlowAdvectedMonolithicConjugateEnergy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralRecoveryImplementation {
    Tet4ConstantStrain,
    Tet10IntegrationPointProjection,
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
    ElementOwnedThermoelasticStressEnergy,
    AcousticDomainGraphComplexPressure,
    AcousticSolverMeshPressureVelocityEnergy,
    AcousticSolverMeshNodalPressure,
    ElectricalSolverMeshFieldDisplacementEnergy,
    ElectricalSolverMeshFieldCurrentJoulePower,
    MagneticSolverMeshVectorPotentialFluxFieldEnergy,
    MaxwellSolverMeshElectricFluxFieldPower,
    CfdReducedChannelProfile,
    FluidOriginalCellVelocityPressureOwnerFaceFlux,
    ConjugateNodalTemperatureInterfaceHeat,
    FluidStructureNodalDisplacementInterfaceEvidence,
    CoupledResampledFields,
    /// Per-length Maxwell capacitance matrices of a cross-section from the energy form of its
    /// potential solves (with the dielectrics and in vacuum), and the line parameters that
    /// follow from them.
    CrossSectionLineParameterEnergy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyImplementation {
    CanonicalSolverMeshTet4,
    CanonicalSolverMeshTet10,
    /// Linear triangles of a canonical 2D cross-section mesh.
    CanonicalSolverMeshTri3,
    ExplicitStructuralConnectivityBeam,
    ExplicitStructuralConnectivityTriangularShell,
    InferredLattice,
    GeneratedGraph,
    ImplicitChannel,
    DofAdjacency,
    OrientedEdgeIncidence,
    OwnerOrientedFiniteVolumeFaceIncidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceImplementation {
    NotApplicable,
    CanonicalVolumeProjection,
    CanonicalBoundaryProjection,
    CanonicalBoundaryPressureIntegration,
    /// Force, traction or pressure integrated over the deformed boundary at every Newton
    /// iterate, with its load stiffness (total Lagrangian solids only).
    CanonicalFollowerBoundaryIntegration,
    CanonicalPointProjection,
    AcousticHashedTargetGraphProjection,
    BoundaryResultantMomentProjection,
    ReferencePointGeneralizedLoad,
    ExplicitRegionNodalDistribution,
    UniformNodeDistribution,
    PrescribedTemperatureDeltaProfile,
    OrdinalRegionTemperatureDistribution,
    PersistentRegionElementTemperatureMap,
    SolvedThermalStateNodalField,
    OrdinalDistribution,
    SyntheticProfileFallback,
    UnmappedFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTransferImplementation {
    NotApplicable,
    CanonicalBoundaryProjection,
    ExplicitRegionConnectivity,
    GlobalCountProfile,
    ImplicitNaturalBoundaryAndGaugeAnchor,
    FullExteriorTangentialPotentialGroundWithWeakCoulombGauge,
    FullExteriorElectricPerfectConductor,
    HeuristicInterfaceTransfer,
    /// Declared conformal interface: shared nodes or an exact conductance jump.
    DeclaredConformalInterfaceLaw,
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
            "forged_field": "value"
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
