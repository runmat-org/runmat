use runmat_analysis_core::{
    is_exposed_analysis_identity, AnalysisProcedure, BoundaryTransferImplementation,
    ConstitutiveImplementation, CoupledPhysicsFamily, CouplingImplementation,
    ExactFormulationComponent, ExactFormulationRealization, FormulationMaturity, PhysicsFamily,
    RecoveryImplementation, SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, TemporalImplementation, TopologyImplementation,
};

use super::structural::{
    thermomechanical_exact_realization, StructuralRealizationInput,
    ThermomechanicalRealizationInput,
};

/// Enumerates every exact non-structural realization implemented by the current pipelines.
///
/// This is deliberately plural: topology and projection branches are result-changing formulation
/// choices. Callers must report the branch actually selected; the registry never guesses one from
/// family and procedure alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredFormulation {
    realization: ExactFormulationRealization,
    qualification: FormulationMaturity,
}

impl RegisteredFormulation {
    pub(super) const fn new(
        realization: ExactFormulationRealization,
        qualification: FormulationMaturity,
    ) -> Self {
        Self {
            realization,
            qualification,
        }
    }

    pub fn realization(&self) -> &ExactFormulationRealization {
        &self.realization
    }

    pub const fn qualification(&self) -> FormulationMaturity {
        self.qualification
    }
}

pub fn registered_formulations(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
) -> Vec<RegisteredFormulation> {
    if !is_exposed_analysis_identity(family, procedure, SolutionSpace::Direct) {
        return Vec::new();
    }

    let realizations = match family {
        PhysicsFamily::Structural => Vec::new(),
        PhysicsFamily::Thermal => thermal_realizations(procedure),
        PhysicsFamily::Electromagnetic => electromagnetic_realizations(procedure),
        PhysicsFamily::Acoustic => acoustic_realizations(),
        PhysicsFamily::Fluid => fluid_realizations(procedure),
        PhysicsFamily::Coupled(coupled) => coupled_realizations(coupled, procedure),
    };
    realizations
        .into_iter()
        .map(|realization| {
            RegisteredFormulation::new(realization, FormulationMaturity::Experimental)
        })
        .collect()
}

fn thermal_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    let temporal = match procedure {
        AnalysisProcedure::Static => TemporalImplementation::ThermalSteadyLinearSolve,
        AnalysisProcedure::TransientResponse => TemporalImplementation::ThermalBackwardEuler,
        _ => return Vec::new(),
    };
    [
        (
            SpatialOperatorImplementation::ThermalGalerkinConduction,
            ConstitutiveImplementation::IsotropicThermalConduction,
            temporal,
            vec![TopologyImplementation::CanonicalSolverMeshTet4],
            RecoveryImplementation::ThermalSolverMeshGradient,
            vec![SourceImplementation::CanonicalVolumeProjection],
            vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        ),
        (
            SpatialOperatorImplementation::ThermalGalerkinConduction,
            ConstitutiveImplementation::IsotropicThermalConduction,
            temporal,
            vec![TopologyImplementation::CanonicalSolverMeshTet10],
            RecoveryImplementation::ThermalSolverMeshGradient,
            vec![SourceImplementation::CanonicalVolumeProjection],
            vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        ),
    ]
    .into_iter()
    .map(
        |(
            spatial_operator,
            constitutive,
            temporal,
            topologies,
            recovery,
            sources,
            boundary_transfers,
        )| ExactFormulationRealization {
            components: vec![ExactFormulationComponent {
                spatial_operator,
                constitutive,
                recovery,
                topologies,
                sources,
            }],
            temporals: vec![temporal],
            coupling: CouplingImplementation::Uncoupled,
            boundary_transfers,
        },
    )
    .collect()
}

fn electromagnetic_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    if procedure != AnalysisProcedure::HarmonicResponse {
        return Vec::new();
    }
    let mut realizations = Vec::new();
    for topology in [
        TopologyImplementation::CanonicalSolverMeshTet4,
        TopologyImplementation::CanonicalSolverMeshTet10,
    ] {
        for source in [
            SourceImplementation::CanonicalEdgeProjection,
            SourceImplementation::OrdinalDistribution,
            SourceImplementation::SyntheticProfileFallback,
            SourceImplementation::UnmappedFallback,
        ] {
            for boundary_transfer in [
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
                BoundaryTransferImplementation::CanonicalGroundWithOrdinalInsulationProfile,
                BoundaryTransferImplementation::ImplicitNaturalBoundaryAndGaugeAnchor,
                BoundaryTransferImplementation::UnmappedFallback,
            ] {
                realizations.push(ExactFormulationRealization {
                    components: vec![ExactFormulationComponent {
                        spatial_operator: SpatialOperatorImplementation::ReducedEdgeGraphCurlCurl,
                        constitutive: ConstitutiveImplementation::ReducedFrequencyMaterialResponse,
                        recovery: RecoveryImplementation::ReducedEdgeGraphFlux,
                        topologies: vec![topology, TopologyImplementation::OrientedEdgeIncidence],
                        sources: vec![source],
                    }],
                    temporals: vec![
                        TemporalImplementation::ElectromagneticReducedEdgeGraphHarmonicSolve,
                    ],
                    coupling: CouplingImplementation::Uncoupled,
                    boundary_transfers: vec![boundary_transfer],
                });
            }
        }
    }
    for source in [
        SourceImplementation::OrdinalDistribution,
        SourceImplementation::SyntheticProfileFallback,
        SourceImplementation::UnmappedFallback,
    ] {
        for boundary_transfer in [
            BoundaryTransferImplementation::GlobalCountProfile,
            BoundaryTransferImplementation::ImplicitNaturalBoundaryAndGaugeAnchor,
            BoundaryTransferImplementation::UnmappedFallback,
        ] {
            realizations.push(ExactFormulationRealization {
                components: vec![ExactFormulationComponent {
                    spatial_operator: SpatialOperatorImplementation::ReducedEdgeGraphCurlCurl,
                    constitutive: ConstitutiveImplementation::ReducedFrequencyMaterialResponse,
                    recovery: RecoveryImplementation::ReducedEdgeGraphFlux,
                    topologies: vec![
                        TopologyImplementation::DofAdjacency,
                        TopologyImplementation::OrientedEdgeIncidence,
                    ],
                    sources: vec![source],
                }],
                temporals: vec![
                    TemporalImplementation::ElectromagneticReducedEdgeGraphHarmonicSolve,
                ],
                coupling: CouplingImplementation::Uncoupled,
                boundary_transfers: vec![boundary_transfer],
            });
        }
    }
    realizations
}

fn acoustic_realizations() -> Vec<ExactFormulationRealization> {
    [
        SourceImplementation::SyntheticProfileFallback,
        SourceImplementation::AcousticHashedTargetGraphProjection,
    ]
    .into_iter()
    .map(|source| ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator: SpatialOperatorImplementation::ReducedAcousticHelmholtz,
            constitutive: ConstitutiveImplementation::ReducedAcousticMedium,
            recovery: RecoveryImplementation::AcousticDomainGraphComplexPressure,
            topologies: vec![TopologyImplementation::GeneratedGraph],
            sources: vec![source],
        }],
        temporals: vec![TemporalImplementation::AcousticDomainGraphHarmonicSolve],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![BoundaryTransferImplementation::GlobalCountProfile],
    })
    .collect()
}

fn fluid_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    let temporal = match procedure {
        AnalysisProcedure::Static => TemporalImplementation::CfdReducedChannelSteadyIteration,
        AnalysisProcedure::TransientResponse => {
            TemporalImplementation::CfdReducedChannelTimeStepping
        }
        _ => return Vec::new(),
    };
    domain_topology_branches()
        .into_iter()
        .map(|branch| ExactFormulationRealization {
            components: vec![fluid_component(&branch)],
            temporals: vec![temporal],
            coupling: CouplingImplementation::Uncoupled,
            boundary_transfers: branch.boundary_transfers,
        })
        .collect()
}

fn coupled_realizations(
    family: CoupledPhysicsFamily,
    procedure: AnalysisProcedure,
) -> Vec<ExactFormulationRealization> {
    let temporal = match procedure {
        AnalysisProcedure::Static => TemporalImplementation::CoupledSinglePassProfileTransfer,
        AnalysisProcedure::TransientResponse => TemporalImplementation::CoupledFixedPointRelaxation,
        AnalysisProcedure::Nonlinear => TemporalImplementation::CoupledScalarResidualIteration,
        _ => return Vec::new(),
    };
    match family {
        CoupledPhysicsFamily::Thermomechanical => thermomechanical_realizations(procedure),
        CoupledPhysicsFamily::Electrothermal => Vec::new(),
        CoupledPhysicsFamily::ConjugateHeatTransfer => flow_thermal_realizations(
            temporal,
            CouplingImplementation::PartitionedConjugateProjection,
        ),
        CoupledPhysicsFamily::FluidStructureInteraction => domain_topology_branches()
            .into_iter()
            .map(|branch| ExactFormulationRealization {
                components: vec![
                    fluid_component(&branch),
                    ExactFormulationComponent {
                        spatial_operator: SpatialOperatorImplementation::Structural(
                            StructuralOperatorImplementation::SyntheticDiagonalChain,
                        ),
                        constitutive:
                            ConstitutiveImplementation::ReducedScalarStructuralNonlinearity,
                        recovery: RecoveryImplementation::Unavailable,
                        topologies: vec![TopologyImplementation::DofAdjacency],
                        sources: vec![SourceImplementation::OrdinalDistribution],
                    },
                ],
                temporals: vec![temporal],
                coupling: CouplingImplementation::PartitionedFluidStructureRelaxation,
                boundary_transfers: vec![
                    BoundaryTransferImplementation::HeuristicInterfaceTransfer,
                ],
            })
            .collect(),
    }
}

fn thermomechanical_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    let branches = [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet10VolumeAverageStrain),
        ),
        (
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            Some(StructuralRecoveryImplementation::BeamSectionResultants),
        ),
        (
            StructuralOperatorImplementation::Mitc3TriangularShell,
            Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants),
        ),
        (
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
        ),
    ];
    branches
        .into_iter()
        .map(|(operator, recovery)| {
            thermomechanical_exact_realization(ThermomechanicalRealizationInput {
                structural: StructuralRealizationInput {
                    assemblies: vec![StructuralFormulationRealization::new(operator, recovery)],
                    procedure,
                    coupling: CouplingImplementation::OnePassPrescribedThermomechanicalProfile,
                    sources: vec![SourceImplementation::NotApplicable],
                    boundary_transfers: vec![BoundaryTransferImplementation::NotApplicable],
                },
                profile_sources: vec![SourceImplementation::PrescribedTemperatureDeltaProfile],
            })
            .expect("registered thermomechanical procedure and operator must be compatible")
        })
        .collect()
}

fn flow_thermal_realizations(
    temporal: TemporalImplementation,
    coupling: CouplingImplementation,
) -> Vec<ExactFormulationRealization> {
    domain_topology_branches()
        .into_iter()
        .map(|branch| ExactFormulationRealization {
            components: vec![fluid_component(&branch), thermal_component(&branch)],
            temporals: vec![temporal],
            coupling,
            boundary_transfers: vec![BoundaryTransferImplementation::HeuristicInterfaceTransfer],
        })
        .collect()
}

#[derive(Clone)]
struct DomainTopologyBranch {
    primary_topology: TopologyImplementation,
    sources: Vec<SourceImplementation>,
    boundary_transfers: Vec<BoundaryTransferImplementation>,
}

impl DomainTopologyBranch {
    fn is_canonical(&self) -> bool {
        matches!(
            self.primary_topology,
            TopologyImplementation::CanonicalSolverMeshTet4
                | TopologyImplementation::CanonicalSolverMeshTet10
        )
    }
}

fn domain_topology_branches() -> Vec<DomainTopologyBranch> {
    let mut branches = Vec::new();
    for primary_topology in [
        TopologyImplementation::CanonicalSolverMeshTet4,
        TopologyImplementation::CanonicalSolverMeshTet10,
    ] {
        branches.extend([
            DomainTopologyBranch {
                primary_topology,
                sources: vec![SourceImplementation::CanonicalBoundaryProjection],
                boundary_transfers: vec![
                    BoundaryTransferImplementation::CanonicalBoundaryProjection,
                ],
            },
            DomainTopologyBranch {
                primary_topology,
                sources: vec![SourceImplementation::OrdinalDistribution],
                boundary_transfers: vec![BoundaryTransferImplementation::GlobalCountProfile],
            },
            DomainTopologyBranch {
                primary_topology,
                sources: vec![SourceImplementation::UnmappedFallback],
                boundary_transfers: vec![BoundaryTransferImplementation::UnmappedFallback],
            },
        ]);
    }
    branches.push(DomainTopologyBranch {
        primary_topology: TopologyImplementation::ImplicitChannel,
        sources: vec![SourceImplementation::OrdinalDistribution],
        boundary_transfers: vec![BoundaryTransferImplementation::GlobalCountProfile],
    });
    branches
}

fn fluid_component(branch: &DomainTopologyBranch) -> ExactFormulationComponent {
    ExactFormulationComponent {
        spatial_operator: SpatialOperatorImplementation::ReducedChannelFlow,
        constitutive: ConstitutiveImplementation::ReducedIncompressibleFlowClosure,
        recovery: RecoveryImplementation::CfdReducedChannelProfile,
        topologies: vec![branch.primary_topology],
        sources: branch.sources.clone(),
    }
}

fn thermal_component(branch: &DomainTopologyBranch) -> ExactFormulationComponent {
    ExactFormulationComponent {
        spatial_operator: SpatialOperatorImplementation::ReducedThermalDiffusion,
        constitutive: ConstitutiveImplementation::ReducedThermalCoefficientProfile,
        recovery: if branch.is_canonical() {
            RecoveryImplementation::ThermalSolverMeshGradient
        } else {
            RecoveryImplementation::ThermalInferredAxisGradient
        },
        topologies: vec![if branch.is_canonical() {
            branch.primary_topology
        } else {
            TopologyImplementation::InferredLattice
        }],
        sources: branch.sources.clone(),
    }
}
