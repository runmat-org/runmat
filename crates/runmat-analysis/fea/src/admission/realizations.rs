use runmat_analysis_core::{
    is_exposed_analysis_identity, AnalysisProcedure, BoundaryTransferImplementation,
    ConstitutiveImplementation, CoupledPhysicsFamily, CouplingImplementation,
    ExactFormulationComponent, ExactFormulationRealization, PhysicsFamily, RecoveryImplementation,
    SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
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
}

impl RegisteredFormulation {
    pub(super) const fn new(realization: ExactFormulationRealization) -> Self {
        Self { realization }
    }

    pub fn realization(&self) -> &ExactFormulationRealization {
        &self.realization
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
        .map(RegisteredFormulation::new)
        .collect()
}

fn thermal_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    let temporals = match procedure {
        AnalysisProcedure::Static => [
            TemporalImplementation::ThermalSteadyLinearSolve,
            TemporalImplementation::ThermalSteadyDampedNewton,
        ],
        AnalysisProcedure::TransientResponse => [
            TemporalImplementation::ThermalBackwardEuler,
            TemporalImplementation::ThermalBackwardEulerDampedNewton,
        ],
        _ => return Vec::new(),
    };
    temporals
        .into_iter()
        .flat_map(|temporal| {
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
        })
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
    if procedure == AnalysisProcedure::Static {
        let mut realizations = super::electrical::registered_realizations();
        realizations.extend(super::magnetic::registered_realizations());
        return realizations;
    }
    if procedure != AnalysisProcedure::HarmonicResponse {
        return Vec::new();
    }
    super::maxwell::registered_realizations()
}

fn acoustic_realizations() -> Vec<ExactFormulationRealization> {
    let mut realizations = [
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
    .collect::<Vec<_>>();
    for topology in [
        TopologyImplementation::CanonicalSolverMeshTet4,
        TopologyImplementation::CanonicalSolverMeshTet10,
    ] {
        realizations.push(ExactFormulationRealization {
            components: vec![ExactFormulationComponent {
                spatial_operator: SpatialOperatorImplementation::AcousticGalerkinHelmholtz,
                constitutive: ConstitutiveImplementation::ElementOwnedLosslessAcousticFluid,
                recovery: RecoveryImplementation::AcousticSolverMeshPressureVelocityEnergy,
                topologies: vec![topology],
                sources: vec![SourceImplementation::CanonicalBoundaryProjection],
            }],
            temporals: vec![TemporalImplementation::AcousticCanonicalHarmonicSolve],
            coupling: CouplingImplementation::Uncoupled,
            boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        });
    }
    realizations
}

fn fluid_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    match procedure {
        AnalysisProcedure::Static => vec![super::fluid::canonical_fluid_steady_realization()],
        AnalysisProcedure::TransientResponse => {
            vec![super::fluid::canonical_fluid_transient_realization()]
        }
        _ => Vec::new(),
    }
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
        // The former synthetic scalar branch is not a genuine structural constitutive solve.
        // Do not assign the new J2 identity to that unrelated reduced coupling implementation.
        CoupledPhysicsFamily::FluidStructureInteraction => Vec::new(),
    }
}

fn thermomechanical_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    if procedure == AnalysisProcedure::Nonlinear {
        return Vec::new();
    }
    let branches = [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection),
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
        .flat_map(|(operator, recovery)| {
            let mut couplings = vec![(
                CouplingImplementation::OnePassPrescribedThermomechanicalProfile,
                SourceImplementation::PrescribedTemperatureDeltaProfile,
            )];
            if procedure == AnalysisProcedure::Static
                && matches!(
                    operator,
                    StructuralOperatorImplementation::Tet4LinearElasticity
                        | StructuralOperatorImplementation::Tet10LinearElasticity
                )
            {
                couplings.push((
                    CouplingImplementation::OneWaySolvedThermoelasticState,
                    SourceImplementation::SolvedThermalStateNodalField,
                ));
            }
            couplings.into_iter().map(move |(coupling, source)| {
                thermomechanical_exact_realization(ThermomechanicalRealizationInput {
                    structural: StructuralRealizationInput {
                        assemblies: vec![StructuralFormulationRealization::new(operator, recovery)],
                        procedure,
                        kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
                        coupling,
                        sources: vec![SourceImplementation::NotApplicable],
                        boundary_transfers: vec![BoundaryTransferImplementation::NotApplicable],
                    },
                    profile_sources: vec![source],
                })
                .expect("registered thermomechanical procedure and operator must be compatible")
            })
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
            },
            DomainTopologyBranch {
                primary_topology,
                sources: vec![SourceImplementation::OrdinalDistribution],
            },
            DomainTopologyBranch {
                primary_topology,
                sources: vec![SourceImplementation::UnmappedFallback],
            },
        ]);
    }
    branches.push(DomainTopologyBranch {
        primary_topology: TopologyImplementation::ImplicitChannel,
        sources: vec![SourceImplementation::OrdinalDistribution],
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
