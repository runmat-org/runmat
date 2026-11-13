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
        PhysicsFamily::Acoustic => acoustic_realizations(procedure),
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
    if procedure == AnalysisProcedure::LineParameters {
        return vec![super::electrical::line_parameters_realization()];
    }
    if procedure != AnalysisProcedure::HarmonicResponse {
        return Vec::new();
    }
    super::maxwell::registered_realizations()
}

fn acoustic_realizations(procedure: AnalysisProcedure) -> Vec<ExactFormulationRealization> {
    let temporal = match procedure {
        AnalysisProcedure::ModalExtraction => TemporalImplementation::AcousticCanonicalModalSolve,
        AnalysisProcedure::TransientResponse => {
            TemporalImplementation::AcousticCanonicalTransientSolve
        }
        _ => return harmonic_acoustic_realizations(),
    };
    [
        TopologyImplementation::CanonicalSolverMeshTet4,
        TopologyImplementation::CanonicalSolverMeshTet10,
    ]
    .into_iter()
    .map(|topology| ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator: SpatialOperatorImplementation::AcousticGalerkinHelmholtz,
            constitutive: ConstitutiveImplementation::ElementOwnedLosslessAcousticFluid,
            recovery: RecoveryImplementation::AcousticSolverMeshNodalPressure,
            topologies: vec![topology],
            sources: vec![SourceImplementation::CanonicalBoundaryProjection],
        }],
        temporals: vec![temporal],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
    })
    .collect()
}

fn harmonic_acoustic_realizations() -> Vec<ExactFormulationRealization> {
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
    match family {
        CoupledPhysicsFamily::Thermomechanical => thermomechanical_realizations(procedure),
        CoupledPhysicsFamily::Electrothermal => Vec::new(),
        CoupledPhysicsFamily::ConjugateHeatTransfer => match procedure {
            AnalysisProcedure::Static => {
                vec![super::fluid::canonical_conjugate_heat_steady_realization()]
            }
            AnalysisProcedure::TransientResponse => {
                vec![super::fluid::canonical_conjugate_heat_transient_realization()]
            }
            _ => Vec::new(),
        },
        CoupledPhysicsFamily::FluidStructureInteraction => match procedure {
            AnalysisProcedure::Static => {
                vec![super::fluid::canonical_fluid_structure_steady_realization()]
            }
            AnalysisProcedure::TransientResponse => {
                vec![super::fluid::canonical_fluid_structure_transient_realization()]
            }
            _ => Vec::new(),
        },
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
