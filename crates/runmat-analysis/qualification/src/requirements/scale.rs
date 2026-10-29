//! Performance at scale: one thread plan, remote and cooperative execution for every family, and published sizes and times.
use crate::requirement::*;
const PAR: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::ParallelCpu]
);
const REMOTE: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::RemoteExecution]
);
const DECOMP: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::DomainDecomposed]
);
const PUBLIC: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::PublicStudy]
);
const SWEEP: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::RemoteExecution
    ]
);
const DURABLE: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::Durable]
);
const SHAREDPAR: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [ExecutionEnvelope::ParallelCpu]
);
const SHARED: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::ParallelCpu,
        ExecutionEnvelope::RemoteExecution,
        ExecutionEnvelope::DomainDecomposed
    ]
);
const FLUIDPUB: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [ExecutionEnvelope::PublicStudy]
);
const FEA_SCALE_PARALLEL_001: Requirement = Requirement {
    id: "FEA-SCALE-PARALLEL-001",
    title: "Every family saturates the machine through one thread plan",
    stages: &[Stage::Scale],
    acceptance: "Every physics family assembles, solves, recovers and publishes in parallel under the single thread-plan authority, so a study uses all admitted cores of one machine with no serial-only assembly, solve or recovery path, and results match the serial reference under the declared determinism policy and are identical across thread counts.",
    owners: &["crates/runmat-cpu-execution/src/lib.rs", "crates/runmat-sparse-solvers/src/par.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "one_thread_plan_authority",
            "Exactly one thread-plan authority binds the admitted cores for assembly, solve, recovery and publication of every family; no family reserves cores and runs serially, and a study that requests N threads is observed to use N.",
            EvidenceKind::ResourceAdmission,
            PAR,
        ),
        obligation(
            "parallel_assembly_and_recovery",
            "Operator assembly, boundary and excitation realization, field recovery and publication run in parallel and agree with the serial reference within the declared tolerance, bit-for-bit where the policy declares it.",
            EvidenceKind::ExecutionEquivalence,
            PAR,
        ),
        obligation(
            "parallel_solve",
            "Direct (Cholesky, LDLT, LU), Krylov, AMG and block-eigen solves of the family run in parallel on the shared sparse layer and reproduce the serial solution and residual history within tolerance.",
            EvidenceKind::ExecutionEquivalence,
            PAR,
        ),
        obligation(
            "shared_parallel_kernels",
            "The shared parallel kernels under every family agree with their serial references: element assembly into the sparse pattern is bit-identical to the serial loop at 1, 2, 3, 7 and 16 threads; scoped Krylov solves equal the serial solution and reject nested execution; pooled CSR products preserve serial bits and output shape.",
            EvidenceKind::ExecutionEquivalence,
            SHAREDPAR,
        ),
        obligation(
            "thread_count_determinism",
            "The same study yields identical result digests, or differences within the stated tolerance, across 1, 2, 4 and all admitted threads.",
            EvidenceKind::ExecutionEquivalence,
            PAR,
        ),
        obligation(
            "strong_scaling_table",
            "A strong-scaling table from 1 to all cores of the stated machine, produced with the exclusive benchmark gate, is generated and published for the family's published example, with parallel efficiency and the hardware recorded.",
            EvidenceKind::PerformanceMeasurement,
            PAR,
        ),
    ],
};
const FEA_SCALE_DISTRIBUTED_001: Requirement = Requirement {
    id: "FEA-SCALE-DISTRIBUTED-001",
    title: "Remote and cooperative execution for every family",
    stages: &[Stage::Scale, Stage::ExecutionContracts],
    acceptance: "Every physics family runs whole-study on a remote cluster with the same results as local execution, sweep cases and frequency points fan out across workers with results in authored order, and one large problem partitions across workers through domain decomposition with a solution equal to the serial reference.",
    owners: &["crates/runmat-execution-runner-native/src/analysis_placement.rs", "crates/runmat-cli/src/commands/fea/preparation.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "remote_whole_study_every_family",
            "`runmat run --cluster` runs the family's published study remotely with encrypted inputs, no unsupported-physics rejection, and results equal to the local run under the equivalence policy.",
            EvidenceKind::ExecutionEquivalence,
            REMOTE,
        ),
        obligation(
            "sweep_and_frequency_fanout",
            "Sweep cases and independent frequency points run concurrently on local workers or a cluster, overlap in time, publish per-case status and return results in authored order identical to serial execution.",
            EvidenceKind::ExecutionEquivalence,
            SWEEP,
        ),
        obligation(
            "domain_decomposition",
            "One problem partitions into owned and ghost subdomains over at least 4 workers; the cooperative solve equals the serial reference within tolerance, fails collectively with epoch fencing, and is demonstrated at a model size beyond one worker's memory share.",
            EvidenceKind::ExecutionEquivalence,
            DECOMP,
        ),
        obligation(
            "checkpoint_and_mid_phase_restart",
            "A long run checkpoints and restarts mid-phase after worker loss or cancellation to the same final result.",
            EvidenceKind::ProcessLifecycle,
            DURABLE,
        ),
        obligation(
            "shared_distributed_contracts",
            "The shared remote, partitioned and thread-parallel contracts are one implementation reached by every family: no family-specific placement table, and unsupported placements fail typed before result commitment.",
            EvidenceKind::ExecutionEquivalence,
            SHARED,
        ),
    ],
};
const FEA_SCALE_SIZE_001: Requirement = Requirement {
    id: "FEA-SCALE-SIZE-001",
    title: "Millions of degrees of freedom at stated time and memory",
    stages: &[Stage::Scale, Stage::PublicSurface],
    acceptance: "Each family's published example runs at a stated model size, wall time and peak memory on stated hardware, with models of millions of degrees of freedom solved on a workstation and a cluster, and whole-task resources measured rather than inferred from component reservations.",
    owners: &["crates/runmat-analysis/fea/src/algebra/mod.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "million_dof_public_study",
            "Each family solves a model of at least one million degrees of freedom (cells for flow) through the public command within a stated wall time and memory, on stated hardware, with the result checked against the same reference as the small case.",
            EvidenceKind::PerformanceMeasurement,
            PUBLIC,
        ),
        obligation(
            "published_size_and_time_per_example",
            "For every physics page example the node count, degrees of freedom, wall time, peak memory and hardware are generated from the run and published.",
            EvidenceKind::PerformanceMeasurement,
            PUBLIC,
        ),
        obligation(
            "whole_task_resource_measurement",
            "Whole-task peak resident memory is measured across worker processes and compared with the admitted estimate; admission never over-reserves a practical small model (buckling at 3,000 nodes, modal output, blade) and never admits a model that then exceeds its bound.",
            EvidenceKind::ResourceAdmission,
            PUBLIC,
        ),
        obligation(
            "usage_reports_and_failure_classification",
            "The plan reports cores, memory, wall time and core hours before a run; each run records per-stage usage and classifies failures (resource, numerical, input, infrastructure) in typed form, and the stated core-hour figure per guide is measured on the release.",
            EvidenceKind::ResourceAdmission,
            PUBLIC,
        ),
        obligation(
            "fluid_performance_targets",
            "Steady flow on 1 million cells on a 16-core workstation completes in under 3 minutes and 4 GB, 10 million cells in under 45 minutes and 40 GB; transient flow sustains at least 2 million cell-steps per second per CPU node; strong-scaling efficiency is at least 70 percent down to 50,000 cells per core.",
            EvidenceKind::PerformanceMeasurement,
            FLUIDPUB,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_SCALE_PARALLEL_001,
    FEA_SCALE_DISTRIBUTED_001,
    FEA_SCALE_SIZE_001,
];
