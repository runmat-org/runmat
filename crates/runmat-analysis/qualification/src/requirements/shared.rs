//! Standing cross-family acceptance with explicit domain boundaries. Required envelopes
//! without a discovered witness remain open; declarations never imply a test verdict.
use crate::requirement::*;
const KERNEL: &[Envelope] = &[envelope(
    Formulation::SharedInfrastructure,
    MeshOrder::CanonicalMixedTopology,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const ALL: &[Envelope] = &[
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::SharedInfrastructure,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
mod acceptance;
mod execution;
mod inputs_state;
mod resources;
mod spatial;
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    acceptance::FEA_SHARED_OWNERSHIP_001,
    spatial::FEA_SHARED_TOPOLOGY_001,
    spatial::FEA_SHARED_PROPERTIES_001,
    spatial::FEA_SHARED_CONSTRAINTS_001,
    inputs_state::FEA_SHARED_EXCITATION_001,
    inputs_state::FEA_SHARED_SEQUENCES_001,
    inputs_state::FEA_SHARED_OBSERVATIONS_001,
    execution::FEA_SHARED_EXECUTION_001,
    execution::FEA_SHARED_OPERATORS_001,
    execution::FEA_SHARED_ACCELERATORS_001,
    execution::FEA_SHARED_IDENTITY_001,
    resources::FEA_SHARED_RESOURCES_001,
    acceptance::FEA_SHARED_ACCURACY_001,
    acceptance::FEA_SHARED_EXTENSIONS_001,
    acceptance::FEA_SHARED_BASELINE_001,
    acceptance::FEA_SHARED_AUTHORITIES_001,
];
