use std::collections::BTreeSet;

use runmat_analysis_execution::{
    decode_recovered_field_artifact, prepare_structural_linear_program,
    prepare_structural_sequence_step, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity;
use runmat_analysis_fea::execution::SolverResultIdentity;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::planning::resource::{record_resource_usage, ResourceUsageAttempt};
use runmat_analysis_fea::results::accuracy::{AccuracyStatus, PublicationDecision};
use runmat_analysis_fea::results::field::RecoveredFieldCatalog;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::resource::ResourceAccounting;
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn prepare_kernel_output(
    output: &AnalysisKernelOutput,
) -> runmat_analysis_execution::PreparedAnalysisObjects {
    output
        .clone()
        .into_prepared(
            output_policy().chunk_bytes(),
            output_policy().inventory_limits(),
        )
        .unwrap()
}

fn output_policy() -> runmat_analysis_execution::AnalysisOutputPolicy {
    runmat_analysis_execution::AnalysisOutputPolicy::new(4096, ObjectInventoryLimits::default())
        .unwrap()
}

fn test_host_for_inputs(
    source: &AnalysisHostWorkload,
    kind: AnalysisWorkKind,
    inputs: &[runmat_analysis_execution::PreparedAnalysisObjects],
    outputs: Vec<AnalysisNumericalObjectKind>,
    result_input: Option<usize>,
) -> AnalysisHostWorkload {
    let result_inputs = result_input.into_iter().collect::<Vec<_>>();
    test_host_for_input_references(source, kind, inputs, outputs, &result_inputs)
}

fn test_host_for_input_references(
    source: &AnalysisHostWorkload,
    kind: AnalysisWorkKind,
    inputs: &[runmat_analysis_execution::PreparedAnalysisObjects],
    outputs: Vec<AnalysisNumericalObjectKind>,
    result_inputs: &[usize],
) -> AnalysisHostWorkload {
    let workload_inputs = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if result_inputs.contains(&index) {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect();
    AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            source.workload.study,
            source.workload.step,
            source.workload.execution_plan,
            kind,
            workload_inputs,
            outputs,
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap(),
        source.artifact_access.clone(),
        source.output_policy,
    )
    .unwrap()
}

pub(super) fn structural_displacement_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::Field(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
        quantity: runmat_analysis_core::ObservationQuantity::Displacement,
        value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
        target: runmat_analysis_core::ObservationTarget::WholeModel,
        source_location: runmat_analysis_core::FieldLocation::Node,
        output_location: runmat_analysis_core::FieldLocation::Node,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling: runmat_analysis_core::SamplingPolicy::Final,
        reduction: None,
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::Canonical,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        ),
    })
}

fn structural_nodal_von_mises_observation() -> runmat_analysis_core::ObservationRequest {
    let runmat_analysis_core::ObservationRequest::Field(mut spec) =
        structural_displacement_observation()
    else {
        unreachable!("structural displacement helper always returns a field observation")
    };
    spec.observation_id = runmat_analysis_core::ObservationId("nodal_von_mises".into());
    spec.quantity = runmat_analysis_core::ObservationQuantity::VonMisesStress;
    spec.value_shape = runmat_analysis_core::ObservationValueShape::Scalar;
    spec.source_location = runmat_analysis_core::FieldLocation::Element;
    spec.output_location = runmat_analysis_core::FieldLocation::Node;
    spec.recovery = runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage;
    runmat_analysis_core::ObservationRequest::Field(spec)
}

#[path = "tests/conformal_mixed.rs"]
mod conformal_mixed;
#[path = "tests/explicit.rs"]
mod explicit;
#[path = "tests/linear_pipeline.rs"]
mod linear_pipeline;
