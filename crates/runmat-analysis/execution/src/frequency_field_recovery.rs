use std::collections::BTreeSet;

use runmat_analysis_fea::planning::resource::{
    build_structural_frequency_recovery_resource_plan,
    structural_frequency_recovery_payload_bytes_upper_bound, FeaResourcePlan,
    PreparedPhaseResourceFacts, StructuralFrequencyRecoveryResourceInput,
};
use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest, FieldRecoverySource,
        StructuralFrequencyStateInputs,
    },
    results::field::StructuralRecoveredFieldTopology,
};
use runmat_execution::{value::ValueRefKind, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    decode_prepared_structural_frequency_entry_result,
    decode_prepared_structural_frequency_workload, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, ImportedRecoveredFields, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralFrequencyFieldRecoveryPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub prestress_state: Option<PreparedAnalysisObjects>,
    pub source_entry: PreparedAnalysisObjects,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub fn import_structural_frequency_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralFrequencyFieldRecovery,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    crate::field_recovery::import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        limits,
    )
}

pub struct PreparedStructuralFrequencyFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub fn prepare_structural_frequency_field_recovery(
    preparation: StructuralFrequencyFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedStructuralFrequencyFieldRecovery> {
    preparation.source_host.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::SolveStructuralFrequencyEntry
        || preparation.source_host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralFrequencyEntryResult]
        || preparation.workload.manifest.kind
            != AnalysisNumericalObjectKind::StructuralFrequencyWorkload
        || preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh
        || preparation.source_entry.manifest.kind
            != AnalysisNumericalObjectKind::StructuralFrequencyEntryResult
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency recovery requires one completed frequency entry, workload, canonical mesh, and bounded chunks"
                .into(),
        ));
    }
    let workload = decode_prepared_structural_frequency_workload(&preparation.workload)?;
    let mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = decode_prepared_structural_frequency_entry_result(&preparation.source_entry)?;
    let prestress_state = preparation
        .prestress_state
        .as_ref()
        .map(crate::decode_prepared_semantic_state)
        .transpose()?;
    let prepared = match prestress_state.as_ref() {
        Some(state) => runmat_analysis_fea::pipeline::structural_frequency::validate_prestressed_structural_frequency_entry(
            &workload,
            &mesh,
            result.request(),
            state,
        )?,
        None => runmat_analysis_fea::pipeline::structural_frequency::validate_structural_frequency_entry(
            &workload,
            &mesh,
            result.request(),
        )?,
    };
    let topology = StructuralRecoveredFieldTopology::derive(workload.model(), Some(&mesh))
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.workload.manifest.logical_identity != workload.identity().digest()
        || preparation.mesh.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        || preparation.source_entry.manifest.logical_identity != result.identity().digest()
        || result.request().scope() != &prepared.scope
        || result.request().study() != prepared.study
        || result.request().step() != prepared.step
        || result.request().execution_plan() != prepared.execution_plan
        || preparation.source_host.workload.study != result.request().study()
        || preparation.source_host.workload.step != result.request().step()
        || preparation.source_host.workload.execution_plan != result.request().execution_plan()
        || !source_input_matches(
            &preparation.source_host,
            AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
            &preparation.workload,
        )
        || !source_input_matches(
            &preparation.source_host,
            AnalysisNumericalObjectKind::Mesh,
            &preparation.mesh,
        )
        || preparation
            .source_host
            .workload
            .inputs
            .iter()
            .filter(|input| {
                input.kind == AnalysisNumericalObjectKind::StructuralFrequencyEntryRequest
                    && input.logical_identity == result.request().identity().digest()
            })
            .count()
            != 1
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency recovery inputs differ from their source execution",
        ));
    }
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(workload.model(), Some(&mesh))
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let recovery_plan = runmat_analysis_fea::execution::FieldRecoveryPlan::resolve_structural(
        workload.procedure_step()?,
        topology,
        constraints.groups.len(),
    )?;
    let request = FieldRecoveryRequest::new(
        prepared.step,
        FieldRecoverySource::StructuralFrequencyEntry(result.identity()),
        runmat_analysis_fea::execution::FieldTopologyIdentity::Structural {
            topology: topology.identity,
        },
        recovery_plan,
    )?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut phase_inputs = vec![
        request_objects.clone(),
        preparation.workload.clone(),
        preparation.mesh.clone(),
        preparation.source_entry.clone(),
    ];
    phase_inputs.extend(preparation.prestress_state.iter().cloned());
    let resource_plan = build_structural_frequency_recovery_resource_plan(
        &mesh,
        &constraints,
        &workload,
        StructuralFrequencyRecoveryResourceInput {
            request: &request,
            result: &result,
        },
        StructuralFrequencyStateInputs {
            prestress: prestress_state.as_ref(),
            modal_basis: None,
        },
        &[],
        PreparedPhaseResourceFacts {
            inputs: crate::prepared_input_resource_facts(&phase_inputs)?,
            output: crate::projected_output_resource_facts(
                structural_frequency_recovery_payload_bytes_upper_bound(
                    &mesh,
                    &constraints,
                    &workload,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            output_set: crate::projected_output_set_resource_facts(),
        },
        preparation.local_execution,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut typed_inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.workload, ValueRefKind::DriverObject),
        (preparation.mesh, ValueRefKind::DriverObject),
        (preparation.source_entry, ValueRefKind::ResultObject),
        (resource_plan_objects.clone(), ValueRefKind::DriverObject),
    ];
    if let Some(state) = preparation.prestress_state {
        typed_inputs.push((state, ValueRefKind::ResultObject));
    }
    let workload_inputs = typed_inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = typed_inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let workload = AnalysisWorkloadRequest::new(
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        AnalysisWorkKind::RecoverField,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        preparation.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralFrequencyFieldRecovery {
        request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

fn source_input_matches(
    host: &AnalysisHostWorkload,
    kind: AnalysisNumericalObjectKind,
    objects: &PreparedAnalysisObjects,
) -> bool {
    host.workload
        .inputs
        .iter()
        .filter(|input| {
            input.kind == kind
                && input.logical_identity == objects.manifest.logical_identity
                && input.payload_digest == objects.manifest.payload_digest
                && input.manifest_digest == objects.root.digest
                && input.manifest_encoded_length == objects.root.encoded_length
        })
        .count()
        == 1
}
