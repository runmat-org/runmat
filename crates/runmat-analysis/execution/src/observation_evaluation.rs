use std::collections::BTreeSet;

use runmat_analysis_core::{ObservationId, ProcedureStep};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest, FieldRecoverySource,
    ObservationEvaluationRequest, SequenceIdentity, StepIdentity,
};
use runmat_analysis_fea::planning::resource::{FeaResourcePlan, ObservationResourceFacts};
use runmat_analysis_fea::results::field::{
    RecoveredFieldSelection, StructuralRecoveredFieldTopology,
};
use runmat_analysis_fea::results::observation::{
    resolve_structural_observation_plan, ObservationArtifactManifest, ObservationLimits,
    ObservationResult,
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    import_analysis_partition, AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload,
    AnalysisOutputRoot, AnalysisPartitionImport, AnalysisWorkloadInput, ImportedAnalysisPartition,
    ImportedRecoveredFields, PreparedAnalysisObjects,
};

use crate::observation_source::{
    AdmittedStructuralObservationWorkload, StructuralObservationWorkloadSource,
};

pub struct StructuralObservationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: FieldRecoveryRequest,
    pub source_fields: ImportedRecoveredFields,
    pub source_workload: StructuralObservationWorkloadSource,
    pub mesh: Option<PreparedAnalysisObjects>,
    pub resource_plan: PreparedAnalysisObjects,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub limits: ObservationLimits,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub(crate) struct AdmittedStructuralObservation {
    pub request: ObservationEvaluationRequest,
    pub selection: RecoveredFieldSelection,
    pub workload: AdmittedStructuralObservationWorkload,
    pub mesh: Option<SolverMeshArtifact>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservations {
    pub manifest: ObservationArtifactManifest,
    pub catalog_objects: ImportedAnalysisPartition,
    pub output: AnalysisOutputRoot,
    pub access: ArtifactAccess,
}

pub(crate) fn admit_structural_observation(
    preparation: &StructuralObservationPreparation,
) -> AnalysisBridgeResult<AdmittedStructuralObservation> {
    preparation.source_host.validate()?;
    preparation.source_request.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::RecoverField
        || preparation.source_host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
        || preparation.source_fields.output.kind != AnalysisNumericalObjectKind::Field
        || preparation.resource_plan.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan
        || preparation
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh)
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation evaluation requires completed recovered fields, their typed source workload, optional canonical mesh, and bounded chunks".into(),
        ));
    }
    let request_input = unique_workload_input(
        &preparation.source_host,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let source_workload_input =
        unique_workload_input(&preparation.source_host, preparation.source_workload.kind())?;
    let mesh_input =
        optional_workload_input(&preparation.source_host, AnalysisNumericalObjectKind::Mesh)?;
    if request_input.logical_identity != preparation.source_request.identity().digest()
        || preparation.source_host.workload.step != preparation.source_request.producer
        || source_workload_input.logical_identity
            != preparation
                .source_workload
                .objects()
                .manifest
                .logical_identity
        || source_workload_input.payload_digest
            != preparation
                .source_workload
                .objects()
                .manifest
                .payload_digest
        || source_workload_input.manifest_digest
            != preparation.source_workload.objects().root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "observation source differs from its field-recovery workload",
        ));
    }
    match (mesh_input, preparation.mesh.as_ref()) {
        (Some(source), Some(mesh))
            if source.logical_identity == mesh.manifest.logical_identity
                && source.payload_digest == mesh.manifest.payload_digest
                && source.manifest_digest == mesh.root.digest => {}
        (None, None) => {}
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "observation mesh presence or identity differs from the field-recovery workload",
            ))
        }
    }
    let expected_step =
        StepIdentity::from_solver_contract(preparation.sequence, &preparation.step)?;
    if expected_step != preparation.source_request.producer {
        return Err(AnalysisBridgeError::Identity(
            "observation procedure step differs from its producing step",
        ));
    }
    let workload = preparation.source_workload.decode()?;
    match (&workload, preparation.source_request.source) {
        (
            AdmittedStructuralObservationWorkload::Linear(_),
            FieldRecoverySource::SolverResult(_),
        )
        | (
            AdmittedStructuralObservationWorkload::Frequency(_),
            FieldRecoverySource::StructuralFrequencyEntry(_),
        ) => {}
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "observation workload kind differs from the recovered-field source",
            ))
        }
    }
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&preparation.resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mesh = preparation
        .mesh
        .as_ref()
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let topology = StructuralRecoveredFieldTopology::derive(workload.model(), mesh.as_ref())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if topology.identity != preparation.source_request.topology {
        return Err(AnalysisBridgeError::Identity(
            "observation topology differs from its field-recovery request",
        ));
    }
    let fields = &preparation.source_fields.manifest;
    if fields.identity.digest() != preparation.source_fields.output.logical_identity
        || fields.recovery_request != preparation.source_request.identity()
        || fields.source != preparation.source_request.source
    {
        return Err(AnalysisBridgeError::Identity(
            "recovered-field artifact differs from its source request",
        ));
    }
    let definitions = fields
        .fields
        .iter()
        .map(|field| field.definition.clone())
        .collect::<Vec<_>>();
    let constraint_plan =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            workload.model(),
            mesh.as_ref(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let plan = resolve_structural_observation_plan(
        &preparation.step,
        workload.model(),
        mesh.as_ref(),
        &constraint_plan,
        &definitions,
        preparation.limits,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let required_definitions = plan
        .observations
        .iter()
        .map(|observation| observation.source_field_definition_identity)
        .collect::<BTreeSet<_>>();
    let selected_fields = fields
        .fields
        .iter()
        .filter(|entry| required_definitions.contains(&entry.definition.identity()))
        .map(|entry| entry.field)
        .collect::<Vec<_>>();
    let selection = RecoveredFieldSelection::new(fields.clone(), selected_fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let request = ObservationEvaluationRequest::new(
        preparation.source_request.producer,
        &selection,
        topology.identity,
        plan,
    )?;
    let expected_observation_resources =
        ObservationResourceFacts::from_plan(&request.plan, workload.observation_execution())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.input.topology_digest() != request.plan.topology_digest
        || resource_plan.input.observations() != Some(&expected_observation_resources)
    {
        return Err(AnalysisBridgeError::Identity(
            "observation execution policy differs from its canonical resource plan",
        ));
    }
    Ok(AdmittedStructuralObservation {
        request,
        selection,
        workload,
        mesh,
        resource_plan,
    })
}

pub fn import_observation_result(
    source: &impl CacheImport,
    observations: &ImportedObservations,
    observation_id: &ObservationId,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ObservationResult> {
    let entry = observations
        .manifest
        .results
        .iter()
        .find(|entry| &entry.observation_id == observation_id)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "requested observation is absent from the observation manifest".into(),
            )
        })?;
    let imported = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: &observations.output.root,
            expected_kind: AnalysisNumericalObjectKind::Observation,
            expected_logical_identity: observations.output.logical_identity,
            expected_payload_digest: observations.output.payload_digest,
            partition_identity: entry.result.digest(),
            access: &observations.access,
            limits,
        },
    )?;
    let result = ObservationResult::from_canonical_bytes(&imported.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if result.logical_identity != entry.result
        || result.observation_id != entry.observation_id
        || result.quantity != entry.quantity
        || result.location != entry.location
        || result.value_shape != entry.value_shape
        || result.partial_access != entry.partial_access
        || result.plan_identity != observations.manifest.plan_identity
        || result.topology_digest != observations.manifest.topology_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "observation result differs from its semantic manifest entry",
        ));
    }
    Ok(result)
}

fn unique_workload_input(
    host: &AnalysisHostWorkload,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&AnalysisWorkloadInput> {
    let mut matches = host
        .workload
        .inputs
        .iter()
        .filter(|input| input.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

fn optional_workload_input(
    host: &AnalysisHostWorkload,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<Option<&AnalysisWorkloadInput>> {
    let mut matches = host
        .workload
        .inputs
        .iter()
        .filter(|input| input.kind == kind);
    let input = matches.next();
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}
