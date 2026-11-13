use runmat_analysis_core::{ObservationId, ProcedureStep};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest, FieldRecoverySource,
    ObservationEvaluationRequest, SequenceIdentity, StepIdentity, ThermalSolverResultManifest,
};
use runmat_analysis_fea::results::field::StructuralRecoveredFieldTopology;
use runmat_analysis_fea::results::observation::{
    resolve_structural_observation_plan, ObservationArtifactManifest, ObservationFieldCatalog,
    ObservationFieldSelection, ObservationLimits, ObservationResult,
    ObservationSelectionCoordinate,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    import_analysis_partition, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisHostWorkload, AnalysisOutputRoot, AnalysisPartitionImport, AnalysisWorkloadInput,
    ImportedAnalysisPartition, ImportedRecoveredFields, PreparedAnalysisObjects,
};

use crate::observation_source::{AdmittedObservationWorkload, ObservationWorkloadSource};

pub struct ObservationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source: ObservationPublishedSource,
    pub source_workload: ObservationWorkloadSource,
    pub mesh: Option<PreparedAnalysisObjects>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub limits: ObservationLimits,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    /// The admitted local CPU plan independent fan-out work (per observation partition) runs on.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

/// Published metadata and transport reference; thermal variants never own numerical history.
pub enum ObservationPublishedSource {
    Structural {
        recovery_request: FieldRecoveryRequest,
        fields: ImportedRecoveredFields,
    },
    SampledRecovery {
        recovery_request: FieldRecoveryRequest,
        fields: ImportedRecoveredFields,
        sampling: runmat_analysis_fea::results::field::SampledVolumeFieldTopology,
    },
    /// Native nodal/element fields recovered on the canonical solver mesh.
    VolumeMeshRecovery {
        recovery_request: FieldRecoveryRequest,
        fields: ImportedRecoveredFields,
    },
    Thermal {
        manifest: ThermalSolverResultManifest,
        output: AnalysisOutputRoot,
        access: ArtifactAccess,
        sample_ordinal: usize,
    },
}

impl ObservationPublishedSource {
    pub(crate) fn output(&self) -> &AnalysisOutputRoot {
        match self {
            Self::Structural { fields, .. }
            | Self::SampledRecovery { fields, .. }
            | Self::VolumeMeshRecovery { fields, .. } => &fields.output,
            Self::Thermal { output, .. } => output,
        }
    }
    pub(crate) fn access<'a>(&'a self, host: &'a AnalysisHostWorkload) -> &'a ArtifactAccess {
        match self {
            Self::Structural { .. }
            | Self::SampledRecovery { .. }
            | Self::VolumeMeshRecovery { .. } => &host.artifact_access,
            Self::Thermal { access, .. } => access,
        }
    }
}

#[path = "observation_sampled.rs"]
mod sampled;
#[path = "observation_thermal.rs"]
mod thermal;
#[path = "observation_volume_mesh.rs"]
mod volume_mesh;

pub(crate) struct AdmittedObservation {
    pub request: ObservationEvaluationRequest,
    pub selection: ObservationFieldSelection,
    pub field_catalog: PreparedAnalysisObjects,
    pub workload: AdmittedObservationWorkload,
    pub mesh: Option<SolverMeshArtifact>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
    /// The sealed catalog/plan pair of a volume-mesh source, carried to publication.
    pub volume_mesh:
        Option<runmat_analysis_fea::results::observation::VolumeMeshObservationAuthority>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservations {
    pub manifest: ObservationArtifactManifest,
    pub catalog_objects: ImportedAnalysisPartition,
    pub output: AnalysisOutputRoot,
    pub access: ArtifactAccess,
}

pub(crate) fn admit_observation(
    preparation: &ObservationPreparation,
) -> AnalysisBridgeResult<AdmittedObservation> {
    match &preparation.source {
        ObservationPublishedSource::Structural {
            recovery_request,
            fields,
        } => admit_structural(preparation, recovery_request, fields),
        ObservationPublishedSource::SampledRecovery {
            recovery_request,
            fields,
            sampling,
        } => sampled::admit_sampled(preparation, recovery_request, fields, sampling),
        ObservationPublishedSource::VolumeMeshRecovery {
            recovery_request,
            fields,
        } => volume_mesh::admit_volume_mesh(preparation, recovery_request, fields),
        ObservationPublishedSource::Thermal { .. } => thermal::admit_thermal(preparation),
    }
}

fn admit_recovery_source(
    preparation: &ObservationPreparation,
    source_request: &FieldRecoveryRequest,
    source_fields: &ImportedRecoveredFields,
) -> AnalysisBridgeResult<(AdmittedObservationWorkload, Option<SolverMeshArtifact>)> {
    preparation.source_host.validate()?;
    source_request.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::RecoverField
        || preparation.source_host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
        || source_fields.output.kind != AnalysisNumericalObjectKind::Field
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
    if request_input.logical_identity != source_request.identity().digest()
        || preparation.source_host.workload.step != source_request.producer
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
    if expected_step != source_request.producer {
        return Err(AnalysisBridgeError::Identity(
            "observation procedure step differs from its producing step",
        ));
    }
    let workload = preparation.source_workload.decode()?;
    match (&workload, source_request.source) {
        (AdmittedObservationWorkload::Linear(_), FieldRecoverySource::SolverResult(_))
        | (AdmittedObservationWorkload::NonlinearSolid(_), FieldRecoverySource::SolverResult(_))
        | (AdmittedObservationWorkload::Magnetostatic(_), FieldRecoverySource::SolverResult(_))
        | (AdmittedObservationWorkload::MaxwellHarmonic(_), FieldRecoverySource::SolverResult(_))
        | (AdmittedObservationWorkload::VolumeMesh, FieldRecoverySource::SolverResult(_))
        | (
            AdmittedObservationWorkload::Frequency(_),
            FieldRecoverySource::StructuralFrequencyEntry(_),
        ) => {}
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "observation workload kind differs from the recovered-field source",
            ))
        }
    }
    let mesh = preparation
        .mesh
        .as_ref()
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    Ok((workload, mesh))
}

fn admit_structural(
    preparation: &ObservationPreparation,
    source_request: &FieldRecoveryRequest,
    source_fields: &ImportedRecoveredFields,
) -> AnalysisBridgeResult<AdmittedObservation> {
    let (workload, mesh) = admit_recovery_source(preparation, source_request, source_fields)?;
    let model = workload.model()?;
    let topology = StructuralRecoveredFieldTopology::derive(model, mesh.as_ref())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if source_request.topology
        != (runmat_analysis_fea::execution::FieldTopologyIdentity::Structural {
            topology: topology.identity,
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "observation topology differs from its field-recovery request",
        ));
    }
    let fields = &source_fields.manifest;
    if fields.identity.digest() != source_fields.output.logical_identity
        || fields.recovery_request != source_request.identity()
        || fields.source != source_request.source
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
            model,
            mesh.as_ref(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let plan = resolve_structural_observation_plan(
        &preparation.step,
        model,
        mesh.as_ref(),
        &constraint_plan,
        &definitions,
        preparation.limits,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let catalog = ObservationFieldCatalog::structural(source_request.clone(), fields.clone())?;
    let selected_fields = catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::StructuralRecovery)?;
    let selection = ObservationFieldSelection::structural(source_request, fields, selected_fields)?;
    let request = ObservationEvaluationRequest::new(selection.clone(), plan)?;
    catalog.validate_structural_selection(&selection)?;
    let field_catalog = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
        catalog.identity(),
        &catalog.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    for accelerator in &preparation.accelerator_requests {
        accelerator
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    }
    Ok(AdmittedObservation {
        request,
        selection,
        field_catalog,
        workload,
        mesh,
        accelerator_requests: preparation.accelerator_requests.clone(),
        volume_mesh: None,
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
