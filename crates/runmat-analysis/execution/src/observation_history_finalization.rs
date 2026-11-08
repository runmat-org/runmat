//! Governed finalization of a source-bound checkpoint; the driver never reduces its values.
use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind as Kind, AnalysisWorkKind, ObservationHistoryCheckpointRequest,
    ObservationHistoryFinalizationRequest, ObservationHistorySourceContext, ThermalWorkload,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_history_finalization_resource_plan,
    observation_history_finalization_payload_bytes_upper_bound,
    observation_history_series_reduction_payload_bytes_upper_bound, FeaResourcePlan,
};
use runmat_analysis_fea::results::observation::{
    BoundedObservationHistory, ObservationFieldCatalog, ObservationHistoryPublication,
    SeriesReductionPublication, SeriesReductionResult,
};
use runmat_execution::{value::ValueRefKind, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ObjectNamespace, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::observation_source::{AdmittedObservationWorkload, ObservationWorkloadSource};
use crate::{
    analysis_object_references, decode_prepared_observation_history_checkpoint,
    import_analysis_objects, import_analysis_output_set, import_observation_history,
    prepare_analysis_objects, prepare_analysis_program, prepared_input_resource_facts,
    projected_output_resource_facts, projected_output_set_resource_facts, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, ImportedObservationHistory,
    ImportedObservationHistoryStorage, ObservationHistoryImport, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

#[derive(Clone)]
pub enum ObservationHistoryFinalizationSource {
    Structural {
        workload: ObservationWorkloadSource,
        mesh: Option<PreparedAnalysisObjects>,
    },
    Thermal {
        workload: PreparedAnalysisObjects,
        mesh: PreparedAnalysisObjects,
    },
    /// Native node, element or face fields on the canonical solver mesh, of any family: the
    /// volume-mesh field catalog carries the producing step, so the mesh is the only context.
    VolumeMesh { mesh: PreparedAnalysisObjects },
}

pub struct ObservationHistoryFinalizationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub request: ObservationHistoryFinalizationRequest,
    pub checkpoint: PreparedAnalysisObjects,
    pub catalog: PreparedAnalysisObjects,
    pub source: ObservationHistoryFinalizationSource,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedObservationHistoryFinalizationTask {
    pub request: ObservationHistoryFinalizationRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservationHistoryFinalization {
    /// Metadata derived from the validated terminal checkpoint and imported output closure.
    pub publication: ObservationHistoryPublication,
    pub history: ImportedObservationHistory,
    pub series_reduction: Option<SeriesReductionResult>,
    pub series_objects: Option<PreparedAnalysisObjects>,
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

pub fn prepare_observation_history_finalization_task(
    preparation: ObservationHistoryFinalizationPreparation,
) -> AnalysisBridgeResult<PreparedObservationHistoryFinalizationTask> {
    let source_host = &preparation.source_host;
    source_host.validate()?;
    if preparation.chunk_bytes == 0
        || source_host.workload.kind != AnalysisWorkKind::CheckpointObservationHistory
        || source_host.workload.outputs != vec![Kind::ObservationHistoryCheckpoint]
        || source_host.workload.step != preparation.request.producer
        || preparation.catalog.manifest.kind != Kind::ObservationFieldCatalog
    {
        return Err(invalid(
            "history finalization requires a checkpoint task and authoritative field catalog",
        ));
    }
    let checkpoint = decode_prepared_observation_history_checkpoint(&preparation.checkpoint)?;
    let checkpoint_requests = source_host
        .workload
        .inputs
        .iter()
        .filter(|input| input.kind == Kind::ObservationHistoryCheckpointRequest)
        .collect::<Vec<_>>();
    let [checkpoint_request] = checkpoint_requests.as_slice() else {
        return Err(invalid("checkpoint source requires one transition request"));
    };
    // Bind the supplied terminal checkpoint to the producing task's exact transition.
    let terminal_sample = checkpoint
        .samples()
        .last()
        .ok_or_else(|| invalid("empty checkpoint"))?;
    let source_request = ObservationHistoryCheckpointRequest::new(
        preparation.request.plan.clone(),
        preparation.request.producer,
        preparation.request.observation_id.clone(),
        checkpoint.manifest().predecessor,
        checkpoint
            .manifest()
            .observed_sample_count
            .checked_sub(1)
            .ok_or_else(|| invalid("empty checkpoint"))?,
        &terminal_sample.result,
    )?;
    if source_request.identity.digest() != checkpoint_request.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "terminal checkpoint differs from producing task",
        ));
    }
    let catalog = ObservationFieldCatalog::from_canonical_bytes(&preparation.catalog.payload())?;
    if preparation.catalog.manifest.logical_identity != catalog.identity() {
        return Err(AnalysisBridgeError::Identity(
            "history field catalog object identity differs",
        ));
    }
    let mut inputs = vec![
        prepare_analysis_objects(
            Kind::ObservationHistoryFinalizationRequest,
            preparation.request.identity().digest(),
            &preparation.request.canonical_bytes()?,
            preparation.chunk_bytes,
            ObjectNamespace::InputValue,
            preparation.inventory_limits,
        )?,
        preparation.checkpoint,
        preparation.catalog,
    ];
    let (decoded_workload, decoded_mesh) = match preparation.source {
        ObservationHistoryFinalizationSource::Structural { workload, mesh } => {
            let decoded = workload.decode()?;
            let canonical = match &decoded {
                AdmittedObservationWorkload::Linear(linear) => linear.canonical_bytes()?,
                AdmittedObservationWorkload::Frequency(frequency) => {
                    if workload.objects().manifest.logical_identity != frequency.identity().digest()
                    {
                        return Err(AnalysisBridgeError::Identity(
                            "structural frequency workload identity differs",
                        ));
                    }
                    frequency.canonical_bytes()?
                }
                AdmittedObservationWorkload::NonlinearSolid(_) => {
                    return Err(invalid("nonlinear history needs an original accepted-path source; only final fields are admitted"));
                }
                AdmittedObservationWorkload::Thermal(_)
                | AdmittedObservationWorkload::Magnetostatic(_)
                | AdmittedObservationWorkload::VolumeMesh
                | AdmittedObservationWorkload::MaxwellHarmonic(_) => {
                    return Err(invalid(
                        "structural history cannot use a nonstructural workload",
                    ));
                }
            };
            if workload.objects().manifest.payload_digest
                != runmat_execution::Digest::sha256(&canonical)
                || workload.objects().payload() != canonical
            {
                return Err(AnalysisBridgeError::Identity(
                    "structural workload canonical content differs",
                ));
            }
            let decoded_mesh = mesh
                .as_ref()
                .map(|objects| {
                    if objects.manifest.kind != Kind::Mesh {
                        return Err(invalid("structural history mesh has wrong numerical kind"));
                    }
                    let decoded = SolverMeshArtifact::canonical_decode(&objects.payload())
                        .map_err(invalid)?;
                    if objects.manifest.logical_identity
                        != runmat_execution::Digest::from_bytes(*decoded.canonical_digest.bytes())
                    {
                        return Err(AnalysisBridgeError::Identity(
                            "structural mesh identity differs",
                        ));
                    }
                    Ok(decoded)
                })
                .transpose()?;
            inputs.push(workload.objects().clone());
            inputs.extend(mesh);
            (decoded, decoded_mesh)
        }
        ObservationHistoryFinalizationSource::Thermal { workload, mesh } => {
            if workload.manifest.kind != Kind::ThermalWorkload || mesh.manifest.kind != Kind::Mesh {
                return Err(invalid("thermal history requires typed workload and mesh"));
            }
            let decoded_workload = ThermalWorkload::from_canonical_bytes(&workload.payload())?;
            let decoded_mesh =
                SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
            if workload.manifest.logical_identity != decoded_workload.identity()?.digest()
                || mesh.manifest.logical_identity
                    != runmat_execution::Digest::from_bytes(*decoded_mesh.canonical_digest.bytes())
            {
                return Err(AnalysisBridgeError::Identity(
                    "thermal context identity differs",
                ));
            }
            inputs.extend([workload, mesh]);
            (
                AdmittedObservationWorkload::Thermal(decoded_workload),
                Some(decoded_mesh),
            )
        }
        ObservationHistoryFinalizationSource::VolumeMesh { mesh } => {
            if mesh.manifest.kind != Kind::Mesh {
                return Err(invalid("volume-mesh history requires the canonical mesh"));
            }
            let decoded_mesh =
                SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
            if mesh.manifest.logical_identity
                != runmat_execution::Digest::from_bytes(*decoded_mesh.canonical_digest.bytes())
            {
                return Err(AnalysisBridgeError::Identity(
                    "volume-mesh history mesh identity differs",
                ));
            }
            inputs.push(mesh);
            (AdmittedObservationWorkload::VolumeMesh, Some(decoded_mesh))
        }
    };
    let context = match &decoded_workload {
        AdmittedObservationWorkload::NonlinearSolid(_) => {
            return Err(invalid("nonlinear history needs an original accepted-path source; only final fields are admitted"));
        }
        AdmittedObservationWorkload::VolumeMesh => ObservationHistorySourceContext::VolumeMesh {
            mesh: decoded_mesh
                .as_ref()
                .ok_or_else(|| invalid("volume-mesh history mesh absent"))?,
        },
        AdmittedObservationWorkload::Magnetostatic(_)
        | AdmittedObservationWorkload::MaxwellHarmonic(_) => {
            return Err(invalid(
                "electromagnetic workload histories finalize through the volume-mesh source",
            ));
        }
        AdmittedObservationWorkload::Thermal(workload) => {
            ObservationHistorySourceContext::Thermal {
                workload,
                mesh: decoded_mesh
                    .as_ref()
                    .ok_or_else(|| invalid("thermal history mesh absent"))?,
            }
        }
        AdmittedObservationWorkload::Linear(workload) => {
            ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::Linear(
                    workload,
                ),
                mesh: decoded_mesh.as_ref(),
            }
        }
        AdmittedObservationWorkload::Frequency(workload) => {
            ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::Frequency(
                    workload,
                ),
                mesh: decoded_mesh.as_ref(),
            }
        }
    };
    preparation
        .request
        .validate_inputs(&checkpoint, &catalog, context)?;
    let facts = prepared_input_resource_facts(&inputs)?;
    let history_bound = observation_history_finalization_payload_bytes_upper_bound(
        &preparation.request,
        &checkpoint,
        &facts,
    )
    .map_err(invalid)?;
    let series_bound = observation_history_series_reduction_payload_bytes_upper_bound(
        &preparation.request,
        &checkpoint,
        &facts,
    )
    .map_err(invalid)?;
    let resource_plan = build_observation_history_finalization_resource_plan(
        &preparation.request,
        &checkpoint,
        &catalog,
        context,
        runmat_analysis_fea::planning::resource::PreparedPhaseResourceFacts {
            inputs: facts,
            output: projected_output_resource_facts(history_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
        series_bound
            .map(projected_output_resource_facts)
            .transpose()?,
    )
    .map_err(invalid)?;
    let resource_plan_objects = prepare_analysis_objects(
        Kind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes().map_err(invalid)?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    inputs.push(resource_plan_objects.clone());
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if input.manifest.kind == Kind::ObservationHistoryCheckpoint {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect();
    let mut outputs = vec![Kind::History];
    if preparation.request.series_reduction.is_some() {
        outputs.push(Kind::SeriesReductionResult);
    }
    let workload = AnalysisWorkloadRequest::new(
        source_host.workload.study,
        preparation.request.producer,
        source_host.workload.execution_plan,
        AnalysisWorkKind::FinalizeObservationHistory,
        workload_inputs,
        outputs,
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        source_host.artifact_access.clone(),
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedObservationHistoryFinalizationTask {
        request: preparation.request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

pub fn import_observation_history_finalization_task(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedObservationHistoryFinalizationTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedObservationHistoryFinalization> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let expected_count = if prepared.request.series_reduction.is_some() {
        2
    } else {
        1
    };
    if output_set.manifest.outputs.len() != expected_count
        || output_set.manifest.outputs[0].kind != Kind::History
        || (expected_count == 2
            && output_set.manifest.outputs[1].kind != Kind::SeriesReductionResult)
    {
        return Err(invalid("history finalization output inventory differs"));
    }
    let checkpoint_objects = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == Kind::ObservationHistoryCheckpoint)
        .ok_or_else(|| invalid("history finalization checkpoint absent"))?;
    let checkpoint = decode_prepared_observation_history_checkpoint(checkpoint_objects)?;
    let checkpoint_manifest = checkpoint.manifest().clone();
    // Resume/finish validates retained metadata and sample identities, without reducing values.
    let expected = BoundedObservationHistory::resume(
        &prepared.request.plan,
        &prepared.request.observation_id,
        checkpoint,
    )
    .map_err(invalid)?
    .finish()
    .map_err(invalid)?;
    let output = &output_set.manifest.outputs[0];
    let history = import_observation_history(
        source,
        ObservationHistoryImport {
            root: &output.root,
            expected_identity: expected.manifest().identity,
            expected_payload_digest: output.payload_digest,
            partial_access: expected.manifest().partial_access,
            access: &prepared.host.artifact_access,
            limits,
        },
    )?;
    if output.logical_identity != expected.manifest().identity.digest()
        || &history.manifest != expected.manifest()
    {
        return Err(AnalysisBridgeError::Identity(
            "finalized history differs from terminal checkpoint",
        ));
    }
    let mut references = match &history.storage {
        ImportedObservationHistoryStorage::Partitioned(catalog) => {
            let identities = std::iter::once(history.manifest.identity.digest())
                .chain(
                    history
                        .manifest
                        .samples
                        .iter()
                        .map(|sample| sample.sample.digest()),
                )
                .collect::<Vec<_>>();
            if catalog
                .manifest
                .partitions
                .iter()
                .map(|part| part.logical_identity)
                .collect::<Vec<_>>()
                != identities
            {
                return Err(AnalysisBridgeError::Identity(
                    "history partitions differ from semantic catalog",
                ));
            }
            analysis_object_references(
                &catalog.manifest,
                &history.root,
                &history.access,
                ValueRefKind::ResultObject,
            )?
        }
        ImportedObservationHistoryStorage::Whole {
            objects,
            history: actual,
        } => {
            if actual.as_ref() != &expected {
                return Err(AnalysisBridgeError::Identity(
                    "whole history differs from checkpoint",
                ));
            }
            objects.unique_references(&history.access, ValueRefKind::ResultObject)?
        }
    };
    let (series_reduction, series_objects) = if expected_count == 2 {
        let output = &output_set.manifest.outputs[1];
        let objects = import_analysis_objects(
            source,
            &output.root,
            Kind::SeriesReductionResult,
            output.logical_identity,
            output.payload_digest,
            &prepared.host.artifact_access,
            limits,
        )?;
        let reduction =
            SeriesReductionResult::from_canonical_bytes(&objects.payload()).map_err(invalid)?;
        reduction
            .validate_source(&prepared.request.plan, &expected)
            .map_err(invalid)?;
        if reduction.identity.digest() != output.logical_identity {
            return Err(AnalysisBridgeError::Identity(
                "series reduction object identity differs",
            ));
        }
        references.extend(objects.unique_references(&history.access, ValueRefKind::ResultObject)?);
        (Some(reduction), Some(objects))
    } else {
        (None, None)
    };
    crate::object::retain_unique_references(&mut references);
    references.push(output_set.root);
    if references != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "history finalization output closure differs",
        ));
    }
    let publication = ObservationHistoryPublication {
        request: prepared.request.clone(),
        checkpoint: checkpoint_manifest,
        history: history.manifest.clone(),
        series: series_reduction
            .as_ref()
            .map(|series| {
                SeriesReductionPublication::new(series, &expected, &prepared.request.plan)
            })
            .transpose()?,
    };
    Ok(ImportedObservationHistoryFinalization {
        publication,
        history,
        series_reduction,
        series_objects,
    })
}
