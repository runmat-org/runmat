//! Retain original physics-owned field definitions in the same closed run publication.
use super::*;
use runmat_analysis_fea::{
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    },
    results::field::{RecoveredFieldArtifactManifest, RecoveredFieldEntityCounts},
};

pub(super) fn thermal(
    prefix: &ThermalPrefixCompletion,
    context: &OperationContext,
) -> Result<(String, RecoveredFieldArtifactManifest), OperationErrorEnvelope> {
    let CompletedStudyStepRun::Standard(run) = &prefix.completed.run else {
        return Err(completion_error(
            context,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "original thermal fields require a standard thermal run",
        ));
    };
    let manifest = &prefix.thermal.manifest;
    let build = || -> Result<RecoveredFieldArtifactManifest, String> {
        manifest.validate().map_err(|e| e.to_string())?;
        let source = FieldRecoverySource::SolverResult(manifest.identity());
        let request = FieldRecoveryRequest::new(
            manifest.producer,
            source,
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: runmat_meshing_core::StableDigest::from_bytes(
                    *manifest.topology_digest.bytes(),
                ),
            },
            FieldRecoveryPlan::default(),
        )
        .map_err(|e| e.to_string())?;
        RecoveredFieldArtifactManifest::from_entries(
            request.identity(),
            source,
            manifest.topology_digest,
            RecoveredFieldEntityCounts {
                nodes: manifest.ordered_node_ids.len(),
                volume_elements: manifest.ordered_element_ids.len(),
                oriented_faces: 0,
                volume_integration_points: 0,
                oriented_edges: 0,
                beam_elements: 0,
                shell_elements: 0,
                connectors: 0,
                constraint_groups: 0,
            },
            manifest.fields.iter().flatten().cloned().collect(),
        )
        .map_err(|e| e.to_string())
    };
    build()
        .map(|catalog| (run.data.run_id.clone(), catalog))
        .map_err(|error| {
            completion_error(context, "RM.FEA.RUN_STUDY.ORIGINAL_FIELDS_INVALID", &error)
        })
}

pub(super) fn take(
    run: CompletedStudyStepRun,
) -> (
    CompletedStudyStepRun,
    Option<Box<runmat_analysis_fea::results::field::OriginalFieldSourcePlan>>,
) {
    match run {
        CompletedStudyStepRun::OriginalFields { run, source } => {
            (CompletedStudyStepRun::Standard(run), Some(source))
        }
        run => (run, None),
    }
}

pub(super) fn validate_source(
    context: &OperationContext,
    sequence: SequenceIdentity,
    step: &runmat_analysis_core::ProcedureStep,
    result: SolverResultIdentity,
    source: &runmat_analysis_fea::results::field::OriginalFieldSourcePlan,
) -> Result<(), OperationErrorEnvelope> {
    let validate = || -> Result<(), String> {
        source.validate().map_err(|error| error.to_string())?;
        let producer = StepIdentity::from_solver_contract(sequence, step)
            .map_err(|error| error.to_string())?;
        if source.request.producer != producer
            || source.request.source != FieldRecoverySource::SolverResult(result)
        {
            return Err("original field source differs from its completed producer".into());
        }
        Ok(())
    };
    validate().map_err(|error| {
        completion_error(context, "RM.FEA.RUN_STUDY.ORIGINAL_FIELDS_INVALID", error)
    })
}

pub(super) fn stage(
    context: &OperationContext,
    staging: &storage::AnalysisPublicationBuilder,
    run_id: &str,
    thermal: &Option<(String, RecoveredFieldArtifactManifest)>,
    sources: &BTreeMap<String, Box<runmat_analysis_fea::results::field::OriginalFieldSourcePlan>>,
) -> Result<(), OperationErrorEnvelope> {
    let stage = || {
        if let Some((id, catalog)) = thermal {
            if id == run_id {
                staging.stage_original_fields(id, catalog)?;
            }
        }
        if let Some(source) = sources.get(run_id) {
            staging.stage_original_field_source(run_id, source)?;
        }
        Ok::<_, storage::AnalysisPublicationError>(())
    };
    stage().map_err(|error| {
        completion_storage_error(context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
    })
}

pub(in crate::analysis) fn standard(
    run: OperationEnvelope<AnalysisRunResult>,
    source: Option<runmat_analysis_fea::results::field::OriginalFieldSourcePlan>,
) -> CompletedStudyStepRun {
    match source {
        Some(source) => CompletedStudyStepRun::OriginalFields {
            run: Box::new(run),
            source: Box::new(source),
        },
        None => CompletedStudyStepRun::Standard(Box::new(run)),
    }
}
