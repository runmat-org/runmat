//! Remote composition of a non-structural family after exact meshing.
//!
//! The family's governed task graph runs on pool workers through the same coordinators and
//! completion roots as local execution; only placement and object custody differ.

use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_execution::task::RetryPolicy;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::CompletedRemoteExactMeshing;
use super::exact_structural_study::RemoteExactStructuralStudyComposition;
use super::input_materialization::RemoteInputMaterializationOutcome;
use super::RemotePoolDriver;
use crate::analysis_task_pool::RemoteAnalysisTaskPool;
use crate::NativeExecutionResult;

pub(super) struct FamilyComposition<'a> {
    pub pool: Arc<RemotePoolDriver>,
    pub request: &'a ProgramExecutionRequest,
    pub decoded: crate::DecodedStructuralStudyRoot,
    pub meshing_request: runmat_meshing_core::MeshingRequest,
    pub completed: Box<CompletedRemoteExactMeshing>,
    pub regions: runmat_geometry_ops::ResolvedGeometryRegions,
    pub input_outcomes: BTreeMap<runmat_execution::Digest, RemoteInputMaterializationOutcome>,
    pub resources: &'a AllocationResources,
}

pub(super) async fn execute(
    input: FamilyComposition<'_>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let FamilyComposition {
        pool,
        request,
        decoded,
        meshing_request,
        completed,
        regions,
        input_outcomes,
        resources,
    } = input;
    let Some(geometry_output) = completed.materialization.clone() else {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "typed family geometry has no materialization receipt".into(),
        ));
    };
    let plans = match decoded
        .input_materializations
        .iter()
        .map(|plan| plan.resolve(&completed.result.artifact))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(plans) => plans,
        Err(error) => {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                error.to_string(),
            ))
        }
    };
    // The remote worker's admitted task memory is the budget its plan selects within.
    let task_resources = match super::exact_structural_study::admit_analysis_task_resources(
        resources,
        &decoded.analysis_accelerator_requests,
        decoded.local_execution,
        decoded.memory_per_worker_bytes,
    ) {
        Ok(resources) => resources,
        Err(error) => return Ok(RemoteExactStructuralStudyComposition::Failed(error)),
    };
    let mut terminal = None;
    let mut completed_inputs = Vec::new();
    let planned = crate::study_family::plan_family_study(
        crate::study_family::FamilyStudyContext {
            study: &decoded.study,
            materials: &decoded.materials,
            planning: &decoded.planning,
            document: &completed.document,
            regions: &regions,
            mesh: &completed.result.artifact,
            initial_fields: &decoded.initial_fields,
            frozen_project_digest: decoded.frozen_project_digest,
            artifact_access: &decoded.artifact_access,
            program_revision: &request.recipe.program_revision,
            local_execution: decoded.local_execution,
            worker_memory: &|| Ok(task_resources.memory_bytes),
        },
        &mut |model| match super::input_materialization::apply_to_model(
            model,
            &plans,
            &input_outcomes,
        ) {
            Ok(inputs) => {
                completed_inputs = inputs;
                Ok(())
            }
            Err(outcome) => {
                terminal = Some(outcome);
                Err("generated input materialization did not complete".into())
            }
        },
    );
    if let Some(outcome) = terminal {
        return Ok(super::exact_structural_study::signal_composition(outcome));
    }
    let planned = match planned {
        Ok(planned) => planned,
        Err(error) => return Ok(RemoteExactStructuralStudyComposition::Failed(error)),
    };
    let policy = crate::study_family::FamilyStudyExecutionPolicy {
        resources: task_resources,
        retry: RetryPolicy::ExplicitlyIdempotent {
            max_attempts: runmat_analysis_execution::DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
        },
        deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(resources)?),
        priority: 0,
    };
    let mut task_pool = RemoteAnalysisTaskPool::new(Arc::clone(&pool));
    let family =
        match crate::study_family::execute_family_study(&mut task_pool, planned, policy).await {
            Ok(family) => family,
            Err(error) => {
                return Ok(super::exact_structural_study::sequence_error_composition(
                    &pool,
                    error.to_string(),
                ))
            }
        };
    let (response, objects) = crate::family_study_publication::package_family_study_result(
        request,
        &decoded,
        &geometry_output,
        &meshing_request,
        &completed.result,
        &completed_inputs,
        &family,
    )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}
