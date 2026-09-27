//! Thermal-prefix execution after exact geometry and authored input resolution.
use super::*;

pub(super) struct Composition<'a> {
    pub pool: Arc<RemotePoolDriver>,
    pub request: &'a ProgramExecutionRequest,
    pub decoded: &'a crate::DecodedExactStructuralStudyRoot,
    pub meshing: &'a runmat_meshing_execution::ExactMeshingDagRunResult,
    pub region_resolution: &'a runmat_analysis_execution::PreparedAnalysisObjects,
    pub inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
    pub workloads: StructuralStudyWorkloads,
    pub unresolved_workloads_digest: Digest,
    pub policy: crate::StructuralSequenceExecutionPolicy,
}

pub(super) async fn execute(
    input: Composition<'_>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let mut compiled = input.decoded.compiled_study.clone().ok_or_else(|| {
        protocol("thermal prestress execution requires the original typed compilation product")
    })?;
    // Input materialization changes execution references, not authored intent. Keep the
    // original compilation in the outer root and repeat this resolution during cold import.
    compiled.model = input.workloads.authored_model().map_err(protocol)?.clone();
    compiled
        .validate_against(&input.meshing.artifact)
        .map_err(protocol)?;
    let completion = match crate::execute_thermal_prestress_study_remotely(
        Arc::clone(&input.pool),
        crate::ThermalPrestressStudyPreparation {
            structural: runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                authored_study_digest: compiled.definition_digest().map_err(protocol)?,
                frozen_project_digest: input.decoded.frozen_project_digest,
                workloads: input.workloads,
                mesh: input.meshing.artifact.clone(),
                artifact_access: input.decoded.artifact_access.clone(),
                program_revision: input.request.recipe.program_revision.clone(),
                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                inventory_limits: ObjectInventoryLimits::default(),
                required_capabilities: Default::default(),
                accelerator_requests: input.decoded.analysis_accelerator_requests.clone(),
                local_execution: input.decoded.local_execution,
            },
            compiled,
        },
        input.policy,
    )
    .await
    {
        Ok(completed) => completed,
        Err(error) => return Ok(sequence_error_composition(&input.pool, error.to_string())),
    };
    let (response, objects) = crate::exact_structural_study_root::package_exact_structural_study_result(
        input.request,
        input.decoded,
        input.meshing,
        input.region_resolution,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::ThermalPrestress {
            inputs: input.inputs,
            completion: &completion,
            unresolved_workloads_digest: input.unresolved_workloads_digest,
        },
    )?;
    for object in objects {
        input
            .pool
            .register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}
