//! Original static fields and compiler admission; no numerical solve or private codec.
use super::*;
use runmat_analysis_fea::{
    constraints::ConstraintCompilation,
    execution::{AnalysisTransportProfiler, StructuralLinearWorkload},
    planning::resource::{
        build_structural_field_recovery_resource_plan, structural_field_recovery_shape,
        FeaResourceDemand, FeaResourcePlan, PreparedPhaseResourceFacts,
        StructuralFieldRecoveryPlanning,
    },
};
use runmat_meshing_core::SolverMeshArtifact;
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
pub fn require_structural_recovery_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::StructuralFieldRecovery(demand) = &plan.input.demand else {
        return Err(invalid("static recovery resource family differs"));
    };
    let required = [
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        AnalysisNumericalObjectKind::SolverInvocation,
        AnalysisNumericalObjectKind::SolverResult,
        AnalysisNumericalObjectKind::ResourcePlan,
    ];
    let mesh_count = inputs
        .iter()
        .filter(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .count();
    if mesh_count > 1
        || inputs.len() != required.len() + mesh_count
        || required
            .iter()
            .any(|kind| inputs.iter().filter(|i| i.manifest.kind == *kind).count() != 1)
    {
        return Err(invalid("static recovery admitted input kinds differ"));
    }
    for input in inputs {
        runmat_execution_artifact::object::validate_inventory(&input.objects, limits)?;
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan
            && crate::decode_admitted_resource_plan(input, limits)? != *plan
        {
            return Err(invalid("static recovery carried admission differs"));
        }
        if input.manifest.kind == AnalysisNumericalObjectKind::FieldRecoveryRequest
            && input.manifest.logical_identity != demand.request.digest()
        {
            return Err(invalid("static recovery request identity differs"));
        }
        if input.manifest.kind == AnalysisNumericalObjectKind::SolverResult
            && input.manifest.logical_identity != demand.source_result.digest()
        {
            return Err(invalid("static recovery original result identity differs"));
        }
    }
    if crate::prepared_input_resource_facts_without_resource_plan(inputs)? != demand.inputs {
        return Err(invalid(
            "static recovery original source decode profile differs",
        ));
    }
    Ok(())
}
#[expect(
    clippy::too_many_arguments,
    reason = "FEA alpha execution-context structure is deferred under the approved 1.0 hardening scope"
)]
pub fn build_structural_recovery_metadata_resource_plan(
    request: &FieldRecoveryRequest,
    workload: &StructuralLinearWorkload,
    mesh: Option<&SolverMeshArtifact>,
    result: &runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    compilation: &ConstraintCompilation<'_>,
    inputs: &[PreparedAnalysisObjects],
    policy: crate::AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    build_from_refs(
        request,
        workload,
        mesh,
        result,
        compilation,
        inputs.iter(),
        policy,
        local_execution,
    )
}
#[expect(
    clippy::too_many_arguments,
    reason = "FEA alpha execution-context structure is deferred under the approved 1.0 hardening scope"
)]
pub(super) fn build_from_refs<'a>(
    request: &FieldRecoveryRequest,
    workload: &StructuralLinearWorkload,
    mesh: Option<&SolverMeshArtifact>,
    result: &runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    compilation: &ConstraintCompilation<'_>,
    inputs: impl Iterator<Item = &'a PreparedAnalysisObjects>,
    policy: crate::AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let (partitions, _) = structural_field_recovery_shape(request, result)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    Ok(build_structural_field_recovery_resource_plan(
        StructuralFieldRecoveryPlanning {
            request,
            workload,
            source_result: result,
            mesh,
            compilation,
            facts: PreparedPhaseResourceFacts {
                inputs: crate::object::prepared_input_resource_facts_from_refs(
                    inputs.filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan),
                )?,
                output: crate::projected_transport_output_resource_facts(&profile)?,
                output_set: crate::projected_output_set_resource_facts(),
            },
            local_execution,
        },
    )?)
}

/// Cold import reconstructs demand from original source objects, never retained scalar claims.
pub fn build_prepared_structural_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: crate::AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let unique = |kind| {
        let mut found = inputs.iter().filter(|i| i.manifest.kind == kind);
        let input = found
            .next()
            .ok_or_else(|| invalid("static recovery original input missing"))?;
        if found.next().is_some() {
            return Err(invalid("static recovery original input duplicated"));
        }
        Ok(input)
    };
    let plan = crate::decode_admitted_resource_plan(
        unique(AnalysisNumericalObjectKind::ResourcePlan)?,
        policy.inventory_limits(),
    )?;
    require_structural_recovery_decode_budget(inputs, &plan, policy.inventory_limits())?;
    if FieldRecoveryRequest::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?.payload(),
    )? != *request
    {
        return Err(invalid("static recovery original request differs"));
    }
    let workload = StructuralLinearWorkload::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::SolverInvocation)?.payload(),
    )?;
    let mesh = inputs
        .iter()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .map(|i| {
            use runmat_meshing_core::CanonicalMeshingContract;
            SolverMeshArtifact::canonical_decode(&i.payload()).map_err(invalid)
        })
        .transpose()?;
    let result =
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::from_canonical_bytes(
            &unique(AnalysisNumericalObjectKind::SolverResult)?.payload(),
        )?;
    result.validate_thermal_workload(&workload)?;
    result.validate_thermal_source(
        request.producer,
        request
            .source
            .solver_result()
            .ok_or_else(|| invalid("static source result missing"))?,
    )?;
    let available =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            workload
                .procedure_step()?
                .accuracy
                .resources
                .resident_byte_budget()
                .min(plan.request.memory_bytes),
            &crate::prepared_input_resource_facts(inputs)?,
            plan.input.worker.fixed_memory_bytes,
        )?;
    let compilation = runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &workload.model, mesh.as_ref(), available,
    ).map_err(invalid)?;
    result
        .constraint_result
        .validate_against_model(compilation.plan(), &workload.model, mesh.as_ref())
        .map_err(invalid)?;
    build_structural_recovery_metadata_resource_plan(
        request,
        &workload,
        mesh.as_ref(),
        &result,
        &compilation,
        inputs,
        policy,
        local_execution,
    )
}
