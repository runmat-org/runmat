//! Genuine finer continuum qualification; coarse failed bounds remain unchanged.
use super::*;
use crate::execution::{
    StructuralTransientBlockRequest, StructuralTransientNumericalReceipt,
    StructuralTransientResultAssemblyRequest, StructuralTransientSolverResult,
};
use crate::pipeline::transient::{
    prepare_transient_checkpoint_scope, recover_completed_transient_run, run_transient_block,
    CompletedStructuralTransientRunInput, StructuralTransientRunBlockInput,
};

fn governed_history(
    workload: &crate::execution::StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
) -> FeaTransientRunResult {
    // Scope preparation admits original source storage before assembling its operator.
    let scope = prepare_transient_checkpoint_scope(workload, mesh, None).unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let request = StructuralTransientBlockRequest::new(scope, None, terminal).unwrap();
    let plan = crate::planning::resource::build_structural_transient_block_resource_plan(
        workload,
        mesh,
        None,
        &request,
        None,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap_or_else(|error| panic!("finer continuum full-block admission rejected: {error}"));
    eprintln!(
        "finer continuum full-block admitted request={:?}",
        plan.request
    );
    // The runner repeats this same block admission before numerical assembly/history.
    let segment = run_transient_block(StructuralTransientRunBlockInput {
        workload,
        solver_mesh: mesh,
        request: &request,
        predecessor: None,
        modal_state: None,
        observation_cursors: Vec::new(),
    })
    .unwrap();
    let segments = [segment];
    let assembly = StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let result = StructuralTransientSolverResult::assemble(&assembly, &segments).unwrap();
    let receipt = StructuralTransientNumericalReceipt::new(&result, &assembly, &segments).unwrap();
    drop(segments);
    let recovery = crate::planning::resource::build_structural_transient_recovery_resource_plan(
        workload, mesh, None, &result, &receipt,
    )
    .unwrap_or_else(|error| {
        panic!("finer continuum full-history recovery admission rejected: {error}")
    });
    eprintln!(
        "finer continuum full-history recovery admitted request={:?}",
        recovery.request
    );
    recover_completed_transient_run(CompletedStructuralTransientRunInput {
        workload,
        solver_mesh: mesh,
        modal_state: None,
        result,
        numerical_receipt: &receipt,
    })
    .unwrap()
}

/// Keep every continuum metric mandatory. Collect every rung before testing the gates so
/// a failed acceleration bound does not conceal the remaining independent measurements.
fn qualify(quadratic: bool, damped: bool) {
    let cells = if quadratic { [4, 8, 16] } else { [8, 16, 32] };
    let errors = cells.map(|cells| {
        let mesh = mesh(quadratic, cells);
        eprintln!(
            "finer continuum start quadratic={quadratic} damped={damped} cells={cells} nodes={} elements={} intervals=400",
            mesh.topology.nodes.len(), mesh.topology.volume_elements.len()
        );
        let started = std::time::Instant::now();
        // This is source/scope admission only (NoSolve, no recovered fields). Full block
        // and recovery admission below are separate and retain the original resource cap.
        // This borrowed preassembly path derives canonical source/constraint/sparse counts,
        // never assembling an operator or allocating/factorizing a dense numerical matrix.
        let model = model(&mesh, damped);
        let workload = crate::execution::StructuralTransientWorkload {
            schema_version: 2,
        initial_fields: Vec::new(),
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            solver: crate::execution::StructuralTransientSolverPlan::Direct,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 400 },
            residual_target: 1.0e-10,
        };
        let prepared = crate::pipeline::transient::prepare_transient_preassembly(
            &workload.model,
            (&workload).into(),
        ).unwrap();
        let admission = crate::planning::resource::build_structural_transient_scope_resource_plan(
            &workload, &mesh, None, &prepared, None, None,
crate::planning::resource::LocalExecutionPlan::serial(),
);
        let admitted = admission.unwrap_or_else(|error| panic!(
            "finer continuum admission rejected quadratic={quadratic} damped={damped} cells={cells} original_ceiling={} error={error}",
            workload.model.sequence.steps[0].accuracy.resources.resident_byte_budget()
        ));
        eprintln!("finer continuum source/scope admitted (not full history) quadratic={quadratic} damped={damped} cells={cells} request={:?} original_ceiling={}",
            admitted.request, workload.model.sequence.steps[0].accuracy.resources.resident_byte_budget());
        let result = governed_history(&workload, &mesh);
        assert_eq!(result.time_points_s.len(), 401);
        assert_eq!(result.time_points_s[0], 0.0);
        assert_eq!(*result.time_points_s.last().unwrap(), END_TIME);
        let errors = refinement::errors(&mesh, damped, &result);
        eprintln!(
            "finer continuum measured quadratic={quadratic} damped={damped} cells={cells} U/V/A/energy={errors:?} elapsed_s={}",
            started.elapsed().as_secs_f64()
        );
        errors
    });
    // These are precisely the pre-existing coarse-envelope gates, not fitted fine-mesh bounds.
    let factor = if quadratic { 3.0 } else { 1.6 };
    let bound = if quadratic { 0.005 } else { 0.05 };
    for quantity in 0..4 {
        assert!(
            errors[1][quantity] < errors[0][quantity] / factor,
            "quadratic={quadratic} damped={damped} quantity={quantity} cells={cells:?} errors={errors:?} factor={factor}"
        );
        assert!(
            errors[2][quantity] < errors[1][quantity] / factor,
            "quadratic={quadratic} damped={damped} quantity={quantity} cells={cells:?} errors={errors:?} factor={factor}"
        );
        assert!(
            errors[2][quantity] < bound,
            "quadratic={quadratic} damped={damped} quantity={quantity} cells={cells:?} errors={errors:?} bound={bound}"
        );
    }
}

#[test]
fn finer_tet4_undamped_original_sine_continuum() {
    qualify(false, false);
}

#[test]
fn finer_tet4_damped_original_sine_continuum() {
    qualify(false, true);
}

#[test]
fn finer_tet10_undamped_original_sine_continuum() {
    qualify(true, false);
}

#[test]
fn finer_tet10_damped_original_sine_continuum() {
    qualify(true, true);
}
