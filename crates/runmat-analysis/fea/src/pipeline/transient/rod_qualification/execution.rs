//! Cold canonical objects and recovered histories; not process failure or worker replacement.
use super::*;
use crate::execution::{
    StructuralTransientBlockRequest, StructuralTransientCheckpoint,
    StructuralTransientNumericalReceipt, StructuralTransientPhysicalSample,
    StructuralTransientResultAssemblyRequest, StructuralTransientSegment,
    StructuralTransientSegmentManifest, StructuralTransientSegmentSample,
    StructuralTransientSolverPlan, StructuralTransientSolverResult,
    StructuralTransientSolverResultManifest, StructuralTransientWorkload,
};
use crate::pipeline::transient::{
    prepare_transient_checkpoint_scope, recover_completed_transient_run, run_transient_block,
    CompletedStructuralTransientRunInput, StructuralTransientRunBlockInput,
};
use runmat_meshing_core::CanonicalMeshingContract;

#[test]
fn transient_rod_cold_blocks_and_recovered_history_retain_independent_continuum_accuracy() {
    for quadratic in [false, true] {
        // The continuum bound is a spatial accuracy claim: eight quadratic cells keep the nodal
        // acceleration floor (0.0014) below it; four cells sit at 0.0061 for any time step.
        let mesh = mesh(quadratic, 8);
        let model = model(&mesh, true);
        let workload = StructuralTransientWorkload {
            schema_version: 2,
            initial_fields: Vec::new(),
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            solver: StructuralTransientSolverPlan::Direct,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 80 },
            residual_target: 1.0e-10,
        };
        let workload =
            StructuralTransientWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        let mesh = SolverMeshArtifact::canonical_decode(&mesh.canonical_encode().unwrap()).unwrap();
        let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
        let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
        let mut predecessor = None;
        let mut segments = Vec::new();
        for end in [terminal / 2, terminal] {
            let request =
                StructuralTransientBlockRequest::new(scope.clone(), predecessor.as_ref(), end)
                    .unwrap();
            let request = StructuralTransientBlockRequest::from_canonical_bytes(
                &request.canonical_bytes().unwrap(),
            )
            .unwrap();
            let segment = run_transient_block(StructuralTransientRunBlockInput {
                workload: &workload,
                solver_mesh: &mesh,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state: None,
                observation_cursors: Vec::new(),
            })
            .unwrap();
            predecessor = Some(
                StructuralTransientCheckpoint::from_canonical_bytes(
                    &segment
                        .manifest()
                        .terminal_checkpoint
                        .canonical_bytes()
                        .unwrap(),
                )
                .unwrap(),
            );
            let manifest = StructuralTransientSegmentManifest::from_canonical_bytes(
                &segment.manifest().canonical_bytes().unwrap(),
            )
            .unwrap();
            let initial = segment.initial_sample().map(|sample| {
                StructuralTransientPhysicalSample::from_canonical_bytes(
                    &sample.canonical_bytes().unwrap(),
                )
                .unwrap()
            });
            let samples = segment
                .samples()
                .iter()
                .map(|sample| {
                    StructuralTransientSegmentSample::from_canonical_bytes(
                        &sample.canonical_bytes().unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            segments.push(
                StructuralTransientSegment::from_manifest_and_samples(manifest, initial, samples)
                    .unwrap(),
            );
        }
        let request = StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
        let result = StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
        let receipt =
            StructuralTransientNumericalReceipt::new(&result, &request, &segments).unwrap();
        let manifest = StructuralTransientSolverResultManifest::from_canonical_bytes(
            &result.manifest().canonical_bytes().unwrap(),
        )
        .unwrap();
        let samples = result
            .samples()
            .iter()
            .map(|sample| {
                StructuralTransientPhysicalSample::from_canonical_bytes(
                    &sample.canonical_bytes().unwrap(),
                )
                .unwrap()
            })
            .collect();
        let result =
            StructuralTransientSolverResult::from_manifest_and_samples(manifest, samples).unwrap();
        let receipt = StructuralTransientNumericalReceipt::from_canonical_bytes(
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap();
        let plan = crate::planning::resource::build_structural_transient_recovery_resource_plan(
            &workload, &mesh, None, &result, &receipt,
        )
        .unwrap();
        assert!(plan.request.memory_bytes > 0);
        let recovered = recover_completed_transient_run(CompletedStructuralTransientRunInput {
            workload: &workload,
            solver_mesh: &mesh,
            modal_state: None,
            result: result.clone(),
            numerical_receipt: &receipt,
        })
        .unwrap();
        let error = refinement::errors(&mesh, true, &recovered);
        let bound = if quadratic { 0.005 } else { 0.05 };
        assert!(
            error.iter().all(|error| *error < bound),
            "quadratic={quadratic} errors={error:?}"
        );
        let uninterrupted = run(&mesh, true, 80);
        assert_eq!(recovered.time_points_s, uninterrupted.time_points_s);
        assert_eq!(
            recovered.displacement_snapshots,
            uninterrupted.displacement_snapshots
        );
        assert_eq!(
            recovered.velocity_snapshots,
            uninterrupted.velocity_snapshots
        );
        assert_eq!(
            recovered.acceleration_snapshots,
            uninterrupted.acceleration_snapshots
        );
        let mut foreign = workload.clone();
        let InitialCondition::Displacement(condition) = &mut foreign.model.initial_conditions[1]
        else {
            unreachable!()
        };
        condition.value = InitialValue::UniformVector3([
            Displacement::new(0.002),
            Displacement::new(0.0),
            Displacement::new(0.0),
        ]);
        assert!(
            crate::planning::resource::build_structural_transient_recovery_resource_plan(
                &foreign, &mesh, None, &result, &receipt
            )
            .is_err()
        );
        assert!(
            crate::planning::resource::build_structural_transient_recovery_resource_plan(
                &workload,
                &super::mesh(quadratic, 3),
                None,
                &result,
                &receipt
            )
            .is_err()
        );
    }
}
