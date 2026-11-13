use super::*;

#[path = "tests/curved.rs"]
mod curved;
use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::StructuralLinearWorkload,
    planning::resource::{
        build_structural_resource_plan, ResourcePlanningError, StructuralSymbolicOperators,
    },
    post::fields::structural_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
    topology::AnalysisProjection,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};

#[test]
fn connected_preflight_tracks_expanded_proof_and_upper_pattern_fallback() {
    for (levels, expected_dofs, expected_nonzeros) in [(4, 1_035, 27_765), (5, 4_107, 110_709)] {
        let mesh = refine(connected_tetrahedron_solver_mesh(), levels).unwrap();
        let model = patch_model(&mesh);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let symbolic =
            StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh).unwrap();
        assert_eq!(symbolic.stiffness_upper_bound().rows(), expected_dofs);
        assert_eq!(
            symbolic.stiffness_upper_bound().column_indices().len(),
            expected_nonzeros
        );
        let field_definitions = structural_observable_field_definitions().unwrap();
        let observations = resolve_structural_observation_plan(
            &model.sequence.steps[0],
            &model,
            Some(&mesh),
            &constraints,
            &field_definitions,
            ObservationLimits::default(),
        )
        .unwrap();
        let ceiling = model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        let workload = StructuralLinearWorkload {
            schema_version: 2,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution: ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            state_input: None,
        };
        let plan = build_structural_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
            None,
        )
        .unwrap();
        assert!(plan.request.memory_bytes <= ceiling);
        assert!(
            plan.request.memory_bytes > 32 * 1024 * 1024,
            "levels {levels}: {} bytes",
            plan.request.memory_bytes
        );

        if levels == 4 {
            let mut capped = workload;
            capped.model.sequence.steps[0]
                .accuracy
                .resources
                .maximum_resident_bytes = Some(32 * 1024 * 1024);
            let error = build_structural_resource_plan(
                Some(&mesh),
                &constraints,
                &observations,
                &capped,
                &[],
                LocalExecutionPlan::serial(),
                None,
            )
            .expect_err("a ceiling below the resource reservation must reject early");
            assert!(matches!(
                error,
                ResourcePlanningError::AccuracyCeiling { .. }
            ));
        }
    }
}

#[test]
fn refined_connected_tet10_preflight_reserves_and_rejects_a_lower_ceiling() {
    let mesh =
        elevate_connected_tet4_fixture(refine(connected_tetrahedron_solver_mesh(), 4).unwrap());
    mesh.validate_canonical().unwrap();
    let model = patch_model(&mesh);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let fields = structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .unwrap();
    let mut workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    assert!(plan.request.memory_bytes > 1);
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
    assert!(matches!(
        build_structural_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
            None,
        ),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
}

#[test]
fn connected_patch_serial_and_four_worker_fields_match_the_affine_solution() {
    let mesh = refine(connected_tetrahedron_solver_mesh(), 4).unwrap();
    let model = patch_model(&mesh);
    let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
    let scoped = solve_patch(
        &model,
        &mesh,
        LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
    )
    .unwrap();
    assert_static_field_parity(&serial, &scoped);
    assert_affine_displacement(&scoped, &mesh, 1.0e-7);
    assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
}

#[test]
fn connected_patch_4096_elements_solves_with_parallel_parity_and_affine_accuracy() {
    let mesh = refine(connected_tetrahedron_solver_mesh(), 5).unwrap();
    let model = patch_model(&mesh);
    let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
    let scoped = solve_patch(
        &model,
        &mesh,
        LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
    )
    .unwrap();
    assert_static_field_parity(&serial, &scoped);
    assert_affine_displacement(&scoped, &mesh, 2.0e-7);
    assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
}

#[test]
fn refined_connected_tet10_patch_solves_with_parallel_parity_and_affine_accuracy() {
    let mesh =
        elevate_connected_tet4_fixture(refine(connected_tetrahedron_solver_mesh(), 4).unwrap());
    mesh.validate_canonical().unwrap();
    assert_eq!(mesh.topology.volume_elements.len(), 1024);
    let model = patch_model(&mesh);
    let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
    let scoped = solve_patch(
        &model,
        &mesh,
        LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
    )
    .unwrap();
    assert_static_field_parity(&serial, &scoped);
    assert_affine_displacement(&scoped, &mesh, 1.0e-6);
    assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
}

fn assert_static_field_parity(serial: &FeaRunResult, scoped: &FeaRunResult) {
    assert_eq!(serial.fields, scoped.fields, "published field parity");
    assert_eq!(serial.observations, scoped.observations);
    assert_eq!(
        serial.structural_static_balance,
        scoped.structural_static_balance
    );
    assert_eq!(
        serial.structural_constraint_result,
        scoped.structural_constraint_result
    );
    assert_eq!(
        serial.structural_connector_result,
        scoped.structural_connector_result
    );
}

fn assert_affine_displacement(result: &FeaRunResult, mesh: &SolverMeshArtifact, tolerance: f64) {
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    for (actual, node) in displacement.chunks_exact(3).zip(&mesh.topology.nodes) {
        for axis in 0..3 {
            let expected = [0.1, 0.2, 0.3][axis] * node.coordinates_m.canonical_value()[axis];
            assert!(
                (actual[axis] - expected).abs() <= tolerance,
                "node {} axis {axis}: {} != {expected}",
                node.node_id,
                actual[axis]
            );
        }
    }
}

fn assert_affine_element_fields(
    result: &FeaRunResult,
    mesh: &SolverMeshArtifact,
    strain_tolerance: f64,
    stress_tolerance: f64,
) {
    let strain = result
        .field(FEA_FIELD_STRUCTURAL_STRAIN)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let stress = result
        .field(FEA_FIELD_STRUCTURAL_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let energy = result
        .field(FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(strain.len(), mesh.topology.volume_elements.len() * 6);
    assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 6);
    assert_eq!(energy.len(), mesh.topology.volume_elements.len());
    for (element, (actual_strain, actual_stress)) in strain
        .chunks_exact(6)
        .zip(stress.chunks_exact(6))
        .enumerate()
    {
        for (component, (&actual, expected)) in actual_strain
            .iter()
            .zip([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= strain_tolerance,
                "element {element} strain component {component}: {actual} != {expected}"
            );
        }
        for (component, (&actual, expected)) in actual_stress
            .iter()
            .zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0])
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= stress_tolerance,
                "element {element} stress component {component}: {actual} != {expected}"
            );
        }
        assert!((energy[element] - 12.8).abs() <= stress_tolerance);
    }
    let total = result
        .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(total.len(), 1);
    assert!((total[0] - 12.8 / 6.0).abs() <= stress_tolerance);
}
