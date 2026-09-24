use super::*;
use crate::fixtures::{fixture_model, FixtureId};

fn direct_scope_case() -> (
    crate::execution::StructuralTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let workload = crate::execution::StructuralTransientWorkload {
        schema_version: 1,
        model,
        step_id: runmat_analysis_core::ProcedureStepId("transient".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        solver: crate::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 2,
        },
        residual_target: 1.0e-8,
    };
    (workload, mesh)
}

#[test]
fn scope_preparation_rejects_accepted_schedule_over_sample_ceiling_before_assembly() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 2,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_block_basis(&model, &options) {
        Ok(_) => panic!("over-ceiling transient schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires at least 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn motion_breakpoint_samples_count_against_transient_ceiling() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4SupportMotionReference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 1,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_block_basis(&model, &options) {
        Ok(_) => panic!("over-ceiling motion schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn direct_host_transient_checks_schedule_before_initial_assembly() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 2,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match run_transient_with_options(&model, ComputeBackend::Cpu, options) {
        Ok(_) => panic!("over-ceiling direct transient schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires at least 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn direct_host_samples_follow_the_shared_preassembly_schedule() {
    let model = fixture_model(FixtureId::TransientLong);
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 4,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let prepared = prepare_transient_preassembly(&model, &options).unwrap();
    let expected = prepared
        .schedule
        .points
        .iter()
        .map(|point| point.time_s)
        .collect::<Vec<_>>();
    let result = run_transient_with_options(&model, ComputeBackend::Cpu, options).unwrap();
    assert_eq!(result.time_points_s, expected);
}

#[test]
fn checkpoint_scope_resource_preflight_is_deterministic_and_precedes_assembly() {
    let (workload, mesh) = direct_scope_case();
    let prepared =
        prepare_transient_preassembly(&workload.model, &workload.solve_options(mesh.clone()))
            .unwrap();
    let first = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &workload, &mesh, None, &prepared,
    )
    .unwrap();
    let second = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &workload, &mesh, None, &prepared,
    )
    .unwrap();
    assert_eq!(first.identity, second.identity);
    assert_eq!(first.request.cpu_millicores, 1_000);
    assert!(first.request.memory_bytes > 0);
    assert_eq!(first.breakdown.observation_produced_sample_count.0, 0);
    prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();

    let mut rejected = workload;
    rejected.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1;
    let error = prepare_transient_checkpoint_scope(&rejected, &mesh, None).unwrap_err();
    assert!(
        matches!(error, FeaRunError::InvalidModel(message) if message.contains("resident memory bytes"))
    );
}
