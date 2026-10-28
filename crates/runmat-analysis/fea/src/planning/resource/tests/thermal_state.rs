use super::*;
use crate::planning::resource::preflight::{build_static_resource_plan, StaticResourcePreflight};
use runmat_analysis_core::{
    InitialStateConflictPolicy, StateInput, StateKind, StateSource, StateUse,
};

#[test]
fn solved_thermal_inventory_changes_preflight_identity_memory_and_publication() {
    let (model, mesh, _, constraints) = structural_fixture();
    let fields = crate::post::fields::structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .unwrap();
    let build = |step: &runmat_analysis_core::ProcedureStep| {
        // The preflight profiles the actual authored producer declaration, not a detached
        // step that differs from the model's sequence.
        let mut authored_model = model.clone();
        authored_model.sequence.steps[0] = step.clone();
        build_static_resource_plan(StaticResourcePreflight {
            compilation: None,
            prepared_inputs: None,
            model: &authored_model,
            step,
            backend: crate::ComputeBackend::Cpu,
            preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
            observation_execution: crate::results::observation::ObservationExecutionPolicy::serial(
            ),
            mesh: Some(&mesh),
            constraints: &constraints,
            observations: &observations,
            accelerator_requests: &[],
            local_execution: LocalExecutionPlan::serial(),
        })
    };
    let mut step = model.sequence.steps[0].clone();
    let baseline = build(&step).unwrap();
    // Resource inventory follows the typed input declaration; physical source/transition
    // validation remains the executor's responsibility, not an estimator-side solve.
    step.state_inputs.push(StateInput {
        input_id: "temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: step.step_id.clone(),
            state_id: runmat_analysis_core::StateId("thermal_state".into()),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let thermal = build(&step).unwrap();
    let baseline_operator = baseline.input.structural_operator().unwrap();
    let thermal_operator = thermal.input.structural_operator().unwrap();
    // One Tet4: four source nodes and fourteen retained scalar values + local-vector allowance.
    let source_and_recovery_bytes = 4 * 112 + 14 * 8 + 32 + (12 + 1) * 8;
    let new_output_bytes = (4 + 24 + 24 + 1 + 12 + 1) * 8;
    assert_eq!(
        thermal.input.worker.fixed_memory_bytes.0,
        baseline.input.worker.fixed_memory_bytes.0 + source_and_recovery_bytes
    );
    assert_eq!(
        thermal_operator.phase_resources.retained_bytes.0,
        baseline_operator.phase_resources.retained_bytes.0 + new_output_bytes
    );
    assert_eq!(
        thermal_operator.publication.result_artifact_bytes.0,
        baseline_operator.publication.result_artifact_bytes.0 + new_output_bytes
    );
    assert!(
        thermal.request.memory_bytes >= baseline.request.memory_bytes + source_and_recovery_bytes
    );
    assert_ne!(thermal.identity, baseline.identity);
    step.accuracy.resources.maximum_resident_bytes = Some(baseline.request.memory_bytes);
    assert!(matches!(
        build(&step),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
}
