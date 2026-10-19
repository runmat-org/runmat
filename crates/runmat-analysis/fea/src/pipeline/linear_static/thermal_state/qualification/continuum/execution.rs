//! The continuum fixture uses the existing prepared constraint/observation/resource authority.
use super::*;
use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    pipeline::linear_static::{
        run_linear_static_from_thermal_state_with_prepared_constraint_plan,
        PreparedThermoelasticStaticInput, ThermoelasticStaticResult,
    },
    planning::resource::{build_static_resource_plan, LocalExecutionPlan, StaticResourcePreflight},
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    StaticResourceAdmission,
};

pub(super) fn solve(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    source: &CanonicalSemanticState,
) -> ThermoelasticStaticResult {
    let consumer = &model.sequence.steps[1];
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(model, Some(mesh)).unwrap();
    let definitions = structural_static_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        consumer,
        model,
        Some(mesh),
        &constraints,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let options = options(mesh);
    let plan = build_static_resource_plan(StaticResourcePreflight {
        compilation: None,
        prepared_inputs: None,
        model,
        step: consumer,
        backend: ComputeBackend::Cpu,
        preconditioner: options.preconditioner_kind,
        observation_execution: ObservationExecutionPolicy::serial(),
        mesh: Some(mesh),
        constraints: &constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: LocalExecutionPlan::serial(),
    })
    .unwrap();
    let result = run_linear_static_from_thermal_state_with_prepared_constraint_plan(
        model,
        ComputeBackend::Cpu,
        options,
        PreparedThermoelasticStaticInput {
            consumer_step_id: &consumer.step_id,
            source,
            constraint_plan: &constraints,
            admitted: Some(StaticResourceAdmission {
                prepared_inputs: None,
                plan: &plan,
                observation_execution: ObservationExecutionPolicy::serial(),
            }),
        },
    )
    .unwrap();
    assert_eq!(result.thermal_transition.consumer_step_id, consumer.step_id);
    result
}
