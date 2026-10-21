//! Prepared execution selects one consumer without dropping the source sequence or admission.

use super::*;
use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    planning::resource::{build_static_resource_plan, LocalExecutionPlan, StaticResourcePreflight},
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    StaticResourceAdmission,
};

#[test]
fn prepared_thermal_solve_selects_each_static_consumer_and_preserves_admission() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mesh) = model_and_mesh(order, true);
        let mut second = ProcedureStep::coupled(
            "second_expansion",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        );
        second.state_inputs = model.sequence.steps[1].state_inputs.clone();
        model.sequence.dependencies.push(SequenceDependency {
            predecessor: model.sequence.steps[0].step_id.clone(),
            successor: second.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        });
        let mut transition = model.sequence.transitions[0].clone();
        transition.transition_id = TransitionId("second_temperature_reuse".into());
        transition.consumer_step = second.step_id.clone();
        model.sequence.transitions.push(transition);
        model.sequence.steps.push(second);
        let source = solve_temperature(&model, &mesh);
        let source_bytes = source.canonical_bytes().unwrap();
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..LinearStaticSolveOptions::default()
        };
        // The convenience API intentionally rejects ambiguity; the prepared path names its step.
        assert!(run_linear_static_from_thermal_state(
            &model,
            ComputeBackend::Cpu,
            options.clone(),
            &source
        )
        .is_err());
        let mut results = Vec::new();
        for consumer in &model.sequence.steps[1..] {
            let fields = structural_static_observable_field_definitions().unwrap();
            let observations = resolve_structural_observation_plan(
                consumer,
                &model,
                Some(&mesh),
                &constraints,
                &fields,
                ObservationLimits::default(),
            )
            .unwrap();
            let plan = build_static_resource_plan(StaticResourcePreflight {
                compilation: None,
                prepared_inputs: None,
                model: &model,
                step: consumer,
                backend: ComputeBackend::Cpu,
                preconditioner: options.preconditioner_kind,
                observation_execution: ObservationExecutionPolicy::serial(),
                mesh: Some(&mesh),
                constraints: &constraints,
                observations: &observations,
                accelerator_requests: &[],
                local_execution: LocalExecutionPlan::serial(),
            })
            .unwrap();
            let execute = |constraint_plan, admitted| {
                run_linear_static_from_thermal_state_with_prepared_constraint_plan(
                    &model,
                    ComputeBackend::Cpu,
                    options.clone(),
                    PreparedThermoelasticStaticInput {
                        consumer_step_id: &consumer.step_id,
                        source: &source,
                        constraint_plan,
                        admitted,
                    },
                )
            };
            let result = execute(
                &constraints,
                Some(StaticResourceAdmission {
                    prepared_inputs: None,
                    plan: &plan,
                    observation_execution: ObservationExecutionPolicy::serial(),
                }),
            )
            .unwrap();
            assert_eq!(result.thermal_transition.consumer_step_id, consumer.step_id);
            assert_eq!(result.thermal_transition.source_state, source.identity);
            let expected_consumer = crate::execution::StepIdentity::from_solver_contract(
                crate::execution::SequenceIdentity::from_solver_contract(&model.sequence).unwrap(),
                consumer,
            )
            .unwrap();
            assert_eq!(result.thermal_transition.consumer, expected_consumer);
            let mut wrong_plan = plan.clone();
            wrong_plan.request.cpu_millicores += 1;
            assert!(execute(
                &constraints,
                Some(StaticResourceAdmission {
                    prepared_inputs: None,
                    plan: &wrong_plan,
                    observation_execution: ObservationExecutionPolicy::serial()
                })
            )
            .is_err());
            // A malformed source would fail binding. The incompatible admission must fail
            // first, before source validation/alignment starts allocating its node lookup.
            let mut invalid_source = source.clone();
            invalid_source.kind = StateKind::StructuralPrimary;
            let rejected = run_linear_static_from_thermal_state_with_prepared_constraint_plan(
                &model,
                ComputeBackend::Cpu,
                options.clone(),
                PreparedThermoelasticStaticInput {
                    consumer_step_id: &consumer.step_id,
                    source: &invalid_source,
                    constraint_plan: &constraints,
                    admitted: Some(StaticResourceAdmission {
                        prepared_inputs: None,
                        plan: &wrong_plan,
                        observation_execution: ObservationExecutionPolicy::serial(),
                    }),
                },
            );
            assert!(matches!(rejected, Err(FeaRunError::Resource(_))));
            let mut wrong_model = model.clone();
            wrong_model
                .constraints
                .retain(|constraint| matches!(constraint, AnalysisConstraint::Thermal(_)));
            let wrong_constraints =
                CanonicalStructuralConstraintPlan::compile_for_model(&wrong_model, Some(&mesh))
                    .unwrap();
            assert!(execute(&wrong_constraints, None).is_err());
            results.push(result);
        }
        assert_ne!(
            results[0].solver_result_identity,
            results[1].solver_result_identity
        );
        assert_eq!(results[0].run.fields, results[1].run.fields);
        assert_eq!(source.canonical_bytes().unwrap(), source_bytes);
    }
}
