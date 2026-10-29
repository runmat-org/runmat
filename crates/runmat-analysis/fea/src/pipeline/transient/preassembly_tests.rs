use super::*;
use crate::fixtures::{fixture_model, FixtureId};

fn direct_scope_case() -> (
    crate::execution::StructuralTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let workload = crate::execution::StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: runmat_analysis_core::ProcedureStepId("transient".into()),
        backend: ComputeBackend::Cpu,
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
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_preassembly(&model, (&options).into()) {
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
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_preassembly(&model, (&options).into()) {
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
        ..TransientSolveOptions::default()
    };
    let prepared = prepare_transient_preassembly(&model, (&options).into()).unwrap();
    let expected = prepared
        .schedule
        .points
        .iter()
        .map(|point| point.time_s.canonical_value())
        .collect::<Vec<_>>();
    let result = run_transient_with_options(&model, ComputeBackend::Cpu, options).unwrap();
    assert_eq!(result.time_points_s, expected);
}

#[test]
fn checkpoint_scope_resource_preflight_is_deterministic_and_precedes_assembly() {
    let (workload, mesh) = direct_scope_case();
    let prepared = prepare_transient_preassembly(&workload.model, (&workload).into()).unwrap();
    let first = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &workload,
        &mesh,
        None,
        &prepared,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let second = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &workload,
        &mesh,
        None,
        &prepared,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    assert_eq!(first.identity, second.identity);
    assert_eq!(first.request.cpu_millicores, 1_000);
    assert!(first.request.memory_bytes > 0);
    assert!(first.request.max_artifact_bytes > 0);
    assert_eq!(
        first.request.max_artifact_bytes,
        first.request.max_egress_bytes
    );
    assert_eq!(first.breakdown.observation_produced_sample_count.0, 0);
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
    let actual_payload_bytes = u64::try_from(scope.canonical_bytes().unwrap().len()).unwrap();
    let planned_payload_bytes =
        crate::planning::resource::structural_transient_scope_payload_bytes_upper_bound(
            &prepared.schedule,
        )
        .unwrap();
    assert!(actual_payload_bytes <= planned_payload_bytes);
    assert_eq!(
        first.request.max_artifact_bytes,
        planned_payload_bytes
            + crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES
            + crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES
    );

    let mut longer = workload.clone();
    longer.time_step_policy =
        crate::solve::transient::StructuralTimeStepPolicy::Fixed { interval_count: 64 };
    let longer_prepared = prepare_transient_preassembly(&longer.model, (&longer).into()).unwrap();
    let longer_plan = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &longer,
        &mesh,
        None,
        &longer_prepared,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let longer_scope = prepare_transient_checkpoint_scope(&longer, &mesh, None).unwrap();
    assert!(longer_plan.request.max_artifact_bytes > first.request.max_artifact_bytes);
    assert!(
        u64::try_from(longer_scope.canonical_bytes().unwrap().len()).unwrap()
            <= crate::planning::resource::structural_transient_scope_payload_bytes_upper_bound(
                &longer_prepared.schedule,
            )
            .unwrap()
    );

    let mut rejected = workload;
    rejected.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = prepare_transient_checkpoint_scope(&rejected, &mesh, None).unwrap_err();
    assert!(matches!(
        error,
        FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                ..
            }
        )
    ));
}

#[test]
fn transient_block_checks_scope_residency_before_its_initial_assembly() {
    let (mut workload, mesh) = direct_scope_case();
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
    let request = crate::execution::StructuralTransientBlockRequest::new(scope, None, 2).unwrap();
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = run_transient_block(StructuralTransientRunBlockInput {
        workload: &workload,
        solver_mesh: &mesh,
        request: &request,
        predecessor: None,
        modal_state: None,
        observation_cursors: Vec::new(),
    })
    .unwrap_err();
    assert!(
        matches!(
            &error,
            FeaRunError::Resource(
                crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "resident memory bytes",
                    required,
                    maximum: 1,
                }
            ) if *required > 1
        ),
        "{error:?}"
    );
}

#[test]
fn transient_block_rejects_its_own_memory_ceiling_before_initial_assembly() {
    let (mut workload, mesh) = direct_scope_case();
    // An authored ceiling with the same canonical integer width as the value set below, so the
    // workload encoding (and the plan derived from it) does not change with the value.
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(u64::from(u32::MAX));
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
    let request = crate::execution::StructuralTransientBlockRequest::new(scope, None, 2).unwrap();
    let block_plan = crate::planning::resource::build_structural_transient_block_resource_plan(
        &workload,
        &mesh,
        None,
        &request,
        None,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    // The block's accuracy ceiling is checked by the same planner that run_transient_block
    // calls before it assembles any operator; one byte short of its residency is rejected.
    let mut constrained = workload.clone();
    constrained.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(block_plan.request.memory_bytes - 1);
    // The authored ceiling is part of the workload, so the constrained block has its own scope.
    let constrained_scope = prepare_transient_checkpoint_scope(&constrained, &mesh, None).unwrap();
    let constrained_request =
        crate::execution::StructuralTransientBlockRequest::new(constrained_scope, None, 2).unwrap();
    let error = crate::planning::resource::build_structural_transient_block_resource_plan(
        &constrained,
        &mesh,
        None,
        &constrained_request,
        None,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn transient_block_resource_plan_bounds_a_physical_segment() {
    let (workload, mesh) = direct_scope_case();
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
    let request = crate::execution::StructuralTransientBlockRequest::new(scope, None, 2).unwrap();
    let plan = crate::planning::resource::build_structural_transient_block_resource_plan(
        &workload,
        &mesh,
        None,
        &request,
        None,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let crate::planning::resource::FeaResourceDemand::StructuralTransientBlock(demand) =
        &plan.input.demand
    else {
        panic!("expected transient block demand");
    };
    assert_eq!(plan.request.cpu_millicores, 1_000);
    assert_eq!(demand.time_point_count.0, 2);
    assert!(matches!(
        demand.operator.solve,
        crate::planning::resource::SolveResourcePolicy::TransientBlock { .. }
    ));
    let segment = run_transient_block(StructuralTransientRunBlockInput {
        workload: &workload,
        solver_mesh: &mesh,
        request: &request,
        predecessor: None,
        modal_state: None,
        observation_cursors: Vec::new(),
    })
    .unwrap();
    let mut actual_bytes =
        u64::try_from(segment.manifest().canonical_bytes().unwrap().len()).unwrap();
    if let Some(initial) = segment.initial_sample() {
        actual_bytes += u64::try_from(initial.canonical_bytes().unwrap().len()).unwrap();
    }
    for sample in segment.samples() {
        actual_bytes += u64::try_from(sample.canonical_bytes().unwrap().len()).unwrap();
    }
    assert!(actual_bytes <= demand.segment_output.encoded_payload_bytes_upper_bound.0);
    assert_eq!(
        plan.request.max_artifact_bytes,
        demand.segment_output.encoded_payload_bytes_upper_bound.0
            + crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES
            + crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES
    );

    let mut understated_output = plan.input.clone();
    let crate::planning::resource::FeaResourceDemand::StructuralTransientBlock(understated) =
        &mut understated_output.demand
    else {
        unreachable!();
    };
    understated
        .segment_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(crate::planning::resource::build_resource_plan(understated_output).is_err());

    let mut wrong_inputs = demand.inputs.clone();
    wrong_inputs.encoded_payload_bytes.0 += 1;
    assert!(
        crate::planning::resource::build_structural_transient_block_resource_plan(
            &workload,
            &mesh,
            None,
            &request,
            None,
            Some(wrong_inputs),
            None,
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .is_err()
    );
}

#[test]
fn scope_task_plan_uses_the_prepared_input_closure_and_output_set() {
    let (workload, mesh) = direct_scope_case();
    let prepared = prepare_transient_preassembly(&workload.model, (&workload).into()).unwrap();
    let baseline = crate::planning::resource::build_structural_transient_scope_resource_plan(
        &workload,
        &mesh,
        None,
        &prepared,
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let crate::planning::resource::FeaResourceDemand::StructuralTransientScopePreparation(demand) =
        &baseline.input.demand
    else {
        panic!("expected scope preparation demand");
    };
    let mut input_facts = demand.inputs.clone();
    input_facts.transport_closure_bytes.0 += 1_024;
    let output_set = crate::planning::resource::PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: crate::planning::resource::ByteCount(
            demand.scope_output.encoded_payload_bytes_upper_bound.0 + 4_096,
        ),
    };
    let task_plan = plan_transient_checkpoint_scope_task(
        &workload,
        &mesh,
        None,
        input_facts.clone(),
        output_set.clone(),
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let crate::planning::resource::FeaResourceDemand::StructuralTransientScopePreparation(
        task_demand,
    ) = &task_plan.input.demand
    else {
        panic!("expected scope task demand");
    };
    assert_eq!(task_demand.inputs, input_facts);
    assert_eq!(task_plan.input.output_set, Some(output_set.clone()));
    // The supplied output-set root bound replaces the default root bound one for one.
    assert_eq!(
        task_plan.request.max_artifact_bytes
            + crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES,
        baseline.request.max_artifact_bytes + output_set.encoded_bytes_upper_bound.0
    );
    assert_eq!(
        task_plan.input.demand.output_facts(
            crate::execution::AnalysisNumericalObjectKind::StructuralTransientCheckpointScope
        ),
        Some(&task_demand.scope_output)
    );
    assert!(matches!(
        plan_transient_checkpoint_scope_task(
            &workload,
            &mesh,
            None,
            input_facts.clone(),
            crate::planning::resource::PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: crate::planning::resource::ByteCount(
                    crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES + 1,
                ),
            },
            crate::planning::resource::LocalExecutionPlan::serial(),
        ),
        Err(FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::Invalid(_)
        ))
    ));

    input_facts.encoded_payload_bytes.0 -= 1;
    assert!(matches!(
        plan_transient_checkpoint_scope_task(
            &workload,
            &mesh,
            None,
            input_facts,
            output_set,
            crate::planning::resource::LocalExecutionPlan::serial(),
        ),
        Err(FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::Invalid(_)
        ))
    ));
}
