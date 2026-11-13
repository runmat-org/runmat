use super::*;
use crate::{
    planning::resource::{FeaResourceDemand, FeaResourcePhase},
    solve::nonlinear::solid::continuum_qualification::fixture,
    topology::{CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule},
};

#[test]
fn complete_point_history_and_accepted_cap_have_their_own_resource_authority() {
    for quadratic in [false, true] {
        let (model, mesh, _) = fixture::prepare(1, quadratic);
        let mut options = NonlinearSolveOptions {
            solver_mesh: Some(mesh.clone()),
            max_accepted_increments: 8,
            ..NonlinearSolveOptions::default()
        };
        let first = prepare(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
        )
        .unwrap();
        assert_eq!(
            first.input.phase,
            FeaResourcePhase::StructuralNonlinearPipeline
        );
        let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &first.input.demand else {
            panic!("nonlinear demand missing")
        };
        let integration =
            CanonicalSolidIntegrationPlan::derive(&mesh, CanonicalSolidIntegrationRule::Keast15)
                .unwrap();
        assert!(demand.constraint_compilation_peak_bytes.0 > 0);
        assert_eq!(demand.integration_identity, integration.identity());
        assert_eq!(demand.point_count.0, integration.total_point_count() as u64);
        assert_eq!(demand.maximum_accepted_increments.0, 8);
        assert!(demand.tangent_evaluation_limit.0 > 8);
        assert!(demand.additional_persistent_bytes.0 > 0);
        let crate::planning::resource::SolveResourcePolicy::Iterative { admission, .. } =
            &demand.operator.solve
        else {
            panic!("Newton tangent admission missing")
        };
        // These fixtures use direct fixed-DOF constraints. Newton nevertheless
        // solves the reduced system rather than linear direct elimination's
        // full-size matrix, so its admitted certificate dimension must differ.
        assert!(admission.dimension.0 > 0);
        assert!(admission.dimension.0 < demand.operator.sparse.rows.0);
        assert!(first.breakdown.phase_retained_bytes.0 >= demand.additional_persistent_bytes.0);
        options.max_accepted_increments = 16;
        let second = prepare(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
        )
        .unwrap();
        assert!(second.request.memory_bytes > first.request.memory_bytes);
        assert_ne!(second.identity, first.identity);
        let decoded =
            FeaResourcePlan::from_canonical_bytes(&first.canonical_bytes().unwrap()).unwrap();
        assert_eq!(decoded, first);
    }
}

#[test]
fn authored_memory_ceiling_rejects_before_physical_assembly_and_solve() {
    let (mut model, mesh, _) = fixture::prepare(1, false);
    let options = NonlinearSolveOptions {
        solver_mesh: Some(mesh),
        max_accepted_increments: 8,
        ..NonlinearSolveOptions::default()
    };
    let plan = prepare(
        &model,
        &model.sequence.steps[0],
        ComputeBackend::Cpu,
        &options,
    )
    .unwrap();
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
    let error = run_nonlinear_with_options(&model, ComputeBackend::Cpu, options).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("accuracy resource ceiling exceeded"),
        "{error}"
    );
}

#[test]
fn retained_context_capacity_and_overflow_remain_typed_before_physical_preparation() {
    use crate::planning::resource::ResourcePlanningError;
    let (mut model, mesh, _) = fixture::prepare(1, false);
    // An authored ceiling: the default is unbounded and placement applies machine capacity.
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(8 << 30);
    let options = NonlinearSolveOptions {
        solver_mesh: Some(mesh),
        max_accepted_increments: 8,
        ..NonlinearSolveOptions::default()
    };
    let step = &model.sequence.steps[0];
    let plan = prepare(&model, step, ComputeBackend::Cpu, &options).unwrap();
    let maximum = step.accuracy.resources.resident_byte_budget();
    let retained = maximum - plan.request.memory_bytes + 1;
    let error = super::super::preparation::prepare_with_retained(
        &model,
        ComputeBackend::Cpu,
        &options,
        retained,
    )
    .err()
    .expect("retained context must reject");
    assert!(
        matches!(error, FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required, maximum: ceiling, .. })
        if required == maximum + 1 && ceiling == maximum)
    );
    let overflow = super::super::preparation::prepare_with_retained(
        &model,
        ComputeBackend::Cpu,
        &options,
        u64::MAX,
    )
    .err()
    .expect("overflow must reject");
    assert!(matches!(
        overflow,
        FeaRunError::Resource(ResourcePlanningError::Overflow(_))
    ));
}

#[test]
fn readonly_recovery_has_distinct_identity_and_rejects_forged_solve_or_work() {
    use crate::planning::resource::{build_resource_plan, FeaResourcePlan, SolveResourcePolicy};
    for quadratic in [false, true] {
        let (mut model, mesh, _) = fixture::prepare(1, quadratic);
        // An authored ceiling: the default is unbounded and placement applies machine capacity.
        model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(8 << 30);
        let mut options = NonlinearSolveOptions {
            solver_mesh: Some(mesh),
            max_accepted_increments: 8,
            ..NonlinearSolveOptions::default()
        };
        let step = &model.sequence.steps[0];
        let recovery = prepare_recovery(&model, step, ComputeBackend::Cpu, &options, 4).unwrap();
        let solve =
            prepare_with_output_limit(&model, step, ComputeBackend::Cpu, &options, 4).unwrap();
        assert_eq!(
            recovery.input.phase,
            FeaResourcePhase::StructuralNonlinearRecovery
        );
        assert_ne!(recovery.identity, solve.identity);
        assert!(recovery.request.memory_bytes < solve.request.memory_bytes);
        assert_eq!(recovery.breakdown.resource_plan_decoded_resident_bytes.0, 0);
        assert_eq!(
            recovery.breakdown.resource_plan_transport_closure_bytes.0,
            0
        );
        assert!(solve.breakdown.resource_plan_decoded_resident_bytes.0 > 0);
        assert_eq!(recovery.breakdown.preconditioner_bytes.0, 0);
        assert_eq!(recovery.breakdown.factorization_bytes.0, 0);
        let FeaResourceDemand::StructuralNonlinearRecovery(demand) = &recovery.input.demand else {
            panic!("missing readonly authority")
        };
        let FeaResourceDemand::StructuralNonlinearPipeline(solve_demand) = &solve.input.demand
        else {
            unreachable!()
        };
        assert!(demand.constraint_compilation_peak_bytes.0 > 0);
        assert_eq!(
            demand.constraint_compilation_peak_bytes,
            solve_demand.constraint_compilation_peak_bytes
        );
        assert_eq!(demand.physical_evaluation_limit.0, 2);
        assert!(matches!(
            demand.operator.solve,
            SolveResourcePolicy::NoSolve { .. }
        ));
        assert_eq!(demand.operator.publication.result_artifact_bytes.0, 0);
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&recovery.canonical_bytes().unwrap()).unwrap(),
            recovery
        );
        let mut forged = recovery.input.clone();
        forged.structural_operator_mut().unwrap().solve =
            solve.input.structural_operator().unwrap().solve.clone();
        assert!(build_resource_plan(forged).is_err());
        let mut forged = recovery.input.clone();
        let FeaResourceDemand::StructuralNonlinearRecovery(demand) = &mut forged.demand else {
            unreachable!()
        };
        demand.physical_evaluation_limit.0 = 1;
        assert!(build_resource_plan(forged).is_err());
        options.max_accepted_increments = 4096;
        options.line_search = !options.line_search;
        let same = prepare_recovery(&model, step, ComputeBackend::Cpu, &options, 4).unwrap();
        assert_eq!(
            same, recovery,
            "readonly work must not depend on Newton caps/search"
        );
        let maximum = step.accuracy.resources.resident_byte_budget();
        let error = super::super::preparation::prepare_recovery(
            &model,
            &options,
            4,
            maximum - recovery.request.memory_bytes + 1,
        )
        .err()
        .unwrap();
        assert!(
            matches!(error, FeaRunError::Resource(crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
            required, maximum: ceiling, .. }) if required == maximum + 1 && ceiling == maximum)
        );
    }
}

#[test]
fn resource_scaling_plans_every_output_envelope_without_a_fixed_codec_ceiling() {
    for cells in [2, 4, 6] {
        let (model, mesh, _) = fixture::prepare(cells, true);
        let mut model = model;
        model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = None;
        let options = NonlinearSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..NonlinearSolveOptions::default()
        };
        let mut previous_memory = 0;
        for limit in [8usize, 4096] {
            let planned = prepare_with_output_limit(
                &model,
                &model.sequence.steps[0],
                ComputeBackend::Cpu,
                &options,
                limit,
            );
            // No fixed codec item ceiling caps the output envelope: with no authored memory
            // cap every size plans, and the run's admitted budget decides at admission.
            let plan = planned.unwrap();
            let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &plan.input.demand else {
                panic!()
            };
            let integration = crate::topology::CanonicalSolidIntegrationPlan::derive(
                &mesh,
                crate::topology::CanonicalSolidIntegrationRule::Keast15,
            )
            .unwrap();
            let properties =
                crate::properties::compile_structural_property_map(&model, Some(&mesh))
                    .unwrap()
                    .identity;
            let storage =
                crate::assembly::nonlinear_solid::resources::NonlinearSolidStorageDemand::derive(
                    &integration,
                    &properties,
                    mesh.topology.nodes.len() * 3,
                    1,
                    limit,
                    options.line_search,
                )
                .unwrap();
            assert!(plan.request.memory_bytes > previous_memory);
            assert!(demand.additional_persistent_bytes.0 >= storage.point_history_bytes);
            assert!(demand.operator.publication.result_artifact_bytes.0 > 0);
            assert!(plan.request.memory_bytes <= runmat_types::wire_integer::MAX_WIRE_INTEGER);
            previous_memory = plan.request.memory_bytes;
        }
    }
}
