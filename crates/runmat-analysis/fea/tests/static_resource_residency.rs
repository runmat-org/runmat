//! Isolated Rust-heap evidence for the admitted CPU physical static pipeline.
//! Scheduler memory-byte-millis records reservations, not this measured heap peak.

#[path = "support/peak_allocator.rs"]
mod peak_allocator;

use peak_allocator::{begin_measurement, end_measurement, process_peak_rss_bytes};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, Displacement, FormulationRequirement,
    RegionTarget, StructuralConstraint, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, StructuralTranslationDof,
};
use runmat_analysis_fea::{
    constraints::{CanonicalStructuralConstraintPlan, ConstraintEnforcement},
    execution::StructuralLinearWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{
        build_structural_resource_plan, LocalExecutionPlan, LocalThreadCount, SolveResourcePolicy,
    },
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    run_linear_static_with_prepared_constraint_plan, ComputeBackend, LinearStaticSolveOptions,
    StaticResourceAdmission,
};

fn equation(terms: Vec<StructuralConstraintEquationTerm>) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "measured_equation".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
            terms,
            right_hand_side: Displacement::new(0.0),
        }),
    })
}

fn ux_term(node_id: u32, coefficient: f64) -> StructuralConstraintEquationTerm {
    StructuralConstraintEquationTerm {
        target: RegionTarget::point(format!("node:{node_id}")),
        dof: StructuralTranslationDof::Ux,
        coefficient,
    }
}

fn static_model(enforcement: ConstraintEnforcement, node_count: u32) -> AnalysisModel {
    assert!(node_count >= 34);
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let structural = model.structural.as_mut().unwrap();
    let template = structural.elements[0].clone();
    for node_id in 3..=node_count {
        structural.nodes.push(runmat_analysis_core::StructuralNode {
            node_id,
            coordinates_m: [1.0, f64::from(node_id - 2) * 0.01, 0.0],
            solid_attachment: None,
        });
        let mut element = template.clone();
        element.element_id = format!("beam_{}", node_id - 1);
        let runmat_analysis_core::StructuralElementKind::Beam(beam) = &mut element.kind else {
            unreachable!("fixture provides a beam")
        };
        beam.node_ids = [1, node_id];
        structural.elements.push(element);
    }
    match enforcement {
        ConstraintEnforcement::DirectElimination => {}
        ConstraintEnforcement::AffineReduction => {
            model
                .constraints
                .push(equation(vec![ux_term(2, 1.0), ux_term(3, -1.0)]));
        }
        ConstraintEnforcement::LagrangeMultiplier => {
            model.constraints.push(equation(
                (2..=node_count)
                    .map(|node_id| ux_term(node_id, 1.0 / f64::from(node_count - 1)))
                    .collect(),
            ));
        }
    }
    model
}

fn measure_static_case(
    enforcement: ConstraintEnforcement,
    local_execution: LocalExecutionPlan,
    node_count: u32,
) -> (u64, u64) {
    let model = static_model(enforcement, node_count);
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    assert_eq!(constraints.enforcement, enforcement);
    let fields = structural_static_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        None,
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .unwrap();
    let options = LinearStaticSolveOptions {
        formulation_requirement: FormulationRequirement::Experimental,
        local_execution,
        ..LinearStaticSolveOptions::default()
    };
    let workload = StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: ComputeBackend::Cpu,
        formulation_requirement: options.formulation_requirement,
        preconditioner: options.preconditioner_kind,
        algebra_backend: options.algebra_backend_kind,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: options.require_solver_mesh_for_solid,
        initial_guess_transition: None,
    };
    let plan = build_structural_resource_plan(
        None,
        &constraints,
        &observations,
        &workload,
        &[],
        local_execution,
    )
    .unwrap();
    match enforcement {
        ConstraintEnforcement::LagrangeMultiplier => assert!(matches!(
            plan.input.structural_operator().unwrap().solve,
            SolveResourcePolicy::SaddlePointGmres { .. }
        )),
        _ => assert!(matches!(
            plan.input.structural_operator().unwrap().solve,
            SolveResourcePolicy::Iterative { .. }
        )),
    }

    let baseline = begin_measurement();
    let run = run_linear_static_with_prepared_constraint_plan(
        &model,
        ComputeBackend::Cpu,
        options,
        &constraints,
        Some(StaticResourceAdmission {
            plan: &plan,
            observation_execution: ObservationExecutionPolicy::serial(),
        }),
    )
    .unwrap();
    let peak = end_measurement();
    assert!(run.structural_constraint_result.is_some());
    let incremental_peak = peak.saturating_sub(baseline);
    let numerical_request = plan
        .request
        .memory_bytes
        .checked_sub(plan.breakdown.worker_overhead_bytes.0)
        .unwrap();
    eprintln!(
        "static {enforcement:?} ({local_execution:?}, nodes={node_count}): baseline_live_heap={baseline} peak_live_heap={peak} incremental_peak={incremental_peak} process_peak_rss={:?} numerical_request={numerical_request} planned_memory={}",
        process_peak_rss_bytes(), plan.request.memory_bytes,
    );
    assert!(
        incremental_peak <= numerical_request,
        "measured Rust heap increment {incremental_peak} exceeds admitted numerical memory {numerical_request} for {enforcement:?}",
    );
    (incremental_peak, numerical_request)
}

#[test]
fn admitted_cpu_static_heap_peaks_fit_ordinary_affine_and_multiplier_requests() {
    let serial = LocalExecutionPlan::serial();
    let (ordinary_peak, ordinary_plan) =
        measure_static_case(ConstraintEnforcement::DirectElimination, serial, 34);
    let (affine_peak, affine_plan) =
        measure_static_case(ConstraintEnforcement::AffineReduction, serial, 34);
    let (multiplier_peak, multiplier_plan) =
        measure_static_case(ConstraintEnforcement::LagrangeMultiplier, serial, 34);
    assert!(ordinary_peak > 0 && affine_peak > 0 && multiplier_peak > 0);
    assert!(ordinary_plan > 0 && affine_plan > 0 && multiplier_plan > 0);

    let scoped = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    measure_static_case(ConstraintEnforcement::DirectElimination, scoped, 34);
    measure_static_case(ConstraintEnforcement::AffineReduction, scoped, 34);
    measure_static_case(ConstraintEnforcement::LagrangeMultiplier, scoped, 34);

    for (enforcement, small_peak, small_plan) in [
        (
            ConstraintEnforcement::DirectElimination,
            ordinary_peak,
            ordinary_plan,
        ),
        (
            ConstraintEnforcement::AffineReduction,
            affine_peak,
            affine_plan,
        ),
        (
            ConstraintEnforcement::LagrangeMultiplier,
            multiplier_peak,
            multiplier_plan,
        ),
    ] {
        let (larger_peak, larger_plan) = measure_static_case(enforcement, serial, 68);
        assert!(
            larger_peak > small_peak,
            "{enforcement:?} heap peak did not grow"
        );
        assert!(
            larger_plan > small_plan,
            "{enforcement:?} plan did not grow"
        );
    }
}
