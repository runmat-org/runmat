//! Isolated Rust-heap and whole-process RSS evidence for admitted CPU physical static solves.
//! Scheduler memory-byte-millis records reservations, not these measured peaks.

#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use std::sync::Mutex;

use peak_allocator::{begin_measurement, end_measurement, process_peak_rss_bytes};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, Displacement, RegionTarget,
    StructuralConstraint, StructuralConstraintEquation, StructuralConstraintEquationTerm,
    StructuralConstraintKind, StructuralTranslationDof,
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

// The counting allocator is process-global; concurrent tests would contaminate each other's peak.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

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
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([1.0, f64::from(node_id - 2) * 0.01, 0.0]),
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

struct StaticMeasurement {
    incremental_heap_peak: u64,
    numerical_request: u64,
    process_peak_rss_before: Option<u64>,
    process_peak_rss_after: Option<u64>,
}

fn measure_static_case(
    enforcement: ConstraintEnforcement,
    local_execution: LocalExecutionPlan,
    node_count: u32,
) -> StaticMeasurement {
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
        local_execution,
        ..LinearStaticSolveOptions::default()
    };
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: ComputeBackend::Cpu,
        preconditioner: options.preconditioner_kind,
        algebra_backend: options.algebra_backend_kind,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: options.require_solver_mesh_for_solid,
        state_input: None,
    };
    let plan = build_structural_resource_plan(
        None,
        &constraints,
        &observations,
        &workload,
        &[],
        local_execution,
        None,
    )
    .unwrap();
    // All three enforcement forms use the canonical affine-reduced SPD path;
    // multiplier recovery does not select a separate saddle-point GMRES solve.
    assert!(matches!(
        plan.input.structural_operator().unwrap().solve,
        SolveResourcePolicy::Iterative { .. }
    ));

    let process_peak_rss_before = process_peak_rss_bytes();
    let baseline = begin_measurement();
    let run = run_linear_static_with_prepared_constraint_plan(
        &model,
        ComputeBackend::Cpu,
        options,
        &constraints,
        Some(StaticResourceAdmission {
            prepared_inputs: None,
            plan: &plan,
            observation_execution: ObservationExecutionPolicy::serial(),
        }),
    )
    .unwrap();
    let peak = end_measurement();
    let process_peak_rss_after = process_peak_rss_bytes();
    assert!(run.structural_constraint_result.is_some());
    let incremental_peak = peak.saturating_sub(baseline);
    let numerical_request = plan
        .request
        .memory_bytes
        .checked_sub(plan.breakdown.worker_overhead_bytes.0)
        .unwrap();
    eprintln!(
        "static {enforcement:?} ({local_execution:?}, nodes={node_count}): baseline_live_heap={baseline} peak_live_heap={peak} incremental_peak={incremental_peak} process_peak_rss_before={process_peak_rss_before:?} process_peak_rss_after={process_peak_rss_after:?} numerical_request={numerical_request} planned_memory={}",
        plan.request.memory_bytes,
    );
    assert!(
        incremental_peak <= numerical_request,
        "measured Rust heap increment {incremental_peak} exceeds admitted numerical memory {numerical_request} for {enforcement:?}",
    );
    StaticMeasurement {
        incremental_heap_peak: incremental_peak,
        numerical_request,
        process_peak_rss_before,
        process_peak_rss_after,
    }
}

#[test]
fn admitted_cpu_static_heap_peaks_fit_ordinary_affine_and_multiplier_requests() {
    let _measurement_guard = MEASUREMENT_LOCK.lock().unwrap();
    let serial = LocalExecutionPlan::serial();
    let ordinary = measure_static_case(ConstraintEnforcement::DirectElimination, serial, 34);
    let affine = measure_static_case(ConstraintEnforcement::AffineReduction, serial, 34);
    let multiplier = measure_static_case(ConstraintEnforcement::LagrangeMultiplier, serial, 34);
    assert!(
        ordinary.incremental_heap_peak > 0
            && affine.incremental_heap_peak > 0
            && multiplier.incremental_heap_peak > 0
    );
    assert!(
        ordinary.numerical_request > 0
            && affine.numerical_request > 0
            && multiplier.numerical_request > 0
    );

    let scoped = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    measure_static_case(ConstraintEnforcement::DirectElimination, scoped, 34);
    measure_static_case(ConstraintEnforcement::AffineReduction, scoped, 34);
    measure_static_case(ConstraintEnforcement::LagrangeMultiplier, scoped, 34);

    for (enforcement, small) in [
        (ConstraintEnforcement::DirectElimination, ordinary),
        (ConstraintEnforcement::AffineReduction, affine),
        (ConstraintEnforcement::LagrangeMultiplier, multiplier),
    ] {
        let larger = measure_static_case(enforcement, serial, 68);
        assert!(
            larger.incremental_heap_peak > small.incremental_heap_peak,
            "{enforcement:?} heap peak did not grow"
        );
        assert!(
            larger.numerical_request > small.numerical_request,
            "{enforcement:?} plan did not grow"
        );
    }
}

/// A fresh test process gives each branch its own process-lifetime RSS high-water mark. The
/// increase above the pre-solve mark includes native allocations, but cannot attribute shared
/// runtime/library memory or prove the production worker's placement and usage accounting.
#[test]
fn isolated_static_process_rss_growth_fits_admitted_request() {
    let _measurement_guard = MEASUREMENT_LOCK.lock().unwrap();
    if !cfg!(unix) {
        return;
    }
    const CASE_ENV: &str = "RUNMAT_FEA_STATIC_RSS_CASE";
    if let Some(case) = std::env::var_os(CASE_ENV) {
        let enforcement = match case.to_str().unwrap() {
            "ordinary" => ConstraintEnforcement::DirectElimination,
            "affine" => ConstraintEnforcement::AffineReduction,
            "multiplier" => ConstraintEnforcement::LagrangeMultiplier,
            other => panic!("unknown isolated static RSS case: {other}"),
        };
        let measured = measure_static_case(enforcement, LocalExecutionPlan::serial(), 200);
        let before = measured.process_peak_rss_before.unwrap();
        let after = measured.process_peak_rss_after.unwrap();
        assert!(
            after > before,
            "solve did not raise process RSS high-water mark"
        );
        assert!(
            after - before <= measured.numerical_request,
            "process RSS high-water increase exceeded numerical request for {enforcement:?}"
        );
        return;
    }

    for case in ["ordinary", "affine", "multiplier"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "isolated_static_process_rss_growth_fits_admitted_request",
                "--nocapture",
            ])
            .env(CASE_ENV, case)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated {case} RSS check failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        eprintln!("{case}: {}", String::from_utf8_lossy(&output.stderr));
    }
}
