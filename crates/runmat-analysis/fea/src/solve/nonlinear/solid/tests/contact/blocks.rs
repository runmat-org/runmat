//! Existing controller on the original full authored interface model.
use super::*;
use crate::solve::nonlinear::{
    driver::ContactBlockExecution, run_nonlinear_load_block, LoadIncrementSchedule,
    NonlinearExecution, NonlinearLoadBlockRequest, NonlinearLoadBlockStart,
    NonlinearLoadContinuation, NonlinearSolveOptions,
};
use crate::ComputeBackend;
use runmat_analysis_core::ProcedureStep;
mod constraints;

// witness: FEA-NONLINEAR-STATE-001/points @
//   small_strain_j2_penalty_coulomb_contact/tet4|tet10/prepared_controller;
//   boundary: Prepared constrained affine-body fixture; trusted test loads. No public/native
//   contact or generic contact convergence credit.
// witness: FEA-NONLINEAR-STATE-001/initial_outputs @
//   small_strain_j2_penalty_coulomb_contact/tet4|tet10/prepared_controller;
//   boundary: Prepared constrained affine-body fixture; trusted test loads. No public/native
//   contact or generic contact convergence credit.
// witness: FEA-NONLINEAR-STATE-001/chronology @
//   small_strain_j2_penalty_coulomb_contact/tet4|tet10/prepared_controller;
//   boundary: Prepared constrained affine-body fixture; trusted test loads. No public/native
//   contact or generic contact convergence credit.
#[test]
fn full_authored_contact_cold_blocks_preserve_schedule_original_points_and_work() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut f = CoupledFixture::new(order, true);
        f.model.interfaces = f.surface.interfaces.clone();
        f.model.sequence.steps = vec![ProcedureStep::structural_nonlinear("contact")];
        constraints::author_reference_equalities(&mut f);
        let load = f.summary.physical_load.clone();
        f.summary = assemble_structural_system_borrowed(
            &f.model,
            &RealizedExcitations::default(),
            Some(&f.surface.mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        f.summary.physical_load = load;
        let operator = CanonicalContactOperator::prepare(
            &f.surface.projection,
            &f.model.interfaces,
            &f.surface.rows,
            f.summary.dof_count,
            &[ContactOperatorTestFixture::control(true)],
            ContactOperatorTestFixture::budget(),
        )
        .unwrap();
        let p = policy();
        let options = NonlinearSolveOptions {
            solver_mesh: Some(f.surface.mesh.clone()),
            increment_count: 2,
            max_accepted_increments: 3,
            max_newton_iters: p.max_iterations,
            tolerance: p.relative_residual_tolerance,
            linear_tolerance: p.linear_relative_tolerance,
            max_load_cutbacks: p.max_cutbacks,
            minimum_load_increment: p.minimum_load_increment,
            line_search: p.line_search,
            max_line_search_backtracks: p.max_backtracks,
            line_search_reduction: p.backtrack_reduction,
            contact_enforcement: None,
        };
        let schedule = LoadIncrementSchedule::canonical(&[], &[0.1, 0.5], 0., 1.).unwrap();
        assert!(run(
            &f,
            &operator,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            3,
            0
        )
        .is_err());
        let whole = run(
            &f,
            &operator,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            3,
            1_000_000,
        )
        .unwrap();
        assert!(whole.complete);
        assert_eq!(whole.result.load_factors, vec![0.1, 0.5, 1.]);
        let mut block = run(
            &f,
            &operator,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            1,
            1_000_000,
        )
        .unwrap();
        for step in 0..3 {
            assert_eq!(
                block.result.load_factors,
                vec![whole.result.load_factors[step]]
            );
            assert_eq!(
                block.result.contact_pressure_snapshots[0],
                whole.result.contact_pressure_snapshots[step]
            );
            assert_eq!(
                block.result.contact_gap_snapshots[0],
                whole.result.contact_gap_snapshots[step]
            );
            assert_eq!(
                block.result.displacement_snapshots[0],
                whole.result.displacement_snapshots[step]
            );
            assert_eq!(
                block.result.contact_gap_snapshots[0].len(),
                operator.point_count()
            );
            for (&pressure, &gap) in block.result.contact_pressure_snapshots[0]
                .iter()
                .zip(&block.result.contact_gap_snapshots[0])
            {
                near(pressure, (-10000. * gap).max(0.), 1e-10);
            }
            if step == 2 {
                break;
            }
            let expected_digest = block.continuation.content_digest();
            let original = block.continuation.canonical_bytes().unwrap();
            let cold = NonlinearLoadContinuation::from_canonical_bytes(&original, expected_digest)
                .unwrap();
            let previous_pressure = block.result.contact_pressure_snapshots[0].clone();
            let previous_gap = block.result.contact_gap_snapshots[0].clone();
            block = run(
                &f,
                &operator,
                &options,
                &schedule,
                NonlinearLoadBlockStart::Continue {
                    continuation: &cold,
                    expected_digest,
                },
                1,
                1_000_000,
            )
            .unwrap();
            assert_eq!(block.result.initial_contact_pressure, previous_pressure);
            assert_eq!(block.result.initial_contact_gap, previous_gap);
            assert_eq!(original, cold.canonical_bytes().unwrap());
            assert!(run(
                &f,
                &operator,
                &options,
                &schedule,
                NonlinearLoadBlockStart::Continue {
                    continuation: &cold,
                    expected_digest,
                },
                1,
                2_000_000
            )
            .is_err());
        }
        assert!(block.complete);
        assert_eq!(
            block
                .continuation
                .accepted_state()
                .contact_configuration()
                .unwrap()
                .ordinal(),
            block.continuation.work().accepted_increments() as u64
        );
        assert_eq!(
            whole.continuation.accepted_state(),
            block.continuation.accepted_state()
        );
        // Extra cold prefix/end proofs are real work, not reset counters.
        assert!(
            block.continuation.work().linear_solves() > whole.continuation.work().linear_solves()
        );
        assert_eq!(
            block.result.preconditioner,
            crate::contracts::SolverPreconditioner::Ilu0
        );
    }
}

#[test]
fn full_source_composition_rejects_foreign_interfaces_and_mutated_geometry_with_same_hashes() {
    let mut f = CoupledFixture::new(ElementOrder::Tet10, true);
    f.model.interfaces = f.surface.interfaces.clone();
    let operator = f.operator(true);
    assert!(CanonicalNonlinearSolidBasis::prepare_composed(
        &f.model,
        &f.surface.mesh,
        &f.summary,
        Some(&operator),
    )
    .is_ok());
    let mut foreign = f.model.clone();
    foreign.interfaces[0].interface_id.push_str("-foreign");
    assert!(CanonicalNonlinearSolidBasis::prepare_composed(
        &foreign,
        &f.surface.mesh,
        &f.summary,
        Some(&operator),
    )
    .is_err());
    let mut projection = f.surface.projection.clone();
    projection.boundary_faces[0].area_m2 = runmat_types::physical::Quantity::new(
        projection.boundary_faces[0].area_m2.canonical_value() * (2.),
    );
    assert_eq!(
        projection.projection_digest,
        f.surface.projection.projection_digest
    );
    let altered = CanonicalContactOperator::prepare(
        &projection,
        &f.model.interfaces,
        &f.surface.rows,
        f.summary.dof_count,
        &[ContactOperatorTestFixture::control(true)],
        ContactOperatorTestFixture::budget(),
    )
    .unwrap();
    assert!(CanonicalNonlinearSolidBasis::prepare_composed(
        &f.model,
        &f.surface.mesh,
        &f.summary,
        Some(&altered),
    )
    .is_err());
    assert!(CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).is_err());
}

fn run<'a>(
    f: &'a CoupledFixture,
    operator: &'a CanonicalContactOperator<'a>,
    options: &'a NonlinearSolveOptions,
    schedule: &'a LoadIncrementSchedule,
    input: NonlinearLoadBlockStart<'a>,
    limit: usize,
    output_bytes: u64,
) -> Result<crate::solve::nonlinear::driver::NonlinearLoadBlockResult, crate::FeaRunError> {
    run_nonlinear_load_block(NonlinearLoadBlockRequest {
        model: &f.model,
        summary: &f.summary,
        options,
        execution: NonlinearExecution {
            backend: ComputeBackend::Cpu,
            solver_admission: None,
            contact: Some(ContactBlockExecution {
                coupling: CoupledFixture::coupling(operator),
                maximum_retained_output_bytes: output_bytes,
            }),
        },
        start: 0.,
        end: 1.,
        schedule,
        input,
        accepted_block_limit: limit,
    })
}
