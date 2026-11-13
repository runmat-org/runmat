use super::*;
use crate::solve::nonlinear::solid::continuum_qualification::fixture;

#[test]
fn admitted_nonuniform_blocks_preserve_every_original_point_and_output() {
    for quadratic in [false, true] {
        let (model, mesh, _) = fixture::prepare(1, quadratic);
        let options = NonlinearSolveOptions {
            solver_mesh: Some(mesh),
            increment_count: 2,
            max_accepted_increments: 8,
            max_newton_iters: 32,
            tolerance: 1e-9,
            linear_tolerance: 1e-12,
            max_line_search_backtracks: 16,
            ..NonlinearSolveOptions::default()
        };
        let whole = run_nonlinear_block(
            &model,
            ComputeBackend::Cpu,
            &options,
            NonlinearBlockInput::Fresh,
            8,
        )
        .unwrap();
        let first = run_nonlinear_block(
            &model,
            ComputeBackend::Cpu,
            &options,
            NonlinearBlockInput::Fresh,
            1,
        )
        .unwrap();
        assert!(!first.complete);
        let expected = first.checkpoint.content_digest();
        let cold = NonlinearLoadContinuation::from_canonical_bytes(
            &first.checkpoint.canonical_bytes().unwrap(),
            expected,
        )
        .unwrap();
        let last = run_nonlinear_block(
            &model,
            ComputeBackend::Cpu,
            &options,
            NonlinearBlockInput::Continue {
                checkpoint: &cold,
                expected_digest: expected,
            },
            8,
        )
        .unwrap();
        assert!(last.complete);
        assert_eq!(
            last.checkpoint.accepted_state(),
            whole.checkpoint.accepted_state()
        );
        assert_eq!(
            last.result.initial_equivalent_plastic_strain,
            *first
                .result
                .equivalent_plastic_strain_snapshots
                .last()
                .unwrap()
        );
        assert_eq!(
            last.result.stress_snapshots.last(),
            whole.result.stress_snapshots.last()
        );
        assert_eq!(
            last.result.internal_force_snapshots.last(),
            whole.result.internal_force_snapshots.last()
        );
        // This is exact original-trajectory parity, not independent continuum
        // accuracy: the original nonuniform qualification failures remain open.
        assert_eq!(
            last.checkpoint.work().accepted_increments(),
            whole.checkpoint.work().accepted_increments()
        );
        let alphas = whole
            .checkpoint
            .accepted_state()
            .point_history()
            .map(|(_, _, p)| p.equivalent_plastic_strain)
            .collect::<Vec<_>>();
        assert!(alphas.iter().any(|a| *a != alphas[0]));
    }
}

#[test]
fn blocked_execution_rejects_resource_and_control_substitution() {
    let (mut model, mesh, _) = fixture::prepare(1, false);
    let options = NonlinearSolveOptions {
        solver_mesh: Some(mesh),
        max_accepted_increments: 8,
        ..NonlinearSolveOptions::default()
    };
    assert!(run_nonlinear_block(
        &model,
        ComputeBackend::Gpu,
        &options,
        NonlinearBlockInput::Fresh,
        1
    )
    .is_err());
    assert!(run_nonlinear_block(
        &model,
        ComputeBackend::Cpu,
        &options,
        NonlinearBlockInput::Fresh,
        0
    )
    .is_err());
    assert!(run_nonlinear_block(
        &model,
        ComputeBackend::Cpu,
        &options,
        NonlinearBlockInput::Fresh,
        9
    )
    .is_err());
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = run_nonlinear_block(
        &model,
        ComputeBackend::Cpu,
        &options,
        NonlinearBlockInput::Fresh,
        1,
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("accuracy resource ceiling exceeded"),
        "{error}"
    );
}
