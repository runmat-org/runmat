use super::*;

mod yielded_substitution;

fn block(
    model: &AnalysisModel,
    summary: &AssemblySummary,
    options: &NonlinearSolveOptions,
    schedule: &LoadIncrementSchedule,
    input: NonlinearLoadBlockStart<'_>,
    limit: usize,
) -> Result<NonlinearLoadBlockResult, FeaRunError> {
    run_nonlinear_load_block(NonlinearLoadBlockRequest {
        model,
        summary,
        options,
        execution: NonlinearExecution {
            backend: ComputeBackend::Cpu,
            contact: None,
            solver_admission: None,
        },
        start: 0.0,
        end: 1.0,
        schedule,
        input,
        accepted_block_limit: limit,
    })
}

// witness: FEA-NONLINEAR-STATE-001/points @ small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Homogeneous core cold blocks; original histories and outputs, not continuum accuracy.
// witness: FEA-NONLINEAR-STATE-001/initial_outputs @
//   small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Homogeneous core cold blocks; original histories and outputs, not continuum accuracy.
// witness: FEA-NONLINEAR-STATE-001/chronology @ small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Homogeneous core cold blocks; original histories and outputs, not continuum accuracy.
#[test]
fn cold_blocks_preserve_original_point_history_schedule_and_physical_outputs() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, summary, mut options) = uniaxial(order);
        options.increment_count = 2;
        options.max_accepted_increments = 3;
        let schedule = LoadIncrementSchedule::canonical(&[], &[0.95], 0.0, 1.0).unwrap();
        let whole = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            3,
        )
        .unwrap();
        let first = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            2,
        )
        .unwrap();
        assert!(!first.complete);
        assert!(
            first
                .result
                .equivalent_plastic_strain_snapshots
                .last()
                .unwrap()[0]
                > 0.0
        );
        let bytes = first.continuation.canonical_bytes().unwrap();
        let cold = NonlinearLoadContinuation::from_canonical_bytes(
            &bytes,
            first.continuation.content_digest(),
        )
        .unwrap();
        assert!(NonlinearLoadContinuation::from_canonical_bytes(
            &bytes,
            Digest::sha256(b"foreign-root")
        )
        .is_err());
        assert!(NonlinearLoadContinuation::from_canonical_bytes(
            &bytes[..bytes.len() - 1],
            first.continuation.content_digest()
        )
        .is_err());
        assert_eq!(cold, first.continuation);
        let resumed = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &cold,
                expected_digest: cold.content_digest(),
            },
            1,
        )
        .unwrap();
        assert!(resumed.complete);
        assert_eq!(
            resumed.result.initial_equivalent_plastic_strain,
            *first
                .result
                .equivalent_plastic_strain_snapshots
                .last()
                .unwrap()
        );
        assert_eq!(
            resumed.result.initial_displacement,
            cold.accepted_state().displacement()
        );
        assert_eq!(
            resumed.continuation.accepted_state(),
            whole.continuation.accepted_state()
        );
        assert_eq!(resumed.continuation.work().accepted_increments(), 3);
        assert_eq!(
            resumed.continuation.maximum_relative_residual(),
            whole
                .result
                .residual_norms
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert_eq!(
            resumed.continuation.maximum_relative_increment(),
            whole
                .result
                .relative_increment_norms
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert!(
            resumed.continuation.maximum_relative_residual() >= cold.maximum_relative_residual()
        );
        assert!(
            resumed.continuation.maximum_relative_increment() >= cold.maximum_relative_increment()
        );
        assert_eq!(
            resumed.continuation.work().newton_iterations(),
            whole.continuation.work().newton_iterations()
        );
        assert_eq!(
            resumed.continuation.work().tangent_evaluations(),
            whole.continuation.work().tangent_evaluations()
        );
        for (actual, expected) in [
            (
                &resumed.result.displacement_snapshots,
                &whole.result.displacement_snapshots,
            ),
            (
                &resumed.result.internal_force_snapshots,
                &whole.result.internal_force_snapshots,
            ),
            (
                &resumed.result.stress_snapshots,
                &whole.result.stress_snapshots,
            ),
            (
                &resumed.result.plastic_strain_snapshots,
                &whole.result.plastic_strain_snapshots,
            ),
            (
                &resumed.result.equivalent_plastic_strain_snapshots,
                &whole.result.equivalent_plastic_strain_snapshots,
            ),
        ] {
            assert_eq!(actual.last(), expected.last());
        }
        // Independent uniaxial hardening law: sigma=5.6, alpha=(5.6-5)/30=.02.
        assert!((resumed.result.stress_snapshots[0][0] - 5.6).abs() < 2e-9);
        assert!((resumed.result.equivalent_plastic_strain_snapshots[0][0] - 0.02).abs() < 2e-9);
        assert!(block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &resumed.continuation,
                expected_digest: resumed.continuation.content_digest()
            },
            1
        )
        .is_err());
    }
}

// witness: FEA-NONLINEAR-STATE-001/cold_validation @ small_strain_j2/tet10/prepared_controller;
//   boundary: Altered controller and original point state reject; no independent past-trajectory
//   proof.
#[test]
fn altered_cold_controller_and_point_state_reject_without_mutating_original() {
    let (model, summary, options) = uniaxial(ElementOrder::Tet10);
    let schedule = LoadIncrementSchedule::canonical(&[], &[0.3], 0.0, 1.0).unwrap();
    let first = block(
        &model,
        &summary,
        &options,
        &schedule,
        NonlinearLoadBlockStart::Fresh,
        1,
    )
    .unwrap();
    assert_eq!(first.result.load_factors, [0.25]);
    let original = first.continuation;
    for substitution in 0..7 {
        let mut changed = original.clone();
        match substitution {
            0 => changed.pending_target = Some(0.9),
            1 => changed.work.accepted_increments = options.max_accepted_increments,
            2 => changed.work.newton_iterations = usize::MAX,
            3 => changed
                .accepted_state
                .constitutive
                .point_identities
                .swap(0, 1),
            4 => changed.accepted_state.constitutive.points[0].equivalent_plastic_strain += 0.1,
            5 => changed.controller_scope = Digest::sha256(b"foreign-controller"),
            _ => changed.accepted_state.displacement[0] += 1.0,
        }
        changed.content_digest = changed.derive_digest().unwrap();
        assert!(
            block(
                &model,
                &summary,
                &options,
                &schedule,
                NonlinearLoadBlockStart::Continue {
                    continuation: &changed,
                    expected_digest: changed.content_digest()
                },
                1
            )
            .is_err(),
            "substitution={substitution}"
        );
    }
    assert!(block(
        &model,
        &summary,
        &options,
        &schedule,
        NonlinearLoadBlockStart::Continue {
            continuation: &original,
            expected_digest: Digest::sha256(b"wrong-object")
        },
        1
    )
    .is_err());
    let next = block(
        &model,
        &summary,
        &options,
        &schedule,
        NonlinearLoadBlockStart::Continue {
            continuation: &original,
            expected_digest: original.content_digest(),
        },
        1,
    )
    .unwrap();
    assert_eq!(next.result.load_factors, [0.3]);
    assert_eq!(original.work().accepted_increments(), 1);
}

// witness: FEA-NONLINEAR-STATE-001/chronology @ small_strain_j2/tet4/prepared_controller;
//   boundary: One-increment cold replay costs count toward original global limits.
#[test]
fn one_increment_blocks_count_replay_cost_without_resetting_global_limits() {
    let (model, summary, mut options) = uniaxial(ElementOrder::Tet4);
    options.max_accepted_increments = 4;
    let schedule = LoadIncrementSchedule::canonical(&[], &[], 0.0, 1.0).unwrap();
    let mut current = block(
        &model,
        &summary,
        &options,
        &schedule,
        NonlinearLoadBlockStart::Fresh,
        1,
    )
    .unwrap();
    for _ in 1..4 {
        current = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &current.continuation,
                expected_digest: current.continuation.content_digest(),
            },
            1,
        )
        .unwrap();
    }
    assert!(current.complete);
    assert_eq!(current.continuation.work().accepted_increments(), 4);
    assert_eq!(current.continuation.work().controller_evaluations(), 16);
}

// witness: FEA-NONLINEAR-STATE-001/chronology @ small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Original cutback target and failed work survive cold blocks.
#[test]
fn cutback_boundary_retains_original_target_and_failed_work_across_cold_blocks() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, summary, mut options) = uniaxial(order);
        options.increment_count = 1;
        options.max_newton_iters = 1;
        options.max_accepted_increments = 16;
        let schedule = LoadIncrementSchedule::canonical(&[], &[], 0.0, 1.0).unwrap();
        let first = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            1,
        )
        .unwrap();
        assert_eq!(first.result.load_factors, [0.5]);
        assert_eq!(first.continuation.pending_target(), Some(1.0));
        assert_eq!(first.continuation.work().rejected_increments(), 1);
        assert_eq!(first.continuation.work().newton_iterations(), 2);
        let cold = NonlinearLoadContinuation::from_canonical_bytes(
            &first.continuation.canonical_bytes().unwrap(),
            first.continuation.content_digest(),
        )
        .unwrap();
        let next = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &cold,
                expected_digest: cold.content_digest(),
            },
            1,
        )
        .unwrap();
        assert_eq!(next.result.load_factors, [0.75]);
        assert_eq!(next.continuation.pending_target(), Some(1.0));
        assert_eq!(next.continuation.work().rejected_increments(), 2);
        assert_eq!(next.continuation.work().newton_iterations(), 4);
        assert_eq!(cold.work().accepted_increments(), 1);
    }
}

// witness: FEA-NONLINEAR-STATE-001/rollback @ small_strain_j2/tet4/prepared_controller;
//   boundary: Cancelled block preserves original prefix and publishes no accepted suffix.
#[test]
fn cancelled_continuation_does_not_mutate_or_publish_an_accepted_suffix() {
    let (model, summary, options) = uniaxial(ElementOrder::Tet4);
    let schedule = LoadIncrementSchedule::canonical(&[], &[], 0.0, 1.0).unwrap();
    let first = block(
        &model,
        &summary,
        &options,
        &schedule,
        NonlinearLoadBlockStart::Fresh,
        1,
    )
    .unwrap();
    let before = first.continuation.canonical_bytes().unwrap();
    let context = crate::progress::FeaProgressContext::default();
    context.cancellation.cancel();
    let guard = crate::progress::replace_fea_progress_context(context);
    assert!(matches!(
        block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &first.continuation,
                expected_digest: first.continuation.content_digest()
            },
            1
        ),
        Err(FeaRunError::Cancelled)
    ));
    drop(guard);
    assert_eq!(before, first.continuation.canonical_bytes().unwrap());
}
