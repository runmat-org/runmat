//! Isolated requested-Rust-heap evidence for borrowed constraint-rank classification.
//! Does not measure constraint compilation, native allocations, allocator overhead or task RSS.
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;
#[path = "../support/constraint_rank_reference.rs"]
mod reference;

use runmat_analysis_core::StructuralTranslationDof;
use runmat_analysis_fea::constraints::{
    classify_constraint_rows, constraint_rank_graph_workspace_bytes,
    constraint_rank_workspace_profile, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    ConstraintResidualQuantity, ReactionRecovery,
};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

fn equation(
    id: String,
    terms: impl IntoIterator<Item = (usize, f64)>,
) -> CanonicalStructuralConstraintRow {
    CanonicalStructuralConstraintRow {
        identity: Digest::sha256(id.as_bytes()),
        constraint_id: id,
        terms: terms
            .into_iter()
            .map(|(index, coefficient)| {
                let mut identity = [0; 32];
                identity[..8].copy_from_slice(&(index as u64).to_be_bytes());
                CanonicalStructuralConstraintTerm {
                    owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes(
                        identity,
                    )),
                    dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
                    coefficient,
                }
            })
            .collect(),
        right_hand_side: 0.0,
        residual_quantity: ConstraintResidualQuantity::Displacement,
        // Deliberately not an authority for the borrowed classifier.
        reaction_recovery: ReactionRecovery::RedundantEquation,
    }
}

fn connected_rows(count: usize, label_bytes: usize) -> Vec<CanonicalStructuralConstraintRow> {
    let prefix = "x".repeat(label_bytes);
    let mut rows = vec![equation(
        format!("{prefix}-0"),
        (0..count).map(|index| (index, 1.0)),
    )];
    for index in 1..count {
        rows.push(equation(
            format!("{prefix}-{index}"),
            [(0, 1.0), (index, 2.0)],
        ));
    }
    rows.push(equation(
        format!("{prefix}-duplicate"),
        (0..count).map(|index| (index, 1.0)),
    ));
    rows
}

fn measure(
    rows: &[CanonicalStructuralConstraintRow],
    active: usize,
    redundant: usize,
) -> (u64, u64) {
    let original = rows.to_vec();
    let baseline = peak_allocator::begin_measurement();
    let graph_bound = constraint_rank_graph_workspace_bytes(rows);
    let count_peak = peak_allocator::end_measurement().saturating_sub(baseline);
    let graph_bound = graph_bound.unwrap();
    assert_eq!(
        count_peak, 0,
        "count-only graph preflight must not allocate"
    );
    let baseline = peak_allocator::begin_measurement();
    let profile = constraint_rank_workspace_profile(rows);
    let profile_peak = peak_allocator::end_measurement().saturating_sub(baseline);
    let profile = profile.unwrap();
    assert!(profile.graph_bytes <= graph_bound);
    assert!(
        profile_peak <= profile.graph_bytes,
        "profile peak={profile_peak}, graph bound={}",
        profile.graph_bytes
    );

    let baseline = peak_allocator::begin_measurement();
    let classification = classify_constraint_rows(rows);
    let peak = peak_allocator::end_measurement().saturating_sub(baseline);
    let classification = classification.unwrap();
    assert!(
        peak <= profile.total_requested_bytes,
        "rows={} peak={peak}, bound={}",
        rows.len(),
        profile.total_requested_bytes
    );
    assert_eq!(classification.active_row_count, active);
    assert_eq!(classification.redundant_row_count, redundant);
    assert_eq!(classification.classifications.len(), rows.len());
    assert_eq!(rows, original);
    println!(
        "rows={} profile_peak={profile_peak} peak={peak} bound={}",
        rows.len(),
        profile.total_requested_bytes
    );
    (peak, profile.total_requested_bytes)
}

// One test keeps the binary-local allocator's measurement windows non-overlapping.
#[test]
fn connected_component_rank_workspace_bounds_actual_requested_heap() {
    verify_legacy_elimination_parity();
    measure(&[], 0, 0);
    for count in [1, 64, 4096] {
        let rows = (0..count)
            .map(|index| equation(format!("fixed-{index}"), [(index, 1.0)]))
            .collect::<Vec<_>>();
        measure(&rows, count, 0);
    }
    for count in [16, 64, 128] {
        let short = measure(&connected_rows(count, 1), count, 1);
        let long = measure(&connected_rows(count, 4096), count, 1);
        assert_eq!(
            short, long,
            "constraint labels must be borrowed, not copied into provenance"
        );
    }
    let repeated = (0..512)
        .map(|index| equation(format!("repeat-{index}"), [(0, 1.0)]))
        .collect::<Vec<_>>();
    measure(&repeated, 1, 511);
    for star in [false, true] {
        let mut previous_peak = None;
        for count in [256, 1024] {
            let rows = (0..count)
                .map(|index| {
                    equation(
                        format!("sparse-{index}"),
                        [(index, 1.0), (if star { count } else { index + 1 }, 1.0)],
                    )
                })
                .collect::<Vec<_>>();
            let (peak, _) = measure(&rows, count, 0);
            if let Some(previous) = previous_peak {
                assert!(peak <= previous * 5,
                    "4x larger sparse connected input must not force quadratic allocation: {previous} -> {peak}");
            }
            previous_peak = Some(peak);
        }
    }
    println!(
        "process_peak_rss_bytes={:?}",
        peak_allocator::process_peak_rss_bytes()
    );
}

fn verify_legacy_elimination_parity() {
    let mut random = 17_u64;
    for case in 0..128 {
        let mut rows = Vec::new();
        for ordinal in 0..20 {
            let scale = [1.0e-12, 1.0, 1.0e12][(case + ordinal) % 3];
            let terms = (0..6)
                .filter_map(|index| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let coefficient = ((random >> 32) % 7) as i32 - 3;
                    (coefficient != 0).then_some((index, f64::from(coefficient) * scale))
                })
                .collect::<Vec<_>>();
            if terms.is_empty() {
                continue;
            }
            let rhs = terms
                .iter()
                .map(|(index, coefficient)| (*index + 1) as f64 * coefficient)
                .sum();
            let mut row = equation(format!("case-{case}-row-{ordinal}"), terms);
            row.right_hand_side = rhs;
            rows.push(row);
        }
        if case % 2 == 1 {
            rows.reverse();
        }
        // Also compare rejection, including exact conflict provenance.
        if case % 4 == 0 {
            let mut contradiction = rows[0].clone();
            contradiction.constraint_id = "contradiction".into();
            contradiction.right_hand_side += 100.0;
            rows.push(contradiction);
        }
        let mut legacy_rows = rows.clone();
        for row in &mut legacy_rows {
            row.reaction_recovery = if row.terms.len() == 1 {
                ReactionRecovery::UnconstrainedEquilibriumResidual
            } else {
                ReactionRecovery::ConstraintMultiplier
            };
        }
        let legacy = reference::classify(&mut legacy_rows);
        let borrowed = classify_constraint_rows(&rows);
        match (legacy, borrowed) {
            (Ok((active, redundant)), Ok(actual)) => {
                assert_eq!(
                    (active, redundant),
                    (actual.active_row_count, actual.redundant_row_count),
                    "case={case}"
                );
                assert_eq!(
                    legacy_rows
                        .iter()
                        .map(|row| row.reaction_recovery)
                        .collect::<Vec<_>>(),
                    actual.classifications,
                    "case={case}"
                );
            }
            (Err(expected), Err(actual)) => assert_eq!(expected, actual, "case={case}"),
            (expected, actual) => panic!("case={case}: legacy={expected:?}, borrowed={actual:?}"),
        }
    }
}
