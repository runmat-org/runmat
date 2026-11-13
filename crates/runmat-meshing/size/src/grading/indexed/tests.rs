use super::*;
use crate::metric::{MetricSourceKind, MetricTensor3};

fn evaluation(length: f64) -> ResolvedMetricEvaluation {
    ResolvedMetricEvaluation {
        metric: MetricTensor3::isotropic_length_m(length).unwrap(),
        active_sources: vec![MetricSourceKind::Global],
        applied_contribution_count: 3,
        clipped_contribution_count: 2,
        rejected_contribution_count: 1,
    }
}

// Independent whole-graph relaxation is intentionally not the production heap.
fn reference(
    ratio: f64,
    offsets: &[usize],
    neighbors: &[usize],
    values: &[ResolvedMetricEvaluation],
) -> Vec<ResolvedMetricEvaluation> {
    let original = values
        .iter()
        .map(|value| value.metric.conservative_minimum_length_m().unwrap())
        .collect::<Vec<_>>();
    let mut caps = original.clone();
    loop {
        let previous = caps.clone();
        for index in 0..caps.len() {
            for &neighbor in &neighbors[offsets[index]..offsets[index + 1]] {
                let candidate = previous[index] * ratio;
                if candidate.is_finite() && candidate < caps[neighbor] {
                    caps[neighbor] = candidate;
                }
            }
        }
        if caps == previous {
            break;
        }
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let mut result = value.clone();
            if caps[index] < original[index] {
                let density = MetricTensor3::isotropic_length_m(caps[index]).unwrap().xx;
                let delta = (density - result.metric.maximum_row_density()).max(0.0);
                result.metric.xx += delta;
                result.metric.yy += delta;
                result.metric.zz += delta;
                result.clipped_contribution_count += 1;
            }
            result
        })
        .collect()
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One-, eight- and 257-vertex paths with reversed identity order, anisotropy and tied
//   sources compare every resulting metric and evidence count with independent full-graph
//   relaxation at grading ratios one, 1.2, two and the finite overflow boundary. This supports the
//   shared indexed grading representation, not full physical mesh admission, native memory or
//   arbitrary input scale.
#[test]
fn indexed_grading_matches_independent_propagation_and_preserves_metadata() {
    for count in [1, 8, 257] {
        let mut offsets = vec![0];
        let mut neighbors = Vec::new();
        for index in 0..count {
            if index > 0 {
                neighbors.push(index - 1);
            }
            if index + 1 < count {
                neighbors.push(index + 1);
            }
            offsets.push(neighbors.len());
        }
        let identities = (0..count).rev().collect::<Vec<_>>();
        for ratio in [1.0, 1.2, 2.0, f64::MAX] {
            let mut values = (0..count)
                .map(|index| evaluation(if index % 7 == 0 { 0.01 } else { 1.0 }))
                .collect::<Vec<_>>();
            values[0].metric.xx *= 3.0;
            let expected = reference(ratio, &offsets, &neighbors, &values);
            grade_indexed_metric_evaluations(ratio, &identities, &offsets, &neighbors, &mut values)
                .unwrap();
            assert_eq!(values, expected);
        }
    }
}

#[test]
fn indexed_grading_rejects_malformed_support_without_mutation() {
    let cases = [
        (vec![0, 1], vec![0, 1], vec![1]),
        (vec![0, 1], vec![0, 2, 1], vec![1]),
        (vec![0, 1], vec![0, 1, 1], vec![1]),
        (vec![0, 1], vec![0, 1, 2], vec![2, 0]),
        (vec![0, 1], vec![0, 1, 2], vec![0, 1]),
        (vec![0, 1], vec![0, 2, 3], vec![1, 1, 0]),
        (vec![0, 0], vec![0, 1, 2], vec![1, 0]),
    ];
    for (identities, offsets, columns) in cases {
        let mut values = vec![evaluation(0.01), evaluation(1.0)];
        let original = values.clone();
        assert!(grade_indexed_metric_evaluations(
            1.2,
            &identities,
            &offsets,
            &columns,
            &mut values
        )
        .is_err());
        assert_eq!(values, original);
    }
    let mut empty = Vec::new();
    grade_indexed_metric_evaluations(1.2, &[] as &[u32], &[0], &[], &mut empty).unwrap();
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A 4,097-vertex star uses non-Clone identities and preserves every original
//   active-source allocation while indexed grading matches independent whole-graph relaxation
//   exactly. Scratch support consists of CSR indices, scalar caps and ranked indices. This
//   establishes neither measured allocator demand nor native study memory, cancellation, arbitrary
//   topology mutation locality or whole-task scaling.
#[test]
fn indexed_grading_handles_high_valence_without_identity_or_metadata_copies() {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct NonCloneIdentity(usize);
    let count = 4097;
    let identities = (0..count).map(NonCloneIdentity).collect::<Vec<_>>();
    let mut offsets = vec![0];
    let mut neighbors = (1..count).collect::<Vec<_>>();
    offsets.push(neighbors.len());
    for _ in 1..count {
        neighbors.push(0);
        offsets.push(neighbors.len());
    }
    let mut values = (0..count)
        .map(|index| evaluation(if index == 0 { 0.01 } else { 1.0 }))
        .collect::<Vec<_>>();
    let pointers = values
        .iter()
        .map(|value| value.active_sources.as_ptr())
        .collect::<Vec<_>>();
    let expected = reference(2.0, &offsets, &neighbors, &values);
    grade_indexed_metric_evaluations(2.0, &identities, &offsets, &neighbors, &mut values).unwrap();
    assert_eq!(values, expected);
    assert!(values
        .iter()
        .zip(pointers)
        .all(|(value, original)| value.active_sources.as_ptr() == original));
}

#[test]
fn closed_grading_bounds_preserve_values_and_still_reject_invalid_support() {
    for count in [1, 8, 257] {
        let identities = (0..count).rev().collect::<Vec<_>>();
        let mut offsets = vec![0];
        let mut neighbors = Vec::new();
        for index in 0..count {
            if index > 0 {
                neighbors.push(index - 1);
            }
            if index + 1 < count {
                neighbors.push(index + 1);
            }
            offsets.push(neighbors.len());
        }
        for ratio in [1.0, 1.25, f64::MAX] {
            for near_uniform in [false, true] {
                let mut values = (0..count)
                    .map(|index| {
                        evaluation(if near_uniform && index % 2 == 0 {
                            1.1
                        } else {
                            1.0
                        })
                    })
                    .collect::<Vec<_>>();
                let expected = reference(ratio, &offsets, &neighbors, &values);
                grade_indexed_metric_evaluations(
                    ratio,
                    &identities,
                    &offsets,
                    &neighbors,
                    &mut values,
                )
                .unwrap();
                assert_eq!(values, expected);
            }
        }
    }
    let mut constant = vec![evaluation(1.0), evaluation(1.0)];
    let original = constant.clone();
    assert!(
        grade_indexed_metric_evaluations(1.25, &[0, 1], &[0, 1, 1], &[1], &mut constant).is_err()
    );
    assert_eq!(constant, original);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A positive tensor scaling preserves every directional ratio while reaching the
//   original finest-length cap, and repeated grading is unchanged. Isotropic cap realization
//   remains exact. This component case does not establish spatial adaptation, arbitrary
//   floating-point dominance or whole-study throughput.
#[test]
fn anisotropic_grading_preserves_directional_ratios_and_is_idempotent() {
    let mut values = vec![evaluation(0.125), evaluation(1.0)];
    values[1].metric = MetricTensor3 {
        xx: 4.0,
        yy: 3.0,
        zz: 2.0,
        xy: 0.5,
        xz: 0.25,
        yz: -0.125,
    };
    let original = values[1].metric;
    grade_indexed_metric_evaluations(2.0, &[0, 1], &[0, 1, 2], &[1, 0], &mut values).unwrap();
    let factor = values[1].metric.xx / original.xx;
    assert!(factor > 1.0);
    for (before, after) in [
        (original.yy, values[1].metric.yy),
        (original.zz, values[1].metric.zz),
        (original.xy, values[1].metric.xy),
        (original.xz, values[1].metric.xz),
        (original.yz, values[1].metric.yz),
    ] {
        assert!((after / before - factor).abs() <= factor * 8.0 * f64::EPSILON);
    }
    assert!(values[1].metric.maximum_row_density() >= 16.0);
    assert!(values[1].metric.maximum_row_density() <= 16.0 * (1.0 + 32.0 * f64::EPSILON));
    assert!(
        values[1].metric.xx >= original.xx
            && values[1].metric.yy >= original.yy
            && values[1].metric.zz >= original.zz
    );
    let once = values.clone();
    grade_indexed_metric_evaluations(2.0, &[0, 1], &[0, 1, 2], &[1, 0], &mut values).unwrap();
    assert_eq!(values, once);
}
