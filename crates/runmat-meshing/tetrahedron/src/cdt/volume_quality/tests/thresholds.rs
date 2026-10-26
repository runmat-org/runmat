//! Scheduling equivalence does not change the configured physical acceptance set.
use super::*;
use crate::cdt::{mutation::TopologyMutation, volume_quality::working::WorkingVolumeQuality};

fn baseline() -> (
    DelaunayVolumeTopology,
    MetricFieldRequest,
    DelaunayVolumeProvenance,
    DelaunayVolumeQuality,
) {
    let topology = topology();
    let request = metric_request();
    let provenance = metric_contexts(&topology);
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (topology, request, provenance, quality)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Above-bound ratios inside the same scheduling bucket, including the next
//   representable value above one, remain physical violations. Nonfinite ratios fail closed. This
//   checks the acceptance predicate; it does not establish meshing convergence or a physical error
//   bound for arbitrary geometry.
#[test]
fn scheduling_ties_do_not_waive_full_value_physical_bounds() {
    let (_, _, _, baseline) = baseline();
    let mut cell = baseline.tetrahedra[0].clone();
    for offset in [0, 1, 8, 1 << 22] {
        let ratio = f64::from_bits(1.0_f64.to_bits() + offset);
        assert_eq!(violation_priority_key(ratio), violation_priority_key(1.0));
        cell.refinement_violation_ratio = ratio;
        cell.optimization_violation_ratio = ratio;
        assert_eq!(cell.requires_refinement(), offset != 0);
        assert_eq!(cell.requires_optimization(), offset != 0);
    }
    for ratio in [0.0, f64::from_bits(1.0_f64.to_bits() - 1), 1.0] {
        assert!(!violates_quality_bound(ratio));
    }
    for ratio in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(violates_quality_bound(ratio));
    }
}

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Size and shape violations inside a scheduling tie are established from primitive
//   physical inequalities, independently admitted and retained by the mutable work queue. This
//   finite mesh family does not qualify an original CAD model or substitute for final physical
//   admission.
#[test]
fn independent_admission_and_working_queues_preserve_near_boundary_violations() {
    let (topology, request, provenance, baseline) = baseline();
    for objective in [QualityObjective::Refinement, QualityObjective::Shape] {
        let mut options = DelaunayVolumeQualityOptions {
            maximum_metric_edge_length: baseline.maximum_metric_edge_length * 2.0,
            minimum_metric_scaled_jacobian: baseline.minimum_metric_scaled_jacobian / 2.0,
            ..Default::default()
        };
        match objective {
            QualityObjective::Refinement => {
                options.maximum_metric_edge_length =
                    baseline.maximum_metric_edge_length * (1.0 - 1.0e-10)
            }
            QualityObjective::Shape => {
                options.minimum_metric_scaled_jacobian =
                    baseline.minimum_metric_scaled_jacobian * (1.0 + 1.0e-10)
            }
        }
        let quality = evaluate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            options,
            &NeverCancelled,
        )
        .unwrap();
        // Assert the primitive physical inequality independently of both queues.
        let ratio = match objective {
            QualityObjective::Refinement => {
                quality.maximum_metric_edge_length / options.maximum_metric_edge_length
            }
            QualityObjective::Shape => {
                options.minimum_metric_scaled_jacobian / quality.minimum_metric_scaled_jacobian
            }
        };
        assert!(ratio > 1.0);
        assert_eq!(violation_priority_key(ratio), violation_priority_key(1.0));
        let expected = match objective {
            QualityObjective::Refinement => quality.worst_refinement_tetrahedron,
            QualityObjective::Shape => quality.worst_optimization_tetrahedron,
        };
        assert!(
            expected.is_some(),
            "a configured physical bound is violated"
        );
        validate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            &quality,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let working_topology = TopologyMutation::from_canonical(topology.clone());
        let working =
            WorkingVolumeQuality::new(quality, &working_topology, options, &NeverCancelled)
                .unwrap();
        assert_eq!(
            working.worst(objective).map(|cell| cell.node_identities),
            expected
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A substituted ratio and missing work source cannot hide a primitive size-bound
//   violation, even when the substitution lies within numeric field-comparison allowance. This
//   separates approximate evidence comparison from physical acceptance; other tampering and
//   whole-study envelopes require their own witnesses.
#[test]
fn near_bound_report_substitution_cannot_hide_an_independent_physical_violation() {
    let (topology, request, provenance, baseline) = baseline();
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: baseline.maximum_metric_edge_length * (1.0 - 1.0e-12),
        minimum_metric_scaled_jacobian: baseline.minimum_metric_scaled_jacobian / 2.0,
        ..Default::default()
    };
    let mut quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert!(quality.maximum_metric_edge_length > options.maximum_metric_edge_length);
    assert!(quality.worst_refinement_tetrahedron.is_some());
    // The perturbation is inside primitive-comparison roundoff allowance, but
    // that allowance cannot erase an independently established bound violation.
    for cell in &mut quality.tetrahedra {
        if cell.maximum_metric_edge_length > options.maximum_metric_edge_length {
            cell.refinement_violation_ratio = 1.0;
        }
    }
    quality.worst_refinement_tetrahedron = None;
    assert!(validate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        &quality,
        options,
        &NeverCancelled
    )
    .is_err());
}
