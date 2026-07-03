use runmat_meshing_core::predicate::tetrahedron_scaled_jacobian;

use super::super::*;

#[test]
fn missing_face_local_cap_quality_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_quality(&cavity, &nodes, refill_options())
        .expect("local cap diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.pass_face_count, 0);
    assert_eq!(diagnostic.failed_face_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert!(diagnostic.candidate_source_bins.is_empty());
    assert_eq!(diagnostic.max_scaled_jacobian, 0.0);
    assert_eq!(diagnostic.max_failed_face_scaled_jacobian, 0.0);
    assert!(diagnostic.failed_face_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.failed_face_source_bins.is_empty());
    assert!(diagnostic.rejected_by_reason.is_empty());
}

#[test]
fn local_cap_apex_candidates_include_optimized_normal_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.18, 0.72, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);

    let quality_for = |candidate: &LocalCapApexCandidate| {
        tetrahedron_scaled_jacobian([
            nodes[&face[0]],
            nodes[&face[1]],
            nodes[&face[2]],
            candidate.coordinates_m,
        ])
    };
    let best_discrete_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_discrete_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);

    assert!(best_optimized_positive >= best_discrete_positive);
    assert!(best_optimized_negative >= best_discrete_negative);
}

#[test]
fn local_cap_apex_candidates_include_inplane_inward_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.2, 0.8, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);
    let inplane_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward")
        .collect::<Vec<_>>();
    let optimized_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward_optimized")
        .collect::<Vec<_>>();

    assert!(!inplane_candidates.is_empty());
    assert!(!optimized_candidates.is_empty());
    assert!(inplane_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
    assert!(optimized_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
}
