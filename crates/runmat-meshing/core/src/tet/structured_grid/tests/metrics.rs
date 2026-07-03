use super::common::*;
use super::*;

#[test]
fn boundary_projection_accepts_exact_quality_non_regressing_move() {
    let (input, nodes, volume_elements, boundary_faces) = boundary_projection_fixture();
    let mut original_quality = quality_report(
        element_quality_for_nodes(&volume_elements, &nodes).expect("quality"),
        boundary_projection_errors(&input, &boundary_faces, &nodes),
    );
    original_quality.min_exact_scaled_jacobian = 0.0;
    for element in &mut original_quality.elements {
        element.exact_scaled_jacobian = 0.0;
    }

    let (projected_nodes, projected_quality) = project_boundary_nodes_if_quality_improves(
        &input,
        nodes,
        &volume_elements,
        &boundary_faces,
        original_quality.clone(),
    );

    assert!(
        projected_quality.max_boundary_projection_error_m
            < original_quality.max_boundary_projection_error_m
    );
    assert!(
        projected_quality.min_scaled_jacobian >= QualityThresholds::default().min_scaled_jacobian
    );
    assert!(projected_nodes[0].coordinates_m[2] < 0.5);
}
#[test]
fn boundary_projection_accepts_above_threshold_exact_quality_regression() {
    let (input, nodes, volume_elements, boundary_faces) = boundary_projection_fixture();
    let mut original_quality = quality_report(
        element_quality_for_nodes(&volume_elements, &nodes).expect("quality"),
        boundary_projection_errors(&input, &boundary_faces, &nodes),
    );
    original_quality.min_exact_scaled_jacobian = 1.0;
    for element in &mut original_quality.elements {
        element.exact_scaled_jacobian = 1.0;
    }

    let (projected_nodes, projected_quality) = project_boundary_nodes_if_quality_improves(
        &input,
        nodes.clone(),
        &volume_elements,
        &boundary_faces,
        original_quality.clone(),
    );

    assert_ne!(projected_nodes, nodes);
    assert!(
        projected_quality.max_boundary_projection_error_m
            < original_quality.max_boundary_projection_error_m
    );
    assert!(
        projected_quality.min_exact_scaled_jacobian
            >= QualityThresholds::default().min_scaled_jacobian
    );
}
