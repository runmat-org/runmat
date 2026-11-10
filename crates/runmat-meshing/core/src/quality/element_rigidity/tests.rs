use super::*;

// Boundary: the derived floor separates straight slivers that assembly's rigidity test accepts
// from those it refuses, for each element order; Tet10 needs a far higher floor than Tet4.
#[test]
fn assembly_floor_separates_admitted_and_refused_slivers() {
    let tet10 = assembly_minimum_scaled_jacobian(ElementOrder::Tet10);
    let tet4 = assembly_minimum_scaled_jacobian(ElementOrder::Tet4);
    assert!((1.0e-3..1.0e-2).contains(&tet10), "Tet10 floor {tet10}");
    assert!(tet4 < 1.0e-5, "Tet4 floor {tet4}");
    // The canonical sliver's scaled Jacobian is proportional to its height near zero.
    let height_at =
        |floor: f64| floor / tetrahedron_scaled_jacobian(&canonical_sliver(1.0e-3)) * 1.0e-3;
    let at = height_at(tet10);
    assert!(straight_element_passes(
        ElementOrder::Tet10,
        canonical_sliver(at * 1.01)
    ));
    assert!(!straight_element_passes(
        ElementOrder::Tet10,
        canonical_sliver(at * 0.99)
    ));
    // The volume's former default floor admits a Tet10 that assembly refuses.
    assert!(!straight_element_passes(
        ElementOrder::Tet10,
        canonical_sliver(height_at(1.0e-3))
    ));
    assert!(straight_element_passes(
        ElementOrder::Tet4,
        canonical_sliver(height_at(1.0e-3))
    ));
}

#[test]
fn rigidity_test_names_a_degenerate_element() {
    let corners = [[0.0, 0.0, 0.0]; 4];
    assert_eq!(
        element_rigidity(&corners, &[0.0; 144]),
        Err(ElementRigidityShortfall::DegenerateGeometry)
    );
}
