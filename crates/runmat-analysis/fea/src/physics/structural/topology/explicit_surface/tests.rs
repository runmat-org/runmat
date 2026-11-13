use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::StructuralNode;

#[test]
fn two_orthogonal_shells_preserve_area_measure_normal_and_borrowed_labels() {
    let mut structural = fixture_model(FixtureId::StructuralShellPlateMomentReference)
        .structural
        .unwrap();
    structural.nodes.push(StructuralNode {
        node_id: 4,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new([
                0.0, 0.0, 1.0,
            ]),
        solid_attachment: None,
    });
    let mut second = structural.elements[0].clone();
    second.element_id = "shell_2".into();
    let StructuralElementKind::Shell(shell) = &mut second.kind else {
        panic!("shell")
    };
    shell.node_ids = [1, 2, 4];
    structural.elements.push(second);
    let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
    let geometry = explicit_shell_surface_geometry(
        &structural,
        &topology,
        &runmat_meshing_core::fixtures::fixture_face("shell_panel"),
    )
    .unwrap();
    assert_eq!(geometry.area_m2.canonical_value(), 1.0);
    assert_eq!(geometry.triangles.len(), 2);
    assert_eq!(
        geometry.triangles[0].oriented_area_vector_m2,
        [0.0, 0.0, 0.5]
    );
    assert_eq!(
        geometry.triangles[1].oriented_area_vector_m2,
        [0.0, -0.5, 0.0]
    );
    assert!(std::ptr::eq(
        geometry.triangles[1].element_id.as_ptr(),
        structural.elements[1].element_id.as_ptr()
    ));
    for index in 0..4 {
        let node = &geometry.nodes[&topology.nodes[index].identity];
        assert_eq!(
            node.coordinates_m.canonical_value(),
            structural.nodes[index].coordinates_m.canonical_value()
        );
        let measure = if index < 2 { 1.0 / 3.0 } else { 1.0 / 6.0 };
        assert_eq!(node.nodal_measure_m2, measure);
        let expected = match index {
            0 | 1 => [
                0.0,
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            2 => [0.0, 0.0, 1.0],
            _ => [0.0, -1.0, 0.0],
        };
        for (actual, expected) in node.oriented_normal.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-16);
        }
    }
    let retained = geometry.nodes.len()
        * std::mem::size_of::<(StableDigest, ExplicitShellSurfaceNode)>()
        + geometry.triangles.len() * std::mem::size_of::<ExplicitShellTriangle<'_>>();
    assert!(
        shell_surface_workspace_bytes(
            &structural,
            &runmat_meshing_core::fixtures::fixture_face("shell_panel")
        )
        .unwrap()
            > retained as u64
    );
}

#[test]
fn opposing_faces_reject_undefined_normal_and_degenerate_first_face_precedes_later_missing_node() {
    let mut structural = fixture_model(FixtureId::StructuralShellPlateMomentReference)
        .structural
        .unwrap();
    let mut reverse = structural.elements[0].clone();
    reverse.element_id = "reverse".into();
    let StructuralElementKind::Shell(shell) = &mut reverse.kind else {
        panic!("shell")
    };
    shell.node_ids = [1, 3, 2];
    structural.elements.push(reverse);
    let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
    assert!(matches!(
        explicit_shell_surface_geometry(
            &structural,
            &topology,
            &runmat_meshing_core::fixtures::fixture_face("shell_panel")
        ),
        Err(ExplicitShellSurfaceError::UndefinedNodeNormal { .. })
    ));
    structural.nodes[2].coordinates_m =
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
            structural.nodes[1].coordinates_m.canonical_value(),
        );
    let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
    let StructuralElementKind::Shell(shell) = &mut structural.elements[1].kind else {
        panic!("shell")
    };
    shell.node_ids[0] = u32::MAX;
    assert!(matches!(
        explicit_shell_surface_geometry(
            &structural,
            &topology,
            &runmat_meshing_core::fixtures::fixture_face("shell_panel")
        ),
        Err(ExplicitShellSurfaceError::DegenerateTriangle { .. })
    ));
}
