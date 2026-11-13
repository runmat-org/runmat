use super::*;
use runmat_analysis_core::{ContactFriction, ContactInterfaceModel};
use runmat_meshing_core::{fixtures, ElementOrder, PersistentEntityKind};
fn interface(projection: &AnalysisProjection) -> AnalysisInterface {
    let target = |index: usize| {
        let entity = projection.boundary_faces[index]
            .provenance
            .iter()
            .find(|e| e.kind == PersistentEntityKind::Face)
            .unwrap();
        RegionTarget::boundary(entity.analysis_selection_id().unwrap())
    };
    AnalysisInterface {
        interface_id: "contact".to_owned(),
        primary: target(0),
        secondary: target(1),
        kind: AnalysisInterfaceKind::Contact(ContactInterfaceModel {
            friction: ContactFriction::Frictionless,
        }),
    }
}
// witness: FEA-CONTACT-OPERATOR-001/surface_search_gap @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Canonical contact surfaces bind original face occurrences and current geometry for
//   both element orders.
#[test]
fn canonical_contact_surfaces_bind_original_face_occurrences_and_current_geometry() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let interface = interface(&projection);
        let plan = CanonicalContactSurfaces::derive(&projection, &interface).unwrap();
        assert_eq!(plan.primary_face_indices(), &[0]);
        assert_eq!(plan.secondary_face_indices(), &[1]);
        assert_eq!(
            CanonicalContactSurfaces::resident_bytes(&projection, &interface).unwrap(),
            2 * std::mem::size_of::<usize>() as u64
        );
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value().map(|p| p + 2.0))
            .collect();
        let search = plan.primary_search_index(&positions, 1).unwrap();
        let face = &projection.boundary_faces[0];
        let n = face.oriented_area_vector_m2;
        let length = n[0].hypot(n[1]).hypot(n[2]);
        let query = std::array::from_fn(|i| face.centroid_m[i] + 2.0 + 0.01 * n[i] / length);
        let matched = search
            .closest_point(
                query,
                crate::topology::BoundaryPairingPolicy {
                    narrow: crate::topology::BoundarySearchPolicy {
                        distance_tolerance_m: 1e-8,
                        maximum_patches: 1000,
                        maximum_depth: 20,
                    },
                    maximum_distance_m: 0.1,
                    distance_separation_m: 1e-6,
                    maximum_faces: 1,
                    maximum_total_patches: 1000,
                },
            )
            .unwrap();
        assert_eq!(matched.face_identity, face.stable_identity);
        assert_eq!(
            plan.primary_snapshot(0, &positions)
                .unwrap()
                .face_identity(),
            projection.boundary_faces[0].stable_identity
        );
    }
}
#[test]
fn empty_and_overlapping_contact_selections_reject_before_execution() {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut interface = interface(&projection);
    interface.secondary = interface.primary.clone();
    assert!(matches!(
        CanonicalContactSurfaces::derive(&projection, &interface),
        Err(ContactSurfaceError::SurfaceConflict)
    ));
    interface.secondary =
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("missing"));
    assert!(matches!(
        CanonicalContactSurfaces::derive(&projection, &interface),
        Err(ContactSurfaceError::EmptySurface)
    ));
}
