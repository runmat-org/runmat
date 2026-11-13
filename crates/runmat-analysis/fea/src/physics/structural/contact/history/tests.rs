use super::*;
use crate::topology::BoundarySearchPolicy;
use runmat_analysis_core::{
    AnalysisInterfaceKind, ContactFriction, ContactInterfaceModel, FrictionCoefficient,
    RegionTarget,
};
use runmat_meshing_core::{fixtures, ElementOrder, PersistentEntityKind};
fn policy() -> BoundaryProjectionDifferentialPolicy {
    BoundaryProjectionDifferentialPolicy {
        stationarity_tolerance_m: 1e-10,
        minimum_barycentric: 1e-7,
        minimum_metric_ratio: 1e-10,
    }
}
fn search() -> BoundarySearchPolicy {
    BoundarySearchPolicy {
        distance_tolerance_m: 1e-10,
        maximum_patches: 1000,
        maximum_depth: 20,
    }
}
fn near(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} != {b}");
}
struct Fixture {
    projection: AnalysisProjection,
    interface: AnalysisInterface,
    positions: Vec<[f64; 3]>,
    primary: usize,
    secondary: usize,
    barycentric: [f64; 3],
}
impl Fixture {
    fn new(order: ElementOrder, coefficient: f64) -> Self {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let mut selected = None;
        for primary in 0..projection.boundary_faces.len() {
            for secondary in 0..projection.boundary_faces.len() {
                if primary == secondary {
                    continue;
                }
                let query = projection
                    .boundary_surface_snapshot(secondary, &positions)
                    .unwrap()
                    .evaluate([1.0 / 3.0; 3])
                    .unwrap()
                    .position_m;
                let point = projection
                    .boundary_surface_snapshot(primary, &positions)
                    .unwrap()
                    .closest_point(query, search())
                    .unwrap();
                if point.barycentric.iter().all(|x| *x > 0.05) {
                    selected = Some((primary, secondary, point.barycentric));
                    break;
                }
            }
            if selected.is_some() {
                break;
            }
        }
        let (primary, secondary, barycentric) = selected.unwrap();
        let target = |index: usize| {
            RegionTarget::boundary(
                projection.boundary_faces[index]
                    .provenance
                    .iter()
                    .find(|p| p.kind == PersistentEntityKind::Face)
                    .unwrap()
                    .analysis_selection_id()
                    .unwrap(),
            )
        };
        let interface = AnalysisInterface {
            interface_id: "point-contact".into(),
            primary: target(primary),
            secondary: target(secondary),
            kind: AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                friction: ContactFriction::Coulomb {
                    coefficient: FrictionCoefficient::new(coefficient),
                },
            }),
        };
        Self {
            projection,
            interface,
            positions,
            primary,
            secondary,
            barycentric,
        }
    }
    fn law(&self) -> PenaltyContactLaw {
        let AnalysisInterfaceKind::Contact(model) = &self.interface.kind else {
            unreachable!()
        };
        PenaltyContactLaw::derive(
            model,
            ContactPenaltyPolicy {
                normal_stiffness_pa_per_m:
                    runmat_types::physical::TractionDisplacementStiffness::new(10000.0),
                tangential_stiffness_pa_per_m: Some(
                    runmat_types::physical::TractionDisplacementStiffness::new(10000.0),
                ),
            },
        )
        .unwrap()
    }
    fn footprint(&self, binding: &ContactPointBinding<'_>, positions: &[[f64; 3]]) -> [f64; 3] {
        let query = self
            .projection
            .boundary_surface_snapshot(self.secondary, positions)
            .unwrap()
            .evaluate(binding.reference.reference_barycentric())
            .unwrap()
            .position_m;
        self.projection
            .boundary_surface_snapshot(self.primary, positions)
            .unwrap()
            .closest_point(query, search())
            .unwrap()
            .barycentric
    }
    fn force(&self, trial: &ContactPointTrial) -> Vec<[f64; 3]> {
        let mut force = vec![[0.0; 3]; self.positions.len()];
        for (&i, f) in self.projection.boundary_faces[self.primary]
            .node_indices
            .iter()
            .zip(trial.primary_internal_force_n())
        {
            for (out, value) in force[i].iter_mut().zip(f) {
                *out += value;
            }
        }
        for (&i, f) in self.projection.boundary_faces[self.secondary]
            .node_indices
            .iter()
            .zip(trial.secondary_internal_force_n())
        {
            for (out, value) in force[i].iter_mut().zip(f) {
                *out += value;
            }
        }
        force
    }
}
// witness: FEA-CONTACT-OPERATOR-001/history_search_diagnostics @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Original point, frame and ordinal cold revalidation reject tampering for both orders;
//   search diagnostics for initial penetration are in the operator resource tests.
#[test]
fn original_point_frame_and_ordinal_cold_revalidation_reject_tampering() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, 0.3);
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
        let inventory = ContactReferenceInventory::derive(&surfaces, 1000).unwrap();
        let binding = ContactPointBinding::derive(&inventory, 0, f.law(), policy()).unwrap();
        let original = binding
            .initial_history(f.primary, f.barycentric, &f.positions)
            .unwrap();
        let wire = serde_json::to_value(&original).unwrap();
        let cold: ContactMaterialPointHistory = serde_json::from_value(wire.clone()).unwrap();
        binding.validate_history(&cold, 0, &f.positions).unwrap();
        assert_eq!(cold, original);
        assert!(binding.validate_history(&cold, 1, &f.positions).is_err());
        for (key, value) in [
            (
                "scope",
                serde_json::to_value(Digest::sha256(b"foreign")).unwrap(),
            ),
            ("primary_barycentric", serde_json::json!([1.0, 0.0, 0.0])),
            (
                "relative_material_vector_m",
                serde_json::json!([1.0, 2.0, 3.0]),
            ),
            ("schema", serde_json::json!(2)),
        ] {
            let mut bad = wire.clone();
            bad[key] = value;
            let bad: ContactMaterialPointHistory = serde_json::from_value(bad).unwrap();
            assert!(binding.validate_history(&bad, 0, &f.positions).is_err());
        }
        let mut bad = wire.clone();
        bad["frame"]["axes"][0][0] = serde_json::json!(4.0);
        let bad: ContactMaterialPointHistory = serde_json::from_value(bad).unwrap();
        assert!(binding.validate_history(&bad, 0, &f.positions).is_err());
        let mut moved = f.positions.clone();
        moved[0][0] += 0.001;
        assert!(binding.validate_history(&cold, 0, &moved).is_err());
        let trial = binding
            .trial(&cold, 0, &f.positions, &f.positions, f.barycentric)
            .unwrap();
        assert_eq!(trial.history().ordinal(), 1);
        assert_eq!(original, cold);
        binding
            .validate_history(trial.history(), 1, &f.positions)
            .unwrap();
    }
}
#[test]
fn objective_contact_history_has_zero_rigid_slip_work_force_and_moment_defects() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, 0.3);
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
        let inventory = ContactReferenceInventory::derive(&surfaces, 1000).unwrap();
        let binding = ContactPointBinding::derive(&inventory, 0, f.law(), policy()).unwrap();
        let accepted = binding
            .initial_history(f.primary, f.barycentric, &f.positions)
            .unwrap();
        let map = |p: [f64; 3]| [2.0 + p[2], -3.0 + p[0], 4.0 + p[1]];
        let moved: Vec<_> = f.positions.iter().copied().map(map).collect();
        let trial = binding
            .trial(&accepted, 0, &f.positions, &moved, f.barycentric)
            .unwrap();
        near(
            trial
                .point_response()
                .frictional_dissipation_increment_j_per_m2
                .canonical_value(),
            0.0,
            1e-10,
        );
        for q in trial
            .point_response()
            .trial_state
            .tangential_traction_pa()
            .canonical_value()
        {
            near(q, 0.0, 1e-10);
        }
        let force = f.force(&trial);
        let mut resultant = [0.0; 3];
        let mut moment = [0.0; 3];
        let mut work = 0.0;
        for (p, force) in moved.iter().zip(force) {
            for k in 0..3 {
                resultant[k] += force[k];
            }
            moment[0] += p[1] * force[2] - p[2] * force[1];
            moment[1] += p[2] * force[0] - p[0] * force[2];
            moment[2] += p[0] * force[1] - p[1] * force[0];
            let velocity = [-p[1], p[0], 0.0];
            work += velocity.iter().zip(force).map(|(v, f)| v * f).sum::<f64>();
        }
        for x in resultant.into_iter().chain(moment) {
            near(x, 0.0, 1e-9);
        }
        near(work, 0.0, 1e-9);
    }
}

mod tangent;
