use super::*;
use runmat_analysis_core::{
    ContactFriction, ContactInterfaceModel, FrictionCoefficient, RegionTarget,
};
use runmat_meshing_core::{fixtures, ElementOrder, PersistentEntityKind};
mod cold;
mod reference;
mod resources;
pub(crate) struct Fixture {
    pub(crate) mesh: runmat_meshing_core::SolverMeshArtifact,
    pub(crate) projection: AnalysisProjection,
    pub(crate) interfaces: Vec<AnalysisInterface>,
    pub(crate) rows: Vec<[usize; 3]>,
    pub(crate) positions: Vec<[f64; 3]>,
}
impl Fixture {
    pub(crate) fn new(order: ElementOrder, friction: bool) -> Self {
        let mut mesh = fixtures::three_region_tetrahedron_solver_mesh_with_order(order);
        for node in &mut mesh.topology.nodes {
            if (100..200).contains(&node.node_id) {
                let [x, y, z] = node.coordinates_m.canonical_value();
                node.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([
                    0.2 + 0.2 * (x - 2.),
                    0.4 - 0.2 * y,
                    -0.05 - 0.2 * z,
                ]);
            }
        }
        for (id, name) in [
            (10, "contact-primary"),
            (11, "contact-primary-side"),
            (110, "contact-secondary"),
        ] {
            let face = mesh
                .topology
                .boundary_faces
                .iter_mut()
                .find(|f| f.face_id == id)
                .unwrap();
            let mut entity = face
                .provenance
                .iter()
                .find(|e| e.kind == PersistentEntityKind::Face)
                .unwrap()
                .clone();
            entity.source_topology_id = name.into();
            let original = face
                .provenance
                .iter()
                .find(|e| e.kind == PersistentEntityKind::Face)
                .unwrap()
                .clone();
            let midside = face.node_ids.iter().skip(3).copied().collect::<Vec<_>>();
            face.provenance.push(entity.clone());
            face.provenance.sort();
            face.provenance.dedup();
            for node in mesh
                .topology
                .nodes
                .iter_mut()
                .filter(|n| midside.contains(&n.node_id))
            {
                let mut parameter=node.exact_parameters.iter().find(|p|matches!(p,runmat_meshing_core::SolverNodeExactParameter::Surface{source_face_id,..} if *source_face_id==original)).unwrap().clone();
                if let runmat_meshing_core::SolverNodeExactParameter::Surface {
                    source_face_id,
                    ..
                } = &mut parameter
                {
                    *source_face_id = entity.clone();
                }
                node.exact_parameters.push(parameter);
                runmat_meshing_core::sort_solver_node_exact_parameters(&mut node.exact_parameters);
                node.provenance.push(entity.clone());
                node.provenance.sort();
                node.provenance.dedup();
            }
        }
        mesh.seal_canonical_digest().unwrap();
        mesh.validate().unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let target = |name: &str| {
            let mut target = RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(name)),
            );
            let entity = projection
                .boundary_faces
                .iter()
                .flat_map(|f| &f.provenance)
                .find(|e| e.source_topology_id == name)
                .unwrap()
                .clone();
            target.bind_resolved_entities(vec![entity]).unwrap();
            target
        };
        let interfaces = vec![AnalysisInterface {
            interface_id: "two-body-contact".into(),
            primary: target("contact-primary"),
            secondary: target("contact-secondary"),
            kind: AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                friction: if friction {
                    ContactFriction::Coulomb {
                        coefficient: FrictionCoefficient::new(0.3),
                    }
                } else {
                    ContactFriction::Frictionless
                },
            }),
        }];
        let positions = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let rows = (0..projection.nodes.len())
            .map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        Self {
            mesh,
            projection,
            interfaces,
            rows,
            positions,
        }
    }
    pub(crate) fn control(friction: bool) -> ContactInterfacePolicy {
        ContactInterfacePolicy {
            penalty: ContactPenaltyPolicy {
                normal_stiffness_pa_per_m:
                    runmat_types::physical::TractionDisplacementStiffness::new(10000.),
                tangential_stiffness_pa_per_m: friction.then_some(
                    runmat_types::physical::TractionDisplacementStiffness::new(10000.),
                ),
            },
            pairing: BoundaryPairingPolicy {
                narrow: crate::topology::BoundarySearchPolicy {
                    distance_tolerance_m: 1e-8,
                    maximum_patches: 10000,
                    maximum_depth: 20,
                },
                maximum_distance_m: 0.2,
                distance_separation_m: 1e-6,
                maximum_faces: 4,
                maximum_total_patches: 10000,
            },
            stationary: BoundaryStationaryPolicy {
                differential: crate::topology::BoundaryProjectionDifferentialPolicy {
                    stationarity_tolerance_m: 1e-10,
                    minimum_barycentric: 1e-7,
                    minimum_metric_ratio: 1e-10,
                },
                maximum_iterations: 12,
                maximum_backtracks: 12,
            },
            maximum_initial_penetration_m: 0.,
        }
    }
    pub(crate) fn budget() -> ContactOperatorPolicy {
        ContactOperatorPolicy {
            maximum_dofs: 1000,
            maximum_reference_points: 1000,
            maximum_face_pairs: 16,
            maximum_coordinates: 20000,
            maximum_patches_per_trial: 100000,
            maximum_refinement_evaluations_per_trial: 100000,
            maximum_preparation_resident_bytes: 100_000_000,
            maximum_preparation_scalar_operations: 100_000_000,
        }
    }
    fn operator(&self, friction: bool) -> CanonicalContactOperator<'_> {
        CanonicalContactOperator::prepare(
            &self.projection,
            &self.interfaces,
            &self.rows,
            3 * self.rows.len(),
            &[Self::control(friction)],
            Self::budget(),
        )
        .unwrap()
    }
    fn moved(&self, dz: f64, dx: f64) -> Vec<[f64; 3]> {
        self.positions
            .iter()
            .zip(&self.projection.nodes)
            .map(|(p, n)| {
                if (100..200).contains(&n.node_id) {
                    [p[0] + dx, p[1], p[2] + dz]
                } else {
                    *p
                }
            })
            .collect()
    }
    fn secondary_force(&self, force: &[f64]) -> [f64; 3] {
        std::array::from_fn(|k| {
            self.projection
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| (100..200).contains(&n.node_id))
                .map(|(i, _)| force[self.rows[i][k]])
                .sum()
        })
    }
}
fn near(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} != {b}");
}
