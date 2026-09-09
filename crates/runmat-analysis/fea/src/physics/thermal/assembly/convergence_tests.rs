use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisConstraint, PrescribedValue, RegionTarget, Temperature, ThermalConstraint,
    ThermalConstraintKind,
};
use runmat_meshing_core::{
    solver_boundary_face_identity, solver_volume_element_identity, BoundaryFaceRole,
    BoundaryTriangleOrder, ElementOrder, FieldTopologyLocation, FieldTopologyMap, MeshNeighbor,
    MeshRegion, PersistentEntityId, PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact,
    SolverMeshNode, SolverVolumeElement, StableDigest,
};

use super::assemble;
use crate::{
    excitation::{ExcitationSamplePoint, RealizedExcitations},
    fixtures::{fixture_model, FixtureId},
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_operator_system,
        preconditioner::SpdPreconditionerKind,
    },
};

const TETRAHEDRON_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

#[test]
fn transient_slab_has_second_order_spatial_convergence() {
    let final_time_s = 0.1;
    let time_step_s = final_time_s / 256.0;
    let mut errors = Vec::new();
    let mut midspan_temperatures = Vec::new();

    let refinements = [2, 3, 4];
    for cells in refinements {
        let mesh = slab_mesh(cells);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        model.boundaries.clear();
        model.excitations.clear();
        model.materials[0].thermal.conductivity_w_per_mk = 1.0;
        model.materials[0].thermal.specific_heat_j_per_kgk = 1.0;
        model.materials[0].mechanical.density_kg_per_m3 = 1.0;
        for (id, region) in [
            ("left_temperature", "slab_left"),
            ("right_temperature", "slab_right"),
        ] {
            model
                .constraints
                .push(AnalysisConstraint::Thermal(ThermalConstraint {
                    constraint_id: id.into(),
                    kind: ThermalConstraintKind::Prescribed {
                        target: RegionTarget::boundary(region),
                        value: PrescribedValue::Constant(Temperature::new(300.0)),
                    },
                }));
        }
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let mut temperature = assembly
            .projection
            .nodes
            .iter()
            .map(|node| 300.0 + 20.0 * (std::f64::consts::PI * node.coordinates_m[0]).sin())
            .collect::<Vec<_>>();
        for _ in 0..256 {
            let operator = assembly
                .transient_operator(time_step_s, &temperature)
                .unwrap();
            let solve = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                Some(&temperature),
            );
            assert!(solve.converged);
            temperature = solve.solution;
        }
        let decay = (-std::f64::consts::PI.powi(2) * final_time_s).exp();
        let squared_error = assembly
            .projection
            .nodes
            .iter()
            .zip(&temperature)
            .map(|(node, numerical)| {
                let exact =
                    300.0 + 20.0 * (std::f64::consts::PI * node.coordinates_m[0]).sin() * decay;
                (numerical - exact).powi(2)
            })
            .sum::<f64>();
        errors.push((squared_error / temperature.len() as f64).sqrt());
        midspan_temperatures.push(assembly.projection.nodes.iter().zip(&temperature).find_map(
            |(node, value)| ((node.coordinates_m[0] - 0.5).abs() <= 1.0e-12).then_some(*value),
        ));
    }

    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "spatial errors: {errors:?}; midspan temperatures: {midspan_temperatures:?}"
    );
    let observed_orders = errors
        .windows(2)
        .zip(refinements.windows(2))
        .map(|(error, cells)| (error[0] / error[1]).ln() / (cells[1] as f64 / cells[0] as f64).ln())
        .collect::<Vec<_>>();
    assert!(
        observed_orders.iter().all(|order| *order > 1.3),
        "spatial errors: {errors:?}; observed orders: {observed_orders:?}"
    );
}

fn slab_mesh(cell_count: usize) -> SolverMeshArtifact {
    let mut artifact =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let solid = entity(PersistentEntityKind::Solid, "slab");
    let region = entity(PersistentEntityKind::Region, "region");
    let nodes_per_axis = cell_count + 1;
    let mut nodes = Vec::with_capacity(nodes_per_axis.pow(3));
    for x_index in 0..=cell_count {
        for y_index in 0..=cell_count {
            for z_index in 0..=cell_count {
                let index = nodes.len();
                nodes.push(SolverMeshNode {
                    node_id: index as u64 + 1,
                    stable_identity: node_digest(index),
                    coordinates_m: [
                        x_index as f64 / cell_count as f64,
                        y_index as f64 / cell_count as f64,
                        z_index as f64 / cell_count as f64,
                    ],
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
            }
        }
    }

    let node_id = |x: usize, y: usize, z: usize| {
        (x * nodes_per_axis * nodes_per_axis + y * nodes_per_axis + z) as u64 + 1
    };
    let mut elements = Vec::with_capacity(cell_count.pow(3) * 6);
    for x in 0..cell_count {
        for y in 0..cell_count {
            for z in 0..cell_count {
                let v000 = node_id(x, y, z);
                let v001 = node_id(x, y, z + 1);
                let v010 = node_id(x, y + 1, z);
                let v011 = node_id(x, y + 1, z + 1);
                let v100 = node_id(x + 1, y, z);
                let v101 = node_id(x + 1, y, z + 1);
                let v110 = node_id(x + 1, y + 1, z);
                let v111 = node_id(x + 1, y + 1, z + 1);
                for mut corners in [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ] {
                    if signed_six_volume(corners, &nodes) < 0.0 {
                        corners.swap(1, 2);
                    }
                    let element_id = elements.len() as u64 + 1;
                    elements.push(SolverVolumeElement {
                        element_id,
                        stable_identity: solver_volume_element_identity(
                            corners.map(|id| nodes[id as usize - 1].stable_identity),
                        ),
                        order: ElementOrder::Tet4,
                        node_ids: corners.to_vec(),
                        region_id: region.clone(),
                        provenance: vec![solid.clone()],
                    });
                }
            }
        }
    }

    let mut face_owners = BTreeMap::<[u64; 3], Vec<(u64, u8)>>::new();
    for element in &elements {
        for (local_face, local_nodes) in TETRAHEDRON_FACES.iter().enumerate() {
            let mut face = local_nodes.map(|index| element.node_ids[index]);
            face.sort_unstable();
            face_owners
                .entry(face)
                .or_default()
                .push((element.element_id, local_face as u8));
        }
    }
    let neighbors = elements
        .iter()
        .flat_map(|element| {
            TETRAHEDRON_FACES
                .iter()
                .enumerate()
                .map(|(local_face, local_nodes)| {
                    let mut face = local_nodes.map(|index| element.node_ids[index]);
                    face.sort_unstable();
                    let adjacent_element_id = face_owners[&face]
                        .iter()
                        .find_map(|(owner, _)| (*owner != element.element_id).then_some(*owner));
                    MeshNeighbor {
                        element_id: element.element_id,
                        local_face_index: local_face as u8,
                        adjacent_element_id,
                    }
                })
        })
        .collect::<Vec<_>>();
    let boundary_faces = face_owners
        .iter()
        .filter(|(_, owners)| owners.len() == 1)
        .enumerate()
        .map(|(index, (node_ids, owners))| {
            let x_values = node_ids.map(|id| nodes[id as usize - 1].coordinates_m[0]);
            let source = if x_values.iter().all(|x| x.abs() <= 1.0e-12) {
                "slab_left"
            } else if x_values.iter().all(|x| (*x - 1.0).abs() <= 1.0e-12) {
                "slab_right"
            } else {
                "slab_insulated"
            };
            SolverBoundaryFace {
                face_id: index as u64 + 1,
                stable_identity: solver_boundary_face_identity(
                    node_ids.map(|id| nodes[id as usize - 1].stable_identity),
                ),
                order: BoundaryTriangleOrder::Tri3,
                node_ids: node_ids.to_vec(),
                adjacent_volume_element_ids: vec![owners[0].0],
                role: BoundaryFaceRole::Exterior,
                provenance: vec![entity(PersistentEntityKind::Face, source)],
            }
        })
        .collect::<Vec<_>>();
    let node_ids = nodes.iter().map(|node| node.node_id).collect::<Vec<_>>();
    let element_ids = elements
        .iter()
        .map(|element| element.element_id)
        .collect::<Vec<_>>();
    let face_ids = boundary_faces
        .iter()
        .map(|face| face.face_id)
        .collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = elements;
    artifact.topology.neighbors = neighbors;
    artifact.topology.boundary_faces = boundary_faces;
    artifact.topology.boundary_edges.clear();
    artifact.topology.regions = vec![MeshRegion {
        region_id: region,
        element_ids: element_ids.clone(),
    }];
    artifact.topology.field_topologies = vec![
        field("nodes", FieldTopologyLocation::Node, node_ids),
        field(
            "elements",
            FieldTopologyLocation::VolumeElement,
            element_ids,
        ),
        field("faces", FieldTopologyLocation::BoundaryFace, face_ids),
        field("edges", FieldTopologyLocation::BoundaryEdge, Vec::new()),
    ];
    artifact.resolved_request.resources.maximum_nodes = artifact.topology.nodes.len() as u64;
    artifact.resolved_request.resources.maximum_elements =
        artifact.topology.volume_elements.len() as u64;
    artifact.seal_canonical_digest().unwrap();
    artifact
}

fn signed_six_volume(corners: [u64; 4], nodes: &[SolverMeshNode]) -> f64 {
    let [a, b, c, d] = corners.map(|id| nodes[id as usize - 1].coordinates_m);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
}

fn node_digest(index: usize) -> StableDigest {
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&(index as u64 + 1).to_le_bytes());
    StableDigest::from_bytes(bytes)
}

fn entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source.into(),
        assembly_path: vec!["root".into()],
    }
}

fn field(
    topology_id: &str,
    location: FieldTopologyLocation,
    ordered_entity_ids: Vec<u64>,
) -> FieldTopologyMap {
    FieldTopologyMap {
        topology_id: topology_id.into(),
        location,
        ordered_entity_ids,
    }
}
