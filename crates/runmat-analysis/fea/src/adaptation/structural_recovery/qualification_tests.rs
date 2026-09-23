use std::collections::BTreeMap;

use runmat_analysis_core::MaterialElasticityModel;
use runmat_execution::Digest;
use runmat_meshing_core::{
    solver_boundary_face_identity, solver_midside_node_identity, solver_volume_element_identity,
    sort_solver_node_exact_parameters, BoundaryFaceRole, BoundaryTriangleOrder, ElementOrder,
    FieldTopologyLocation, FieldTopologyMap, MeshNeighbor, MeshRegion, PersistentEntityId,
    PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode,
    SolverNodeExactParameter, SolverVolumeElement, StableDigest, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use super::{estimate_structural_recovery_error, StructuralRecoveryEstimatorOptions};
use crate::{
    contracts::{FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS},
    fixtures::{fixture_model, FixtureId},
    topology::AnalysisProjection,
    ComputeBackend, LinearStaticSolveOptions,
};

const TETRAHEDRON_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

#[test]
fn solved_tet10_recovery_estimator_decreases_under_uniform_refinement() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut errors = Vec::new();
    let mut peak_displacements = Vec::new();
    for divisions in [1, 2, 3] {
        let mesh = structured_tet10_cantilever(divisions);
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("the refined physical Tet10 cantilever must solve");
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .expect("the solved field must retain physical integration-point stress");
        peak_displacements.push(
            result
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap()
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max),
        );
        assert_eq!(
            stress.shape,
            vec![mesh.topology.volume_elements.len() * 4, 6]
        );
        errors.push(
            estimate_structural_recovery_error(
                &mesh,
                "elements",
                stress,
                StructuralRecoveryEstimatorOptions::default(),
            )
            .expect("the solved point field must drive the estimator")
            .total_error,
        );
    }

    assert!(
        errors.windows(2).all(|pair| pair[1] < 0.9 * pair[0]),
        "{errors:?}"
    );
    let coarse_change = (peak_displacements[1] - peak_displacements[0]).abs();
    let fine_change = (peak_displacements[2] - peak_displacements[1]).abs();
    assert!(fine_change < 0.25 * coarse_change, "{peak_displacements:?}");
}

#[test]
fn solved_heterogeneous_tet4_stress_matches_an_independent_constitutive_reference() {
    let model = fixture_model(FixtureId::MultiMaterialAssembly);
    let mesh = crate::tests::multi_region_tetrahedron_solver_mesh();
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("the heterogeneous reference must solve");
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let point_stress = result
        .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .unwrap();
    let point_values = point_stress.as_host_f64().unwrap();
    assert_eq!(
        point_stress.shape,
        vec![mesh.topology.volume_elements.len() * 4, 6]
    );

    for (element_index, element) in mesh.topology.volume_elements.iter().enumerate() {
        let indices = element.node_ids[..4]
            .iter()
            .map(|node_id| {
                projection
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *node_id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let coordinates = indices
            .iter()
            .map(|index| projection.nodes[*index].coordinates_m)
            .collect::<Vec<_>>();
        let origin = coordinates[0];
        assert_eq!(coordinates[1], [origin[0] + 1.0, origin[1], origin[2]]);
        assert_eq!(coordinates[2], [origin[0], origin[1] + 1.0, origin[2]]);
        assert_eq!(coordinates[3], [origin[0], origin[1], origin[2] + 1.0]);
        let value = |node: usize, component: usize| displacement[indices[node] * 3 + component];
        let strain = [
            value(1, 0) - value(0, 0),
            value(2, 1) - value(0, 1),
            value(3, 2) - value(0, 2),
            value(3, 1) - value(0, 1) + value(2, 2) - value(0, 2),
            value(3, 0) - value(0, 0) + value(1, 2) - value(0, 2),
            value(2, 0) - value(0, 0) + value(1, 1) - value(0, 1),
        ];
        let assignment = model
            .material_assignments
            .iter()
            .find(|assignment| {
                assignment.selected_region_id()
                    == Some(element.region_id.source_topology_id.as_str())
            })
            .unwrap();
        let material = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .unwrap();
        let MaterialElasticityModel::IsotropicLinear(mechanical) =
            material.elasticity.as_ref().unwrap()
        else {
            panic!("reference fixture must use isotropic elasticity");
        };
        let shear = mechanical.youngs_modulus_pa / (2.0 * (1.0 + mechanical.poisson_ratio));
        let lambda = mechanical.youngs_modulus_pa * mechanical.poisson_ratio
            / ((1.0 + mechanical.poisson_ratio) * (1.0 - 2.0 * mechanical.poisson_ratio));
        let trace = strain[0] + strain[1] + strain[2];
        let expected = [
            lambda * trace + 2.0 * shear * strain[0],
            lambda * trace + 2.0 * shear * strain[1],
            lambda * trace + 2.0 * shear * strain[2],
            shear * strain[3],
            shear * strain[4],
            shear * strain[5],
        ];
        for point in 0..4 {
            let actual = &point_values[(element_index * 4 + point) * 6..][..6];
            for (actual, expected) in actual.iter().zip(expected) {
                let scale = expected.abs().max(1.0);
                assert!((actual - expected).abs() <= 1.0e-11 * scale);
            }
        }
    }

    let estimate = estimate_structural_recovery_error(
        &mesh,
        "elements",
        point_stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate.total_error, 0.0);
}

fn structured_tet10_cantilever(divisions: usize) -> SolverMeshArtifact {
    assert!(divisions > 0);
    let mut artifact =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let solid = entity(PersistentEntityKind::Solid, "cantilever");
    let region = entity(PersistentEntityKind::Region, "region");
    let nodes_per_axis = divisions + 1;
    let mut nodes = Vec::with_capacity(nodes_per_axis.pow(3));
    for x in 0..=divisions {
        for y in 0..=divisions {
            for z in 0..=divisions {
                let index = nodes.len();
                nodes.push(SolverMeshNode {
                    node_id: index as u64 + 1,
                    stable_identity: digest("corner", index as u64),
                    coordinates_m: [
                        x as f64 / divisions as f64,
                        y as f64 / divisions as f64,
                        z as f64 / divisions as f64,
                    ],
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
            }
        }
    }
    let corner_id = |x: usize, y: usize, z: usize| {
        (x * nodes_per_axis * nodes_per_axis + y * nodes_per_axis + z) as u64 + 1
    };
    let mut corner_elements = Vec::with_capacity(divisions.pow(3) * 6);
    for x in 0..divisions {
        for y in 0..divisions {
            for z in 0..divisions {
                let v000 = corner_id(x, y, z);
                let v001 = corner_id(x, y, z + 1);
                let v010 = corner_id(x, y + 1, z);
                let v011 = corner_id(x, y + 1, z + 1);
                let v100 = corner_id(x + 1, y, z);
                let v101 = corner_id(x + 1, y, z + 1);
                let v110 = corner_id(x + 1, y + 1, z);
                let v111 = corner_id(x + 1, y + 1, z + 1);
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
                    corner_elements.push(corners);
                }
            }
        }
    }

    let mut midpoint_by_edge = BTreeMap::<[u64; 2], u64>::new();
    let mut volume_elements = Vec::with_capacity(corner_elements.len());
    for corners in &corner_elements {
        let mut node_ids = corners.to_vec();
        for endpoints in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            let mut edge = endpoints.map(|endpoint| corners[endpoint]);
            edge.sort_unstable();
            let midpoint_id = *midpoint_by_edge.entry(edge).or_insert_with(|| {
                let [left, right] = edge.map(|id| nodes[id as usize - 1].clone());
                let node_id = nodes.len() as u64 + 1;
                nodes.push(SolverMeshNode {
                    node_id,
                    stable_identity: solver_midside_node_identity([
                        left.stable_identity,
                        right.stable_identity,
                    ]),
                    coordinates_m: std::array::from_fn(|axis| {
                        0.5 * (left.coordinates_m[axis] + right.coordinates_m[axis])
                    }),
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
                node_id
            });
            node_ids.push(midpoint_id);
        }
        let element_id = volume_elements.len() as u64 + 1;
        volume_elements.push(SolverVolumeElement {
            element_id,
            stable_identity: solver_volume_element_identity(
                corners.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order: ElementOrder::Tet10,
            node_ids,
            region_id: region.clone(),
            provenance: vec![solid.clone()],
        });
    }

    let mut face_owners = BTreeMap::<[u64; 3], Vec<(u64, u8)>>::new();
    for element in &volume_elements {
        for (local_face, local_nodes) in TETRAHEDRON_FACES.iter().enumerate() {
            let mut face = local_nodes.map(|index| element.node_ids[index]);
            face.sort_unstable();
            face_owners
                .entry(face)
                .or_default()
                .push((element.element_id, local_face as u8));
        }
    }
    let neighbors = volume_elements
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
    let mut boundary_faces = Vec::new();
    for (corners, owners) in face_owners.iter().filter(|(_, owners)| owners.len() == 1) {
        let coordinates = corners.map(|id| nodes[id as usize - 1].coordinates_m);
        let source = if coordinates.iter().all(|point| point[0].abs() <= 1.0e-12) {
            "root"
        } else if coordinates
            .iter()
            .all(|point| (point[0] - 1.0).abs() <= 1.0e-12)
        {
            "tip"
        } else {
            "free_surface"
        };
        let source_face = entity(PersistentEntityKind::Face, source);
        let mut node_ids = corners.to_vec();
        for mut edge in [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[0]],
        ] {
            edge.sort_unstable();
            let midpoint_id = midpoint_by_edge[&edge];
            let midpoint = &mut nodes[midpoint_id as usize - 1];
            midpoint.provenance.push(source_face.clone());
            midpoint.provenance.sort();
            midpoint.provenance.dedup();
            midpoint
                .exact_parameters
                .push(SolverNodeExactParameter::Surface {
                    source_face_id: source_face.clone(),
                    chart_id: digest(source, 0),
                    evaluator_uv: [0.5, 0.5],
                });
            sort_solver_node_exact_parameters(&mut midpoint.exact_parameters);
            midpoint.exact_parameters.dedup();
            node_ids.push(midpoint_id);
        }
        boundary_faces.push(SolverBoundaryFace {
            face_id: boundary_faces.len() as u64 + 1,
            stable_identity: solver_boundary_face_identity(
                corners.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order: BoundaryTriangleOrder::Tri6,
            node_ids,
            adjacent_volume_element_ids: vec![owners[0].0],
            role: BoundaryFaceRole::Exterior,
            provenance: vec![source_face],
        });
    }

    let node_ids = nodes.iter().map(|node| node.node_id).collect::<Vec<_>>();
    let element_ids = volume_elements
        .iter()
        .map(|element| element.element_id)
        .collect::<Vec<_>>();
    let face_ids = boundary_faces
        .iter()
        .map(|face| face.face_id)
        .collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = volume_elements;
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

fn digest(domain: &str, ordinal: u64) -> StableDigest {
    let mut bytes = domain.as_bytes().to_vec();
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    StableDigest::from_bytes(*Digest::sha256(bytes).bytes())
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
