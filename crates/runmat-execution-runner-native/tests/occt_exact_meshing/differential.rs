use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_geometry_io::{import_geometry, GeometryImportOptions};
use runmat_meshing_core::{BoundaryTriangleOrder, ElementOrder, SolverMeshArtifact};

use super::{execute_box, BOX};

const EXPECTED_BOX_AREA: f64 = 22.0;
const EXPECTED_BOX_VOLUME: f64 = 6.0;
const EXPECTED_LINEAR_ENERGY: f64 = 84.0;
const TOLERANCE: f64 = 1.0e-10;

#[test]
fn runmat_and_independent_meshers_agree_on_box_invariants() {
    let (topology, result) = execute_box();
    let artifact = &result.artifact;
    let oracle = import_geometry("box.brep", BOX, GeometryImportOptions::default()).unwrap();
    let oracle_surface = oracle.asset.surface_meshes.first().unwrap();

    assert_watertight(&oracle_surface.vertices, &oracle_surface.triangles);
    let runmat_coordinates = solver_coordinates(artifact);
    let runmat_triangles = runmat_boundary_triangles(artifact);
    assert_watertight(&runmat_coordinates, &runmat_triangles);
    assert_eq!(oracle_surface.triangles.len(), 12);
    assert_eq!(artifact.topology.boundary_faces.len(), 16);
    assert_eq!(
        bounds(&oracle_surface.vertices),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    );
    assert_eq!(
        bounds(
            &artifact
                .topology
                .nodes
                .iter()
                .map(|node| node.coordinates_m)
                .collect::<Vec<_>>(),
        ),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    );

    let oracle_areas = triangle_areas(&oracle_surface.vertices, &oracle_surface.triangles);
    let runmat_areas = triangle_areas(&runmat_coordinates, &runmat_triangles);
    assert_close(oracle_areas.iter().sum(), EXPECTED_BOX_AREA);
    assert_close(runmat_areas.iter().sum(), EXPECTED_BOX_AREA);
    let area_median_ratio = median(&runmat_areas) / median(&oracle_areas);
    assert!((0.25..=4.0).contains(&area_median_ratio));

    assert_close(
        surface_volume(&oracle_surface.vertices, &oracle_surface.triangles),
        EXPECTED_BOX_VOLUME,
    );
    assert_close(solver_volume(artifact), EXPECTED_BOX_VOLUME);
    assert_close(reference_box_volume(), EXPECTED_BOX_VOLUME);
    assert_close(solver_linear_energy(artifact), EXPECTED_LINEAR_ENERGY);
    assert_close(reference_box_linear_energy(), EXPECTED_LINEAR_ENERGY);

    assert!(maximum_box_surface_deviation(&oracle_surface.vertices) <= TOLERANCE);
    let boundary_node_ids = artifact
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().take(3).copied())
        .collect::<BTreeSet<_>>();
    let boundary_coordinates = artifact
        .topology
        .nodes
        .iter()
        .filter(|node| boundary_node_ids.contains(&node.node_id))
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    assert!(maximum_box_surface_deviation(&boundary_coordinates) <= TOLERANCE);

    let exact_edges = topology
        .edges
        .iter()
        .map(|edge| edge.id.clone())
        .collect::<BTreeSet<_>>();
    let recovered_edges = provenance_by_kind(artifact, PersistentEntityKind::Edge);
    assert!(exact_edges.is_subset(&recovered_edges));
    let exact_faces = topology
        .faces
        .iter()
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    let recovered_faces = provenance_by_kind(artifact, PersistentEntityKind::Face);
    assert_eq!(recovered_faces, exact_faces);
}

fn solver_coordinates(artifact: &SolverMeshArtifact) -> Vec<[f64; 3]> {
    artifact
        .topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect()
}

fn runmat_boundary_triangles(artifact: &SolverMeshArtifact) -> Vec<[u32; 3]> {
    let indices = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, u32::try_from(index).unwrap()))
        .collect::<BTreeMap<_, _>>();
    artifact
        .topology
        .boundary_faces
        .iter()
        .map(|face| {
            assert_eq!(face.order, BoundaryTriangleOrder::Tri3);
            [
                indices[&face.node_ids[0]],
                indices[&face.node_ids[1]],
                indices[&face.node_ids[2]],
            ]
        })
        .collect()
}

fn provenance_by_kind(
    artifact: &SolverMeshArtifact,
    kind: PersistentEntityKind,
) -> BTreeSet<PersistentEntityId> {
    artifact
        .topology
        .boundary_edges
        .iter()
        .flat_map(|edge| &edge.provenance)
        .chain(
            artifact
                .topology
                .boundary_faces
                .iter()
                .flat_map(|face| &face.provenance),
        )
        .filter(|id| id.kind == kind)
        .cloned()
        .collect()
}

fn assert_watertight(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) {
    let mut incidence = BTreeMap::<[[u64; 3]; 2], usize>::new();
    for triangle in triangles {
        for indices in [
            [triangle[0], triangle[1]],
            [triangle[1], triangle[2]],
            [triangle[2], triangle[0]],
        ] {
            let pair = indices.map(|index| vertices[index as usize].map(f64::to_bits));
            let edge = if pair[0] < pair[1] {
                pair
            } else {
                [pair[1], pair[0]]
            };
            *incidence.entry(edge).or_default() += 1;
        }
    }
    assert!(!incidence.is_empty());
    assert!(incidence.values().all(|count| *count == 2));
}

fn triangle_areas(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Vec<f64> {
    triangles
        .iter()
        .map(|triangle| {
            let a = vertices[triangle[0] as usize];
            let b = vertices[triangle[1] as usize];
            let c = vertices[triangle[2] as usize];
            0.5 * norm(cross(subtract(b, a), subtract(c, a)))
        })
        .collect()
}

fn surface_volume(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles
        .iter()
        .map(|triangle| {
            let a = vertices[triangle[0] as usize];
            let b = vertices[triangle[1] as usize];
            let c = vertices[triangle[2] as usize];
            dot(a, cross(b, c)) / 6.0
        })
        .sum::<f64>()
        .abs()
}

fn solver_volume(artifact: &SolverMeshArtifact) -> f64 {
    solver_tetrahedra(artifact).map(tetra_volume).sum()
}

fn solver_linear_energy(artifact: &SolverMeshArtifact) -> f64 {
    solver_tetrahedra(artifact).map(tetra_energy).sum()
}

fn solver_tetrahedra(artifact: &SolverMeshArtifact) -> std::vec::IntoIter<[[f64; 3]; 4]> {
    let coordinates = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    artifact
        .topology
        .volume_elements
        .iter()
        .map(|element| {
            assert_eq!(element.order, ElementOrder::Tet4);
            std::array::from_fn(|index| coordinates[&element.node_ids[index]])
        })
        .collect::<Vec<_>>()
        .into_iter()
}

fn reference_box_tetrahedra() -> impl Iterator<Item = [[f64; 3]; 4]> {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 3.0],
        [0.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ];
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ]
    .into_iter()
    .map(move |tetrahedron| std::array::from_fn(|index| vertices[tetrahedron[index]]))
}

fn reference_box_volume() -> f64 {
    reference_box_tetrahedra().map(tetra_volume).sum()
}

fn reference_box_linear_energy() -> f64 {
    reference_box_tetrahedra().map(tetra_energy).sum()
}

fn tetra_volume(points: [[f64; 3]; 4]) -> f64 {
    dot(
        subtract(points[1], points[0]),
        cross(
            subtract(points[2], points[0]),
            subtract(points[3], points[0]),
        ),
    )
    .abs()
        / 6.0
}

fn tetra_energy(points: [[f64; 3]; 4]) -> f64 {
    let e1 = subtract(points[1], points[0]);
    let e2 = subtract(points[2], points[0]);
    let e3 = subtract(points[3], points[0]);
    let values = points.map(|point| point[0] + 2.0 * point[1] + 3.0 * point[2]);
    let differences = [
        values[1] - values[0],
        values[2] - values[0],
        values[3] - values[0],
    ];
    let determinant = dot(e1, cross(e2, e3));
    let gradient = scale(
        add(
            add(
                scale(cross(e2, e3), differences[0]),
                scale(cross(e3, e1), differences[1]),
            ),
            scale(cross(e1, e2), differences[2]),
        ),
        1.0 / determinant,
    );
    tetra_volume(points) * dot(gradient, gradient)
}

fn bounds(vertices: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex[axis]);
            maximum[axis] = maximum[axis].max(vertex[axis]);
        }
    }
    (minimum, maximum)
}

fn maximum_box_surface_deviation(vertices: &[[f64; 3]]) -> f64 {
    vertices
        .iter()
        .map(|point| {
            [
                point[0],
                1.0 - point[0],
                point[1],
                2.0 - point[1],
                point[2],
                3.0 - point[2],
            ]
            .into_iter()
            .map(f64::abs)
            .reduce(f64::min)
            .unwrap()
        })
        .reduce(f64::max)
        .unwrap()
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "expected {expected}, got {actual}"
    );
}
