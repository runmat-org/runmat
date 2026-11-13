//! Structured multi-region tetrahedral boxes with conformal interfaces and named faces,
//! for analytic electromagnetic checks independent of CAD import.
use runmat_meshing_core::{
    solver_boundary_face_identity, solver_volume_element_identity, BoundaryFaceRole,
    BoundaryTriangleOrder, ElementOrder, FieldTopologyLocation, FieldTopologyMap, MeshNeighbor,
    MeshRegion, PersistentEntityId, PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact,
    SolverMeshNode, SolverVolumeElement, StableDigest,
};
use std::collections::BTreeMap;

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

pub(crate) fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.into(), vec!["root".into()])
}

/// Box [0,xs.last] x [0,ys.last] x [0,zs.last] on the given grid lines. `region` names the
/// body of each cell from its center; `face` names each exterior or interface face from its
/// centroid, unit normal (outward for exterior faces) and owning region names.
pub(crate) fn mesh(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    region: impl Fn([f64; 3]) -> &'static str,
    face: impl Fn([f64; 3], [f64; 3], &[&str]) -> String,
) -> SolverMeshArtifact {
    let mut artifact =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let (nx, ny, nz) = (xs.len(), ys.len(), zs.len());
    let index = |x: usize, y: usize, z: usize| (x * ny + y) * nz + z;
    let mut nodes = Vec::with_capacity(nx * ny * nz);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let i = nodes.len();
                let mut bytes = [0_u8; 32];
                bytes[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
                bytes[31] = 0x5a;
                nodes.push(SolverMeshNode {
                    node_id: i as u64 + 1,
                    stable_identity: StableDigest::from_bytes(bytes),
                    coordinates_m: runmat_types::physical::Quantity::new([xs[x], ys[y], zs[z]]),
                    provenance: Vec::new(),
                    exact_parameters: Vec::new(),
                });
            }
        }
    }
    let mut elements: Vec<SolverVolumeElement> = Vec::new();
    let mut names: Vec<&'static str> = Vec::new();
    for x in 0..nx - 1 {
        for y in 0..ny - 1 {
            for z in 0..nz - 1 {
                let id = |dx, dy, dz| index(x + dx, y + dy, z + dz) as u64 + 1;
                let center = [
                    0.5 * (xs[x] + xs[x + 1]),
                    0.5 * (ys[y] + ys[y + 1]),
                    0.5 * (zs[z] + zs[z + 1]),
                ];
                let name = region(center);
                let (v000, v001, v010, v011) = (id(0, 0, 0), id(0, 0, 1), id(0, 1, 0), id(0, 1, 1));
                let (v100, v101, v110, v111) = (id(1, 0, 0), id(1, 0, 1), id(1, 1, 0), id(1, 1, 1));
                for mut corners in [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ] {
                    if six_volume(corners, &nodes) < 0.0 {
                        corners.swap(1, 2);
                    }
                    elements.push(SolverVolumeElement {
                        element_id: elements.len() as u64 + 1,
                        stable_identity: solver_volume_element_identity(
                            corners.map(|id| nodes[id as usize - 1].stable_identity),
                        ),
                        order: ElementOrder::Tet4,
                        node_ids: corners.to_vec(),
                        region_id: entity(PersistentEntityKind::Region, name),
                        provenance: vec![entity(PersistentEntityKind::Solid, name)],
                    });
                    names.push(name);
                }
            }
        }
    }
    // Each node belongs to the bodies of the elements that use it.
    for element in &elements {
        for &node in &element.node_ids {
            nodes[node as usize - 1]
                .provenance
                .extend(element.provenance.iter().cloned());
        }
    }
    for node in &mut nodes {
        node.provenance.sort();
        node.provenance.dedup();
    }
    let mut owners = BTreeMap::<[u64; 3], Vec<(u64, usize)>>::new();
    for (e, element) in elements.iter().enumerate() {
        for local in FACES {
            let mut key = local.map(|i| element.node_ids[i]);
            key.sort_unstable();
            owners.entry(key).or_default().push((element.element_id, e));
        }
    }
    let neighbors = elements
        .iter()
        .flat_map(|element| {
            let owners = &owners;
            FACES.iter().enumerate().map(move |(local, nodes)| {
                let mut key = nodes.map(|i| element.node_ids[i]);
                key.sort_unstable();
                MeshNeighbor {
                    element_id: element.element_id,
                    local_face_index: local as u8,
                    adjacent_element_id: owners[&key]
                        .iter()
                        .find_map(|(owner, _)| (*owner != element.element_id).then_some(*owner)),
                }
            })
        })
        .collect::<Vec<_>>();
    let mut boundary_faces = Vec::new();
    let mut interfaces = BTreeMap::<(String, &str, &str), Vec<u64>>::new();
    for (key, adjacent) in &owners {
        let regions: Vec<&str> = adjacent.iter().map(|(_, e)| names[*e]).collect();
        let role = match adjacent.len() {
            1 => BoundaryFaceRole::Exterior,
            2 if regions[0] != regions[1] => BoundaryFaceRole::ConformalInterface,
            _ => continue,
        };
        let points = key.map(|id| nodes[id as usize - 1].coordinates_m);
        let centroid: [f64; 3] = std::array::from_fn(|a| {
            (points[0].canonical_value()[a]
                + points[1].canonical_value()[a]
                + points[2].canonical_value()[a])
                / 3.0
        });
        let u: [f64; 3] = std::array::from_fn(|a| {
            points[1].canonical_value()[a] - points[0].canonical_value()[a]
        });
        let v: [f64; 3] = std::array::from_fn(|a| {
            points[2].canonical_value()[a] - points[0].canonical_value()[a]
        });
        let mut normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let length = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        normal = normal.map(|n| n / length);
        // Orient outward from the first owner.
        let owner = &elements[adjacent[0].1];
        let inside: [f64; 3] = std::array::from_fn(|a| {
            owner
                .node_ids
                .iter()
                .map(|id| nodes[*id as usize - 1].coordinates_m.canonical_value()[a])
                .sum::<f64>()
                / 4.0
        });
        if (0..3)
            .map(|a| normal[a] * (centroid[a] - inside[a]))
            .sum::<f64>()
            < 0.0
        {
            normal = normal.map(|n| -n);
        }
        let name = face(centroid, normal, &regions);
        if role == BoundaryFaceRole::ConformalInterface {
            let (a, b) = if regions[0] < regions[1] {
                (regions[0], regions[1])
            } else {
                (regions[1], regions[0])
            };
            interfaces
                .entry((name.clone(), a, b))
                .or_default()
                .push(boundary_faces.len() as u64 + 1);
        }
        boundary_faces.push(SolverBoundaryFace {
            face_id: boundary_faces.len() as u64 + 1,
            stable_identity: solver_boundary_face_identity(
                key.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order: BoundaryTriangleOrder::Tri3,
            node_ids: key.to_vec(),
            adjacent_volume_element_ids: adjacent.iter().map(|(id, _)| *id).collect(),
            role,
            provenance: vec![entity(PersistentEntityKind::Face, &name)],
        });
    }
    let mut regions = BTreeMap::<&str, Vec<u64>>::new();
    for (element, name) in elements.iter().zip(&names) {
        regions.entry(name).or_default().push(element.element_id);
    }
    let node_ids = nodes.iter().map(|n| n.node_id).collect::<Vec<_>>();
    let element_ids = elements.iter().map(|e| e.element_id).collect::<Vec<_>>();
    let face_ids = boundary_faces.iter().map(|f| f.face_id).collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = elements;
    artifact.topology.neighbors = neighbors;
    artifact.topology.boundary_faces = boundary_faces;
    artifact.topology.boundary_edges.clear();
    let mut conformal = interfaces
        .into_iter()
        .map(
            |((name, a, b), boundary_face_ids)| runmat_meshing_core::ConformalInterface {
                source_face_id: entity(PersistentEntityKind::Face, &name),
                side_a_region_id: entity(PersistentEntityKind::Region, a),
                side_b_region_id: entity(PersistentEntityKind::Region, b),
                boundary_face_ids,
            },
        )
        .collect::<Vec<_>>();
    conformal.sort_by(|x, y| {
        (&x.source_face_id, &x.side_a_region_id, &x.side_b_region_id).cmp(&(
            &y.source_face_id,
            &y.side_a_region_id,
            &y.side_b_region_id,
        ))
    });
    artifact.topology.conformal_interfaces = conformal;
    artifact.topology.contacts.clear();
    artifact.topology.regions = regions
        .into_iter()
        .map(|(name, element_ids)| MeshRegion {
            region_id: entity(PersistentEntityKind::Region, name),
            element_ids,
        })
        .collect();
    artifact.topology.field_topologies = vec![
        topology("nodes", FieldTopologyLocation::Node, node_ids),
        topology(
            "elements",
            FieldTopologyLocation::VolumeElement,
            element_ids,
        ),
        topology("faces", FieldTopologyLocation::BoundaryFace, face_ids),
        topology("edges", FieldTopologyLocation::BoundaryEdge, Vec::new()),
    ];
    artifact.resolved_request.limits.maximum_nodes = artifact.topology.nodes.len() as u64;
    artifact.resolved_request.limits.maximum_elements =
        artifact.topology.volume_elements.len() as u64;
    artifact.seal_canonical_digest().unwrap();
    artifact
}

fn topology(id: &str, location: FieldTopologyLocation, ids: Vec<u64>) -> FieldTopologyMap {
    FieldTopologyMap {
        topology_id: id.into(),
        location,
        ordered_entity_ids: ids,
    }
}

fn six_volume(corners: [u64; 4], nodes: &[SolverMeshNode]) -> f64 {
    let p = corners.map(|id| nodes[id as usize - 1].coordinates_m);
    let a: [f64; 3] =
        std::array::from_fn(|i| p[1].canonical_value()[i] - p[0].canonical_value()[i]);
    let b: [f64; 3] =
        std::array::from_fn(|i| p[2].canonical_value()[i] - p[0].canonical_value()[i]);
    let c: [f64; 3] =
        std::array::from_fn(|i| p[3].canonical_value()[i] - p[0].canonical_value()[i]);
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Evenly spaced grid lines.
pub(crate) fn lines(start: f64, end: f64, cells: usize) -> Vec<f64> {
    (0..=cells)
        .map(|i| start + (end - start) * i as f64 / cells as f64)
        .collect()
}
