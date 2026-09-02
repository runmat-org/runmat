use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient2d, orient3d, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

use super::{
    error, resource, DelaunayPointSetError, DelaunayPointSetErrorKind, DelaunayPointSetOptions,
    DelaunayVolumeNode,
};

pub(super) struct ConvexHullSeed {
    pub(super) nodes: Vec<DelaunayVolumeNode>,
    pub(super) tetrahedra: Vec<[u32; 4]>,
    pub(super) identities: BTreeSet<StableDigest>,
}

pub(super) fn convex_hull_seed(
    nodes: &[DelaunayVolumeNode],
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ConvexHullSeed, DelaunayPointSetError> {
    let mut work = HullWork::new(options, cancellation);
    let initial = initial_tetrahedron(nodes, &mut work)?;
    let interior = std::array::from_fn(|axis| {
        initial
            .iter()
            .map(|index| nodes[*index].coordinates_m[axis] * 0.25)
            .sum()
    });
    let mut faces = initial_faces(initial);
    for point_index in 0..nodes.len() {
        if initial.contains(&point_index) {
            continue;
        }
        let mut visible = Vec::new();
        for face in &faces {
            if face_visible(*face, point_index, interior, nodes, &mut work)? {
                visible.push(*face);
            }
        }
        if visible.is_empty() {
            continue;
        }
        let mut edge_uses = BTreeMap::<[usize; 2], usize>::new();
        for face in &visible {
            for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
                edge.sort_unstable();
                *edge_uses.entry(edge).or_default() += 1;
            }
        }
        for face in visible {
            faces.remove(&face);
        }
        for (edge, uses) in edge_uses {
            if uses != 1 {
                continue;
            }
            let mut face = [edge[0], edge[1], point_index];
            face.sort_unstable();
            if face_sign(face, interior, nodes, &mut work)? != PredicateSign::Zero {
                faces.insert(face);
            }
        }
    }
    build_seed(nodes, faces, initial[0], &mut work)
}

fn initial_tetrahedron(
    nodes: &[DelaunayVolumeNode],
    work: &mut HullWork<'_>,
) -> Result<[usize; 4], DelaunayPointSetError> {
    let first = nodes[0].coordinates_m;
    let second = nodes[1].coordinates_m;
    let mut third = None;
    for (index, node) in nodes.iter().enumerate().skip(2) {
        let point = node.coordinates_m;
        let projections = [
            [
                [first[0], first[1]],
                [second[0], second[1]],
                [point[0], point[1]],
            ],
            [
                [first[1], first[2]],
                [second[1], second[2]],
                [point[1], point[2]],
            ],
            [
                [first[2], first[0]],
                [second[2], second[0]],
                [point[2], point[0]],
            ],
        ];
        let mut noncollinear = false;
        for projection in projections {
            noncollinear |= work.orient2d(projection)? != PredicateSign::Zero;
        }
        if noncollinear {
            third = Some(index);
            break;
        }
    }
    let third = third.ok_or_else(|| {
        error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "all point-set nodes are collinear",
        )
    })?;
    for index in (third + 1)..nodes.len() {
        if work.orient3d([
            first,
            second,
            nodes[third].coordinates_m,
            nodes[index].coordinates_m,
        ])? != PredicateSign::Zero
        {
            return Ok([0, 1, third, index]);
        }
    }
    Err(error(
        DelaunayPointSetErrorKind::InsufficientDimension,
        "all point-set nodes are coplanar",
    ))
}

fn initial_faces(tetrahedron: [usize; 4]) -> BTreeSet<[usize; 3]> {
    (0..4)
        .map(|opposite| {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            face
        })
        .collect()
}

fn face_visible(
    face: [usize; 3],
    point_index: usize,
    interior: [f64; 3],
    nodes: &[DelaunayVolumeNode],
    work: &mut HullWork<'_>,
) -> Result<bool, DelaunayPointSetError> {
    let interior_sign = face_sign(face, interior, nodes, work)?;
    if interior_sign == PredicateSign::Zero {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull interior witness is coplanar with a candidate hull face",
        ));
    }
    let point_sign = face_sign(face, nodes[point_index].coordinates_m, nodes, work)?;
    Ok(point_sign != PredicateSign::Zero && point_sign != interior_sign)
}

fn face_sign(
    face: [usize; 3],
    point: [f64; 3],
    nodes: &[DelaunayVolumeNode],
    work: &mut HullWork<'_>,
) -> Result<PredicateSign, DelaunayPointSetError> {
    work.orient3d([
        nodes[face[0]].coordinates_m,
        nodes[face[1]].coordinates_m,
        nodes[face[2]].coordinates_m,
        point,
    ])
}

fn build_seed(
    nodes: &[DelaunayVolumeNode],
    faces: BTreeSet<[usize; 3]>,
    apex: usize,
    work: &mut HullWork<'_>,
) -> Result<ConvexHullSeed, DelaunayPointSetError> {
    let mut full_tetrahedra = Vec::new();
    for face in faces {
        if face.contains(&apex) {
            continue;
        }
        let tetrahedron = [face[0], face[1], face[2], apex];
        if work.orient3d(tetrahedron.map(|index| nodes[index].coordinates_m))?
            != PredicateSign::Zero
        {
            full_tetrahedra.push(tetrahedron);
        }
    }
    if full_tetrahedra.is_empty() {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull star did not produce a finite tetrahedron",
        ));
    }
    let identities = full_tetrahedra
        .iter()
        .flat_map(|tetrahedron| tetrahedron.iter().map(|index| nodes[*index].identity))
        .collect::<BTreeSet<_>>();
    let mut seed_nodes = nodes
        .iter()
        .copied()
        .filter(|node| identities.contains(&node.identity))
        .collect::<Vec<_>>();
    seed_nodes.sort_by_key(|node| node.identity);
    let seed_indices = seed_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let tetrahedra = full_tetrahedra
        .into_iter()
        .map(|tetrahedron| tetrahedron.map(|index| seed_indices[&nodes[index].identity]))
        .collect();
    Ok(ConvexHullSeed {
        nodes: seed_nodes,
        tetrahedra,
        identities,
    })
}

struct HullWork<'a> {
    options: DelaunayPointSetOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicates: u64,
}

impl<'a> HullWork<'a> {
    fn new(
        options: DelaunayPointSetOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            predicates: 0,
        }
    }

    fn checkpoint(&mut self) -> Result<(), DelaunayPointSetError> {
        self.predicates = self
            .predicates
            .checked_add(1)
            .ok_or_else(|| resource("convex-hull predicate count overflow"))?;
        if self.predicates > self.options.insertion.maximum_predicate_evaluations {
            return Err(resource("convex-hull predicate limit exceeded"));
        }
        if self
            .predicates
            .is_multiple_of(self.options.insertion.topology.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        Ok(())
    }

    fn orient2d(&mut self, points: [[f64; 2]; 3]) -> Result<PredicateSign, DelaunayPointSetError> {
        self.checkpoint()?;
        orient2d(points).map_err(|failure| predicate_error("planar", failure))
    }

    fn orient3d(&mut self, points: [[f64; 3]; 4]) -> Result<PredicateSign, DelaunayPointSetError> {
        self.checkpoint()?;
        orient3d(points).map_err(|failure| predicate_error("spatial", failure))
    }
}

fn predicate_error(kind: &str, failure: impl std::fmt::Debug) -> DelaunayPointSetError {
    error(
        DelaunayPointSetErrorKind::InvalidTopology,
        format!("convex-hull {kind} predicate rejected admitted coordinates: {failure:?}"),
    )
}
