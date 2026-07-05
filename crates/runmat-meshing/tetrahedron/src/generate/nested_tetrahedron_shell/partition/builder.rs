use std::collections::BTreeMap;

use runmat_meshing_core::{
    contracts::{MeshingStage, TopologyEntityId},
    quality::predicate::Point3,
};

use crate::cavity::constrained::ConstrainedCavityNode;

const BARYCENTRIC_KEY_SCALE: f64 = 1.0e12;

#[derive(Debug, Clone)]
pub(super) struct PartitionBuilder {
    pub(super) cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
    pub(super) barycentric_by_id: BTreeMap<u32, [f64; 4]>,
    cavity_id_by_barycentric: BTreeMap<[i64; 4], u32>,
    pub(super) generated_nodes: Vec<ConstrainedCavityNode>,
    outer_points: Vec<Point3>,
    next_cavity_id: u32,
}

impl PartitionBuilder {
    pub(super) fn new(
        cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
        outer_points: Vec<Point3>,
    ) -> Self {
        let next_cavity_id = cavity_id_to_node_id
            .keys()
            .next_back()
            .map(|id| id + 1)
            .unwrap_or(0);
        Self {
            cavity_id_to_node_id,
            barycentric_by_id: BTreeMap::new(),
            cavity_id_by_barycentric: BTreeMap::new(),
            generated_nodes: Vec::new(),
            outer_points,
            next_cavity_id,
        }
    }

    pub(super) fn insert_existing_node(&mut self, barycentric: [f64; 4], cavity_id: u32) {
        self.cavity_id_by_barycentric
            .insert(barycentric_key(barycentric), cavity_id);
        self.barycentric_by_id.insert(cavity_id, barycentric);
    }

    pub(super) fn insert_node(&mut self, barycentric: [f64; 4]) -> u32 {
        let key = barycentric_key(barycentric);
        if let Some(node_id) = self.cavity_id_by_barycentric.get(&key) {
            return *node_id;
        }
        self.insert_generated_node(barycentric)
    }

    pub(super) fn insert_generated_node(&mut self, barycentric: [f64; 4]) -> u32 {
        let key = barycentric_key(barycentric);
        if let Some(node_id) = self.cavity_id_by_barycentric.get(&key) {
            return *node_id;
        }
        let node_id = self.next_cavity_id;
        self.next_cavity_id += 1;
        let coordinates_m = self.point(barycentric);
        self.cavity_id_by_barycentric.insert(key, node_id);
        self.barycentric_by_id.insert(node_id, barycentric);
        self.cavity_id_to_node_id.insert(
            node_id,
            TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("nested_tetrahedron_shell_partition_node_{node_id}"),
            },
        );
        self.generated_nodes.push(ConstrainedCavityNode {
            node_id,
            coordinates_m,
        });
        node_id
    }

    fn point(&self, barycentric: [f64; 4]) -> Point3 {
        let mut point = [0.0; 3];
        for (index, weight) in barycentric.iter().enumerate() {
            for (axis, coordinate) in point.iter_mut().enumerate() {
                *coordinate += weight * self.outer_points[index][axis];
            }
        }
        point
    }

    pub(super) fn coordinates(&self, node_id: u32) -> Point3 {
        self.point(self.barycentric_by_id[&node_id])
    }

    pub(super) fn barycentric_for_topology_node(
        &self,
        node_id: &TopologyEntityId,
    ) -> Option<[f64; 4]> {
        self.cavity_id_to_node_id
            .iter()
            .find_map(|(cavity_id, candidate_id)| {
                (candidate_id == node_id).then(|| self.barycentric_by_id.get(cavity_id).copied())
            })
            .flatten()
    }
}

fn barycentric_key(barycentric: [f64; 4]) -> [i64; 4] {
    barycentric.map(|value| (value * BARYCENTRIC_KEY_SCALE).round() as i64)
}
