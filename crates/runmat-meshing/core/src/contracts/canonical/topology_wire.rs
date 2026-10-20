//! Compact canonical wire form of a solver-mesh topology.
//!
//! The in-memory topology repeats geometry entity identities on every node,
//! element and boundary entity, and a 32-byte identity serializes as 32 Serde
//! integers. On the wire, entities live once in a sorted table and records
//! refer to them by index; identities are four big-endian words; per-record
//! fields become columns. Serialization borrows the topology and allocates
//! only the entity table. Decoding rebuilds the identical in-memory value.
use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::artifact_types::{
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, ConformalInterface, ContactPair,
    FieldTopologyMap, MeshNeighbor, MeshRegion, SolverBoundaryEdge, SolverBoundaryFace,
    SolverMeshNode, SolverMeshTopology, SolverNodeExactParameter, SolverVolumeElement,
};
use super::{ElementOrder, PersistentEntityId, StableDigest};

type Words = [u64; 4];

fn words(digest: &StableDigest) -> Words {
    std::array::from_fn(|word| {
        u64::from_be_bytes(digest.0[word * 8..word * 8 + 8].try_into().expect("eight bytes"))
    })
}

fn digest(words: &Words) -> StableDigest {
    let mut bytes = [0; 32];
    for (word, value) in words.iter().enumerate() {
        bytes[word * 8..word * 8 + 8].copy_from_slice(&value.to_be_bytes());
    }
    StableDigest(bytes)
}

/// Serializes the items an iterator factory yields, without collecting them.
struct Each<F>(F);
impl<F, I> Serialize for Each<F>
where
    F: Fn() -> I,
    I: ExactSizeIterator,
    I::Item: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((self.0)())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum ExactWire {
    #[serde(rename = "c")]
    Curve(u32, f64),
    #[serde(rename = "s")]
    Surface(u32, Words, [f64; 2]),
}

struct EntityTable<'a> {
    entities: Vec<&'a PersistentEntityId>,
    index: BTreeMap<&'a PersistentEntityId, u32>,
}
impl<'a> EntityTable<'a> {
    fn new(topology: &'a SolverMeshTopology) -> Self {
        let mut index = BTreeMap::new();
        let mut note = |entity: &'a PersistentEntityId| {
            index.entry(entity).or_insert(0);
        };
        for node in &topology.nodes {
            node.provenance.iter().for_each(&mut note);
            for parameter in &node.exact_parameters {
                match parameter {
                    SolverNodeExactParameter::Curve { source_edge_id, .. } => note(source_edge_id),
                    SolverNodeExactParameter::Surface { source_face_id, .. } => {
                        note(source_face_id)
                    }
                }
            }
        }
        for element in &topology.volume_elements {
            note(&element.region_id);
            element.provenance.iter().for_each(&mut note);
        }
        for face in &topology.boundary_faces {
            face.provenance.iter().for_each(&mut note);
        }
        for edge in &topology.boundary_edges {
            edge.provenance.iter().for_each(&mut note);
        }
        let entities = index.keys().copied().collect::<Vec<_>>();
        for (position, entity) in entities.iter().enumerate() {
            index.insert(entity, position as u32);
        }
        Self { entities, index }
    }
    fn of(&self, entity: &PersistentEntityId) -> u32 {
        self.index[entity]
    }
    fn all<'b>(&'b self, entities: &'b [PersistentEntityId]) -> impl Serialize + 'b {
        Each(move || entities.iter().map(|entity| self.of(entity)))
    }
}

impl Serialize for SolverMeshTopology {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.nodes.len() > u32::MAX as usize {
            return Err(S::Error::custom("solver topology exceeds its wire index range"));
        }
        let table = EntityTable::new(self);
        let t = &table;
        let nodes = &self.nodes;
        let elements = &self.volume_elements;
        let faces = &self.boundary_faces;
        let edges = &self.boundary_edges;
        let neighbors = &self.neighbors;

        #[derive(Serialize)]
        struct Nodes<A, B, C, D, E> {
            node_ids: A,
            stable_identities: B,
            coordinates_m: C,
            provenance: D,
            exact_parameters: E,
        }
        #[derive(Serialize)]
        struct Elements<A, B, C, D, E, F> {
            element_ids: A,
            stable_identities: B,
            orders: C,
            node_ids: D,
            region_ids: E,
            provenance: F,
        }
        #[derive(Serialize)]
        struct Neighbors<A, B, C> {
            element_ids: A,
            local_face_indices: B,
            adjacent_element_ids: C,
        }
        #[derive(Serialize)]
        struct Faces<A, B, C, D, E, F, G> {
            face_ids: A,
            stable_identities: B,
            orders: C,
            node_ids: D,
            adjacent_volume_element_ids: E,
            roles: F,
            provenance: G,
        }
        #[derive(Serialize)]
        struct Edges<A, B, C, D, E, F> {
            edge_ids: A,
            stable_identities: B,
            orders: C,
            node_ids: D,
            adjacent_boundary_face_ids: E,
            provenance: F,
        }
        #[derive(Serialize)]
        struct Wire<'a, N, V, M, F, E> {
            entities: &'a [&'a PersistentEntityId],
            nodes: N,
            volume_elements: V,
            neighbors: M,
            boundary_faces: F,
            boundary_edges: E,
            regions: &'a [MeshRegion],
            conformal_interfaces: &'a [ConformalInterface],
            contacts: &'a [ContactPair],
            field_topologies: &'a [FieldTopologyMap],
        }
        Wire {
            entities: &t.entities,
            nodes: Nodes {
                node_ids: Each(|| nodes.iter().map(|node| node.node_id)),
                stable_identities: Each(|| nodes.iter().map(|node| words(&node.stable_identity))),
                coordinates_m: Each(|| nodes.iter().map(|node| node.coordinates_m)),
                provenance: Each(|| nodes.iter().map(|node| t.all(&node.provenance))),
                exact_parameters: Each(|| {
                    nodes.iter().map(|node| {
                        Each(move || {
                            node.exact_parameters.iter().map(|parameter| match parameter {
                                SolverNodeExactParameter::Curve {
                                    source_edge_id,
                                    parameter,
                                } => ExactWire::Curve(t.of(source_edge_id), *parameter),
                                SolverNodeExactParameter::Surface {
                                    source_face_id,
                                    chart_id,
                                    evaluator_uv,
                                } => ExactWire::Surface(
                                    t.of(source_face_id),
                                    words(chart_id),
                                    *evaluator_uv,
                                ),
                            })
                        })
                    })
                }),
            },
            volume_elements: Elements {
                element_ids: Each(|| elements.iter().map(|element| element.element_id)),
                stable_identities: Each(|| {
                    elements.iter().map(|element| words(&element.stable_identity))
                }),
                orders: Each(|| elements.iter().map(|element| element.order)),
                node_ids: Each(|| elements.iter().map(|element| &element.node_ids)),
                region_ids: Each(|| elements.iter().map(|element| t.of(&element.region_id))),
                provenance: Each(|| elements.iter().map(|element| t.all(&element.provenance))),
            },
            neighbors: Neighbors {
                element_ids: Each(|| neighbors.iter().map(|neighbor| neighbor.element_id)),
                local_face_indices: Each(|| {
                    neighbors.iter().map(|neighbor| neighbor.local_face_index)
                }),
                adjacent_element_ids: Each(|| {
                    neighbors.iter().map(|neighbor| neighbor.adjacent_element_id)
                }),
            },
            boundary_faces: Faces {
                face_ids: Each(|| faces.iter().map(|face| face.face_id)),
                stable_identities: Each(|| faces.iter().map(|face| words(&face.stable_identity))),
                orders: Each(|| faces.iter().map(|face| face.order)),
                node_ids: Each(|| faces.iter().map(|face| &face.node_ids)),
                adjacent_volume_element_ids: Each(|| {
                    faces.iter().map(|face| &face.adjacent_volume_element_ids)
                }),
                roles: Each(|| faces.iter().map(|face| face.role)),
                provenance: Each(|| faces.iter().map(|face| t.all(&face.provenance))),
            },
            boundary_edges: Edges {
                edge_ids: Each(|| edges.iter().map(|edge| edge.edge_id)),
                stable_identities: Each(|| edges.iter().map(|edge| words(&edge.stable_identity))),
                orders: Each(|| edges.iter().map(|edge| edge.order)),
                node_ids: Each(|| edges.iter().map(|edge| &edge.node_ids)),
                adjacent_boundary_face_ids: Each(|| {
                    edges.iter().map(|edge| &edge.adjacent_boundary_face_ids)
                }),
                provenance: Each(|| edges.iter().map(|edge| t.all(&edge.provenance))),
            },
            regions: &self.regions,
            conformal_interfaces: &self.conformal_interfaces,
            contacts: &self.contacts,
            field_topologies: &self.field_topologies,
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodesWire {
    node_ids: Vec<u64>,
    stable_identities: Vec<Words>,
    coordinates_m: Vec<[f64; 3]>,
    provenance: Vec<Vec<u32>>,
    exact_parameters: Vec<Vec<ExactWire>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ElementsWire {
    element_ids: Vec<u64>,
    stable_identities: Vec<Words>,
    orders: Vec<ElementOrder>,
    node_ids: Vec<Vec<u64>>,
    region_ids: Vec<u32>,
    provenance: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NeighborsWire {
    element_ids: Vec<u64>,
    local_face_indices: Vec<u8>,
    adjacent_element_ids: Vec<Option<u64>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FacesWire {
    face_ids: Vec<u64>,
    stable_identities: Vec<Words>,
    orders: Vec<BoundaryTriangleOrder>,
    node_ids: Vec<Vec<u64>>,
    adjacent_volume_element_ids: Vec<Vec<u64>>,
    roles: Vec<BoundaryFaceRole>,
    provenance: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgesWire {
    edge_ids: Vec<u64>,
    stable_identities: Vec<Words>,
    orders: Vec<BoundaryEdgeOrder>,
    node_ids: Vec<Vec<u64>>,
    adjacent_boundary_face_ids: Vec<Vec<u64>>,
    provenance: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TopologyWire {
    entities: Vec<PersistentEntityId>,
    nodes: NodesWire,
    volume_elements: ElementsWire,
    neighbors: NeighborsWire,
    boundary_faces: FacesWire,
    boundary_edges: EdgesWire,
    regions: Vec<MeshRegion>,
    conformal_interfaces: Vec<ConformalInterface>,
    contacts: Vec<ContactPair>,
    field_topologies: Vec<FieldTopologyMap>,
}

fn same_lengths(lengths: &[usize]) -> Result<(), String> {
    if lengths.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("solver topology wire columns differ in length".into());
    }
    Ok(())
}

impl TopologyWire {
    fn into_topology(self) -> Result<SolverMeshTopology, String> {
        let entities = self.entities;
        if entities.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("solver topology entity table is not canonical".into());
        }
        let entity = |index: &u32| {
            entities
                .get(*index as usize)
                .cloned()
                .ok_or_else(|| "solver topology references an absent entity".to_string())
        };
        let resolve = |indices: Vec<u32>| indices.iter().map(entity).collect::<Result<Vec<_>, _>>();

        let n = self.nodes;
        same_lengths(&[
            n.node_ids.len(),
            n.stable_identities.len(),
            n.coordinates_m.len(),
            n.provenance.len(),
            n.exact_parameters.len(),
        ])?;
        let nodes = n
            .node_ids
            .into_iter()
            .zip(n.stable_identities)
            .zip(n.coordinates_m)
            .zip(n.provenance)
            .zip(n.exact_parameters)
            .map(|((((node_id, identity), coordinates_m), provenance), exact)| {
                Ok(SolverMeshNode {
                    node_id,
                    stable_identity: digest(&identity),
                    coordinates_m,
                    provenance: resolve(provenance)?,
                    exact_parameters: exact
                        .into_iter()
                        .map(|parameter| {
                            Ok(match parameter {
                                ExactWire::Curve(edge, parameter) => {
                                    SolverNodeExactParameter::Curve {
                                        source_edge_id: entity(&edge)?,
                                        parameter,
                                    }
                                }
                                ExactWire::Surface(face, chart, evaluator_uv) => {
                                    SolverNodeExactParameter::Surface {
                                        source_face_id: entity(&face)?,
                                        chart_id: digest(&chart),
                                        evaluator_uv,
                                    }
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let e = self.volume_elements;
        same_lengths(&[
            e.element_ids.len(),
            e.stable_identities.len(),
            e.orders.len(),
            e.node_ids.len(),
            e.region_ids.len(),
            e.provenance.len(),
        ])?;
        let volume_elements = e
            .element_ids
            .into_iter()
            .zip(e.stable_identities)
            .zip(e.orders)
            .zip(e.node_ids)
            .zip(e.region_ids)
            .zip(e.provenance)
            .map(
                |(((((element_id, identity), order), node_ids), region), provenance)| {
                    Ok(SolverVolumeElement {
                        element_id,
                        stable_identity: digest(&identity),
                        order,
                        node_ids,
                        region_id: entity(&region)?,
                        provenance: resolve(provenance)?,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let m = self.neighbors;
        same_lengths(&[
            m.element_ids.len(),
            m.local_face_indices.len(),
            m.adjacent_element_ids.len(),
        ])?;
        let neighbors = m
            .element_ids
            .into_iter()
            .zip(m.local_face_indices)
            .zip(m.adjacent_element_ids)
            .map(
                |((element_id, local_face_index), adjacent_element_id)| MeshNeighbor {
                    element_id,
                    local_face_index,
                    adjacent_element_id,
                },
            )
            .collect();

        let f = self.boundary_faces;
        same_lengths(&[
            f.face_ids.len(),
            f.stable_identities.len(),
            f.orders.len(),
            f.node_ids.len(),
            f.adjacent_volume_element_ids.len(),
            f.roles.len(),
            f.provenance.len(),
        ])?;
        let boundary_faces = f
            .face_ids
            .into_iter()
            .zip(f.stable_identities)
            .zip(f.orders)
            .zip(f.node_ids)
            .zip(f.adjacent_volume_element_ids)
            .zip(f.roles)
            .zip(f.provenance)
            .map(
                |((((((face_id, identity), order), node_ids), adjacent), role), provenance)| {
                    Ok(SolverBoundaryFace {
                        face_id,
                        stable_identity: digest(&identity),
                        order,
                        node_ids,
                        adjacent_volume_element_ids: adjacent,
                        role,
                        provenance: resolve(provenance)?,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let g = self.boundary_edges;
        same_lengths(&[
            g.edge_ids.len(),
            g.stable_identities.len(),
            g.orders.len(),
            g.node_ids.len(),
            g.adjacent_boundary_face_ids.len(),
            g.provenance.len(),
        ])?;
        let boundary_edges = g
            .edge_ids
            .into_iter()
            .zip(g.stable_identities)
            .zip(g.orders)
            .zip(g.node_ids)
            .zip(g.adjacent_boundary_face_ids)
            .zip(g.provenance)
            .map(
                |(((((edge_id, identity), order), node_ids), adjacent), provenance)| {
                    Ok(SolverBoundaryEdge {
                        edge_id,
                        stable_identity: digest(&identity),
                        order,
                        node_ids,
                        adjacent_boundary_face_ids: adjacent,
                        provenance: resolve(provenance)?,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        Ok(SolverMeshTopology {
            nodes,
            volume_elements,
            neighbors,
            boundary_faces,
            boundary_edges,
            regions: self.regions,
            conformal_interfaces: self.conformal_interfaces,
            contacts: self.contacts,
            field_topologies: self.field_topologies,
        })
    }
}

impl<'de> Deserialize<'de> for SolverMeshTopology {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        TopologyWire::deserialize(deserializer)?
            .into_topology()
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_words_round_trip_every_byte() {
        let original = StableDigest(std::array::from_fn(|byte| (byte * 7 + 3) as u8));
        assert_eq!(digest(&words(&original)), original);
    }
}
