use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, ExactEdge, ExactFace, PersistentEntityId};

pub struct TopologyMetricIncidence {
    coedges_by_edge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    coedges_by_face: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    coedges_by_wire: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    edges_by_coedge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    vertices_by_edge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    faces_by_edge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    wires_by_coedge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    shells_by_face: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    solids_by_shell: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    regions_by_solid: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    lumps_by_solid: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    bodies_by_lump: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    bodies_by_sheet_shell: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    assemblies_by_body: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    instances_by_assembly: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    parents_by_assembly: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    contacts_by_face: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    known_entities: BTreeSet<PersistentEntityId>,
}

impl TopologyMetricIncidence {
    pub fn new(topology: &ExactBRepTopology) -> Self {
        let mut incidence = Self {
            coedges_by_edge: BTreeMap::new(),
            coedges_by_face: BTreeMap::new(),
            coedges_by_wire: BTreeMap::new(),
            edges_by_coedge: BTreeMap::new(),
            vertices_by_edge: BTreeMap::new(),
            faces_by_edge: BTreeMap::new(),
            wires_by_coedge: BTreeMap::new(),
            shells_by_face: BTreeMap::new(),
            solids_by_shell: BTreeMap::new(),
            regions_by_solid: BTreeMap::new(),
            lumps_by_solid: BTreeMap::new(),
            bodies_by_lump: BTreeMap::new(),
            bodies_by_sheet_shell: BTreeMap::new(),
            assemblies_by_body: BTreeMap::new(),
            instances_by_assembly: BTreeMap::new(),
            parents_by_assembly: BTreeMap::new(),
            contacts_by_face: BTreeMap::new(),
            known_entities: BTreeSet::new(),
        };
        for coedge in &topology.coedges {
            push(&mut incidence.coedges_by_edge, &coedge.edge_id, &coedge.id);
            push(&mut incidence.coedges_by_face, &coedge.face_id, &coedge.id);
            push(&mut incidence.edges_by_coedge, &coedge.id, &coedge.edge_id);
            push(
                &mut incidence.faces_by_edge,
                &coedge.edge_id,
                &coedge.face_id,
            );
        }
        for wire in &topology.wires {
            for coedge_id in &wire.coedge_ids {
                push(&mut incidence.wires_by_coedge, coedge_id, &wire.id);
                push(&mut incidence.coedges_by_wire, &wire.id, coedge_id);
            }
        }
        for edge in &topology.edges {
            for vertex_id in edge.start_vertex_id.iter().chain(&edge.end_vertex_id) {
                push(&mut incidence.vertices_by_edge, &edge.id, vertex_id);
            }
        }
        for shell in &topology.shells {
            for face_use in &shell.face_uses {
                push(
                    &mut incidence.shells_by_face,
                    &face_use.entity_id,
                    &shell.id,
                );
            }
        }
        for solid in &topology.solids {
            push(
                &mut incidence.solids_by_shell,
                &solid.outer_shell_id,
                &solid.id,
            );
            for shell_id in &solid.void_shell_ids {
                push(&mut incidence.solids_by_shell, shell_id, &solid.id);
            }
        }
        for region in &topology.regions {
            push(
                &mut incidence.regions_by_solid,
                &region.solid_id,
                &region.id,
            );
        }
        for lump in &topology.lumps {
            for solid_id in &lump.solid_ids {
                push(&mut incidence.lumps_by_solid, solid_id, &lump.id);
            }
        }
        for body in &topology.bodies {
            for lump_id in &body.lump_ids {
                push(&mut incidence.bodies_by_lump, lump_id, &body.id);
            }
            for shell_id in &body.sheet_shell_ids {
                push(&mut incidence.bodies_by_sheet_shell, shell_id, &body.id);
            }
        }
        for assembly in &topology.assemblies {
            for body_id in &assembly.body_ids {
                push(&mut incidence.assemblies_by_body, body_id, &assembly.id);
            }
        }
        for instance in &topology.instances {
            push(
                &mut incidence.instances_by_assembly,
                &instance.instantiated_assembly_id,
                &instance.id,
            );
            push(
                &mut incidence.parents_by_assembly,
                &instance.instantiated_assembly_id,
                &instance.parent_assembly_id,
            );
        }
        for contact in &topology.contacts {
            for face_id in contact
                .side_a_face_ids
                .iter()
                .chain(&contact.side_b_face_ids)
            {
                push(&mut incidence.contacts_by_face, face_id, &contact.id);
            }
        }
        incidence.collect_known(topology);
        incidence
    }

    pub fn knows(&self, entity_id: &PersistentEntityId) -> bool {
        self.known_entities.contains(entity_id)
    }

    pub fn incident_edge_entities(&self, edge: &ExactEdge) -> BTreeSet<PersistentEntityId> {
        let mut entities = BTreeSet::from([edge.id.clone()]);
        entities.extend(edge.start_vertex_id.iter().cloned());
        entities.extend(edge.end_vertex_id.iter().cloned());
        let coedges = values(&self.coedges_by_edge, &edge.id);
        entities.extend(coedges.iter().cloned());
        let faces = values(&self.faces_by_edge, &edge.id);
        entities.extend(faces.iter().cloned());
        let wires = expand(&self.wires_by_coedge, &coedges);
        entities.extend(wires);
        let shells = expand(&self.shells_by_face, &faces);
        entities.extend(shells.iter().cloned());
        let solids = expand(&self.solids_by_shell, &shells);
        entities.extend(solids.iter().cloned());
        entities.extend(expand(&self.regions_by_solid, &solids));
        let lumps = expand(&self.lumps_by_solid, &solids);
        entities.extend(lumps.iter().cloned());
        let mut bodies = expand(&self.bodies_by_lump, &lumps);
        bodies.extend(expand(&self.bodies_by_sheet_shell, &shells));
        entities.extend(bodies.iter().cloned());
        let assemblies = expand(&self.assemblies_by_body, &bodies);
        entities.extend(assemblies.iter().cloned());
        self.extend_assembly_ancestry(&assemblies, &mut entities);
        entities.extend(expand(&self.contacts_by_face, &faces));
        entities
    }

    pub fn incident_face_entities(&self, face: &ExactFace) -> BTreeSet<PersistentEntityId> {
        let mut entities = BTreeSet::from([face.id.clone()]);
        let wires = std::iter::once(&face.outer_wire_id)
            .chain(&face.inner_wire_ids)
            .cloned()
            .collect::<Vec<_>>();
        entities.extend(wires.iter().cloned());
        let mut coedges = values(&self.coedges_by_face, &face.id);
        coedges.extend(expand(&self.coedges_by_wire, &wires));
        coedges.sort();
        coedges.dedup();
        entities.extend(coedges.iter().cloned());
        let edges = expand(&self.edges_by_coedge, &coedges);
        entities.extend(edges.iter().cloned());
        entities.extend(expand(&self.vertices_by_edge, &edges));
        let faces = vec![face.id.clone()];
        let shells = expand(&self.shells_by_face, &faces);
        entities.extend(shells.iter().cloned());
        let solids = expand(&self.solids_by_shell, &shells);
        entities.extend(solids.iter().cloned());
        entities.extend(expand(&self.regions_by_solid, &solids));
        let lumps = expand(&self.lumps_by_solid, &solids);
        entities.extend(lumps.iter().cloned());
        let mut bodies = expand(&self.bodies_by_lump, &lumps);
        bodies.extend(expand(&self.bodies_by_sheet_shell, &shells));
        bodies.sort();
        bodies.dedup();
        entities.extend(bodies.iter().cloned());
        let assemblies = expand(&self.assemblies_by_body, &bodies);
        entities.extend(assemblies.iter().cloned());
        self.extend_assembly_ancestry(&assemblies, &mut entities);
        entities.extend(expand(&self.contacts_by_face, &faces));
        entities
    }

    pub fn edge_adjacency(
        topology: &ExactBRepTopology,
    ) -> BTreeMap<PersistentEntityId, BTreeSet<PersistentEntityId>> {
        let mut adjacency = topology
            .edges
            .iter()
            .map(|edge| (edge.id.clone(), BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let mut edges_by_vertex =
            BTreeMap::<PersistentEntityId, BTreeSet<PersistentEntityId>>::new();
        for edge in &topology.edges {
            for vertex in edge.start_vertex_id.iter().chain(&edge.end_vertex_id) {
                edges_by_vertex
                    .entry(vertex.clone())
                    .or_default()
                    .insert(edge.id.clone());
            }
        }
        for edges in edges_by_vertex.into_values() {
            for edge in &edges {
                adjacency
                    .get_mut(edge)
                    .expect("topology edge inventory")
                    .extend(edges.iter().filter(|neighbor| *neighbor != edge).cloned());
            }
        }
        adjacency
    }

    pub fn face_adjacency(
        topology: &ExactBRepTopology,
    ) -> BTreeMap<PersistentEntityId, BTreeSet<PersistentEntityId>> {
        let mut adjacency = topology
            .faces
            .iter()
            .map(|face| (face.id.clone(), BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let mut faces_by_edge = BTreeMap::<PersistentEntityId, BTreeSet<PersistentEntityId>>::new();
        for coedge in &topology.coedges {
            faces_by_edge
                .entry(coedge.edge_id.clone())
                .or_default()
                .insert(coedge.face_id.clone());
        }
        for faces in faces_by_edge.into_values() {
            for face in &faces {
                adjacency
                    .get_mut(face)
                    .expect("topology face inventory")
                    .extend(faces.iter().filter(|neighbor| *neighbor != face).cloned());
            }
        }
        adjacency
    }

    fn extend_assembly_ancestry(
        &self,
        assemblies: &[PersistentEntityId],
        entities: &mut BTreeSet<PersistentEntityId>,
    ) {
        let mut pending = assemblies.to_vec();
        while let Some(assembly) = pending.pop() {
            entities.extend(values(&self.instances_by_assembly, &assembly));
            for parent in values(&self.parents_by_assembly, &assembly) {
                if entities.insert(parent.clone()) {
                    pending.push(parent);
                }
            }
        }
    }

    fn collect_known(&mut self, topology: &ExactBRepTopology) {
        self.known_entities
            .insert(topology.root_assembly_id.clone());
        self.known_entities
            .extend(topology.assemblies.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.instances.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.bodies.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.lumps.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.solids.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.regions.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.shells.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.faces.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.wires.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.coedges.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.edges.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.vertices.iter().map(|value| value.id.clone()));
        self.known_entities
            .extend(topology.contacts.iter().map(|value| value.id.clone()));
    }
}

fn push(
    index: &mut BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    key: &PersistentEntityId,
    value: &PersistentEntityId,
) {
    let values = index.entry(key.clone()).or_default();
    if !values.contains(value) {
        values.push(value.clone());
        values.sort();
    }
}

fn values(
    index: &BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    key: &PersistentEntityId,
) -> Vec<PersistentEntityId> {
    index.get(key).cloned().unwrap_or_default()
}

fn expand(
    index: &BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    keys: &[PersistentEntityId],
) -> Vec<PersistentEntityId> {
    keys.iter()
        .flat_map(|key| index.get(key).into_iter().flatten().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests;
