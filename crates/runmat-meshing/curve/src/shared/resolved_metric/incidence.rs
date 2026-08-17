use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, ExactEdge, PersistentEntityId};

pub(super) struct TopologyIncidence {
    coedges_by_edge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    faces_by_edge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    wires_by_coedge: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    shells_by_face: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    solids_by_shell: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    regions_by_solid: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    lumps_by_solid: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    bodies_by_lump: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    bodies_by_sheet_shell: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    assemblies_by_body: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    contacts_by_face: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    known_entities: BTreeSet<PersistentEntityId>,
}

impl TopologyIncidence {
    pub fn new(topology: &ExactBRepTopology) -> Self {
        let mut incidence = Self {
            coedges_by_edge: BTreeMap::new(),
            faces_by_edge: BTreeMap::new(),
            wires_by_coedge: BTreeMap::new(),
            shells_by_face: BTreeMap::new(),
            solids_by_shell: BTreeMap::new(),
            regions_by_solid: BTreeMap::new(),
            lumps_by_solid: BTreeMap::new(),
            bodies_by_lump: BTreeMap::new(),
            bodies_by_sheet_shell: BTreeMap::new(),
            assemblies_by_body: BTreeMap::new(),
            contacts_by_face: BTreeMap::new(),
            known_entities: BTreeSet::new(),
        };
        for coedge in &topology.coedges {
            push(&mut incidence.coedges_by_edge, &coedge.edge_id, &coedge.id);
            push(
                &mut incidence.faces_by_edge,
                &coedge.edge_id,
                &coedge.face_id,
            );
        }
        for wire in &topology.wires {
            for coedge_id in &wire.coedge_ids {
                push(&mut incidence.wires_by_coedge, coedge_id, &wire.id);
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

    pub fn incident_entities(&self, edge: &ExactEdge) -> BTreeSet<PersistentEntityId> {
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
        entities.extend(expand(&self.assemblies_by_body, &bodies));
        entities.extend(expand(&self.contacts_by_face, &faces));
        entities
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
