use std::collections::{BTreeMap, BTreeSet};

use super::{
    exact_topology_assembly_validation::validate_assembly_occurrences,
    exact_topology_validation_support::*, ExactBRepModel, ExactBRepTopology, GeometryContractError,
    PersistentEntityKind, EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
};

const MAX_TOPOLOGY_ENTITIES: usize = 10_000_000;
const MAX_TOPOLOGY_INCIDENCES: usize = 100_000_000;

impl ExactBRepTopology {
    pub fn validate_against(&self, model: &ExactBRepModel) -> Result<(), GeometryContractError> {
        if self.schema_version != EXACT_BREP_TOPOLOGY_SCHEMA_VERSION {
            return Err(invalid("exact topology schema", "unsupported version"));
        }
        let entity_count = self
            .assemblies
            .len()
            .saturating_add(self.instances.len())
            .saturating_add(self.bodies.len())
            .saturating_add(self.lumps.len())
            .saturating_add(self.solids.len())
            .saturating_add(self.regions.len())
            .saturating_add(self.shells.len())
            .saturating_add(self.faces.len())
            .saturating_add(self.wires.len())
            .saturating_add(self.coedges.len())
            .saturating_add(self.edges.len())
            .saturating_add(self.vertices.len())
            .saturating_add(self.interfaces.len())
            .saturating_add(self.contacts.len());
        if entity_count == 0 || entity_count > MAX_TOPOLOGY_ENTITIES {
            return Err(invalid(
                "exact topology entity count",
                "topology must be non-empty and within the hard entity bound",
            ));
        }
        let incidence_count = self
            .assemblies
            .iter()
            .map(|value| {
                value
                    .body_ids
                    .len()
                    .saturating_add(value.child_instance_ids.len())
            })
            .chain(self.bodies.iter().map(|value| {
                value
                    .lump_ids
                    .len()
                    .saturating_add(value.sheet_shell_ids.len())
            }))
            .chain(self.lumps.iter().map(|value| value.solid_ids.len()))
            .chain(
                self.solids
                    .iter()
                    .map(|value| 1usize.saturating_add(value.void_shell_ids.len())),
            )
            .chain(self.shells.iter().map(|value| value.face_uses.len()))
            .chain(
                self.faces
                    .iter()
                    .map(|value| 1usize.saturating_add(value.inner_wire_ids.len())),
            )
            .chain(self.wires.iter().map(|value| value.coedge_ids.len()))
            .chain(self.contacts.iter().map(|value| {
                value
                    .side_a_face_ids
                    .len()
                    .saturating_add(value.side_b_face_ids.len())
            }))
            .fold(0usize, usize::saturating_add);
        if incidence_count > MAX_TOPOLOGY_INCIDENCES {
            return Err(invalid(
                "exact topology incidence count",
                "topology exceeds the hard aggregate incidence bound",
            ));
        }
        require_count("assembly", self.assemblies.len(), model.assembly_count)?;
        require_count("instance", self.instances.len(), model.instance_count)?;
        require_count("body", self.bodies.len(), model.body_count)?;
        require_count("lump", self.lumps.len(), model.lump_count)?;
        require_count("solid", self.solids.len(), model.solid_count)?;
        require_count("region", self.regions.len(), model.region_count)?;
        require_count("shell", self.shells.len(), model.shell_count)?;
        require_count("face", self.faces.len(), model.face_count)?;
        require_count("wire", self.wires.len(), model.wire_count)?;
        require_count("coedge", self.coedges.len(), model.coedge_count)?;
        require_count("edge", self.edges.len(), model.edge_count)?;
        require_count("vertex", self.vertices.len(), model.vertex_count)?;
        require_count("interface", self.interfaces.len(), model.interface_count)?;
        require_count("contact", self.contacts.len(), model.contact_count)?;

        let assemblies = collect_ids(
            "assemblies",
            PersistentEntityKind::Assembly,
            self.assemblies.iter().map(|value| &value.id),
        )?;
        let instances = collect_ids(
            "instances",
            PersistentEntityKind::Instance,
            self.instances.iter().map(|value| &value.id),
        )?;
        let bodies = collect_ids(
            "bodies",
            PersistentEntityKind::Body,
            self.bodies.iter().map(|value| &value.id),
        )?;
        let lumps = collect_ids(
            "lumps",
            PersistentEntityKind::Lump,
            self.lumps.iter().map(|value| &value.id),
        )?;
        let solids = collect_ids(
            "solids",
            PersistentEntityKind::Solid,
            self.solids.iter().map(|value| &value.id),
        )?;
        let regions = collect_ids(
            "regions",
            PersistentEntityKind::Region,
            self.regions.iter().map(|value| &value.id),
        )?;
        let shells = collect_ids(
            "shells",
            PersistentEntityKind::Shell,
            self.shells.iter().map(|value| &value.id),
        )?;
        let faces = collect_ids(
            "faces",
            PersistentEntityKind::Face,
            self.faces.iter().map(|value| &value.id),
        )?;
        let wires = collect_ids(
            "wires",
            PersistentEntityKind::Wire,
            self.wires.iter().map(|value| &value.id),
        )?;
        let coedges = collect_ids(
            "coedges",
            PersistentEntityKind::Coedge,
            self.coedges.iter().map(|value| &value.id),
        )?;
        let edges = collect_ids(
            "edges",
            PersistentEntityKind::Edge,
            self.edges.iter().map(|value| &value.id),
        )?;
        let vertices = collect_ids(
            "vertices",
            PersistentEntityKind::Vertex,
            self.vertices.iter().map(|value| &value.id),
        )?;
        let face_by_id = self
            .faces
            .iter()
            .map(|face| (&face.id, face))
            .collect::<BTreeMap<_, _>>();
        let coedge_by_id = self
            .coedges
            .iter()
            .map(|coedge| (&coedge.id, coedge))
            .collect::<BTreeMap<_, _>>();
        validate_assembly_occurrences(self, &assemblies, &instances, &bodies)?;

        let mut claimed_lumps = BTreeSet::new();
        let mut claimed_sheet_shells = BTreeSet::new();
        for body in &self.bodies {
            body.mass_properties_evaluator_id.validate()?;
            if body.is_sheet_body == body.sheet_shell_ids.is_empty()
                || (body.is_sheet_body && !body.lump_ids.is_empty())
            {
                return Err(invalid(
                    "body topology",
                    "sheet bodies own sheet shells; solid bodies own lumps",
                ));
            }
            require_ordered_refs(
                "body lumps",
                &body.lump_ids,
                PersistentEntityKind::Lump,
                &lumps,
                !body.is_sheet_body,
            )?;
            require_same_scope("body lumps", &body.id, &body.lump_ids)?;
            claim_unique("lump ownership", &body.lump_ids, &mut claimed_lumps)?;
            require_ordered_refs(
                "body sheet shells",
                &body.sheet_shell_ids,
                PersistentEntityKind::Shell,
                &shells,
                body.is_sheet_body,
            )?;
            require_same_scope("body sheet shells", &body.id, &body.sheet_shell_ids)?;
            claim_unique(
                "sheet shell ownership",
                &body.sheet_shell_ids,
                &mut claimed_sheet_shells,
            )?;
        }
        if claimed_lumps != lumps {
            return Err(invalid(
                "lump ownership",
                "every lump must have one body owner",
            ));
        }
        let mut claimed_solids = BTreeSet::new();
        for lump in &self.lumps {
            require_ordered_refs(
                "lump solids",
                &lump.solid_ids,
                PersistentEntityKind::Solid,
                &solids,
                true,
            )?;
            require_same_scope("lump solids", &lump.id, &lump.solid_ids)?;
            claim_unique("solid ownership", &lump.solid_ids, &mut claimed_solids)?;
        }
        if claimed_solids != solids {
            return Err(invalid(
                "solid ownership",
                "every solid must have one lump owner",
            ));
        }
        let mut region_solids = BTreeSet::new();
        for region in &self.regions {
            require_reference(
                "region solid",
                &region.solid_id,
                PersistentEntityKind::Solid,
                &solids,
            )?;
            require_same_scope(
                "region solid",
                &region.id,
                std::iter::once(&region.solid_id),
            )?;
            if !region_solids.insert(region.solid_id.clone()) {
                return Err(invalid(
                    "region ownership",
                    "a solid has multiple exact regions",
                ));
            }
        }
        if region_solids != solids {
            return Err(invalid(
                "region ownership",
                "every solid must own exactly one exact region",
            ));
        }
        let mut claimed_shells = claimed_sheet_shells;
        for solid in &self.solids {
            require_reference(
                "solid outer shell",
                &solid.outer_shell_id,
                PersistentEntityKind::Shell,
                &shells,
            )?;
            require_same_scope(
                "solid outer shell",
                &solid.id,
                std::iter::once(&solid.outer_shell_id),
            )?;
            if !claimed_shells.insert(solid.outer_shell_id.clone()) {
                return Err(invalid(
                    "shell ownership",
                    "a shell has multiple solid owners",
                ));
            }
            require_ordered_refs(
                "solid void shells",
                &solid.void_shell_ids,
                PersistentEntityKind::Shell,
                &shells,
                false,
            )?;
            require_same_scope("solid void shells", &solid.id, &solid.void_shell_ids)?;
            claim_unique(
                "shell ownership",
                &solid.void_shell_ids,
                &mut claimed_shells,
            )?;
        }
        if claimed_shells != shells {
            return Err(invalid(
                "shell ownership",
                "every shell must have one solid or sheet-body owner",
            ));
        }
        let mut face_use_orientations = BTreeMap::new();
        for shell in &self.shells {
            if shell.face_uses.is_empty() {
                return Err(invalid(
                    "shell faces",
                    "shells must contain oriented face uses",
                ));
            }
            require_ordered_refs(
                "shell faces",
                &shell
                    .face_uses
                    .iter()
                    .map(|value| value.entity_id.clone())
                    .collect::<Vec<_>>(),
                PersistentEntityKind::Face,
                &faces,
                true,
            )?;
            require_same_scope(
                "shell faces",
                &shell.id,
                shell.face_uses.iter().map(|face_use| &face_use.entity_id),
            )?;
            for face_use in &shell.face_uses {
                face_use_orientations
                    .entry(face_use.entity_id.clone())
                    .or_insert_with(Vec::new)
                    .push(face_use.orientation);
            }
        }

        let mut wire_owners = BTreeMap::new();
        for face in &self.faces {
            face.surface_evaluator_id.validate()?;
            face.trim_classifier_id.validate()?;
            require_reference(
                "face outer wire",
                &face.outer_wire_id,
                PersistentEntityKind::Wire,
                &wires,
            )?;
            require_same_scope(
                "face outer wire",
                &face.id,
                std::iter::once(&face.outer_wire_id),
            )?;
            claim_wire(&mut wire_owners, &face.outer_wire_id, &face.id)?;
            require_ordered_refs(
                "face inner wires",
                &face.inner_wire_ids,
                PersistentEntityKind::Wire,
                &wires,
                false,
            )?;
            require_same_scope("face inner wires", &face.id, &face.inner_wire_ids)?;
            for wire in &face.inner_wire_ids {
                claim_wire(&mut wire_owners, wire, &face.id)?;
            }
        }
        if wire_owners.len() != wires.len() {
            return Err(invalid(
                "wire ownership",
                "every wire must have one face owner",
            ));
        }
        let mut claimed_coedges = BTreeSet::new();
        for wire in &self.wires {
            require_canonical_cycle_refs(
                "wire coedges",
                &wire.coedge_ids,
                PersistentEntityKind::Coedge,
                &coedges,
            )?;
            require_same_scope("wire coedges", &wire.id, &wire.coedge_ids)?;
            claim_unique("coedge ownership", &wire.coedge_ids, &mut claimed_coedges)?;
            let owner = wire_owners
                .get(&wire.id)
                .ok_or_else(|| invalid("wire ownership", "wire owner index is incomplete"))?;
            for coedge_id in &wire.coedge_ids {
                let coedge = coedge_by_id.get(coedge_id).ok_or_else(|| {
                    invalid("coedge ownership", "coedge reference index is incomplete")
                })?;
                if &coedge.face_id != owner {
                    return Err(invalid(
                        "coedge face incidence",
                        "coedge face must own the containing wire",
                    ));
                }
            }
        }
        if claimed_coedges != coedges {
            return Err(invalid(
                "coedge ownership",
                "every coedge must have one wire owner",
            ));
        }
        for coedge in &self.coedges {
            require_reference(
                "coedge face",
                &coedge.face_id,
                PersistentEntityKind::Face,
                &faces,
            )?;
            require_same_scope("coedge face", &coedge.id, std::iter::once(&coedge.face_id))?;
            require_reference(
                "coedge edge",
                &coedge.edge_id,
                PersistentEntityKind::Edge,
                &edges,
            )?;
            require_same_scope("coedge edge", &coedge.id, std::iter::once(&coedge.edge_id))?;
            coedge.pcurve_evaluator_id.validate()?;
            if coedge.seam_image.is_some_and(|image| image > 1) {
                return Err(invalid(
                    "coedge seam image",
                    "seam image must be zero or one",
                ));
            }
            if coedge.seam_image.is_some()
                && !face_by_id
                    .get(&coedge.face_id)
                    .is_some_and(|face| face.periodic_u || face.periodic_v)
            {
                return Err(invalid(
                    "coedge seam image",
                    "seam images are only valid on periodic surfaces",
                ));
            }
        }
        for edge in &self.edges {
            edge.curve_evaluator_id.validate()?;
            for vertex in [&edge.start_vertex_id, &edge.end_vertex_id]
                .into_iter()
                .flatten()
            {
                require_reference(
                    "edge vertex",
                    vertex,
                    PersistentEntityKind::Vertex,
                    &vertices,
                )?;
            }
            require_same_scope(
                "edge vertices",
                &edge.id,
                [&edge.start_vertex_id, &edge.end_vertex_id]
                    .into_iter()
                    .flatten(),
            )?;
            if !edge.is_degenerate
                && (edge.start_vertex_id.is_none() || edge.end_vertex_id.is_none())
            {
                return Err(invalid(
                    "edge endpoints",
                    "nondegenerate edges require both endpoint uses",
                ));
            }
            if edge.is_periodic && !edge.is_closed {
                return Err(invalid(
                    "periodic edge",
                    "periodic edges must explicitly be closed",
                ));
            }
        }
        for vertex in &self.vertices {
            if vertex.point_m.iter().any(|value| !value.is_finite())
                || !vertex.tolerance_m.is_finite()
                || vertex.tolerance_m < 0.0
            {
                return Err(invalid(
                    "exact vertex geometry",
                    "point and non-negative tolerance must be finite",
                ));
            }
        }
        validate_interfaces(self, &faces, &regions)?;
        let interface_faces = self
            .interfaces
            .iter()
            .map(|interface| &interface.face_id)
            .collect::<BTreeSet<_>>();
        for face in &faces {
            let expected_uses = usize::from(interface_faces.contains(face)) + 1;
            if face_use_orientations.get(face).map(Vec::len) != Some(expected_uses) {
                return Err(invalid(
                    "shell face incidence",
                    "ordinary faces require one shell use and shared interfaces require two",
                ));
            }
        }
        for interface in &self.interfaces {
            let uses = face_use_orientations
                .get(&interface.face_id)
                .ok_or_else(|| {
                    invalid(
                        "shared interface shell uses",
                        "face-use index is incomplete",
                    )
                })?;
            if !uses.contains(&interface.side_a_orientation)
                || !uses.contains(&interface.side_b_orientation)
            {
                return Err(invalid(
                    "shared interface shell uses",
                    "interface region orientations must match the two shell face uses",
                ));
            }
        }
        validate_contacts(self, &faces)?;
        Ok(())
    }
}
