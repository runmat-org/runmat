use std::collections::BTreeMap;

use runmat_geometry_core::{
    BodyMassProperties, ExactBRepModel, ExactBRepTopology, ExactCoedge, ExactEdge,
    ExactEvaluatorRegistry, ExactFace, ExactGeometryCapabilities, ExactLump, ExactShell,
    ExactSolid, ExactVertex, ExactWire, GeometryDigest, GeometryObjectRef, OrientedEntityUse,
    PersistentEntityKind, EXACT_BREP_MEDIA_TYPE, EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use super::{
    exact_persistent_names::PersistentNameIndex,
    exact_projection_evaluators::project_evaluators,
    exact_projection_identity::*,
    exact_projection_interfaces::project_regions_and_interfaces,
    exact_projection_occurrence::{BodyPartitions, OccurrenceIndex},
    ffi::bridge,
};
use crate::{exact::exact_representation_digest, import::GeometryImportError};

pub(super) struct ExactProjection {
    pub topology: ExactBRepTopology,
    pub evaluators: ExactEvaluatorRegistry,
    pub model: ExactBRepModel,
    pub kernel_body_shapes: BTreeMap<runmat_geometry_core::MassPropertiesEvaluatorId, Vec<u64>>,
}

pub(super) fn project_exact_contracts(
    payload: &bridge::OcctExactShapePayload,
    meters_per_source_unit: f64,
    solid_mass_properties: Option<&BodyMassProperties>,
) -> Result<ExactProjection, GeometryImportError> {
    let representation_digest = exact_representation_digest(&payload.representation);
    validate_projection_shape(payload)?;
    let occurrences = OccurrenceIndex::new(payload, representation_digest)?;
    let names = PersistentNameIndex::new(payload)?;
    let body_partitions = BodyPartitions::new(payload, &occurrences, &names)?;
    let assembly_projection = occurrences.project_assemblies(&body_partitions)?;

    let mut topology = ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: assembly_projection.root_id,
        assemblies: assembly_projection.assemblies,
        instances: assembly_projection.instances,
        bodies: Vec::new(),
        lumps: Vec::new(),
        solids: Vec::new(),
        regions: Vec::new(),
        shells: Vec::new(),
        faces: Vec::new(),
        wires: Vec::new(),
        coedges: Vec::new(),
        edges: Vec::new(),
        vertices: Vec::new(),
        interfaces: Vec::new(),
        contacts: Vec::new(),
    };

    topology.vertices = payload
        .vertices
        .iter()
        .map(|vertex| {
            let path = occurrences.path(vertex.occurrence_index)?;
            Ok(ExactVertex {
                id: names.shape_id(
                    PersistentEntityKind::Vertex,
                    vertex.shape_key,
                    vertex.occurrence_index,
                    path,
                )?,
                point_m: [vertex.point_x, vertex.point_y, vertex.point_z]
                    .map(|coordinate| coordinate * meters_per_source_unit),
                tolerance_m: vertex.tolerance * meters_per_source_unit,
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    topology
        .vertices
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.edges = payload
        .edges
        .iter()
        .map(|edge| {
            let path = occurrences.path(edge.occurrence_index)?;
            Ok(ExactEdge {
                id: names.shape_id(
                    PersistentEntityKind::Edge,
                    edge.shape_key,
                    edge.occurrence_index,
                    path,
                )?,
                curve_evaluator_id: names.curve_id(edge.shape_key)?,
                start_vertex_id: names.optional_shape_id(
                    PersistentEntityKind::Vertex,
                    edge.start_vertex_key,
                    edge.occurrence_index,
                    path,
                )?,
                end_vertex_id: names.optional_shape_id(
                    PersistentEntityKind::Vertex,
                    edge.end_vertex_key,
                    edge.occurrence_index,
                    path,
                )?,
                is_closed: edge.closed,
                is_periodic: edge.periodic,
                is_degenerate: edge.degenerate,
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    topology.edges.sort_by(|left, right| left.id.cmp(&right.id));

    let coedges_by_wire = payload.coedges.iter().fold(
        BTreeMap::<(u64, u64), Vec<&bridge::OcctExactCoedgePayload>>::new(),
        |mut index, coedge| {
            index
                .entry((coedge.occurrence_index, coedge.wire_key))
                .or_default()
                .push(coedge);
            index
        },
    );
    for wire in &payload.wires {
        let path = occurrences.path(wire.occurrence_index)?;
        let mut coedges = coedges_by_wire
            .get(&(wire.occurrence_index, wire.shape_key))
            .cloned()
            .unwrap_or_default();
        coedges.sort_by_key(|coedge| coedge.coedge_key);
        let mut coedge_ids = coedges
            .iter()
            .map(|coedge| names.coedge_id(wire.shape_key, coedge.coedge_key, path))
            .collect::<Result<Vec<_>, _>>()?;
        coedge_ids.sort();
        topology.wires.push(ExactWire {
            id: names.shape_id(
                PersistentEntityKind::Wire,
                wire.shape_key,
                wire.occurrence_index,
                path,
            )?,
            orientation: orientation(wire.reversed),
            coedge_ids,
        });
        let projected_coedges = coedges
            .into_iter()
            .map(|coedge| {
                Ok(ExactCoedge {
                    id: names.coedge_id(wire.shape_key, coedge.coedge_key, path)?,
                    face_id: names.shape_id(
                        PersistentEntityKind::Face,
                        coedge.face_key,
                        coedge.occurrence_index,
                        path,
                    )?,
                    edge_id: names.shape_id(
                        PersistentEntityKind::Edge,
                        coedge.edge_key,
                        coedge.occurrence_index,
                        path,
                    )?,
                    orientation: orientation(coedge.reversed),
                    pcurve_evaluator_id: names.pcurve_id(wire.shape_key, coedge.coedge_key)?,
                    seam_image: u8::try_from(coedge.seam_image).ok(),
                })
            })
            .collect::<Result<Vec<_>, GeometryImportError>>()?;
        topology.coedges.extend(projected_coedges);
    }
    topology.wires.sort_by(|left, right| left.id.cmp(&right.id));
    topology
        .coedges
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.faces = payload
        .faces
        .iter()
        .map(|face| -> Result<ExactFace, GeometryImportError> {
            let path = occurrences.path(face.occurrence_index)?;
            let mut inner_wire_ids = face
                .inner_wire_keys
                .iter()
                .map(|key| {
                    names.shape_id(
                        PersistentEntityKind::Wire,
                        *key,
                        face.occurrence_index,
                        path,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            inner_wire_ids.sort();
            Ok(ExactFace {
                id: names.shape_id(
                    PersistentEntityKind::Face,
                    face.shape_key,
                    face.occurrence_index,
                    path,
                )?,
                orientation: orientation(face.reversed),
                surface_evaluator_id: names.surface_id(face.shape_key)?,
                trim_classifier_id: names.trim_id(face.shape_key)?,
                outer_wire_id: names.shape_id(
                    PersistentEntityKind::Wire,
                    face.outer_wire_key,
                    face.occurrence_index,
                    path,
                )?,
                inner_wire_ids,
                periodic_u: face.periodic_u,
                periodic_v: face.periodic_v,
                has_singularity: face.singular,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    topology.faces.sort_by(|left, right| left.id.cmp(&right.id));

    topology.shells = payload
        .shells
        .iter()
        .map(|shell| -> Result<ExactShell, GeometryImportError> {
            let path = occurrences.path(shell.occurrence_index)?;
            let mut face_uses = shell
                .face_keys
                .iter()
                .zip(&shell.face_reversed)
                .map(|(key, reversed)| {
                    Ok(OrientedEntityUse {
                        entity_id: names.shape_id(
                            PersistentEntityKind::Face,
                            *key,
                            shell.occurrence_index,
                            path,
                        )?,
                        orientation: orientation(*reversed),
                    })
                })
                .collect::<Result<Vec<_>, GeometryImportError>>()?;
            face_uses.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
            Ok(ExactShell {
                id: names.shape_id(
                    PersistentEntityKind::Shell,
                    shell.shape_key,
                    shell.occurrence_index,
                    path,
                )?,
                orientation: orientation(shell.reversed),
                face_uses,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    topology
        .shells
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.solids = payload
        .solids
        .iter()
        .map(|solid| -> Result<ExactSolid, GeometryImportError> {
            let path = occurrences.path(solid.occurrence_index)?;
            let mut void_shell_ids = solid
                .void_shell_keys
                .iter()
                .map(|key| {
                    names.shape_id(
                        PersistentEntityKind::Shell,
                        *key,
                        solid.occurrence_index,
                        path,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            void_shell_ids.sort();
            Ok(ExactSolid {
                id: names.shape_id(
                    PersistentEntityKind::Solid,
                    solid.shape_key,
                    solid.occurrence_index,
                    path,
                )?,
                outer_shell_id: names.shape_id(
                    PersistentEntityKind::Shell,
                    solid.outer_shell_key,
                    solid.occurrence_index,
                    path,
                )?,
                void_shell_ids,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    topology
        .solids
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.lumps = payload
        .lumps
        .iter()
        .map(|lump| {
            let path = occurrences.path(lump.occurrence_index)?;
            let mut solid_ids = lump
                .solid_keys
                .iter()
                .map(|key| {
                    names.shape_id(
                        PersistentEntityKind::Solid,
                        *key,
                        lump.occurrence_index,
                        path,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            solid_ids.sort();
            Ok(ExactLump {
                id: names.lump_id(lump, path)?,
                solid_ids,
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    topology.lumps.sort_by(|left, right| left.id.cmp(&right.id));
    let body_projection = body_partitions.project_bodies(
        &occurrences,
        solid_mass_properties,
        representation_digest,
    )?;
    topology.bodies = body_projection.bodies;
    let region_projection = project_regions_and_interfaces(&topology)?;
    topology.regions = region_projection.regions;
    topology.interfaces = region_projection.interfaces;

    let evaluators = project_evaluators(
        payload,
        &names,
        representation_digest,
        body_projection.mass_properties,
    )?;

    let model = validation_model(&topology, payload, representation_digest);
    evaluators
        .validate_against(&topology, &model)
        .map_err(|error| invalid_contract("projected exact geometry", error))?;
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid_contract("projected solid shell boundaries", error))?;
    Ok(ExactProjection {
        topology,
        evaluators,
        model,
        kernel_body_shapes: body_projection.kernel_shapes,
    })
}

fn validate_projection_shape(
    payload: &bridge::OcctExactShapePayload,
) -> Result<(), GeometryImportError> {
    if payload.coedges.iter().any(|coedge| !coedge.has_pcurve) {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT exact face use has no pcurve".into(),
        ));
    }
    if payload
        .shells
        .iter()
        .any(|shell| shell.face_keys.len() != shell.face_reversed.len())
    {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT exact shell incidence is incomplete".into(),
        ));
    }
    Ok(())
}

fn validation_model(
    topology: &ExactBRepTopology,
    payload: &bridge::OcctExactShapePayload,
    representation_digest: [u8; 32],
) -> ExactBRepModel {
    ExactBRepModel {
        artifact: GeometryObjectRef {
            digest: GeometryDigest::from_bytes(representation_digest),
            encoded_length: payload.representation.len() as u64,
            media_type: EXACT_BREP_MEDIA_TYPE.into(),
            schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
        },
        kernel_abi: payload.kernel_abi.clone(),
        capabilities: ExactGeometryCapabilities {
            curve_point: true,
            curve_tangent: true,
            curve_curvature: true,
            curve_arc_length: true,
            curve_inverse_projection: true,
            pcurve_point: true,
            pcurve_derivatives: true,
            surface_point: true,
            surface_first_derivatives: true,
            surface_second_derivatives: true,
            surface_normal: true,
            surface_principal_curvature: true,
            surface_uv_bounds: true,
            surface_periodicity: true,
            surface_closest_point: true,
            trim_domain_classification: true,
            mass_properties: true,
        },
        assembly_count: topology.assemblies.len() as u64,
        instance_count: topology.instances.len() as u64,
        body_count: topology.bodies.len() as u64,
        lump_count: topology.lumps.len() as u64,
        solid_count: topology.solids.len() as u64,
        region_count: topology.regions.len() as u64,
        shell_count: topology.shells.len() as u64,
        face_count: topology.faces.len() as u64,
        wire_count: topology.wires.len() as u64,
        coedge_count: topology.coedges.len() as u64,
        edge_count: topology.edges.len() as u64,
        vertex_count: topology.vertices.len() as u64,
        interface_count: topology.interfaces.len() as u64,
        contact_count: topology.contacts.len() as u64,
    }
}

pub(super) fn mass_validation_digest(
    representation_digest: [u8; 32],
    properties: &BodyMassProperties,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"runmat.exact-geometry.occt-mass-validation\0");
    hasher.update(representation_digest);
    for value in std::iter::once(properties.volume_m3)
        .chain(std::iter::once(properties.surface_area_m2))
        .chain(properties.centroid_m)
        .chain(properties.inertia_about_centroid_m5)
    {
        hasher.update(value.to_bits().to_be_bytes());
    }
    hasher.finalize().into()
}

fn invalid_contract(role: &str, error: impl std::fmt::Display) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(format!("{role} failed independent validation: {error}"))
}
