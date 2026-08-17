use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    BodyMassProperties, CurveEvaluatorId, ExactAssembly, ExactBRepModel, ExactBRepTopology,
    ExactBody, ExactCoedge, ExactCurveEvaluatorRecord, ExactCurveImplementation, ExactEdge,
    ExactEvaluatorRegistry, ExactFace, ExactGeometryCapabilities, ExactLump,
    ExactMassPropertiesImplementation, ExactMassPropertiesRecord, ExactPcurveEvaluatorRecord,
    ExactPcurveImplementation, ExactShell, ExactSolid, ExactSurfaceEvaluatorRecord,
    ExactSurfaceImplementation, ExactTrimClassifierImplementation, ExactTrimClassifierRecord,
    ExactVertex, ExactWire, GeometryDigest, GeometryObjectRef, MassPropertiesEvaluatorId,
    OrientedEntityUse, PcurveEvaluatorId, PersistentEntityId, PersistentEntityKind,
    SurfaceEvaluatorId, TopologicalOrientation, TrimClassifierId, EXACT_BREP_MEDIA_TYPE,
    EXACT_BREP_TOPOLOGY_SCHEMA_VERSION, EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use super::ffi::bridge;
use crate::{exact::exact_representation_digest, import::GeometryImportError};

const ROOT_SCOPE: &str = "root";

pub(super) fn project_exact_contracts(
    payload: &bridge::OcctExactShapePayload,
    meters_per_source_unit: f64,
    solid_mass_properties: Option<&BodyMassProperties>,
) -> Result<(ExactBRepTopology, ExactEvaluatorRegistry), GeometryImportError> {
    let representation_digest = exact_representation_digest(&payload.representation);
    validate_projection_shape(payload)?;

    let root_assembly_id = fixed_id(PersistentEntityKind::Assembly, "assembly:root");
    let body_id = fixed_id(PersistentEntityKind::Body, "body:root");
    let lump_id = fixed_id(PersistentEntityKind::Lump, "lump:root");
    let mass_id = MassPropertiesEvaluatorId::new("mass:body:root")
        .map_err(|error| invalid_contract("mass-properties identity", error))?;

    let mut topology = ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: root_assembly_id.clone(),
        assemblies: vec![ExactAssembly {
            id: root_assembly_id,
            definition_digest: digest(
                b"runmat.exact-geometry.occt-assembly-definition\0",
                [representation_digest.as_slice()],
            ),
            body_ids: vec![body_id.clone()],
            child_instance_ids: Vec::new(),
        }],
        instances: Vec::new(),
        bodies: Vec::new(),
        lumps: Vec::new(),
        solids: Vec::new(),
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
        .map(|vertex| ExactVertex {
            id: shape_id(PersistentEntityKind::Vertex, vertex.shape_key),
            point_m: [vertex.point_x, vertex.point_y, vertex.point_z]
                .map(|coordinate| coordinate * meters_per_source_unit),
            tolerance_m: vertex.tolerance * meters_per_source_unit,
        })
        .collect();
    topology
        .vertices
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.edges = payload
        .edges
        .iter()
        .map(|edge| ExactEdge {
            id: shape_id(PersistentEntityKind::Edge, edge.shape_key),
            curve_evaluator_id: curve_id(edge.shape_key),
            start_vertex_id: optional_shape_id(PersistentEntityKind::Vertex, edge.start_vertex_key),
            end_vertex_id: optional_shape_id(PersistentEntityKind::Vertex, edge.end_vertex_key),
            is_closed: edge.closed,
            is_periodic: edge.periodic,
            is_degenerate: edge.degenerate,
        })
        .collect();
    topology.edges.sort_by(|left, right| left.id.cmp(&right.id));

    let coedges_by_wire = payload.coedges.iter().fold(
        BTreeMap::<u64, Vec<&bridge::OcctExactCoedgePayload>>::new(),
        |mut index, coedge| {
            index.entry(coedge.wire_key).or_default().push(coedge);
            index
        },
    );
    for wire in &payload.wires {
        let mut coedges = coedges_by_wire
            .get(&wire.shape_key)
            .cloned()
            .unwrap_or_default();
        coedges.sort_by_key(|coedge| coedge.coedge_key);
        topology.wires.push(ExactWire {
            id: shape_id(PersistentEntityKind::Wire, wire.shape_key),
            orientation: orientation(wire.reversed),
            coedge_ids: coedges
                .iter()
                .map(|coedge| coedge_id(wire.shape_key, coedge.coedge_key))
                .collect(),
        });
        topology
            .coedges
            .extend(coedges.into_iter().map(|coedge| ExactCoedge {
                id: coedge_id(wire.shape_key, coedge.coedge_key),
                face_id: shape_id(PersistentEntityKind::Face, coedge.face_key),
                edge_id: shape_id(PersistentEntityKind::Edge, coedge.edge_key),
                orientation: orientation(coedge.reversed),
                pcurve_evaluator_id: pcurve_id(wire.shape_key, coedge.coedge_key),
                seam_image: u8::try_from(coedge.seam_image).ok(),
            }));
    }
    topology.wires.sort_by(|left, right| left.id.cmp(&right.id));
    topology
        .coedges
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.faces = payload
        .faces
        .iter()
        .map(|face| {
            let mut inner_wire_ids = face
                .inner_wire_keys
                .iter()
                .map(|key| shape_id(PersistentEntityKind::Wire, *key))
                .collect::<Vec<_>>();
            inner_wire_ids.sort();
            ExactFace {
                id: shape_id(PersistentEntityKind::Face, face.shape_key),
                orientation: orientation(face.reversed),
                surface_evaluator_id: surface_id(face.shape_key),
                trim_classifier_id: trim_id(face.shape_key),
                outer_wire_id: shape_id(PersistentEntityKind::Wire, face.outer_wire_key),
                inner_wire_ids,
                periodic_u: face.periodic_u,
                periodic_v: face.periodic_v,
                has_singularity: face.singular,
            }
        })
        .collect();
    topology.faces.sort_by(|left, right| left.id.cmp(&right.id));

    topology.shells = payload
        .shells
        .iter()
        .map(|shell| {
            let mut face_uses = shell
                .face_keys
                .iter()
                .zip(&shell.face_reversed)
                .map(|(key, reversed)| OrientedEntityUse {
                    entity_id: shape_id(PersistentEntityKind::Face, *key),
                    orientation: orientation(*reversed),
                })
                .collect::<Vec<_>>();
            face_uses.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
            ExactShell {
                id: shape_id(PersistentEntityKind::Shell, shell.shape_key),
                orientation: orientation(shell.reversed),
                face_uses,
            }
        })
        .collect();
    topology
        .shells
        .sort_by(|left, right| left.id.cmp(&right.id));

    topology.solids = payload
        .solids
        .iter()
        .map(|solid| {
            let mut void_shell_ids = solid
                .void_shell_keys
                .iter()
                .map(|key| shape_id(PersistentEntityKind::Shell, *key))
                .collect::<Vec<_>>();
            void_shell_ids.sort();
            ExactSolid {
                id: shape_id(PersistentEntityKind::Solid, solid.shape_key),
                outer_shell_id: shape_id(PersistentEntityKind::Shell, solid.outer_shell_key),
                void_shell_ids,
            }
        })
        .collect();
    topology
        .solids
        .sort_by(|left, right| left.id.cmp(&right.id));

    let is_sheet_body = topology.solids.is_empty();
    let mut solid_ids = topology
        .solids
        .iter()
        .map(|solid| solid.id.clone())
        .collect::<Vec<_>>();
    solid_ids.sort();
    let mut sheet_shell_ids = if is_sheet_body {
        topology
            .shells
            .iter()
            .map(|shell| shell.id.clone())
            .collect()
    } else {
        Vec::new()
    };
    sheet_shell_ids.sort();
    if !is_sheet_body {
        topology.lumps.push(ExactLump {
            id: lump_id.clone(),
            solid_ids,
        });
    }
    topology.bodies.push(ExactBody {
        id: body_id,
        mass_properties_evaluator_id: mass_id.clone(),
        lump_ids: (!is_sheet_body)
            .then_some(vec![lump_id])
            .unwrap_or_default(),
        is_sheet_body,
        sheet_shell_ids,
    });

    let evaluator_ref = |entity_token: String| runmat_geometry_core::KernelEvaluatorRef {
        entity_token,
        representation_digest,
    };
    let mut evaluators = ExactEvaluatorRegistry {
        schema_version: EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
        kernel_abi: payload.kernel_abi.clone(),
        curves: payload
            .edges
            .iter()
            .map(|edge| ExactCurveEvaluatorRecord {
                id: curve_id(edge.shape_key),
                implementation: ExactCurveImplementation::Kernel {
                    reference: evaluator_ref(format!("edge:{:020}", edge.shape_key)),
                },
            })
            .collect(),
        pcurves: payload
            .coedges
            .iter()
            .map(|coedge| ExactPcurveEvaluatorRecord {
                id: pcurve_id(coedge.wire_key, coedge.coedge_key),
                implementation: ExactPcurveImplementation::Kernel {
                    reference: evaluator_ref(format!(
                        "face:{:020}:wire:{:020}:coedge:{:020}:seam:{}",
                        coedge.face_key, coedge.wire_key, coedge.coedge_key, coedge.seam_image
                    )),
                },
            })
            .collect(),
        surfaces: payload
            .faces
            .iter()
            .map(|face| ExactSurfaceEvaluatorRecord {
                id: surface_id(face.shape_key),
                implementation: ExactSurfaceImplementation::Kernel {
                    reference: evaluator_ref(format!("face:{:020}", face.shape_key)),
                },
            })
            .collect(),
        trim_classifiers: payload
            .faces
            .iter()
            .map(|face| ExactTrimClassifierRecord {
                id: trim_id(face.shape_key),
                implementation: ExactTrimClassifierImplementation::Kernel {
                    reference: evaluator_ref(format!("face:{:020}", face.shape_key)),
                },
            })
            .collect(),
        mass_properties: vec![ExactMassPropertiesRecord {
            id: mass_id,
            implementation: match solid_mass_properties {
                Some(properties) => ExactMassPropertiesImplementation::KernelValidated {
                    properties: *properties,
                    validation_digest: mass_validation_digest(representation_digest, properties),
                },
                None => ExactMassPropertiesImplementation::Kernel {
                    reference: evaluator_ref("body:root".into()),
                },
            },
        }],
    };
    evaluators
        .curves
        .sort_by(|left, right| left.id.cmp(&right.id));
    evaluators
        .pcurves
        .sort_by(|left, right| left.id.cmp(&right.id));
    evaluators
        .surfaces
        .sort_by(|left, right| left.id.cmp(&right.id));
    evaluators
        .trim_classifiers
        .sort_by(|left, right| left.id.cmp(&right.id));

    let model = validation_model(&topology, payload, representation_digest);
    evaluators
        .validate_against(&topology, &model)
        .map_err(|error| invalid_contract("projected exact geometry", error))?;
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid_contract("projected solid shell boundaries", error))?;
    Ok((topology, evaluators))
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
    let owned_shells = payload
        .solids
        .iter()
        .flat_map(|solid| {
            std::iter::once(solid.outer_shell_key).chain(solid.void_shell_keys.iter().copied())
        })
        .collect::<BTreeSet<_>>();
    if !payload.solids.is_empty() && owned_shells.len() != payload.shells.len() {
        return Err(GeometryImportError::InvalidGeometry(
            "mixed solid and sheet topology requires separate exact bodies".into(),
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

fn fixed_id(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.into(),
        assembly_path: vec![ROOT_SCOPE.into()],
    }
}

fn shape_id(kind: PersistentEntityKind, key: u64) -> PersistentEntityId {
    fixed_id(kind, &format!("brep-shape:{key:020}"))
}

fn optional_shape_id(kind: PersistentEntityKind, key: u64) -> Option<PersistentEntityId> {
    (key != 0).then(|| shape_id(kind, key))
}

fn coedge_id(wire_key: u64, position: u64) -> PersistentEntityId {
    fixed_id(
        PersistentEntityKind::Coedge,
        &format!("brep-wire:{wire_key:020}:coedge:{position:020}"),
    )
}

fn curve_id(key: u64) -> CurveEvaluatorId {
    CurveEvaluatorId(format!("curve:brep-shape:{key:020}"))
}

fn pcurve_id(wire_key: u64, position: u64) -> PcurveEvaluatorId {
    PcurveEvaluatorId(format!(
        "pcurve:brep-wire:{wire_key:020}:coedge:{position:020}"
    ))
}

fn surface_id(key: u64) -> SurfaceEvaluatorId {
    SurfaceEvaluatorId(format!("surface:brep-shape:{key:020}"))
}

fn trim_id(key: u64) -> TrimClassifierId {
    TrimClassifierId(format!("trim:brep-shape:{key:020}"))
}

const fn orientation(reversed: bool) -> TopologicalOrientation {
    if reversed {
        TopologicalOrientation::Reversed
    } else {
        TopologicalOrientation::Forward
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

fn digest<'a>(domain: &[u8], parts: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn invalid_contract(role: &str, error: impl std::fmt::Display) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(format!("{role} failed independent validation: {error}"))
}
