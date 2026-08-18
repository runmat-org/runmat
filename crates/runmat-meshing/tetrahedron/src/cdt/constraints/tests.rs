use runmat_geometry_core::*;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_curve::shared_curve_vertex_node_id;
use runmat_meshing_surface::{
    ExactFaceMeshBoundarySegment, ExactFaceMeshNode, ExactFaceMeshTriangle, ExactSurfaceMesh,
    ExactSurfaceShellEvidence, EXACT_SURFACE_MESH_SCHEMA_VERSION,
};

use super::*;
use crate::cdt::{
    build_delaunay_volume_point_set, build_delaunay_volume_provenance, carve_delaunay_volume,
    recover_delaunay_facets, recover_delaunay_segments,
    validate_delaunay_volume_provenance_sources, DelaunayCarvingOptions,
    DelaunayFacetRecoveryOptions, DelaunayPointSetOptions, DelaunaySegmentRecoveryOptions,
    DelaunayVolumeProvenanceOptions,
};

const FACETS: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
const EDGE_VERTICES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

#[test]
fn exact_surface_build_is_canonical_and_retains_persistent_provenance() {
    let (topology, surface) = tetrahedron();
    let constraints = build_delaunay_constraints(
        &topology,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(constraints.nodes.len(), 4);
    assert_eq!(constraints.segments.len(), 6);
    assert_eq!(constraints.facets.len(), 4);
    assert!(constraints
        .nodes
        .iter()
        .all(|node| node.source_vertex_id.is_some()));
    assert!(constraints
        .segments
        .iter()
        .all(|segment| segment.source_edge_id.is_some()));
    let region = entity(PersistentEntityKind::Region, "region");
    assert!(constraints.facets.iter().all(|facet| {
        facet.positive_side == DelaunayConstraintFacetSide::Exterior
            && facet.negative_side == DelaunayConstraintFacetSide::Region(region.clone())
    }));

    let volume = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(volume.tetrahedra.len(), 1);
    assert_eq!(volume.incidence.boundary_facets.len(), 4);

    let segments = recover_delaunay_segments(
        volume,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let facets = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let carved = carve_delaunay_volume(
        &facets,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(carved.topology.incidence.regions.len(), 1);
    assert_eq!(
        carved.topology.incidence.regions[0].region_id,
        entity(PersistentEntityKind::Region, "region")
    );
    let provenance = build_delaunay_volume_provenance(
        &facets,
        &constraints,
        &carved,
        DelaunayCarvingOptions::default(),
        DelaunayVolumeProvenanceOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(provenance.nodes.len(), 4);
    assert_eq!(provenance.segments.len(), 6);
    assert_eq!(provenance.facets.len(), 4);
    validate_delaunay_volume_provenance_sources(
        &facets,
        &constraints,
        &carved,
        &provenance,
        DelaunayCarvingOptions::default(),
        DelaunayVolumeProvenanceOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let mut stale_source = provenance;
    stale_source.facets[0].entity_ids = vec![entity(PersistentEntityKind::Face, "face:stale")];
    assert_eq!(
        validate_delaunay_volume_provenance_sources(
            &facets,
            &constraints,
            &carved,
            &stale_source,
            DelaunayCarvingOptions::default(),
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        crate::cdt::DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );
    assert_eq!(
        build_delaunay_volume_provenance(
            &facets,
            &constraints,
            &carved,
            DelaunayCarvingOptions::default(),
            DelaunayVolumeProvenanceOptions {
                maximum_facet_bindings: 3,
                ..DelaunayVolumeProvenanceOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        crate::cdt::DelaunayVolumeProvenanceErrorKind::ResourceLimit
    );
    assert_eq!(
        build_delaunay_volume_provenance(
            &facets,
            &constraints,
            &carved,
            DelaunayCarvingOptions::default(),
            DelaunayVolumeProvenanceOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        crate::cdt::DelaunayVolumeProvenanceErrorKind::Cancelled
    );
}

#[test]
fn shell_orientation_places_outer_region_on_the_positive_side_when_reversed() {
    let (mut topology, surface) = tetrahedron();
    topology.shells[0].orientation = TopologicalOrientation::Reversed;
    let constraints = build_delaunay_constraints(
        &topology,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let region = entity(PersistentEntityKind::Region, "region");

    assert!(constraints.facets.iter().all(|facet| {
        facet.positive_side == DelaunayConstraintFacetSide::Region(region.clone())
            && facet.negative_side == DelaunayConstraintFacetSide::Exterior
    }));
}

#[test]
fn interface_and_contact_metadata_remain_typed_and_persistent() {
    let (mut topology, surface) = tetrahedron();
    let second_solid = entity(PersistentEntityKind::Solid, "solid:b");
    let second_region = entity(PersistentEntityKind::Region, "region:b");
    let second_shell = entity(PersistentEntityKind::Shell, "shell:b");
    topology.lumps[0].solid_ids.push(second_solid.clone());
    topology.solids.push(ExactSolid {
        id: second_solid.clone(),
        outer_shell_id: second_shell.clone(),
        void_shell_ids: Vec::new(),
    });
    topology.regions.push(ExactRegion {
        id: second_region.clone(),
        solid_id: second_solid,
    });
    topology.shells.push(ExactShell {
        id: second_shell,
        orientation: TopologicalOrientation::Forward,
        face_uses: topology
            .faces
            .iter()
            .map(|face| OrientedEntityUse {
                entity_id: face.id.clone(),
                orientation: TopologicalOrientation::Reversed,
            })
            .collect(),
    });
    let first_region = topology.regions[0].id.clone();
    topology.interfaces = topology
        .faces
        .iter()
        .map(|face| ExactSharedInterface {
            face_id: face.id.clone(),
            side_a_region_id: first_region.clone(),
            side_b_region_id: second_region.clone(),
            side_a_orientation: TopologicalOrientation::Forward,
            side_b_orientation: TopologicalOrientation::Reversed,
        })
        .collect();
    let contact = entity(PersistentEntityKind::Contact, "contact");
    topology.contacts.push(ExactContactPair {
        id: contact.clone(),
        side_a_face_ids: vec![topology.faces[0].id.clone()],
        side_b_face_ids: vec![topology.faces[1].id.clone()],
        pairing_schema_version: EXACT_CONTACT_PAIRING_SCHEMA_VERSION,
        pairing_contract_digest: [7; 32],
    });

    let constraints = build_delaunay_constraints(
        &topology,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert!(constraints.facets.iter().all(|facet| {
        facet.positive_side == DelaunayConstraintFacetSide::Region(second_region.clone())
            && facet.negative_side == DelaunayConstraintFacetSide::Region(first_region.clone())
    }));
    for facet in &constraints.facets {
        let expected = if facet.source_face_id == topology.faces[0].id
            || facet.source_face_id == topology.faces[1].id
        {
            vec![contact.clone()]
        } else {
            Vec::new()
        };
        assert_eq!(facet.contact_ids, expected);
    }
}

#[test]
fn validation_rejects_tampered_identity_side_and_edge_coverage() {
    let (topology, surface) = tetrahedron();
    let constraints = build_delaunay_constraints(
        &topology,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let mut unordered_nodes = constraints.clone();
    unordered_nodes.nodes.swap(0, 1);
    assert_eq!(
        validate_delaunay_constraints(
            &unordered_nodes,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidIdentity
    );

    let mut invalid_side = constraints.clone();
    invalid_side.facets[0].positive_side = DelaunayConstraintFacetSide::Exterior;
    invalid_side.facets[0].negative_side = DelaunayConstraintFacetSide::Void;
    assert_eq!(
        validate_delaunay_constraints(
            &invalid_side,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidBoundary
    );

    let mut missing_edge = constraints;
    missing_edge.segments.remove(0);
    assert_eq!(
        validate_delaunay_constraints(
            &missing_edge,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidBoundary
    );
}

#[test]
fn build_rejects_surface_mismatch_limits_and_cancellation() {
    let (topology, mut surface) = tetrahedron();
    surface.face_ids.pop();
    assert_eq!(
        build_delaunay_constraints(
            &topology,
            &surface,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidBoundary
    );

    let (topology, surface) = tetrahedron();
    let options = DelaunayConstraintOptions {
        maximum_segments: 5,
        ..DelaunayConstraintOptions::default()
    };
    assert_eq!(
        build_delaunay_constraints(&topology, &surface, options, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayConstraintErrorKind::ResourceLimit
    );
    assert_eq!(
        build_delaunay_constraints(
            &topology,
            &surface,
            DelaunayConstraintOptions {
                cancellation_check_interval: 1,
                ..DelaunayConstraintOptions::default()
            },
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::Cancelled
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

pub(crate) fn tetrahedron() -> (ExactBRepTopology, ExactSurfaceMesh) {
    let root = entity(PersistentEntityKind::Assembly, "root");
    let body = entity(PersistentEntityKind::Body, "body");
    let lump = entity(PersistentEntityKind::Lump, "lump");
    let solid = entity(PersistentEntityKind::Solid, "solid");
    let region = entity(PersistentEntityKind::Region, "region");
    let shell = entity(PersistentEntityKind::Shell, "shell");
    let vertices = (0..4)
        .map(|index| entity(PersistentEntityKind::Vertex, &format!("vertex:{index}")))
        .collect::<Vec<_>>();
    let edges = (0..6)
        .map(|index| entity(PersistentEntityKind::Edge, &format!("edge:{index}")))
        .collect::<Vec<_>>();
    let faces = (0..4)
        .map(|index| entity(PersistentEntityKind::Face, &format!("face:{index}")))
        .collect::<Vec<_>>();
    let wires = (0..4)
        .map(|index| entity(PersistentEntityKind::Wire, &format!("wire:{index}")))
        .collect::<Vec<_>>();
    let mut coedges = Vec::new();
    let mut exact_wires = Vec::new();
    let mut boundary_segments = Vec::new();
    for (face_index, facet) in FACETS.iter().enumerate() {
        let mut coedge_ids = Vec::new();
        for local_edge in 0..3 {
            let from = facet[local_edge];
            let to = facet[(local_edge + 1) % 3];
            let edge_index = edge_index(from, to);
            let coedge_id = entity(
                PersistentEntityKind::Coedge,
                &format!("coedge:{face_index}:{local_edge}"),
            );
            coedge_ids.push(coedge_id.clone());
            coedges.push(ExactCoedge {
                id: coedge_id.clone(),
                face_id: faces[face_index].clone(),
                edge_id: edges[edge_index].clone(),
                orientation: if from < to {
                    TopologicalOrientation::Forward
                } else {
                    TopologicalOrientation::Reversed
                },
                pcurve_evaluator_id: PcurveEvaluatorId::new(format!(
                    "pcurve:{face_index}:{local_edge}"
                ))
                .unwrap(),
                seam_image: None,
            });
            boundary_segments.push(ExactFaceMeshBoundarySegment {
                source_coedge_id: coedge_id,
                source_edge_id: edges[edge_index].clone(),
                node_ids: [
                    shared_curve_vertex_node_id(&vertices[from]),
                    shared_curve_vertex_node_id(&vertices[to]),
                ],
                edge_parameters: [0.0, 1.0],
            });
        }
        exact_wires.push(ExactWire {
            id: wires[face_index].clone(),
            orientation: TopologicalOrientation::Forward,
            coedge_ids,
        });
    }
    let topology = ExactBRepTopology {
        schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
        root_assembly_id: root.clone(),
        assemblies: vec![ExactAssembly {
            id: root,
            definition_digest: [1; 32],
            body_ids: vec![body.clone()],
            child_instance_ids: Vec::new(),
        }],
        instances: Vec::new(),
        bodies: vec![ExactBody {
            id: body,
            mass_properties_evaluator_id: MassPropertiesEvaluatorId::new("mass:tetrahedron")
                .unwrap(),
            lump_ids: vec![lump.clone()],
            is_sheet_body: false,
            sheet_shell_ids: Vec::new(),
        }],
        lumps: vec![ExactLump {
            id: lump,
            solid_ids: vec![solid.clone()],
        }],
        solids: vec![ExactSolid {
            id: solid.clone(),
            outer_shell_id: shell.clone(),
            void_shell_ids: Vec::new(),
        }],
        regions: vec![ExactRegion {
            id: region,
            solid_id: solid,
        }],
        shells: vec![ExactShell {
            id: shell.clone(),
            orientation: TopologicalOrientation::Forward,
            face_uses: faces
                .iter()
                .cloned()
                .map(|entity_id| OrientedEntityUse {
                    entity_id,
                    orientation: TopologicalOrientation::Forward,
                })
                .collect(),
        }],
        faces: faces
            .iter()
            .enumerate()
            .map(|(index, id)| ExactFace {
                id: id.clone(),
                orientation: TopologicalOrientation::Forward,
                surface_evaluator_id: SurfaceEvaluatorId::new(format!("surface:{index}")).unwrap(),
                trim_classifier_id: TrimClassifierId::new(format!("trim:{index}")).unwrap(),
                outer_wire_id: wires[index].clone(),
                inner_wire_ids: Vec::new(),
                periodic_u: false,
                periodic_v: false,
                has_singularity: false,
            })
            .collect(),
        wires: exact_wires,
        coedges,
        edges: EDGE_VERTICES
            .iter()
            .enumerate()
            .map(|(index, endpoints)| ExactEdge {
                id: edges[index].clone(),
                curve_evaluator_id: CurveEvaluatorId::new(format!("curve:{index}")).unwrap(),
                start_vertex_id: Some(vertices[endpoints[0]].clone()),
                end_vertex_id: Some(vertices[endpoints[1]].clone()),
                is_closed: false,
                is_periodic: false,
                is_degenerate: false,
            })
            .collect(),
        vertices: [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, point_m)| ExactVertex {
            id: vertices[index].clone(),
            point_m,
            tolerance_m: 1.0e-10,
        })
        .collect(),
        interfaces: Vec::new(),
        contacts: Vec::new(),
    };
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| ExactFaceMeshNode {
            node_id: shared_curve_vertex_node_id(&vertex.id),
            point_m: vertex.point_m,
            uses: Vec::new(),
        })
        .collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.node_id);
    let triangles = FACETS
        .iter()
        .enumerate()
        .map(|(index, facet)| ExactFaceMeshTriangle {
            triangle_id: StableDigest::from_bytes([(index + 1) as u8; 32]),
            chart_id: StableDigest::from_bytes([(index + 20) as u8; 32]),
            source_face_id: faces[index].clone(),
            node_ids: canonical_triangle(
                facet.map(|vertex| shared_curve_vertex_node_id(&vertices[vertex])),
            ),
            unit_normal: [0.0; 3],
            physical_area_m2: 0.5,
            metric_edge_lengths: [1.0; 3],
            minimum_metric_angle_rad: 0.5,
            physical_aspect_ratio: 1.0,
            chordal_deviation_m: 0.0,
            normal_deviation_rad: 0.0,
            acceptance_sample_count: 1,
            accepted_chordal_deviation_m: 0.0,
            accepted_normal_deviation_rad: 0.0,
        })
        .collect();
    let surface = ExactSurfaceMesh {
        schema_version: EXACT_SURFACE_MESH_SCHEMA_VERSION,
        face_ids: faces,
        nodes,
        triangles,
        boundary_segments,
        shells: vec![ExactSurfaceShellEvidence {
            source_shell_id: shell,
            face_count: 4,
            shared_edge_count: 6,
            open_edge_count: 0,
            nonmanifold_edge_count: 0,
            is_sheet_shell: false,
            is_watertight: true,
        }],
        maximum_chordal_deviation_m: 0.0,
        maximum_normal_deviation_rad: 0.0,
    };
    (topology, surface)
}

fn edge_index(left: usize, right: usize) -> usize {
    let mut endpoints = [left, right];
    endpoints.sort_unstable();
    EDGE_VERTICES
        .iter()
        .position(|candidate| *candidate == endpoints)
        .unwrap()
}

fn canonical_triangle(nodes: [StableDigest; 3]) -> [StableDigest; 3] {
    if nodes[1] < nodes[0] && nodes[1] < nodes[2] {
        [nodes[1], nodes[2], nodes[0]]
    } else if nodes[2] < nodes[0] {
        [nodes[2], nodes[0], nodes[1]]
    } else {
        nodes
    }
}

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: value.to_owned(),
        assembly_path: if kind == PersistentEntityKind::Assembly {
            Vec::new()
        } else {
            vec!["root".to_owned()]
        },
    }
}
