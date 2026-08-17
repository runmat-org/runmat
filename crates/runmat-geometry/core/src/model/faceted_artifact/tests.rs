use super::*;
use crate::{
    FacetedSolidModel, GeometryDigest, GeometryHealingPolicy, GeometryModel, GeometryObjectRef,
    GeometryRevisionIdentity, GeometrySourceFormat, GeometrySourceIdentity,
    GeometryTolerancePolicy, PersistentEntityKind, UnitSystem, FACETED_SOLID_MEDIA_TYPE,
    GEOMETRY_DOCUMENT_SCHEMA_VERSION, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}

fn tetrahedron() -> FacetedSolid {
    let shell_id = id(PersistentEntityKind::Shell, "faceted:shell:0");
    FacetedSolid {
        schema_version: FACETED_SOLID_SCHEMA_VERSION,
        vertices: [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, coordinates_m)| FacetedVertex {
            id: id(
                PersistentEntityKind::Vertex,
                &format!("faceted:vertex:{index}"),
            ),
            coordinates_m,
        })
        .collect(),
        triangles: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
            .into_iter()
            .enumerate()
            .map(|(index, vertex_indices)| FacetedTriangle {
                id: id(PersistentEntityKind::Face, &format!("faceted:face:{index}")),
                vertex_indices,
                shell_id: shell_id.clone(),
            })
            .collect(),
        shells: vec![FacetedShell {
            id: shell_id,
            orientation: FacetedShellOrientation::Outward,
            triangle_indices: vec![0, 1, 2, 3],
        }],
    }
}

fn document() -> GeometryDocument {
    GeometryDocument {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentity {
            content_digest: GeometryDigest::from_bytes([7; 32]),
            format: GeometrySourceFormat::Stl,
            importer_version: "runmat-faceted-import/1".into(),
            kernel_version: None,
            source_units: UnitSystem::Meter,
            meters_per_source_unit: 1.0,
        },
        revision: GeometryRevisionIdentity {
            revision: 1,
            persistent_mapping_version: 1,
            parent_document_digest: None,
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 0.0,
        },
        healing: GeometryHealingPolicy {
            algorithm_version: "none/1".into(),
            sew: false,
            repair_orientation: false,
            consolidate_duplicates: false,
            repair_tolerance_scale_gaps: false,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModel::FacetedSolid {
            model: FacetedSolidModel {
                artifact: GeometryObjectRef {
                    digest: GeometryDigest::from_bytes([1; 32]),
                    encoded_length: 1,
                    media_type: FACETED_SOLID_MEDIA_TYPE.into(),
                    schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
                },
                vertex_count: 4,
                triangle_count: 4,
                shell_count: 1,
                is_watertight: true,
                is_oriented: true,
            },
        },
        display_tessellations: Vec::new(),
    }
}

#[test]
fn faceted_solid_round_trips_with_document_binding() {
    let closure = build_faceted_solid_closure(document(), tetrahedron()).unwrap();
    assert_eq!(
        admit_faceted_solid(&closure.document, &closure.solid_bytes).unwrap(),
        closure.solid
    );
    assert_eq!(
        closure.document.primary_artifact().encoded_length,
        closure.solid_bytes.len() as u64
    );
}

#[test]
fn faceted_solid_rejects_open_or_inconsistently_oriented_boundaries() {
    let mut open = tetrahedron();
    open.triangles.pop();
    open.shells[0].triangle_indices.pop();
    let mut model = match document().model {
        GeometryModel::FacetedSolid { model } => model,
        _ => unreachable!(),
    };
    model.triangle_count = 3;
    assert!(open.validate_against(&model).is_err());

    let mut reversed = tetrahedron();
    reversed.triangles[0].vertex_indices.swap(1, 2);
    assert!(reversed.validate_against(&model_for(&reversed)).is_err());

    let mut wrong_orientation = tetrahedron();
    wrong_orientation.shells[0].orientation = FacetedShellOrientation::Inward;
    assert!(wrong_orientation
        .validate_against(&model_for(&wrong_orientation))
        .is_err());

    let mut wrong_membership = tetrahedron();
    wrong_membership.shells[0].id = id(PersistentEntityKind::Shell, "faceted:shell:other");
    assert!(wrong_membership
        .validate_against(&model_for(&wrong_membership))
        .is_err());
}

#[test]
fn faceted_solid_rejects_unreferenced_vertices() {
    let mut solid = tetrahedron();
    solid.vertices.push(FacetedVertex {
        id: id(PersistentEntityKind::Vertex, "faceted:vertex:unused"),
        coordinates_m: [2.0, 2.0, 2.0],
    });

    assert!(solid.validate_against(&model_for(&solid)).is_err());
}

#[test]
fn faceted_solid_rejects_corrupt_or_mismatched_artifacts() {
    let closure = build_faceted_solid_closure(document(), tetrahedron()).unwrap();
    let mut corrupted = closure.solid_bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    assert!(admit_faceted_solid(&closure.document, &corrupted).is_err());

    let mut wrong_count = closure.document.clone();
    let GeometryModel::FacetedSolid { model } = &mut wrong_count.model else {
        unreachable!()
    };
    model.vertex_count += 1;
    assert!(admit_faceted_solid(&wrong_count, &closure.solid_bytes).is_err());

    let model = match &closure.document.model {
        GeometryModel::FacetedSolid { model } => model,
        _ => unreachable!(),
    };
    let mut trailing = closure.solid_bytes.clone();
    trailing.push(0);
    assert!(decode_faceted_solid(&trailing, model).is_err());
    let wrong_domain = crate::encode_geometry_document(&closure.document).unwrap();
    assert!(decode_faceted_solid(&wrong_domain, model).is_err());
}

fn model_for(solid: &FacetedSolid) -> FacetedSolidModel {
    FacetedSolidModel {
        artifact: GeometryObjectRef {
            digest: GeometryDigest::from_bytes([1; 32]),
            encoded_length: 1,
            media_type: FACETED_SOLID_MEDIA_TYPE.into(),
            schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
        },
        vertex_count: solid.vertices.len() as u64,
        triangle_count: solid.triangles.len() as u64,
        shell_count: solid.shells.len() as u64,
        is_watertight: true,
        is_oriented: true,
    }
}
