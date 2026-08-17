use super::*;

fn digest(seed: u8) -> GeometryDigest {
    GeometryDigest::from_bytes([seed; 32])
}

fn object(seed: u8, media_type: &str) -> GeometryObjectRefV2 {
    GeometryObjectRefV2 {
        digest: digest(seed),
        encoded_length: 4096,
        media_type: media_type.into(),
        schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    }
}

fn complete_capabilities() -> ExactGeometryCapabilitiesV2 {
    ExactGeometryCapabilitiesV2 {
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
    }
}

fn exact_document() -> GeometryDocumentV2 {
    GeometryDocumentV2 {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentityV2 {
            content_digest: digest(1),
            format: GeometrySourceFormatV2::Step,
            importer_version: "step-import/v2".into(),
            kernel_version: Some("occt/7.9".into()),
            source_units: UnitSystem::Millimeter,
            meters_per_source_unit: 0.001,
        },
        revision: GeometryRevisionIdentityV2 {
            revision: 7,
            persistent_mapping_version: 2,
            parent_document_digest: Some(digest(2)),
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        healing: GeometryHealingPolicyV2 {
            algorithm_version: "geometry-healing/v2".into(),
            sew: true,
            repair_orientation: true,
            consolidate_duplicates: true,
            repair_tolerance_scale_gaps: true,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModelV2::ExactBRep {
            model: ExactBRepModelV2 {
                artifact: object(3, EXACT_BREP_MEDIA_TYPE_V2),
                kernel_abi: "occt-v1".into(),
                capabilities: complete_capabilities(),
                assembly_count: 1,
                instance_count: 1,
                body_count: 1,
                lump_count: 1,
                solid_count: 1,
                shell_count: 1,
                face_count: 6,
                wire_count: 6,
                coedge_count: 24,
                edge_count: 12,
                vertex_count: 8,
                interface_count: 0,
                contact_count: 0,
            },
        },
        display_tessellations: vec![DisplayTessellationRefV2 {
            profile_id: "interactive/v2".into(),
            geometry_revision: 7,
            derived_from_primary_digest: digest(3),
            artifact: object(4, DISPLAY_TESSELLATION_MEDIA_TYPE_V2),
        }],
    }
}

#[test]
fn exact_document_is_path_free_and_display_is_derived() {
    let document = exact_document();
    document.validate().unwrap();
    assert!(document.is_exact());
    let encoded = serde_json::to_string(&document).unwrap();
    assert!(!encoded.contains("path"));
    assert!(encoded.contains(EXACT_BREP_MEDIA_TYPE_V2));
    assert!(encoded.contains(DISPLAY_TESSELLATION_MEDIA_TYPE_V2));
    assert_eq!(
        serde_json::from_str::<GeometryDocumentV2>(&encoded).unwrap(),
        document
    );
}

#[test]
fn exact_sources_cannot_degrade_to_faceted_payloads() {
    let mut document = exact_document();
    document.model = faceted_model();
    assert!(document.validate().is_err());
}

#[test]
fn faceted_sources_cannot_claim_exact_capability() {
    let mut document = exact_document();
    document.source.format = GeometrySourceFormatV2::Stl;
    document.source.kernel_version = None;
    document.model = faceted_model();
    document.display_tessellations[0].derived_from_primary_digest = digest(5);
    document.validate().unwrap();
    assert!(!document.is_exact());
    document.source.kernel_version = Some("fabricated-exact-kernel".into());
    assert!(document.validate().is_err());
}

#[test]
fn exact_admission_requires_complete_evaluators() {
    let mut document = exact_document();
    if let GeometryModelV2::ExactBRep { model } = &mut document.model {
        model.capabilities.trim_domain_classification = false;
    }
    assert!(document.validate().is_err());
    if let GeometryModelV2::ExactBRep { model } = &mut document.model {
        model.capabilities.trim_domain_classification = true;
        model.capabilities.pcurve_derivatives = false;
    }
    assert!(document.validate().is_err());
}

#[test]
fn exact_root_parts_and_sheet_models_do_not_require_fake_instances_or_solids() {
    let mut document = exact_document();
    let GeometryModelV2::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.instance_count = 0;
    model.lump_count = 0;
    model.solid_count = 0;
    document.validate().unwrap();
}

#[test]
fn display_cache_cannot_replace_or_rebind_primary_geometry() {
    let mut document = exact_document();
    document.display_tessellations[0].artifact.digest = digest(3);
    assert!(document.validate().is_err());
    document.display_tessellations[0].artifact.digest = digest(4);
    document.display_tessellations[0].derived_from_primary_digest = digest(9);
    assert!(document.validate().is_err());
}

#[test]
fn document_rejects_unknown_fields_and_unbounded_artifacts() {
    let mut value = serde_json::to_value(exact_document()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("physicalPath".into(), serde_json::json!("/tmp/model.step"));
    assert!(serde_json::from_value::<GeometryDocumentV2>(value).is_err());

    let mut document = exact_document();
    let GeometryModelV2::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.artifact.encoded_length = (1_u64 << 40) + 1;
    assert!(document.validate().is_err());
}

fn faceted_model() -> GeometryModelV2 {
    GeometryModelV2::FacetedSolid {
        model: FacetedSolidModelV2 {
            artifact: object(5, FACETED_SOLID_MEDIA_TYPE_V2),
            vertex_count: 4,
            triangle_count: 4,
            shell_count: 1,
            is_watertight: true,
            is_oriented: true,
        },
    }
}
