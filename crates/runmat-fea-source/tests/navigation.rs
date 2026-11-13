use runmat_analysis_language::{LogicalSourcePath, SourceDigest};
use runmat_fea_source::*;
use serde_json::json;

fn checked() -> FeaSourceCheckResult {
    serde_json::from_value(json!({
        "contract": FeaSourceCheckContract::AssessmentsV1,
        "revision": vec![7; 32], "status": "valid", "rootModule": "project/main.fea",
        "program": { "studies": [], "sweeps": [], "executionPolicies": [], "executionBindings": [],
            "materials": [], "geometries": [], "meshes": [], "models": [] },
        "sourceIndex": { "entries": (0..513).map(|index| json!({
            "identity": { "kind": "model_member", "module_path": "project/main.fea", "model_id": "assembly",
                "member_kind": "load", "member_id": format!("load{index}") },
            "location": { "path": "project/main.fea", "span": { "start": index * 20, "end": index * 20 + 10 } }
        })).collect::<Vec<_>>() }, "diagnostics": []
    })).unwrap()
}

#[test]
fn checked_source_requires_an_explicit_supported_contract() {
    let report = checked();
    let mut wire = serde_json::to_value(&report).unwrap();
    assert_eq!(
        serde_json::from_value::<FeaSourceCheckResult>(wire.clone()).unwrap(),
        report,
    );
    wire.as_object_mut().unwrap().remove("contract");
    assert!(serde_json::from_value::<FeaSourceCheckResult>(wire.clone()).is_err());
    wire["contract"] = json!("unsupported_contract");
    assert!(serde_json::from_value::<FeaSourceCheckResult>(wire).is_err());
}

#[test]
fn hundreds_of_members_page_without_loss_and_preserve_source_navigation() {
    let source = checked();
    let mut request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::ModelMember {
            kind: FeaModelMemberKind::Load,
        },
        after: None,
        limit: 100,
        detail: FeaQueryDetail::Navigation,
    };
    let mut entries = Vec::new();
    loop {
        let page = source.browse(&request).unwrap();
        assert_eq!(page.total, 513);
        assert!(page.entries.len() <= 100);
        entries.extend(page.entries);
        request.after = page.next;
        if request.after.is_none() {
            break;
        }
    }
    assert_eq!(entries, source.source_index.entries);
    assert_eq!(entries[512].location.span.start, 10240);
}

#[test]
fn exact_identity_does_not_alias_a_same_named_member_in_another_owner() {
    let mut source = checked();
    let identity = source.source_index.entries[512].identity.clone();
    let mut other = source.source_index.entries[512].clone();
    if let FeaSourceIdentity::ModelMember { model_id, .. } = &mut other.identity {
        *model_id = "other_assembly".into();
    }
    source.source_index.entries.insert(0, other);
    let request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::Identity {
            identity: identity.clone(),
        },
        after: None,
        limit: 1,
        detail: FeaQueryDetail::Navigation,
    };
    let page = source.browse(&request).unwrap();
    assert_eq!(page.total, 1);
    assert_eq!(page.entries[0].identity, identity);
    assert!(page.next.is_none());
}

#[test]
fn cursors_cannot_cross_revision_filter_or_module_and_limits_are_bounded() {
    let source = checked();
    let mut request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::All,
        after: None,
        limit: 1,
        detail: FeaQueryDetail::Navigation,
    };
    request.after = source.browse(&request).unwrap().next;
    request.filter = FeaQueryFilter::Function;
    assert_eq!(source.browse(&request), Err(FeaQueryError::InvalidCursor));
    request.filter = FeaQueryFilter::All;
    request.module_path = Some(LogicalSourcePath::new("project/other.fea").unwrap());
    assert_eq!(source.browse(&request), Err(FeaQueryError::InvalidCursor));
    request.module_path = None;
    request.revision = SourceDigest([8; 32]);
    assert_eq!(source.browse(&request), Err(FeaQueryError::StaleRevision));
    request.revision = source.revision;
    request.limit = 101;
    assert_eq!(source.browse(&request), Err(FeaQueryError::InvalidLimit));
    assert_eq!(
        source.material(SourceDigest([8; 32]), &source.root_module, "steel"),
        Err(FeaQueryError::StaleRevision)
    );
}

#[test]
fn material_definitions_are_canonical_revision_fenced_and_response_bounded() {
    let mut source = checked();
    source.program.materials = serde_json::from_value(json!([{
        "modulePath":"project/materials.fea", "material":{
            "material_id":"steel", "name":"Structural steel", "provenance":{"kind":"authored"},
            "mass_density_kg_per_m3":7850, "elasticity":null, "damping":null,
            "thermal":null, "acoustic":null, "electrical":null, "plastic":null
        }
    }]))
    .unwrap();
    source.source_index.entries = serde_json::from_value(json!([{
        "identity":{"kind":"declaration", "module_path":"project/materials.fea",
            "symbol":{"kind":"material", "name":"steel"}},
        "location":{"path":"project/materials.fea", "span":{"start":0,"end":100}}
    }]))
    .unwrap();
    let mut request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::Declaration {
            kind: runmat_analysis_language::FeaSymbolKind::Material,
        },
        after: None,
        limit: 1,
        detail: FeaQueryDetail::Definition,
    };
    let page = source.browse(&request).unwrap();
    let FeaObjectDefinition::Material(material) = &page.details[0].definition else {
        panic!("material")
    };
    assert_eq!(material, &source.program.materials[0]);
    assert_eq!(page.details[0].identity, page.entries[0].identity);
    request.detail = FeaQueryDetail::Navigation;
    assert!(source.browse(&request).unwrap().details.is_empty());
    request.detail = FeaQueryDetail::Definition;
    request.revision = SourceDigest([8; 32]);
    assert_eq!(source.browse(&request), Err(FeaQueryError::StaleRevision));
    request.revision = source.revision;
    source.program.materials[0].material.name = "x".repeat(65536);
    assert_eq!(source.browse(&request), Err(FeaQueryError::ResponseLimit));
}

#[test]
fn definition_queries_do_not_guess_canonical_objects_for_source_only_entries() {
    let mut source = checked();
    source.source_index.entries = serde_json::from_value(json!([{
        "identity":{"kind":"function","module_path":"project/pulse.m","function_id":"pulse"},
        "location":{"path":"project/pulse.m","span":{"start":0,"end":10}}
    }]))
    .unwrap();
    let request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::All,
        after: None,
        limit: 1,
        detail: FeaQueryDetail::Definition,
    };
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::DefinitionUnavailable)
    );
}

#[test]
fn hundreds_of_definition_queries_return_only_the_requested_members() {
    let mut source = checked();
    for entry in &mut source.source_index.entries {
        let FeaSourceIdentity::ModelMember { member_kind, .. } = &mut entry.identity else {
            panic!("member")
        };
        *member_kind = FeaModelMemberKind::Section;
    }
    source.program.models = serde_json::from_value(json!([{
        "module_path":"project/main.fea", "model_id":"assembly", "thermal":null, "thermo_mechanical":null,
        "inputs":[], "includes":[], "sections": (0..513).map(|i| json!({
            "kind":"rectangular_beam", "section_id":format!("load{i}"), "width_m":0.02, "height_m":0.03
        })).collect::<Vec<_>>(), "layups":[], "orientations":[], "material_assignments":[],
        "property_assignments":[], "constraints":[], "boundaries":[], "connectors":[], "interfaces":[],
        "signals":[], "parameterized_signals":[], "excitations":[], "excitation_cases":[],
        "response_spectra":[], "random_excitations":[], "initial_conditions":[]
    }])).unwrap();
    let mut request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::ModelMember {
            kind: FeaModelMemberKind::Section,
        },
        after: None,
        limit: 25,
        detail: FeaQueryDetail::Definition,
    };
    let mut seen = 0;
    loop {
        let page = source.browse(&request).unwrap();
        assert_eq!(page.details.len(), page.entries.len());
        for (entry, details) in page.entries.iter().zip(&page.details) {
            assert_eq!(entry.identity, details.identity);
            assert!(matches!(
                details.definition,
                FeaObjectDefinition::Section(_)
            ));
        }
        seen += page.details.len();
        request.after = page.next;
        if request.after.is_none() {
            break;
        }
    }
    assert_eq!(seen, 513);
}

#[test]
fn named_region_selection_is_canonical_bounded_and_does_not_export_definitions() {
    let mut source = checked();
    source.program.geometries = serde_json::from_value(json!([{
        "module_path":"project/main.fea", "geometry_id":"assembly",
        "source":{"kind":"constructed","features":[],"result":"body"},
        "regions":(0..5000).map(|index|json!({"region_id":format!("face{index}"),"role":"face",
            "selector":{"kind":"result"},"cardinality":"one_or_more"})).collect::<Vec<_>>()
    }]))
    .unwrap();
    source.source_index.entries = serde_json::from_value(json!([{
        "identity":{"kind":"declaration","module_path":"project/main.fea","symbol":{"kind":"geometry","name":"assembly"}},
        "location":{"path":"project/main.fea","span":{"start":0,"end":20}}
    }])).unwrap();
    let mut request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        after: None,
        limit: 1,
        filter: FeaQueryFilter::GeometryRegionSelection {
            geometry: source.source_index.entries[0].identity.clone(),
            region_ids: vec!["face4999".into()],
            required_role: Some(runmat_geometry_core::GeometryRegionRole::Face),
        },
        detail: FeaQueryDetail::Navigation,
    };
    let page = source.browse(&request).unwrap();
    assert_eq!(page.entries.len(), 1);
    assert!(page.details.is_empty());
    assert!(serde_json::to_vec(&page).unwrap().len() < 1024);
    request.detail = FeaQueryDetail::Definition;
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::InvalidRegionSelection)
    );
    request.detail = FeaQueryDetail::Navigation;
    let FeaQueryFilter::GeometryRegionSelection { required_role, .. } = &mut request.filter else {
        unreachable!()
    };
    *required_role = Some(runmat_geometry_core::GeometryRegionRole::Solid);
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::IncompatibleRegionRole)
    );
    let FeaQueryFilter::GeometryRegionSelection {
        required_role,
        region_ids,
        ..
    } = &mut request.filter
    else {
        unreachable!()
    };
    *required_role = None;
    region_ids.push("missing".into());
    assert_eq!(source.browse(&request), Err(FeaQueryError::NotFound));
    let FeaQueryFilter::GeometryRegionSelection { region_ids, .. } = &mut request.filter else {
        unreachable!()
    };
    *region_ids = vec!["face0".into(); 257];
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::InvalidRegionSelection)
    );
    let FeaQueryFilter::GeometryRegionSelection { region_ids, .. } = &mut request.filter else {
        unreachable!()
    };
    *region_ids = vec!["face0".into(); 2];
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::InvalidRegionSelection)
    );
    request.revision = SourceDigest([8; 32]);
    assert_eq!(source.browse(&request), Err(FeaQueryError::StaleRevision));
}

#[test]
fn indexed_but_unresolved_definition_is_not_reported_as_a_missing_name() {
    let source = checked();
    let request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        filter: FeaQueryFilter::Identity {
            identity: source.source_index.entries[0].identity.clone(),
        },
        after: None,
        limit: 1,
        detail: FeaQueryDetail::Definition,
    };
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::DefinitionUnavailable)
    );
}

#[test]
fn region_query_reports_unresolved_geometry_without_claiming_the_name_is_missing() {
    let mut source = checked();
    source.source_index.entries = serde_json::from_value(json!([{
        "identity":{"kind":"declaration","module_path":"project/main.fea","symbol":{"kind":"geometry","name":"bracket"}},
        "location":{"path":"project/main.fea","span":{"start":0,"end":20}}
    }])).unwrap();
    let request = FeaBrowseRequest {
        revision: source.revision,
        module_path: None,
        after: None,
        limit: 1,
        filter: FeaQueryFilter::GeometryRegionSelection {
            geometry: source.source_index.entries[0].identity.clone(),
            region_ids: vec!["face_000008".into()],
            required_role: None,
        },
        detail: FeaQueryDetail::Navigation,
    };
    assert_eq!(
        source.browse(&request),
        Err(FeaQueryError::DefinitionUnavailable)
    );
}
