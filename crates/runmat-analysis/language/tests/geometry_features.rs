pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

// witness: FEA-AUTHORING-GEOMETRY-001/persistent_topology @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Compiler checks source alias versus persistent-id selectors remain distinct typed
//   values; no native CAD feature history, regeneration or AP242 export evidence.
#[test]
fn imported_source_aliases_are_rejected() {
    let source = r#"geometry part {
        file = "./part.brep"
        units = m
        region support : face { select id "face_000001"; expect exactly_one }
    }"#;
    let program = FeaCompiler::new(&TestHost::default().with_geometry("project/part.brep", 2))
        .check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", source, 1),
            },
        });
    assert!(!program.is_valid());
    assert!(format!("{:?}", program.diagnostics).contains("canonical occurrence-qualified ID"));
}

#[test]
fn canonical_occurrence_selection_is_preserved_by_the_compiler() {
    let entity = runmat_geometry_core::PersistentEntityId::new(
        runmat_geometry_core::PersistentEntityKind::Face,
        "exact-face".into(),
        vec!["root".into(), "part-a".into()],
    );
    let id = entity.analysis_selection_id().unwrap();
    let source = r#"geometry part {
        file = "./part.brep"
        units = m
        region support : face { select persistent_id "CANONICAL"; expect exactly_one }
    }"#
    .replace("CANONICAL", &id);
    let program = FeaCompiler::new(&TestHost::default().with_geometry("project/part.brep", 2))
        .check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", &source, 1),
            },
        });
    assert!(program.is_valid(), "{:?}", program.diagnostics);
    assert!(
        matches!(&program.geometries.values().next().unwrap().regions[0].selector,
        runmat_geometry_core::GeometryRegionSelector::PersistentId(value) if value == &id)
    );
}

#[test]
fn local_edge_regions_lower_into_typed_feature_selections() {
    let source = r#"
geometry rounded_block {
    construct {
        feature stock = box(width = 1 m, depth = 1 m, height = 1 m)
        feature rounded = fillet(
            target = feature.stock,
            edges = geometry.rounded_block.region.stock_edges,
            radius = 10 mm,
        )
        result = feature.rounded
    }
    region stock_edges : edge {
        select feature.stock
        expect one_or_more
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let geometry = program.geometries.values().next().unwrap();
    let runmat_geometry_core::GeometryDefinitionSource::Constructed(graph) = &geometry.source
    else {
        panic!("expected constructed geometry");
    };
    let runmat_geometry_core::GeometryFeatureOperation::Fillet { edges, .. } =
        &graph.features[1].operation
    else {
        panic!("expected fillet operation");
    };
    assert_eq!(edges.role, runmat_geometry_core::GeometryRegionRole::Edge);
    assert!(matches!(
        edges.selector,
        runmat_geometry_core::GeometryRegionSelector::FeatureResult(ref feature)
            if feature.0 == "stock"
    ));
    assert_eq!(
        edges.cardinality,
        runmat_geometry_core::GeometryRegionCardinality::OneOrMore
    );
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A B-rep geometry without authored units fails closed at check with a located diagnostic naming the missing units.
#[test]
fn imported_geometry_without_units_names_the_missing_units() {
    let source = r#"
geometry bracket {
    file = "./part.brep"
}
"#;
    let program = FeaCompiler::new(&TestHost::default().with_geometry("project/part.brep", 2))
        .check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", source, 1),
            },
        });
    assert!(!program.is_valid());
    let messages = program
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("`bracket` needs `units = mm`, `m`, or `in`")),
        "{messages:#?}"
    );
    assert!(
        !messages
            .iter()
            .any(|message| message.contains("no inspection facts")),
        "{messages:#?}"
    );
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Omitted units resolve to the single unit the STEP file declares, without diagnostics.
#[test]
fn imported_geometry_without_units_uses_the_file_declared_unit() {
    let source = r#"
geometry bracket {
    file = "./part.step"
}
"#;
    let inspection = GeometryInspection {
        preview_extent_m: None,
        preview_measure: None,
        unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::Declared {
            contexts: vec![runmat_geometry_core::physical::LengthUnitDeclaration {
                context: "STEP fixture #1".into(),
                source_symbol: "mm".into(),
                unit: runmat_geometry_core::LengthUnit::Millimeter,
            }],
        },
        shared_faces: Default::default(),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
            GeometryInspectionCapability::AuthoredNames,
        ],
        exchange: None,
        resolved_units: Some(runmat_geometry_core::LengthUnit::Millimeter),
        authored_names: Default::default(),
    };
    let program = FeaCompiler::new(
        &TestHost::default().with_geometry_inspection("project/part.step", inspection),
    )
    .check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let geometry = program.geometries.values().next().unwrap();
    assert!(matches!(
        &geometry.source,
        runmat_geometry_core::GeometryDefinitionSource::Imported(imported)
            if imported.unit_evidence.uniform_unit() == Some(runmat_geometry_core::LengthUnit::Millimeter)
    ));
}

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Authored `units = m` over a STEP file declaring millimetres resolves to metres and reports a located units-override warning; the check stays valid.
#[test]
fn select_name_resolves_only_names_authored_in_the_cad_file() {
    let source = r#"geometry bracket { file = "./bracket.step"; units = m
        region body : solid { select name "body"; expect exactly_one }
        region pad : face { select name "pad"; expect exactly_one }
    }
material steel { mass_density = 7850 kg/m^3 }
model loaded_bracket { assign material.steel to geometry.bracket.region.body }
"#;
    let inspection = GeometryInspection {
        authored_names: std::collections::BTreeMap::from([(
            "solid".to_string(),
            vec!["body".to_string()],
        )]),
        preview_extent_m: None,
        preview_measure: None,
        shared_faces: Default::default(),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
            GeometryInspectionCapability::AuthoredNames,
        ],
        exchange: None,
        resolved_units: None,
        unit_evidence: Default::default(),
    };
    let host = TestHost::default().with_geometry_inspection("project/bracket.step", inspection);
    let checked = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    let messages = checked
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    // The authored solid name resolves; the face name the CAD file does not carry is refused,
    // with no other source of names to fall back on.
    assert!(
        !messages.iter().any(|message| message.contains("\"body\"")),
        "{messages:#?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("selects face name \"pad\"")
                && message.contains("names no faces")),
        "{messages:#?}"
    );
    // The CAD file is the only geometry dependency: no selection metadata path is read.
    let dependencies = checked.bound.graph.source_dependencies().unwrap();
    assert!(
        dependencies
            .iter()
            .all(|path| !path.as_str().contains(".runmat/")),
        "{dependencies:?}"
    );
}

#[test]
fn a_name_carried_by_two_entities_fails_an_exactly_one_region_at_check() {
    let source = r#"geometry plate { file = "./plate.step"; units = m
        region coolant : solid { select name "coolant"; expect exactly_one }
        region both : solid { select name "coolant"; expect one_or_more }
    }
"#;
    let inspection = GeometryInspection {
        authored_names: std::collections::BTreeMap::from([(
            "solid".to_string(),
            vec![
                "coolant".to_string(),
                "coolant".to_string(),
                "plate".to_string(),
            ],
        )]),
        preview_extent_m: None,
        preview_measure: None,
        shared_faces: Default::default(),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
            GeometryInspectionCapability::AuthoredNames,
        ],
        exchange: None,
        resolved_units: None,
        unit_evidence: Default::default(),
    };
    let host = TestHost::default().with_geometry_inspection("project/plate.step", inspection);
    let checked = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    let messages = checked
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.contains("which 2 solids"))
            .count(),
        1,
        "{messages:#?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("region `coolant`")),
        "{messages:#?}"
    );
}

fn named_face_inspection(capabilities: Vec<GeometryInspectionCapability>) -> GeometryInspection {
    GeometryInspection {
        authored_names: std::collections::BTreeMap::from([(
            "face".to_string(),
            vec!["mount".to_string(), "top".to_string()],
        )]),
        preview_extent_m: None,
        preview_measure: None,
        shared_faces: Default::default(),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities,
        exchange: None,
        resolved_units: None,
        unit_evidence: Default::default(),
    }
}

fn check_puck_regions(inspection: GeometryInspection) -> CheckedFeaProgram {
    let source = r#"geometry puck { file = "./puck.step"; units = m
        region pressed : face { select name "mounting"; expect exactly_one }
        region seat : face { select name "mount"; expect exactly_one }
    }
"#;
    let host = TestHost::default().with_geometry_inspection("project/puck.step", inspection);
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

// Known authored names are checked on their own capability: persistent IDs are a separate fact.
#[test]
fn authored_names_are_checked_without_persistent_ids() {
    let checked = check_puck_regions(named_face_inspection(vec![
        GeometryInspectionCapability::AuthoredNames,
    ]));
    let errors = checked
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error)
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        errors,
        [
            "region `pressed` selects face name \"mounting\", which `project/puck.step` does not carry; named faces: \"mount\", \"top\""
        ],
        "{:#?}",
        checked.diagnostics
    );
    assert!(checked
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.location.is_some()));
}

// The refusal points at the region's own selector, not the geometry block, and suggests only
// names the author wrote: generated topology references stay matchable but aren't listed.
#[test]
fn an_unknown_name_is_refused_at_its_region_and_lists_only_authored_names() {
    let mut inspection = named_face_inspection(vec![GeometryInspectionCapability::AuthoredNames]);
    inspection
        .authored_names
        .get_mut("face")
        .unwrap()
        .push("feature.puck.face.side(s.c)".to_string());
    let checked = check_puck_regions(inspection);
    let refusal = checked
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error)
        .expect("the unknown name is refused");
    assert_eq!(
        refusal.message,
        "region `pressed` selects face name \"mounting\", which `project/puck.step` does not carry; named faces: \"mount\", \"top\""
    );
    let source = r#"geometry puck { file = "./puck.step"; units = m
        region pressed : face { select name "mounting"; expect exactly_one }"#;
    let region_start = source.find("region pressed").unwrap();
    let span = refusal.location.as_ref().unwrap().span;
    assert!(
        span.start >= region_start && span.start < source.len(),
        "refusal at {span:?}, region starts at {region_start}"
    );
}

// A host that cannot read a file's names says so at the geometry, for each `select name`
// region; the check does not pass the name silently.
#[test]
fn select_name_without_readable_names_is_flagged_as_unconfirmed() {
    let checked = check_puck_regions(named_face_inspection(vec![
        GeometryInspectionCapability::Topology,
        GeometryInspectionCapability::PersistentIds,
    ]));
    let unconfirmed = checked
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "fea.semantic.geometry_names_unconfirmed")
        .collect::<Vec<_>>();
    assert_eq!(unconfirmed.len(), 2, "{:#?}", checked.diagnostics);
    assert!(unconfirmed.iter().all(|diagnostic| {
        diagnostic.severity == FeaDiagnosticSeverity::Warning
            && diagnostic.location.is_some()
            && diagnostic
                .message
                .contains("cannot be confirmed without the exact geometry backend")
            && diagnostic.message.contains("`project/puck.step`")
    }));
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("does not carry")),
        "{:#?}",
        checked.diagnostics
    );
}

#[test]
fn authored_units_override_a_different_file_declaration_with_a_warning() {
    let source = r#"
geometry channel {
    file = "./channel.step"
    units = m
}
"#;
    let inspection = GeometryInspection {
        shared_faces: Default::default(),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
            GeometryInspectionCapability::AuthoredNames,
        ],
        exchange: None,
        preview_extent_m: None,
        preview_measure: None,
        resolved_units: Some(runmat_geometry_core::LengthUnit::Meter),
        unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::Declared {
            contexts: vec![runmat_geometry_core::physical::LengthUnitDeclaration {
                context: "STEP representation #1".into(),
                source_symbol: "mm".into(),
                unit: runmat_geometry_core::LengthUnit::Millimeter,
            }],
        },
        authored_names: Default::default(),
    };
    let program = FeaCompiler::new(
        &TestHost::default().with_geometry_inspection("project/channel.step", inspection),
    )
    .check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let warning = program
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "fea.semantic.geometry_units_override")
        .expect("units override must be reported");
    assert_eq!(warning.severity, FeaDiagnosticSeverity::Warning);
    assert!(warning.location.is_some());
    assert!(
        warning.message.contains("declares millimeters"),
        "{}",
        warning.message
    );
    let geometry = program.geometries.values().next().unwrap();
    assert!(matches!(
        &geometry.source,
        runmat_geometry_core::GeometryDefinitionSource::Imported(imported)
            if imported.authored_units == Some(runmat_geometry_core::LengthUnit::Meter)
                && imported.unit_evidence.overridden_declared_unit()
                    == Some(runmat_geometry_core::LengthUnit::Millimeter)
    ));
}
