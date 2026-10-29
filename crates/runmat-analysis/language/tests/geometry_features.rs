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
        named_selections: None,
        unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::Declared {
            contexts: vec![runmat_geometry_core::physical::LengthUnitDeclaration {
                context: "STEP fixture #1".into(),
                source_symbol: "mm".into(),
                unit: runmat_geometry_core::LengthUnit::Millimeter,
            }],
        },
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
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
fn readable_inline_geometry_uses_asset_owned_named_selections() {
    let entity = runmat_geometry_core::PersistentEntityId::new(
        runmat_geometry_core::PersistentEntityKind::Solid,
        "exact-bracket".into(),
        vec!["root".into()],
    );
    let id = entity.analysis_selection_id().unwrap();
    let metadata = runmat_geometry_core::GeometryNamedSelections {
        schema_version: 1,
        source_digest: runmat_geometry_core::GeometryDigest::from_bytes([3; 32]),
        regions: vec![runmat_geometry_core::GeometryRegionDefinition {
            region_id: "body".into(),
            role: runmat_geometry_core::GeometryRegionRole::Solid,
            selector: runmat_geometry_core::GeometryRegionSelector::PersistentId(id.clone()),
            cardinality: runmat_geometry_core::GeometryRegionCardinality::ExactlyOne,
        }],
    };
    let source = r#"geometry bracket { file = "./bracket.brep"; units = m
        region body : solid { select name "body"; expect exactly_one }
    }
material steel { mass_density = 7850 kg/m^3 }
model loaded_bracket { assign material.steel to geometry.bracket.region.body }
"#;
    let inspection = GeometryInspection {
        authored_names: Default::default(),
        preview_extent_m: None,
        named_selections: Some(metadata),
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
        ],
        exchange: None,
        resolved_units: None,
        unit_evidence: Default::default(),
    };
    let host = TestHost::default().with_geometry_inspection("project/bracket.brep", inspection);
    let checked = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(checked.is_valid(), "{:?}", checked.diagnostics);
    assert!(!source.contains("persistent-entity:"));
    assert!(!source.contains("import geometry"));
    let geometry = checked.geometries.values().next().unwrap();
    assert_eq!(geometry.module_path, "project/main.fea");
    let runmat_geometry_core::GeometryDefinitionSource::Imported(imported) = &geometry.source
    else {
        panic!("imported")
    };
    assert_eq!(
        imported
            .named_selections
            .as_ref()
            .unwrap()
            .selection("body", runmat_geometry_core::GeometryRegionRole::Solid)
            .unwrap(),
        id
    );
    assert!(checked
        .bound
        .graph
        .source_dependencies()
        .unwrap()
        .contains(&LogicalSourcePath::new("project/.runmat/geometry/bracket.brep.json").unwrap()));
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
        named_selections: None,
        content_digest: SourceDigest([3; 32]),
        topology_digest: SourceDigest([4; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
        ],
        exchange: None,
        preview_extent_m: None,
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
