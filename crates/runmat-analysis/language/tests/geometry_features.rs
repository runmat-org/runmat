pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn imported_source_aliases_and_persistent_ids_remain_distinct() {
    let source = r#"
geometry imported_part {
    file = "./part.brep"
    units = m
    region support : face {
        select id "face_000001"
        expect exactly_one
    }
    region canonical_support : face {
        select persistent_id "face:canonical"
        expect exactly_one
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default().with_geometry("project/part.brep", 2))
        .check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", source, 1),
            },
        });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let geometry = program.geometries.values().next().unwrap();
    assert!(matches!(
        geometry.regions[0].selector,
        runmat_geometry_core::GeometryRegionSelector::SourceAlias(ref alias)
            if alias == "face_000001"
    ));
    assert!(matches!(
        geometry.regions[1].selector,
        runmat_geometry_core::GeometryRegionSelector::PersistentId(ref id)
            if id == "face:canonical"
    ));
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
