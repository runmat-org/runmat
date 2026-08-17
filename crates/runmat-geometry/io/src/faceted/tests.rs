use super::*;

const TETRAHEDRON_STL: &str = r#"solid tetra
facet normal 0 0 -1
outer loop
vertex 0 0 0
vertex 0 1 0
vertex 1 0 0
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 0 1
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 0 0 0
vertex 0 0 1
vertex 0 1 0
endloop
endfacet
facet normal 1 1 1
outer loop
vertex 1 0 0
vertex 0 1 0
vertex 0 0 1
endloop
endfacet
endsolid tetra
"#;

const REORDERED_TETRAHEDRON_STL: &str = r#"solid tetra
facet normal 1 1 1
outer loop
vertex 0 1 0
vertex 0 0 1
vertex 1 0 0
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 0 0 1
vertex 0 1 0
vertex 0 0 0
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 1 0 0
vertex 0 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 0 1
endloop
endfacet
endsolid tetra
"#;

const MILLIMETER_TETRAHEDRON_STL: &str = r#"solid tetra
facet normal 0 0 -1
outer loop
vertex 0 0 0
vertex 0 1000 0
vertex 1000 0 0
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 0 0 0
vertex 1000 0 0
vertex 0 0 1000
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 0 0 0
vertex 0 0 1000
vertex 0 1000 0
endloop
endfacet
facet normal 1 1 1
outer loop
vertex 1000 0 0
vertex 0 1000 0
vertex 0 0 1000
endloop
endfacet
endsolid tetra
"#;

#[test]
fn faceted_import_is_canonical_across_facet_order_and_units() {
    let context = GeometryImportContext::new();
    let meter = import_faceted_solid(
        "tetra.stl",
        TETRAHEDRON_STL.as_bytes(),
        &FacetedSolidImportOptions::default(),
        &context,
    )
    .unwrap();
    let reordered = import_faceted_solid(
        "reordered.stl",
        REORDERED_TETRAHEDRON_STL.as_bytes(),
        &FacetedSolidImportOptions::default(),
        &context,
    )
    .unwrap();
    let millimeter = import_faceted_solid(
        "millimeter.stl",
        MILLIMETER_TETRAHEDRON_STL.as_bytes(),
        &FacetedSolidImportOptions {
            source_units: UnitSystem::Millimeter,
            ..FacetedSolidImportOptions::default()
        },
        &context,
    )
    .unwrap();

    assert_eq!(meter.closure.solid, reordered.closure.solid);
    assert_eq!(meter.closure.solid_bytes, reordered.closure.solid_bytes);
    assert_eq!(meter.closure.solid, millimeter.closure.solid);
    assert_eq!(meter.closure.solid_bytes, millimeter.closure.solid_bytes);
    assert_ne!(
        meter.closure.document.source.content_digest,
        reordered.closure.document.source.content_digest
    );
    assert_eq!(meter.closure.solid.vertices.len(), 4);
    assert_eq!(meter.closure.solid.triangles.len(), 4);
    assert_eq!(meter.closure.solid.shells.len(), 1);
}

#[test]
fn faceted_import_rejects_open_or_exact_sources() {
    let context = GeometryImportContext::new();
    let open = "solid open\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid open\n";
    assert!(matches!(
        import_faceted_solid(
            "open.stl",
            open.as_bytes(),
            &FacetedSolidImportOptions::default(),
            &context,
        ),
        Err(GeometryImportError::InvalidGeometry(_))
    ));
    assert!(matches!(
        import_faceted_solid(
            "part.step",
            b"ISO-10303-21;\nHEADER;\nENDSEC;",
            &FacetedSolidImportOptions::default(),
            &context,
        ),
        Err(GeometryImportError::UnsupportedFormat)
    ));
}

#[test]
fn faceted_import_validates_options_before_source_parsing() {
    let error = import_faceted_solid(
        "tetra.stl",
        TETRAHEDRON_STL.as_bytes(),
        &FacetedSolidImportOptions {
            source_units: UnitSystem::Unspecified,
            ..FacetedSolidImportOptions::default()
        },
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(error, GeometryImportError::InvalidOptions(_)));

    let oversized = import_faceted_solid(
        "tetra.stl",
        TETRAHEDRON_STL.as_bytes(),
        &FacetedSolidImportOptions {
            max_triangles: u32::MAX as u64,
            ..FacetedSolidImportOptions::default()
        },
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(oversized, GeometryImportError::InvalidOptions(_)));
}
