//! Geometry import and normalization pipeline.

pub mod cad;
pub mod import;
pub mod normalize;
pub mod report;
pub mod sniff;

pub use import::{import_geometry, GeometryImportOptions};
pub use report::{ImportDiagnostic, ImportDiagnosticSeverity, ImportReport, ImportResult};
pub use sniff::{detect_geometry_format, GeometryFormat};

#[cfg(test)]
mod tests {
    use runmat_geometry_core::{GeometryAsset, MeshKind};

    use crate::{
        import::{import_geometry, GeometryImportOptions},
        normalize::deterministic_import_fingerprint,
        ImportResult,
    };

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";

    const DEGENERATE_STL: &str = "solid deg\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 0 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid deg\n";

    fn import(path: &str, bytes: &[u8], options: GeometryImportOptions) -> ImportResult {
        import_geometry(path, bytes, options).expect("import should succeed")
    }

    fn single_mesh(asset: &GeometryAsset) -> &runmat_geometry_core::MeshDescriptor {
        asset.meshes.first().expect("mesh descriptor must exist")
    }

    #[test]
    fn stl_import_happy_path() {
        let result = import(
            "/model.stl",
            TRIANGLE_STL.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.kind, MeshKind::Surface);
        assert_eq!(mesh.element_count, 1);
        assert_eq!(mesh.vertex_count, 3);
    }

    #[test]
    fn degenerate_triangles_are_removed() {
        let result = import(
            "/deg.stl",
            DEGENERATE_STL.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.element_count, 0);
        assert_eq!(mesh.vertex_count, 0);
    }

    #[test]
    fn capacity_guard_rejects_large_triangle_counts() {
        let options = GeometryImportOptions {
            max_triangles: Some(0),
            ..GeometryImportOptions::default()
        };
        let error = import_geometry("/too-large.stl", TRIANGLE_STL.as_bytes(), options)
            .expect_err("capacity guard should fail");
        assert!(error.to_string().contains("CAPACITY_LIMIT_EXCEEDED"));
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let first = import(
            "/deterministic.stl",
            TRIANGLE_STL.as_bytes(),
            GeometryImportOptions::default(),
        );
        let second = import(
            "/deterministic.stl",
            TRIANGLE_STL.as_bytes(),
            GeometryImportOptions::default(),
        );
        let left = deterministic_import_fingerprint(&first.asset).expect("fingerprint should work");
        let right =
            deterministic_import_fingerprint(&second.asset).expect("fingerprint should work");
        assert_eq!(left, right);
    }
}
