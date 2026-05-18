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
    use runmat_geometry_core::{GeometryAsset, MeshKind, SourceGeometryKind};

    use crate::{
        import::{import_geometry, GeometryImportOptions},
        normalize::deterministic_import_fingerprint,
        ImportResult,
    };

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";

    const DEGENERATE_STL: &str = "solid deg\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 0 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid deg\n";

    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\n#11=PRODUCT('Bracket_B','',(#1));\n#20=MATERIAL_DESIGNATION('Aluminum 6061');\nENDSEC;\nEND-ISO-10303-21;\n";
    const SIMPLE_OBJ: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -3 -2\n";
    const SIMPLE_PLY: &str = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n3 0 2 3\n";
    const SIMPLE_GLTF: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]\n          },\n          \"indices\": [0,1,2,0,2,3]\n        }\n      ]\n    }\n  ]\n}\n";
    const SIMPLE_GLB_HEADER: &[u8] = b"glTF\x02\x00\x00\x00";

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

    #[test]
    fn step_import_mvp_extracts_cad_metadata() {
        let result = import(
            "/assembly.step",
            SIMPLE_STEP.as_bytes(),
            GeometryImportOptions::default(),
        );

        assert_eq!(result.asset.source.importer_version, "step/v1");
        assert_eq!(result.asset.source_geometry.kind, SourceGeometryKind::Cad);
        assert_eq!(result.asset.regions.len(), 2);
        assert_eq!(
            result
                .asset
                .source_geometry
                .assembly
                .as_ref()
                .expect("assembly should exist")
                .children
                .len(),
            2
        );
        assert_eq!(result.asset.source_geometry.material_evidence.len(), 1);
    }

    #[test]
    fn step_fingerprint_is_deterministic() {
        let first = import(
            "/deterministic.step",
            SIMPLE_STEP.as_bytes(),
            GeometryImportOptions::default(),
        );
        let second = import(
            "/deterministic.step",
            SIMPLE_STEP.as_bytes(),
            GeometryImportOptions::default(),
        );
        let left = deterministic_import_fingerprint(&first.asset).expect("fingerprint should work");
        let right =
            deterministic_import_fingerprint(&second.asset).expect("fingerprint should work");
        assert_eq!(left, right);
    }

    #[test]
    fn obj_import_triangulates_faces_deterministically() {
        let result = import(
            "/mesh.obj",
            SIMPLE_OBJ.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(result.asset.source.importer_version, "obj/v1");
        assert_eq!(mesh.kind, MeshKind::Surface);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 3);
    }

    #[test]
    fn obj_fingerprint_is_deterministic() {
        let first = import(
            "/deterministic.obj",
            SIMPLE_OBJ.as_bytes(),
            GeometryImportOptions::default(),
        );
        let second = import(
            "/deterministic.obj",
            SIMPLE_OBJ.as_bytes(),
            GeometryImportOptions::default(),
        );
        let left = deterministic_import_fingerprint(&first.asset).expect("fingerprint should work");
        let right =
            deterministic_import_fingerprint(&second.asset).expect("fingerprint should work");
        assert_eq!(left, right);
    }

    #[test]
    fn ply_import_triangulates_faces_deterministically() {
        let result = import(
            "/mesh.ply",
            SIMPLE_PLY.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(result.asset.source.importer_version, "ply/v1");
        assert_eq!(mesh.kind, MeshKind::Surface);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 3);
    }

    #[test]
    fn ply_fingerprint_is_deterministic() {
        let first = import(
            "/deterministic.ply",
            SIMPLE_PLY.as_bytes(),
            GeometryImportOptions::default(),
        );
        let second = import(
            "/deterministic.ply",
            SIMPLE_PLY.as_bytes(),
            GeometryImportOptions::default(),
        );
        let left = deterministic_import_fingerprint(&first.asset).expect("fingerprint should work");
        let right =
            deterministic_import_fingerprint(&second.asset).expect("fingerprint should work");
        assert_eq!(left, right);
    }

    #[test]
    fn gltf_import_triangulates_inline_meshes_deterministically() {
        let result = import(
            "/mesh.gltf",
            SIMPLE_GLTF.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(result.asset.source.importer_version, "gltf/v1");
        assert_eq!(mesh.kind, MeshKind::Surface);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 2);
    }

    #[test]
    fn gltf_fingerprint_is_deterministic() {
        let first = import(
            "/deterministic.gltf",
            SIMPLE_GLTF.as_bytes(),
            GeometryImportOptions::default(),
        );
        let second = import(
            "/deterministic.gltf",
            SIMPLE_GLTF.as_bytes(),
            GeometryImportOptions::default(),
        );
        let left = deterministic_import_fingerprint(&first.asset).expect("fingerprint should work");
        let right =
            deterministic_import_fingerprint(&second.asset).expect("fingerprint should work");
        assert_eq!(left, right);
    }

    #[test]
    fn glb_payload_reports_typed_parse_error() {
        let error = import_geometry(
            "/binary.glb",
            SIMPLE_GLB_HEADER,
            GeometryImportOptions::default(),
        )
        .expect_err("GLB payload should fail with typed parse error");
        assert!(
            error.to_string().contains("GLB payloads are not supported"),
            "unexpected error message: {error}"
        );
    }
}
