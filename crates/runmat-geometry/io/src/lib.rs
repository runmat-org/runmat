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
    use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
    use base64::Engine;
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
    const BOM_PREFIX: &[u8] = b"\xEF\xBB\xBF";
    const SIMPLE_GLTF_IMPLICIT_INDICES: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[0,1,0]]\n          }\n        }\n      ]\n    }\n  ]\n}\n";
    const BAD_GLTF_IMPLICIT_INDEX_COUNT: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]\n          }\n        }\n      ]\n    }\n  ]\n}\n";
    const NON_TRIANGLE_GLTF: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"mode\": 1,\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0]]\n          },\n          \"indices\": [0,1,2]\n        }\n      ]\n    }\n  ]\n}\n";
    const SIMPLE_GLB_HEADER: &[u8] = b"glTF\x02\x00\x00\x00";

    fn import(path: &str, bytes: &[u8], options: GeometryImportOptions) -> ImportResult {
        import_geometry(path, bytes, options).expect("import should succeed")
    }

    fn single_mesh(asset: &GeometryAsset) -> &runmat_geometry_core::MeshDescriptor {
        asset.meshes.first().expect("mesh descriptor must exist")
    }

    fn single_surface_mesh(asset: &GeometryAsset) -> &runmat_geometry_core::SurfaceMesh {
        asset
            .surface_meshes
            .first()
            .expect("surface mesh payload must exist")
    }

    fn has_diag(result: &ImportResult, code: &str) -> bool {
        result
            .report
            .diagnostics
            .iter()
            .any(|diag| diag.code == code)
    }

    fn binary_triangle_stl() -> Vec<u8> {
        let mut payload = vec![0u8; 84 + 50];
        payload[80..84].copy_from_slice(&1u32.to_le_bytes());
        let tri = 84usize;
        payload[tri + 12..tri + 16].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 16..tri + 20].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 20..tri + 24].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 24..tri + 28].copy_from_slice(&(1.0f32).to_le_bytes());
        payload[tri + 28..tri + 32].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 32..tri + 36].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 36..tri + 40].copy_from_slice(&(0.0f32).to_le_bytes());
        payload[tri + 40..tri + 44].copy_from_slice(&(1.0f32).to_le_bytes());
        payload[tri + 44..tri + 48].copy_from_slice(&(0.0f32).to_le_bytes());
        payload
    }

    fn gltf_accessor_data_uri_payload() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let indices = [0u16, 1, 2, 0, 2, 3];

        let mut buffer = Vec::<u8>::new();
        for vertex in positions {
            buffer.extend_from_slice(&vertex[0].to_le_bytes());
            buffer.extend_from_slice(&vertex[1].to_le_bytes());
            buffer.extend_from_slice(&vertex[2].to_le_bytes());
        }
        let index_offset = buffer.len();
        for index in indices {
            buffer.extend_from_slice(&index.to_le_bytes());
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":{byte_len}}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{positions_len}}},{{\"buffer\":0,\"byteOffset\":{index_offset},\"byteLength\":{indices_len}}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}},{{\"bufferView\":1,\"componentType\":5123,\"count\":6,\"type\":\"SCALAR\"}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}},\"indices\":1}}]}}]}}",
            byte_len = buffer.len(),
            positions_len = index_offset,
            indices_len = buffer.len() - index_offset
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_buffer_view_too_small_payload() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let mut buffer = Vec::<u8>::new();
        for vertex in positions {
            buffer.extend_from_slice(&vertex[0].to_le_bytes());
            buffer.extend_from_slice(&vertex[1].to_le_bytes());
            buffer.extend_from_slice(&vertex[2].to_le_bytes());
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":{byte_len}}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":8}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
            byte_len = buffer.len()
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_missing_buffer_byte_length_payload() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let mut buffer = Vec::<u8>::new();
        for vertex in positions {
            buffer.extend_from_slice(&vertex[0].to_le_bytes());
            buffer.extend_from_slice(&vertex[1].to_le_bytes());
            buffer.extend_from_slice(&vertex[2].to_le_bytes());
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\"}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_normalized_payload() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let mut buffer = Vec::<u8>::new();
        for vertex in positions {
            buffer.extend_from_slice(&vertex[0].to_le_bytes());
            buffer.extend_from_slice(&vertex[1].to_le_bytes());
            buffer.extend_from_slice(&vertex[2].to_le_bytes());
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":48}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\",\"normalized\":true}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_sparse_payload() -> Vec<u8> {
        let positions = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let mut buffer = Vec::<u8>::new();
        for vertex in positions {
            buffer.extend_from_slice(&vertex[0].to_le_bytes());
            buffer.extend_from_slice(&vertex[1].to_le_bytes());
            buffer.extend_from_slice(&vertex[2].to_le_bytes());
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":48}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\",\"sparse\":{{\"count\":1,\"indices\":{{\"bufferView\":0,\"componentType\":5123}},\"values\":{{\"bufferView\":0}}}}}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_stride_misaligned_payload() -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        for _ in 0..4 {
            buffer.extend_from_slice(&[0u8; 13]);
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":52}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":52,\"byteStride\":13}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
        )
        .into_bytes()
    }

    fn gltf_accessor_data_uri_count_span_overflow_payload() -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        for _ in 0..4 {
            buffer.extend_from_slice(&[0u8; 12]);
        }
        let encoded = BASE64_ENGINE.encode(&buffer);
        format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"buffers\":[{{\"uri\":\"data:application/octet-stream;base64,{encoded}\",\"byteLength\":48}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48,\"byteStride\":12}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":8,\"type\":\"VEC3\"}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}}}}]}}]}}",
        )
        .into_bytes()
    }

    fn binary_ply_payload() -> Vec<u8> {
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n";
        let vertices = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let faces = [[0i32, 1, 2], [0i32, 2, 3]];

        let mut payload = header.to_vec();
        for vertex in vertices {
            payload.extend_from_slice(&vertex[0].to_le_bytes());
            payload.extend_from_slice(&vertex[1].to_le_bytes());
            payload.extend_from_slice(&vertex[2].to_le_bytes());
        }
        for face in faces {
            payload.push(3u8);
            payload.extend_from_slice(&face[0].to_le_bytes());
            payload.extend_from_slice(&face[1].to_le_bytes());
            payload.extend_from_slice(&face[2].to_le_bytes());
        }
        payload
    }

    fn binary_ply_uint_payload() -> Vec<u8> {
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar uint vertex_indices\nend_header\n";
        let vertices = [
            [0.0f32, 0.0f32, 0.0f32],
            [1.0f32, 0.0f32, 0.0f32],
            [1.0f32, 1.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
        ];
        let faces = [[0u32, 1, 2], [0u32, 2, 3]];

        let mut payload = header.to_vec();
        for vertex in vertices {
            payload.extend_from_slice(&vertex[0].to_le_bytes());
            payload.extend_from_slice(&vertex[1].to_le_bytes());
            payload.extend_from_slice(&vertex[2].to_le_bytes());
        }
        for face in faces {
            payload.push(3u8);
            payload.extend_from_slice(&face[0].to_le_bytes());
            payload.extend_from_slice(&face[1].to_le_bytes());
            payload.extend_from_slice(&face[2].to_le_bytes());
        }
        payload
    }

    fn binary_ply_extra_vertex_property_payload() -> Vec<u8> {
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n";
        let mut payload = header.to_vec();
        payload.resize(payload.len() + 4 * 16 + 2 * (1 + 3 * 4), 0u8);
        payload
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
        let surface_mesh = single_surface_mesh(&result.asset);
        assert_eq!(surface_mesh.vertices.len(), 3);
        assert_eq!(surface_mesh.triangles, vec![[0, 1, 2]]);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
    }

    #[test]
    fn binary_stl_import_happy_path() {
        let payload = binary_triangle_stl();
        let result = import("/model.stl", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.kind, MeshKind::Surface);
        assert_eq!(mesh.element_count, 1);
        assert_eq!(mesh.vertex_count, 3);
        assert_eq!(
            single_surface_mesh(&result.asset).triangles,
            vec![[0, 1, 2]]
        );
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
        let surface_mesh = single_surface_mesh(&result.asset);
        assert!(surface_mesh.vertices.is_empty());
        assert!(surface_mesh.triangles.is_empty());
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
        let surface_mesh = single_surface_mesh(&result.asset);
        assert_eq!(surface_mesh.vertices.len(), 4);
        assert_eq!(
            surface_mesh.triangles,
            vec![[0, 1, 2], [0, 2, 3], [0, 1, 2]]
        );
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
    }

    #[test]
    fn mesh_imports_include_default_face_region_mapping() {
        let result = import(
            "/model.stl",
            TRIANGLE_STL.as_bytes(),
            GeometryImportOptions::default(),
        );

        assert_eq!(result.asset.regions.len(), 1);
        assert_eq!(result.asset.regions[0].region_id, "region_default");
        assert_eq!(result.asset.region_entity_mappings.len(), 1);
        let mapping = &result.asset.region_entity_mappings[0];
        assert_eq!(mapping.region_id, "region_default");
        assert_eq!(mapping.mesh_id, "mesh_1");
        assert_eq!(mapping.ranges.len(), 1);
        assert_eq!(mapping.ranges[0].start, 0);
        assert_eq!(mapping.ranges[0].count, 1);
    }

    #[test]
    fn obj_import_maps_grouped_faces_to_regions() {
        let payload =
            "g Fixed Face\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\ng Load Face\nf 1 3 4\n";
        let result = import(
            "/regions.obj",
            payload.as_bytes(),
            GeometryImportOptions::default(),
        );

        assert_eq!(result.asset.regions.len(), 2);
        assert_eq!(result.asset.region_entity_mappings.len(), 2);
        assert_eq!(result.asset.region_entity_mappings[0].ranges[0].start, 0);
        assert_eq!(result.asset.region_entity_mappings[1].ranges[0].start, 1);
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
    fn obj_import_supports_utf8_bom_prefixed_payloads() {
        let mut payload = BOM_PREFIX.to_vec();
        payload.extend_from_slice(SIMPLE_OBJ.as_bytes());
        let result = import("/mesh.obj", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 3);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
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
        let surface_mesh = single_surface_mesh(&result.asset);
        assert_eq!(surface_mesh.vertices.len(), 4);
        assert_eq!(
            surface_mesh.triangles,
            vec![[0, 1, 2], [0, 2, 3], [0, 2, 3]]
        );
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
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
    fn ply_import_supports_utf8_bom_prefixed_payloads() {
        let mut payload = BOM_PREFIX.to_vec();
        payload.extend_from_slice(SIMPLE_PLY.as_bytes());
        let result = import("/mesh.ply", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 3);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
    }

    #[test]
    fn ply_binary_little_endian_import_triangulates_faces_deterministically() {
        let payload = binary_ply_payload();
        let result = import("/mesh.ply", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(result.asset.source.importer_version, "ply/v1");
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 2);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
    }

    #[test]
    fn ply_binary_little_endian_uint_indices_import_triangulates_faces_deterministically() {
        let payload = binary_ply_uint_payload();
        let result = import("/mesh.ply", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(result.asset.source.importer_version, "ply/v1");
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 2);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
    }

    #[test]
    fn ply_binary_little_endian_extra_vertex_property_reports_typed_parse_error() {
        let payload = binary_ply_extra_vertex_property_payload();
        let error = import_geometry("/mesh.ply", &payload, GeometryImportOptions::default())
            .expect_err("unsupported binary PLY vertex layout should fail");
        assert!(
            error
                .to_string()
                .contains("requires vertex properties exactly"),
            "unexpected error message: {error}"
        );
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
        let surface_mesh = single_surface_mesh(&result.asset);
        assert_eq!(surface_mesh.vertices.len(), 4);
        assert_eq!(surface_mesh.triangles, vec![[0, 1, 2], [0, 2, 3]]);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_VERTEX_COUNT"));
        assert!(has_diag(&result, "GEOMETRY_IMPORT_TRIANGLE_COUNT"));
    }

    #[test]
    fn gltf_import_maps_named_meshes_to_regions() {
        let payload = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\"name\":\"Bracket Body\",\"primitives\":[{\"attributes\":{\"POSITION\":[[0,0,0],[1,0,0],[0,1,0]]},\"indices\":[0,1,2]}]}\n  ]\n}\n";
        let result = import(
            "/named.gltf",
            payload.as_bytes(),
            GeometryImportOptions::default(),
        );

        assert_eq!(result.asset.regions[0].region_id, "region_bracket_body");
        assert_eq!(result.asset.regions[0].name, "Bracket Body");
        assert_eq!(result.asset.region_entity_mappings[0].ranges[0].count, 1);
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
    fn gltf_import_supports_utf8_bom_prefixed_payloads() {
        let mut payload = BOM_PREFIX.to_vec();
        payload.extend_from_slice(SIMPLE_GLTF.as_bytes());
        let result = import("/mesh.gltf", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 2);
        assert!(has_diag(&result, "GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
    }

    #[test]
    fn gltf_import_supports_accessor_data_uri_payloads() {
        let payload = gltf_accessor_data_uri_payload();
        let result = import("/mesh.gltf", &payload, GeometryImportOptions::default());
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.element_count, 2);
        assert!(has_diag(&result, "GEOMETRY_GLTF_ACCESSOR_DATA_URI_USED"));
    }

    #[test]
    fn gltf_accessor_external_uri_reports_typed_parse_error() {
        let payload = "{\"asset\":{\"version\":\"2.0\"},\"buffers\":[{\"uri\":\"mesh.bin\",\"byteLength\":48}],\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}],\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}],\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0}}]}]}";
        let error = import_geometry(
            "/mesh.gltf",
            payload.as_bytes(),
            GeometryImportOptions::default(),
        )
        .expect_err("external GLTF URI should fail with typed parse error");
        assert!(
            error.to_string().contains("data URI"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_buffer_view_bounds_violation_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_buffer_view_too_small_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("bufferView-bounds violation should fail with typed parse error");
        assert!(
            error
                .to_string()
                .contains("declared count/stride exceeds bufferView byte range"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_missing_buffer_byte_length_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_missing_buffer_byte_length_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("missing buffer.byteLength should fail with typed parse error");
        assert!(
            error.to_string().contains("requires buffer.byteLength"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_normalized_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_normalized_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("normalized accessor should fail with typed parse error");
        assert!(
            error
                .to_string()
                .contains("normalized accessors are not supported"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_sparse_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_sparse_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("sparse accessor should fail with typed parse error");
        assert!(
            error
                .to_string()
                .contains("sparse accessors are not supported"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_stride_misaligned_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_stride_misaligned_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("misaligned accessor stride should fail with typed parse error");
        assert!(
            error.to_string().contains("not aligned to component size"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_accessor_count_span_overflow_reports_typed_parse_error() {
        let payload = gltf_accessor_data_uri_count_span_overflow_payload();
        let error = import_geometry("/mesh.gltf", &payload, GeometryImportOptions::default())
            .expect_err("count/stride span overflow should fail with typed parse error");
        assert!(
            error
                .to_string()
                .contains("declared count/stride exceeds bufferView byte range"),
            "unexpected error message: {error}"
        );
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

    #[test]
    fn gltf_non_triangle_mode_reports_typed_parse_error() {
        let error = import_geometry(
            "/bad-mode.gltf",
            NON_TRIANGLE_GLTF.as_bytes(),
            GeometryImportOptions::default(),
        )
        .expect_err("non-triangle GLTF mode should fail");
        assert!(
            error
                .to_string()
                .contains("primitive mode 1 is not supported"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn gltf_import_supports_implicit_indices_and_emits_diagnostic() {
        let result = import(
            "/mesh-implicit.gltf",
            SIMPLE_GLTF_IMPLICIT_INDICES.as_bytes(),
            GeometryImportOptions::default(),
        );
        let mesh = single_mesh(&result.asset);
        assert_eq!(mesh.element_count, 1);
        assert_eq!(mesh.vertex_count, 3);
        assert!(has_diag(&result, "GEOMETRY_GLTF_IMPLICIT_INDICES_USED"));
    }

    #[test]
    fn gltf_implicit_indices_require_triangle_multiple() {
        let error = import_geometry(
            "/mesh-implicit-bad.gltf",
            BAD_GLTF_IMPLICIT_INDEX_COUNT.as_bytes(),
            GeometryImportOptions::default(),
        )
        .expect_err("implicit indices with non-triangle cardinality should fail");
        assert!(
            error.to_string().contains("multiple of 3"),
            "unexpected error message: {error}"
        );
    }
}
