use super::*;
use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";
const SIMPLE_OBJ: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
const SIMPLE_PLY: &str = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n";
const SIMPLE_GLTF: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]\n          },\n          \"indices\": [0,1,2,0,2,3]\n        }\n      ]\n    }\n  ]\n}\n";
const NON_TRIANGLE_GLTF: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"mode\": 1,\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0]]\n          },\n          \"indices\": [0,1,2]\n        }\n      ]\n    }\n  ]\n}\n";
const SIMPLE_GLB_HEADER: &[u8] = b"glTF\x02\x00\x00\x00";
const BOM_PREFIX: &[u8] = b"\xEF\xBB\xBF";
const SIMPLE_GLTF_IMPLICIT_INDICES: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[0,1,0]]\n          }\n        }\n      ]\n    }\n  ]\n}\n";
const BAD_GLTF_IMPLICIT_INDEX_COUNT: &str = "{\n  \"asset\": {\"version\": \"2.0\"},\n  \"meshes\": [\n    {\n      \"primitives\": [\n        {\n          \"attributes\": {\n            \"POSITION\": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]\n          }\n        }\n      ]\n    }\n  ]\n}\n";

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
fn inspect_detects_stl() {
    let result =
        geometry_inspect("/part.stl", TRIANGLE_STL.as_bytes()).expect("inspect should work");
    assert_eq!(result.format, "stl");
}

#[test]
fn inspect_op_returns_typed_metadata() {
    let envelope = geometry_inspect_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-g1".to_string()), Some("request-g1".to_string())),
    )
    .expect("inspect envelope should work");

    assert_eq!(envelope.operation, "geometry.inspect");
    assert_eq!(envelope.op_version, "geometry.inspect/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g1"));
    assert_eq!(envelope.data.format, "stl");
}

#[test]
fn load_and_stats_work_for_stl() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let stats = geometry_compute_stats(&asset).expect("stats should work");
    assert_eq!(stats.mesh_count, 1);
    assert_eq!(stats.total_elements, 1);
}

#[test]
fn inspect_and_load_step_work() {
    let inspect =
        geometry_inspect("/assembly.step", SIMPLE_STEP.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "step");

    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "step/v1");
    assert_eq!(asset.regions.len(), 1);
    assert!(asset.source_geometry.assembly.is_some());
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"CAD_METADATA_PRODUCT_COUNT"));
    assert!(codes.contains(&"CAD_METADATA_MATERIAL_EVIDENCE_COUNT"));
}

#[test]
fn inspect_and_load_binary_stl_work_without_extension() {
    let payload = binary_triangle_stl();
    let inspect = geometry_inspect("/part.mesh", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "stl");

    let asset = geometry_load("/part.mesh", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "stl/v1");
    assert_eq!(asset.meshes.len(), 1);
    assert_eq!(asset.meshes[0].element_count, 1);
    assert_eq!(asset.meshes[0].vertex_count, 3);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_VERTEX_COUNT"));
    assert!(codes.contains(&"GEOMETRY_IMPORT_TRIANGLE_COUNT"));
}

#[test]
fn inspect_and_load_obj_work() {
    let inspect =
        geometry_inspect("/part.obj", SIMPLE_OBJ.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "obj");

    let asset = geometry_load("/part.obj", SIMPLE_OBJ.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "obj/v1");
    assert_eq!(asset.meshes.len(), 1);
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_VERTEX_COUNT"));
    assert!(codes.contains(&"GEOMETRY_IMPORT_TRIANGLE_COUNT"));
}

#[test]
fn inspect_and_load_obj_work_without_extension() {
    let inspect =
        geometry_inspect("/part.dat", SIMPLE_OBJ.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "obj");

    let asset = geometry_load("/part.dat", SIMPLE_OBJ.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "obj/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
}

#[test]
fn inspect_and_load_obj_work_without_extension_with_utf8_bom() {
    let mut payload = BOM_PREFIX.to_vec();
    payload.extend_from_slice(SIMPLE_OBJ.as_bytes());
    let inspect = geometry_inspect("/part.dat", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "obj");

    let asset = geometry_load("/part.dat", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "obj/v1");
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
}

#[test]
fn inspect_and_load_ply_work() {
    let inspect =
        geometry_inspect("/part.ply", SIMPLE_PLY.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "ply");

    let asset = geometry_load("/part.ply", SIMPLE_PLY.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "ply/v1");
    assert_eq!(asset.meshes.len(), 1);
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_VERTEX_COUNT"));
    assert!(codes.contains(&"GEOMETRY_IMPORT_TRIANGLE_COUNT"));
}

#[test]
fn inspect_and_load_ply_work_without_extension() {
    let inspect =
        geometry_inspect("/part.mesh", SIMPLE_PLY.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "ply");

    let asset = geometry_load("/part.mesh", SIMPLE_PLY.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "ply/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
}

#[test]
fn inspect_and_load_ply_work_without_extension_with_utf8_bom() {
    let mut payload = BOM_PREFIX.to_vec();
    payload.extend_from_slice(SIMPLE_PLY.as_bytes());
    let inspect = geometry_inspect("/part.mesh", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "ply");

    let asset = geometry_load("/part.mesh", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "ply/v1");
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
}

#[test]
fn inspect_and_load_binary_ply_work_without_extension() {
    let payload = binary_ply_payload();
    let inspect = geometry_inspect("/part.mesh", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "ply");

    let asset = geometry_load("/part.mesh", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "ply/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
}

#[test]
fn inspect_and_load_binary_ply_uint_indices_work_without_extension() {
    let payload = binary_ply_uint_payload();
    let inspect = geometry_inspect("/part.mesh", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "ply");

    let asset = geometry_load("/part.mesh", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "ply/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
}

#[test]
fn inspect_and_load_gltf_work() {
    let inspect =
        geometry_inspect("/part.gltf", SIMPLE_GLTF.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "gltf");

    let asset = geometry_load("/part.gltf", SIMPLE_GLTF.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "gltf/v1");
    assert_eq!(asset.meshes.len(), 1);
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_VERTEX_COUNT"));
    assert!(codes.contains(&"GEOMETRY_IMPORT_TRIANGLE_COUNT"));
}

#[test]
fn inspect_and_load_gltf_accessor_data_uri_work() {
    let payload = gltf_accessor_data_uri_payload();
    let inspect = geometry_inspect("/part.gltf", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "gltf");

    let asset = geometry_load("/part.gltf", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "gltf/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
    assert_eq!(asset.meshes[0].vertex_count, 4);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_GLTF_ACCESSOR_DATA_URI_USED"));
}

#[test]
fn inspect_and_load_gltf_with_implicit_indices_work() {
    let inspect = geometry_inspect(
        "/part-implicit.gltf",
        SIMPLE_GLTF_IMPLICIT_INDICES.as_bytes(),
    )
    .expect("inspect should work");
    assert_eq!(inspect.format, "gltf");

    let asset = geometry_load(
        "/part-implicit.gltf",
        SIMPLE_GLTF_IMPLICIT_INDICES.as_bytes(),
    )
    .expect("load should work");
    assert_eq!(asset.source.importer_version, "gltf/v1");
    assert_eq!(asset.meshes[0].element_count, 1);
    assert_eq!(asset.meshes[0].vertex_count, 3);
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_GLTF_IMPLICIT_INDICES_USED"));
}

#[test]
fn inspect_and_load_gltf_work_without_extension() {
    let inspect =
        geometry_inspect("/part.data", SIMPLE_GLTF.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "gltf");

    let asset = geometry_load("/part.data", SIMPLE_GLTF.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "gltf/v1");
    assert_eq!(asset.meshes[0].element_count, 2);
}

#[test]
fn inspect_and_load_gltf_work_without_extension_with_utf8_bom() {
    let mut payload = BOM_PREFIX.to_vec();
    payload.extend_from_slice(SIMPLE_GLTF.as_bytes());
    let inspect = geometry_inspect("/part.data", &payload).expect("inspect should work");
    assert_eq!(inspect.format, "gltf");

    let asset = geometry_load("/part.data", &payload).expect("load should work");
    assert_eq!(asset.source.importer_version, "gltf/v1");
    let codes = asset
        .diagnostics
        .iter()
        .map(|diag| diag.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"GEOMETRY_IMPORT_UTF8_BOM_STRIPPED"));
}

#[test]
fn load_op_maps_unsupported_format_error_code() {
    let error = geometry_load_op(
        "/bad.bin",
        b"not geometry",
        OperationContext::new(None, None),
    )
    .expect_err("load should fail");

    assert_eq!(error.error_code, "GEOMETRY_FORMAT_UNSUPPORTED");
    assert_eq!(error.operation, "geometry.load");
    assert_eq!(error.op_version, "geometry.load/v1");
}

#[test]
fn load_op_maps_parse_error_for_glb_payload() {
    let error = geometry_load_op(
        "/mesh.glb",
        SIMPLE_GLB_HEADER,
        OperationContext::new(None, None),
    )
    .expect_err("GLB payload should fail parse");
    assert_eq!(error.error_code, "GEOMETRY_PARSE_FAILED");
    assert!(error.message.contains("GLB payloads are not supported"));
}

#[test]
fn load_op_maps_parse_error_for_non_triangle_gltf_mode() {
    let error = geometry_load_op(
        "/mesh.gltf",
        NON_TRIANGLE_GLTF.as_bytes(),
        OperationContext::new(None, None),
    )
    .expect_err("non-triangle GLTF mode should fail parse");
    assert_eq!(error.error_code, "GEOMETRY_PARSE_FAILED");
    assert!(error.message.contains("mode 1 is not supported"));
}

#[test]
fn load_op_maps_parse_error_for_bad_implicit_gltf_indices() {
    let error = geometry_load_op(
        "/mesh.gltf",
        BAD_GLTF_IMPLICIT_INDEX_COUNT.as_bytes(),
        OperationContext::new(None, None),
    )
    .expect_err("bad implicit GLTF indices should fail parse");
    assert_eq!(error.error_code, "GEOMETRY_PARSE_FAILED");
    assert!(error.message.contains("multiple of 3"));
}

#[test]
fn load_op_maps_parse_error_for_unsupported_binary_ply_layout() {
    let payload = binary_ply_extra_vertex_property_payload();
    let error = geometry_load_op("/mesh.ply", &payload, OperationContext::new(None, None))
        .expect_err("unsupported binary PLY layout should fail parse");
    assert_eq!(error.error_code, "GEOMETRY_PARSE_FAILED");
    assert!(error.message.contains("requires vertex properties exactly"));
}

#[test]
fn load_op_maps_parse_error_for_gltf_accessor_buffer_view_bounds_violation() {
    let payload = gltf_accessor_data_uri_buffer_view_too_small_payload();
    let error = geometry_load_op("/mesh.gltf", &payload, OperationContext::new(None, None))
        .expect_err("gltf accessor bufferView bounds violation should fail parse");
    assert_eq!(error.error_code, "GEOMETRY_PARSE_FAILED");
    assert!(error.message.contains("out of bounds"));
}

#[test]
fn list_regions_op_returns_versioned_envelope() {
    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    let envelope = geometry_list_regions_op(
        &asset,
        OperationContext::new(Some("trace-g2".to_string()), None),
    )
    .expect("list regions should work");

    assert_eq!(envelope.operation, "geometry.list_regions");
    assert_eq!(envelope.op_version, "geometry.list_regions/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g2"));
    assert_eq!(envelope.data.regions.len(), 1);
}

#[test]
fn query_entities_op_returns_deterministic_nodes() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let envelope = geometry_query_entities_op(
        &asset,
        GeometryEntityQuery {
            region_id: None,
            mesh_id: None,
            entity_kind: EntityKind::Node,
            limit: Some(2),
        },
        OperationContext::new(Some("trace-g3".to_string()), None),
    )
    .expect("query should work");

    assert_eq!(envelope.operation, "geometry.query_entities");
    assert_eq!(envelope.op_version, "geometry.query_entities/v1");
    assert_eq!(envelope.data.entities.len(), 2);
    assert!(envelope.data.truncated);
    assert_eq!(envelope.data.entities[0].entity_id, 0);
    assert_eq!(envelope.data.entities[1].entity_id, 1);
}

#[test]
fn query_entities_region_not_found_maps_typed_error() {
    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    let error = geometry_query_entities_op(
        &asset,
        GeometryEntityQuery {
            region_id: Some("missing".to_string()),
            mesh_id: None,
            entity_kind: EntityKind::Node,
            limit: Some(8),
        },
        OperationContext::new(None, None),
    )
    .expect_err("query should fail");

    assert_eq!(error.error_code, "GEOMETRY_REGION_NOT_FOUND");
    assert_eq!(error.operation, "geometry.query_entities");
    assert_eq!(error.op_version, "geometry.query_entities/v1");
}

#[test]
fn capture_view_op_returns_typed_unsupported_error() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let error = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 1024,
            height: 768,
        },
        OperationContext::new(Some("trace-g4".to_string()), None),
    )
    .expect_err("capture should be unsupported for now");

    assert_eq!(error.operation, "geometry.capture_view");
    assert_eq!(error.op_version, "geometry.capture_view/v1");
    assert_eq!(error.error_code, "GEOMETRY_CAPTURE_UNSUPPORTED");
    assert_eq!(error.trace_id.as_deref(), Some("trace-g4"));
}

#[test]
fn capture_view_op_uses_default_svg_renderer_without_adapter() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let envelope = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "svg".to_string(),
            width: 640,
            height: 360,
        },
        OperationContext::new(Some("trace-g4-svg".to_string()), None),
    )
    .expect("svg capture should succeed");

    assert_eq!(envelope.operation, "geometry.capture_view");
    assert_eq!(envelope.op_version, "geometry.capture_view/v1");
    assert_eq!(envelope.data.format, "svg");
    assert_eq!(envelope.data.width, 640);
    assert_eq!(envelope.data.height, 360);
    let payload = std::str::from_utf8(&envelope.data.payload).expect("svg payload utf8");
    assert!(payload.starts_with("<svg"));
    assert!(payload.contains("Geometry Snapshot"));
}

#[test]
fn capture_view_op_validates_non_zero_dimensions() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let error = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 0,
            height: 768,
        },
        OperationContext::new(None, None),
    )
    .expect_err("zero width should be rejected");

    assert_eq!(error.error_code, "GEOMETRY_CAPTURE_INVALID_SPEC");
    assert_eq!(error.operation, "geometry.capture_view");
    assert_eq!(error.op_version, "geometry.capture_view/v1");
}

#[test]
fn capture_view_op_uses_installed_adapter() {
    struct TestCaptureAdapter;

    impl GeometryViewCaptureAdapter for TestCaptureAdapter {
        fn adapter_name(&self) -> &'static str {
            "test-capture"
        }

        fn capture(
            &self,
            _asset: &runmat_geometry_core::GeometryAsset,
            view_spec: &GeometryCaptureViewSpec,
        ) -> Result<GeometryCaptureViewResult, String> {
            Ok(GeometryCaptureViewResult {
                format: view_spec.format.clone(),
                width: view_spec.width,
                height: view_spec.height,
                payload: vec![1, 2, 3, 4],
            })
        }
    }

    static ADAPTER: TestCaptureAdapter = TestCaptureAdapter;
    let _guard = ThreadGeometryCaptureAdapterGuard::set(Some(&ADAPTER));

    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let envelope = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 640,
            height: 480,
        },
        OperationContext::new(Some("trace-g5".to_string()), None),
    )
    .expect("capture should succeed");

    assert_eq!(envelope.operation, "geometry.capture_view");
    assert_eq!(envelope.op_version, "geometry.capture_view/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g5"));
    assert_eq!(envelope.data.format, "png");
    assert_eq!(envelope.data.width, 640);
    assert_eq!(envelope.data.height, 480);
    assert_eq!(envelope.data.payload, vec![1, 2, 3, 4]);
}

#[test]
fn prep_for_analysis_op_returns_versioned_deterministic_result() {
    reset_prep_artifact_store_for_tests();
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let first = geometry_prep_for_analysis_op(
        &asset,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(Some("trace-g6".to_string()), None),
    )
    .expect("prep should work");
    let second = geometry_prep_for_analysis_op(
        &asset,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(Some("trace-g7".to_string()), None),
    )
    .expect("prep should be deterministic");

    assert_eq!(first.operation, "geometry.prep_for_analysis");
    assert_eq!(first.op_version, "geometry.prep_for_analysis/v1");
    assert!(!first.data.prep_artifact_id.is_empty());
    assert_ne!(first.data.prep_artifact_id, second.data.prep_artifact_id);
    assert_eq!(first.data.prep, second.data.prep);
    assert_eq!(
        first.data.prep.schema_version,
        "geometry-prep-for-analysis/v1"
    );
    assert!(!first.data.prep.prepared_meshes.is_empty());
    assert!(!first.data.prep.region_mappings.is_empty());
}

#[test]
fn prep_for_analysis_op_maps_invalid_spec_error() {
    reset_prep_artifact_store_for_tests();
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let error = geometry_prep_for_analysis_op(
        &asset,
        GeometryPrepForAnalysisSpec {
            profile: GeometryPrepProfile::AnalysisReady,
            target_element_budget: 0,
        },
        OperationContext::new(None, None),
    )
    .expect_err("zero element budget should fail");

    assert_eq!(error.operation, "geometry.prep_for_analysis");
    assert_eq!(error.op_version, "geometry.prep_for_analysis/v1");
    assert_eq!(error.error_code, "GEOMETRY_PREP_INVALID_SPEC");
}

#[test]
fn prep_for_analysis_op_supports_adaptive_refine_profile() {
    reset_prep_artifact_store_for_tests();
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let analysis_ready = geometry_prep_for_analysis_op(
        &asset,
        GeometryPrepForAnalysisSpec {
            profile: GeometryPrepProfile::AnalysisReady,
            target_element_budget: 4_000,
        },
        OperationContext::new(None, None),
    )
    .expect("analysis-ready prep should work");
    let adaptive_refine = geometry_prep_for_analysis_op(
        &asset,
        GeometryPrepForAnalysisSpec {
            profile: GeometryPrepProfile::AdaptiveRefine,
            target_element_budget: 4_000,
        },
        OperationContext::new(None, None),
    )
    .expect("adaptive-refine prep should work");

    assert_eq!(adaptive_refine.operation, "geometry.prep_for_analysis");
    assert_eq!(adaptive_refine.op_version, "geometry.prep_for_analysis/v1");
    assert!(
        adaptive_refine.data.prep.quality.min_scaled_jacobian
            >= analysis_ready.data.prep.quality.min_scaled_jacobian
    );
    assert!(
        adaptive_refine.data.prep.quality.mean_aspect_ratio
            <= analysis_ready.data.prep.quality.mean_aspect_ratio
    );
}

#[test]
fn prep_artifact_retention_prunes_old_entries() {
    reset_prep_artifact_store_for_tests();
    std::env::set_var("RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY", "2");
    std::env::remove_var("RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS");
    std::env::remove_var("RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS");

    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let mut ids = Vec::new();
    for _ in 0..4 {
        let prep = geometry_prep_for_analysis_op(
            &asset,
            GeometryPrepForAnalysisSpec::default(),
            OperationContext::new(None, None),
        )
        .expect("prep should succeed");
        ids.push(prep.data.prep_artifact_id);
    }

    assert!(load_prep_artifact(&ids[0])
        .expect("load prep should succeed")
        .is_none());
    assert!(load_prep_artifact(ids.last().expect("latest id"))
        .expect("load latest prep should succeed")
        .is_some());

    std::env::remove_var("RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY");
    reset_prep_artifact_store_for_tests();
}

#[test]
fn prep_artifact_health_reports_metrics_and_distribution() {
    reset_prep_artifact_store_for_tests();
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    for _ in 0..3 {
        let _ = geometry_prep_for_analysis_op(
            &asset,
            GeometryPrepForAnalysisSpec::default(),
            OperationContext::new(None, None),
        )
        .expect("prep should succeed");
    }

    let health = geometry_prep_artifact_health_op(
        GeometryPrepArtifactHealthQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("health op should succeed");
    assert_eq!(health.operation, "geometry.prep_artifact_health");
    assert_eq!(health.op_version, "geometry.prep_artifact_health/v1");
    assert_eq!(
        health.data.schema_version,
        "geometry-prep-artifact-health/v1"
    );
    assert!(health.data.current_artifact_count >= 1);
    assert!(
        health.data.age_p95_seconds.unwrap_or(0.0) >= health.data.age_p50_seconds.unwrap_or(0.0)
    );
    assert!(!health.data.per_geometry.is_empty());

    reset_prep_artifact_store_for_tests();
}
