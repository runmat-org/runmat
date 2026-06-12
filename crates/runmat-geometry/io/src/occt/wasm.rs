use serde::Deserialize;
use wasm_bindgen::prelude::*;

use super::{
    topology_from_raw, OcctCadFormat, OcctCadTopology, OcctRawAssemblyNode, OcctRawFaceSemantic,
    OcctRawTopology,
};
use crate::import::{GeometryImportError, GeometryImportOptions};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_name = __runmatOcctImportCad)]
    fn runmat_occt_import_cad(
        path: &str,
        format: &str,
        bytes: &[u8],
        max_triangles: f64,
    ) -> Result<String, JsValue>;
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmOcctPayload {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default)]
    format_name: String,
    vertices: Vec<f64>,
    triangles: Vec<u32>,
    #[serde(default)]
    triangle_face_ids: Vec<u64>,
    #[serde(default)]
    face_ids: Vec<u64>,
    #[serde(default)]
    face_names: Vec<String>,
    #[serde(default)]
    face_semantics: Vec<WasmOcctFaceSemanticPayload>,
    #[serde(default)]
    assembly_nodes: Vec<WasmOcctAssemblyNodePayload>,
    #[serde(default)]
    warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmOcctFaceSemanticPayload {
    face_id: u64,
    #[serde(default)]
    label_entry: String,
    #[serde(default)]
    label_name: String,
    #[serde(default)]
    label_kind: String,
    #[serde(default)]
    owner_entries: Vec<String>,
    #[serde(default)]
    owner_names: Vec<String>,
    #[serde(default)]
    owner_kinds: Vec<String>,
    #[serde(default)]
    layer_names: Vec<String>,
    #[serde(default)]
    color_type: String,
    #[serde(default)]
    color_hex_rgba: String,
    #[serde(default)]
    material_label_entry: String,
    #[serde(default)]
    material_name: String,
    #[serde(default)]
    material_description: String,
    #[serde(default)]
    material_density: String,
    #[serde(default)]
    material_density_name: String,
    #[serde(default)]
    material_density_value_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmOcctAssemblyNodePayload {
    node_id: String,
    #[serde(default)]
    parent_node_id: String,
    #[serde(default)]
    label: String,
}

pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
) -> Result<OcctCadTopology, GeometryImportError> {
    let max_triangles = options
        .max_triangles
        .map(|value| value as f64)
        .unwrap_or(f64::INFINITY);
    let payload =
        runmat_occt_import_cad(path, format.as_str(), bytes, max_triangles).map_err(|err| {
            GeometryImportError::ParseFailed(format!(
                "OCCT WASM CAD sidecar failed: {}",
                js_error_message(err)
            ))
        })?;
    let payload: WasmOcctPayload = serde_json::from_str(&payload).map_err(|err| {
        GeometryImportError::ParseFailed(format!(
            "OCCT WASM CAD sidecar returned invalid JSON: {err}"
        ))
    })?;

    let triangle_face_ids = if payload.triangle_face_ids.is_empty() {
        vec![0; payload.triangles.len() / 3]
    } else {
        payload.triangle_face_ids
    };
    let face_ids = if payload.face_ids.is_empty() {
        vec![0]
    } else {
        payload.face_ids
    };
    let format_name = if payload.format_name.is_empty() {
        format.as_str().to_string()
    } else {
        payload.format_name
    };

    topology_from_raw(
        OcctRawTopology {
            backend: payload.backend,
            format_name,
            vertices: payload.vertices,
            triangles: payload.triangles,
            triangle_face_ids,
            face_ids,
            face_names: payload.face_names,
            face_semantics: payload
                .face_semantics
                .into_iter()
                .map(|item| OcctRawFaceSemantic {
                    face_id: item.face_id,
                    label_entry: item.label_entry,
                    label_name: item.label_name,
                    label_kind: item.label_kind,
                    owner_entries: item.owner_entries,
                    owner_names: item.owner_names,
                    owner_kinds: item.owner_kinds,
                    layer_names: item.layer_names,
                    color_type: item.color_type,
                    color_hex_rgba: item.color_hex_rgba,
                    material_label_entry: item.material_label_entry,
                    material_name: item.material_name,
                    material_description: item.material_description,
                    material_density: item.material_density,
                    material_density_name: item.material_density_name,
                    material_density_value_type: item.material_density_value_type,
                })
                .collect(),
            assembly_nodes: payload
                .assembly_nodes
                .into_iter()
                .map(|item| OcctRawAssemblyNode {
                    node_id: item.node_id,
                    parent_node_id: item.parent_node_id,
                    label: item.label,
                })
                .collect(),
            warnings: payload.warnings,
        },
        options,
    )
}

fn default_backend() -> String {
    "occt-wasm".to_string()
}

fn js_error_message(value: JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| "unknown JavaScript exception".to_string())
}
