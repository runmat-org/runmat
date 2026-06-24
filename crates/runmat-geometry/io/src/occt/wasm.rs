use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::{
    topology_from_raw, OcctCadFormat, OcctCadTopology, OcctRawAssemblyNode, OcctRawFaceSemantic,
    OcctRawTopology,
};
use crate::import::{
    GeometryImportBudgetPolicy, GeometryImportContext, GeometryImportError, GeometryImportOptions,
};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_name = __runmatOcctImportCad)]
    fn runmat_occt_import_cad(
        path: &str,
        format: &str,
        bytes: &[u8],
        max_triangles: f64,
        linear_deflection: f64,
        angular_deflection: f64,
    ) -> Result<String, JsValue>;
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmOcctPayload {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default)]
    format_name: String,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    triangle_budget: Option<u64>,
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
    context: &GeometryImportContext,
) -> Result<OcctCadTopology, GeometryImportError> {
    context.check_cancelled()?;
    let truncate_preview = options.budget_policy == GeometryImportBudgetPolicy::Truncate;
    let max_triangles = if truncate_preview {
        f64::INFINITY
    } else {
        options
            .max_triangles
            .map(|value| value as f64)
            .unwrap_or(f64::INFINITY)
    };
    let linear_deflection = options
        .tessellation_profile
        .chord_tolerance
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.01);
    let angular_deflection = options
        .tessellation_profile
        .angle_tolerance_deg
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::to_radians)
        .unwrap_or(0.5);
    let payload = runmat_occt_import_cad(
        path,
        format.as_str(),
        bytes,
        max_triangles,
        linear_deflection,
        angular_deflection,
    )
    .map_err(|err| {
        GeometryImportError::ParseFailed(format!(
            "OCCT WASM CAD sidecar failed: {}",
            js_error_message(err)
        ))
    })?;
    context.check_cancelled()?;
    let payload: WasmOcctPayload = serde_json::from_str(&payload).map_err(|err| {
        GeometryImportError::ParseFailed(format!(
            "OCCT WASM CAD sidecar returned invalid JSON: {err}"
        ))
    })?;

    let mut payload = payload;
    let rust_truncated = if truncate_preview {
        truncate_wasm_payload(&mut payload, options.max_triangles, context)?
    } else {
        false
    };
    let triangle_budget = payload
        .triangle_budget
        .or(options.max_triangles)
        .unwrap_or(u64::MAX);
    if rust_truncated {
        payload.warnings.push(format!(
            "OCCT WASM preview tessellation was truncated at the requested triangle budget of {triangle_budget} triangles"
        ));
    }

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
            truncated: payload.truncated || rust_truncated,
            triangle_budget,
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
        context,
    )
}

fn truncate_wasm_payload(
    payload: &mut WasmOcctPayload,
    max_triangles: Option<u64>,
    context: &GeometryImportContext,
) -> Result<bool, GeometryImportError> {
    let Some(max_triangles) = max_triangles else {
        return Ok(false);
    };
    let triangle_count = payload.triangles.len() / 3;
    let max_triangles = usize::try_from(max_triangles).unwrap_or(usize::MAX);
    if triangle_count <= max_triangles {
        return Ok(false);
    }

    let retained_index_count = max_triangles.saturating_mul(3);
    payload.triangles.truncate(retained_index_count);
    if !payload.triangle_face_ids.is_empty() {
        payload.triangle_face_ids.truncate(max_triangles);
    }
    compact_wasm_vertices(payload, context)?;
    payload.truncated = true;
    Ok(true)
}

fn compact_wasm_vertices(
    payload: &mut WasmOcctPayload,
    context: &GeometryImportContext,
) -> Result<(), GeometryImportError> {
    use std::collections::BTreeMap;

    let source_vertex_count = payload.vertices.len() / 3;
    let mut remap = BTreeMap::<u32, u32>::new();
    for (index, vertex_index) in payload.triangles.iter().enumerate() {
        crate::import::check_cancelled_periodic(context, index)?;
        let source_index = *vertex_index as usize;
        if source_index >= source_vertex_count {
            return Err(GeometryImportError::ParseFailed(
                "OCCT WASM sidecar returned a triangle index outside the vertex buffer".to_string(),
            ));
        }
        if !remap.contains_key(vertex_index) {
            let next = u32::try_from(remap.len()).map_err(|_| {
                GeometryImportError::ParseFailed(
                    "OCCT WASM preview exceeded u32 vertex indexing capacity".to_string(),
                )
            })?;
            remap.insert(*vertex_index, next);
        }
    }

    let mut vertices = Vec::with_capacity(remap.len().saturating_mul(3));
    for (source_index, _) in &remap {
        let start = *source_index as usize * 3;
        vertices.extend_from_slice(&payload.vertices[start..start + 3]);
    }
    for triangle_index in &mut payload.triangles {
        *triangle_index = *remap.get(triangle_index).ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "OCCT WASM preview vertex remap missed a triangle index".to_string(),
            )
        })?;
    }
    payload.vertices = vertices;
    Ok(())
}

fn default_backend() -> String {
    "occt-wasm".to_string()
}

fn js_error_message(value: JsValue) -> String {
    if let Some(message) = value.as_string() {
        return message;
    }

    if let Some(error) = value.dyn_ref::<js_sys::Error>() {
        let message = String::from(error.message());
        let stack = js_sys::Reflect::get(error, &JsValue::from_str("stack"))
            .ok()
            .and_then(|value| value.as_string())
            .unwrap_or_default();
        if !stack.is_empty() {
            return stack;
        }
        if !message.is_empty() {
            return message;
        }
    }

    js_sys::JSON::stringify(&value)
        .ok()
        .and_then(|message| message.as_string())
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("{value:?}"))
}
