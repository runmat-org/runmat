use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource, UnsavedRunBuffer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectTestLayoutRequest {
    manifest_path: String,
    manifest_content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectTestLayout {
    source_roots: Vec<String>,
    test_roots: Vec<String>,
    test_paths: Vec<String>,
    test_config_digest: String,
}

#[wasm_bindgen(js_name = projectTestLayout)]
pub fn project_test_layout(value: JsValue) -> Result<JsValue, JsValue> {
    let request: ProjectTestLayoutRequest =
        serde_wasm_bindgen::from_value(value).map_err(|error| {
            JsValue::from_str(&format!("project test layout parse failed: {error}"))
        })?;
    let manifest = if request.manifest_path.ends_with(".json") {
        runmat_config::project::parse_project_manifest_json(&request.manifest_content).map_err(
            |error| JsValue::from_str(&format!("invalid JSON project manifest: {error}")),
        )?
    } else {
        runmat_config::project::parse_project_manifest_toml(&request.manifest_content).map_err(
            |error| JsValue::from_str(&format!("invalid TOML project manifest: {error}")),
        )?
    };
    let test_config_digest = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&manifest.test).map_err(|error| {
            JsValue::from_str(&format!(
                "project test configuration encode failed: {error}"
            ))
        })?)
    );
    serde_wasm_bindgen::to_value(&ProjectTestLayout {
        source_roots: manifest
            .sources
            .roots
            .iter()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .collect(),
        test_roots: manifest
            .test
            .roots
            .iter()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .collect(),
        test_paths: manifest
            .test
            .paths
            .iter()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .collect(),
        test_config_digest,
    })
    .map_err(|error| JsValue::from_str(&format!("project test layout encode failed: {error}")))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FreezeTestSnapshotRequest {
    graph_digest: String,
    base_source_digest: String,
    test_config_digest: String,
    saved_sources: Vec<SavedRunSource>,
    #[serde(default)]
    unsaved_buffers: Vec<UnsavedRunBuffer>,
}

#[wasm_bindgen(js_name = freezeTestSnapshot)]
pub fn freeze_test_snapshot(value: JsValue) -> Result<JsValue, JsValue> {
    let request: FreezeTestSnapshotRequest =
        serde_wasm_bindgen::from_value(value).map_err(|error| {
            JsValue::from_str(&format!("test snapshot request parse failed: {error}"))
        })?;
    let snapshot = FrozenTestRunSnapshot::freeze(
        request.graph_digest,
        request.base_source_digest,
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        request.test_config_digest,
        request.saved_sources,
        request.unsaved_buffers,
    )
    .map_err(|error| JsValue::from_str(&format!("test snapshot freeze failed: {error}")))?;
    serde_wasm_bindgen::to_value(&snapshot)
        .map_err(|error| JsValue::from_str(&format!("test snapshot encode failed: {error}")))
}
