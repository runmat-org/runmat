use runmat_config::document::{
    migrate_legacy_desktop_config, migrate_legacy_desktop_config_between, RunmatConfigDocument,
    RunmatConfigFormat, RunmatConfigPatch,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResolvedRunmatConfig<'a> {
    desktop: &'a runmat_config::desktop::DesktopConfig,
    runtime: &'a runmat_config::runtime::RunMatRuntimeConfig,
}

#[wasm_bindgen(js_name = resolveRunmatConfig)]
pub fn resolve_runmat_config(source: String, format: String) -> Result<JsValue, JsValue> {
    let format = parse_format(&format)?;
    let document = RunmatConfigDocument::parse(source, format).map_err(config_error)?;
    serde_wasm_bindgen::to_value(&ResolvedRunmatConfig {
        desktop: document.desktop(),
        runtime: document.runtime(),
    })
    .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = patchRunmatConfig)]
pub fn patch_runmat_config(
    source: String,
    format: String,
    patch: JsValue,
) -> Result<String, JsValue> {
    let format = parse_format(&format)?;
    let patch: RunmatConfigPatch = serde_wasm_bindgen::from_value(patch)
        .map_err(|error| JsValue::from_str(&format!("invalid RunMat config patch: {error}")))?;
    let document = RunmatConfigDocument::parse(source, format).map_err(config_error)?;
    document
        .patched(&patch)
        .map(RunmatConfigDocument::into_source)
        .map_err(config_error)
}

#[wasm_bindgen(js_name = migrateLegacyRunmatConfig)]
pub fn migrate_legacy_runmat_config(source: String, format: String) -> Result<JsValue, JsValue> {
    let migration =
        migrate_legacy_desktop_config(&source, parse_format(&format)?).map_err(config_error)?;
    serde_wasm_bindgen::to_value(&migration).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = migrateLegacyRunmatConfigInto)]
pub fn migrate_legacy_runmat_config_into(
    legacy_source: String,
    destination_source: String,
    format: String,
) -> Result<JsValue, JsValue> {
    let migration = migrate_legacy_desktop_config_between(
        &legacy_source,
        RunmatConfigFormat::Toml,
        &destination_source,
        parse_format(&format)?,
    )
    .map_err(config_error)?;
    serde_wasm_bindgen::to_value(&migration).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn parse_format(format: &str) -> Result<RunmatConfigFormat, JsValue> {
    match format.to_ascii_lowercase().as_str() {
        "toml" => Ok(RunmatConfigFormat::Toml),
        "json" => Ok(RunmatConfigFormat::Json),
        _ => Err(JsValue::from_str(
            "unsupported RunMat config format; expected `toml` or `json`",
        )),
    }
}

fn config_error(error: runmat_config::document::RunmatConfigDocumentError) -> JsValue {
    JsValue::from_str(&error.to_string())
}
