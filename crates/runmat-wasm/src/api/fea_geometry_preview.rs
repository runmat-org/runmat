//! Browser preview of a checked study's materialized geometry.
//!
//! The browser geometry host (OCCT wasm) materializes the study's geometry with the same
//! program a run prepares, tessellates it under its materialized identities and adds the
//! study's regions. The payload has the shape of the desktop's `previewFeaStudyGeometry`.
use super::fea_execution::exact_execute;
use super::session::{
    geometry_import_options_from_budget, geometry_preview_deadline, preview_payload_from_asset,
    RunMatWasm,
};
use crate::runtime::filesystem::captured::CapturedFilesystem;
use crate::wire::errors::{js_error, js_value_to_string};
use runmat_runtime::geometry::GeometryPreviewBudgetPayload;
use serde::Serialize;
use std::path::Path;
use wasm_bindgen::prelude::*;

/// Triangle ceiling when the caller's budget sets none (the desktop uses the same).
const DEFAULT_PREVIEW_MAX_TRIANGLES: u64 = 16_000_000;

#[wasm_bindgen]
impl RunMatWasm {
    /// `previewFeaStudyGeometry(path, studyId, budget?, checkedRevision?)` resolves to a
    /// `GeometryPreviewPayload` (plain JSON object). With `checkedRevision`, the preview uses
    /// that checked product of `path`; otherwise it checks `path` first.
    #[wasm_bindgen(js_name = previewFeaStudyGeometry)]
    pub async fn preview_fea_study_geometry_js(
        &self,
        path: String,
        study_id: String,
        budget: JsValue,
        checked_revision: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let budget: Option<GeometryPreviewBudgetPayload> = optional_argument(budget, "budget")?;
        let checked_revision: Option<runmat_runtime::analysis::SourceDigest> =
            optional_argument(checked_revision, "checkedRevision")?;
        let (filesystem, path) = CapturedFilesystem::simulation(&path, JsValue::NULL)?;
        let mut session = self.take_session()?;
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let _interrupt =
            super::session::install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        let deadline = geometry_preview_deadline(budget.as_ref());
        let preview = super::session::await_wasm_preparation(&cancellation, async {
            filesystem
                .run(async {
                    let source = match checked_revision {
                        Some(revision) => {
                            let source = session.checked_fea_source(revision)?;
                            if source.physical_path() != Path::new(&path) {
                                return Err(
                                    "checked preview revision belongs to a different source file"
                                        .to_string(),
                                );
                            }
                            source
                        }
                        None => std::sync::Arc::new(
                            session
                                .load_and_check_fea_source_path(Path::new(&path))
                                .await?,
                        ),
                    };
                    materialized_preview(&source, &study_id, budget.as_ref()).await
                })
                .await?
                .map_err(|error| js_error(&error))
        })
        .await?;
        drop(session);
        let inspect = runmat_runtime::geometry::geometry_inspect_payload_from_asset(
            &path,
            "brep",
            preview.kernel_representation_bytes,
            &preview.asset,
        )
        .map_err(|error| js_error(&error))?;
        let payload = preview_payload_from_asset(inspect, preview.asset, budget.as_ref(), deadline)
            .map_err(|error| js_error(&error))?;
        payload
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|error| {
                js_error(&format!(
                    "Failed to serialize FEA study geometry preview: {error}"
                ))
            })
    }
}

async fn materialized_preview(
    source: &runmat_core::CheckedFeaSessionSource,
    study_id: &str,
    budget: Option<&GeometryPreviewBudgetPayload>,
) -> Result<runmat_analysis_execution::MaterializedGeometryPreview, String> {
    let profile = serde_json::from_str(&host_call("{\"operation\":\"profile\"}").await?)
        .map_err(|error| format!("Invalid geometry profile response: {error}"))?;
    let authority = format!("typed-fea-preview:{}", uuid::Uuid::new_v4());
    let prepared = source
        .prepare_fea_study_geometry_preview(study_id, profile, &authority)
        .await?;
    let mut objects = prepared.source.objects;
    objects.push(prepared.program.descriptor);
    let options = geometry_import_options_from_budget(
        budget,
        budget
            .and_then(|item| item.max_triangles)
            .or(Some(DEFAULT_PREVIEW_MAX_TRIANGLES)),
    );
    let request = serde_json::to_string(&serde_json::json!({
        "operation": "materialize_preview",
        "program": prepared.program.program,
        "objects": objects,
        "display_name": format!("{study_id}.brep"),
        "options": options,
    }))
    .map_err(|error| error.to_string())?;
    let preview: runmat_analysis_execution::MaterializedGeometryPreview =
        serde_json::from_str(&host_call(&request).await?)
            .map_err(|error| format!("Invalid geometry preview response: {error}"))?;
    preview.asset.validate().map_err(str::to_string)?;
    Ok(preview)
}

/// One call to the browser geometry host. A preview is not a metered simulation run, so it
/// carries no worker delegation.
async fn host_call(json: &str) -> Result<String, String> {
    exact_execute(json, JsValue::UNDEFINED)
        .await
        .map_err(js_value_to_string)?
        .as_string()
        .ok_or_else(|| "Invalid geometry host response.".to_string())
}

fn optional_argument<T: serde::de::DeserializeOwned>(
    value: JsValue,
    name: &str,
) -> Result<Option<T>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    serde_wasm_bindgen::from_value(value)
        .map(Some)
        .map_err(|error| {
            js_error(&format!(
                "previewFeaStudyGeometry {name} parse failed: {error}"
            ))
        })
}
