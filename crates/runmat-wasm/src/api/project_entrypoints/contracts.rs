use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum Kind {
    Script,
    TypedFunction,
    FeaStudy,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub(super) enum Target {
    Path { path: String },
    ModuleFunction { module: String, function: String },
}

#[derive(Serialize)]
pub(super) struct Failure {
    code: &'static str,
    message: String,
}

#[derive(Serialize)]
pub(super) struct Support {
    supported: bool,
    reason: Option<Failure>,
}
impl Support {
    pub(super) fn available() -> Self {
        Self {
            supported: true,
            reason: None,
        }
    }
    pub(super) fn unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: Some(Failure {
                code,
                message: message.into(),
            }),
        }
    }
}

#[derive(Serialize)]
pub(super) struct Description {
    pub name: String,
    pub kind: Kind,
    pub target: Target,
    pub local: Support,
    pub cluster: Support,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Context {
    pub user_settings_json: Option<String>,
    pub input_manifest: Option<runmat_filesystem::pinned::InputManifest>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct RunOptions {
    pub arguments_json: Option<Vec<u8>>,
    pub requested_outputs: Option<usize>,
    pub user_settings_json: Option<String>,
    pub input_manifest: Option<runmat_filesystem::pinned::InputManifest>,
}

pub(super) fn decode<T: serde::de::DeserializeOwned + Default>(
    value: JsValue,
) -> Result<T, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(value)
        .map_err(|error| failure("project.entrypoint.options", error))
}
pub(super) fn optional_inputs(
    value: JsValue,
) -> Result<Option<runmat_filesystem::pinned::InputManifest>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    serde_wasm_bindgen::from_value(value)
        .map(Some)
        .map_err(|error| failure("project.entrypoint.inputs", error))
}
pub(super) fn failure(code: &'static str, message: impl ToString) -> JsValue {
    crate::wire::json_contract_to_js(&Failure {
        code,
        message: message.to_string(),
    })
    .unwrap_or_else(|_| JsValue::from_str("named entrypoint error could not be encoded"))
}
pub(super) fn result(kind: Kind, value: JsValue) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"kind".into(),
        &crate::wire::json_contract_to_js(&kind)
            .map_err(|error| failure("project.entrypoint.encoding", error))?,
    )?;
    js_sys::Reflect::set(&result, &"result".into(), &value)?;
    Ok(result.into())
}
pub(super) fn code_kind(kind: runmat_execution::ExecutableEntrypointKind) -> Result<Kind, JsValue> {
    match kind {
        runmat_execution::ExecutableEntrypointKind::Script => Ok(Kind::Script),
        runmat_execution::ExecutableEntrypointKind::Function => Ok(Kind::TypedFunction),
        runmat_execution::ExecutableEntrypointKind::DeclarationOnly => Err(failure(
            "project.entrypoint.notExecutable",
            "Named target is declaration-only",
        )),
    }
}
