use js_sys::{Function, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use serde::Serialize;

use super::model::BrowserExecutionCapabilities;

const BROWSER_EXECUTION_SUBMISSION_SCHEMA_V2: u16 = 2;

#[derive(Clone)]
pub(crate) struct BrowserExecutionHost {
    target: JsValue,
    capabilities: BrowserExecutionCapabilities,
}

impl BrowserExecutionHost {
    pub(crate) fn new(target: JsValue) -> Result<Self, JsValue> {
        let capabilities_value = Reflect::get(&target, &JsValue::from_str("capabilities"))?;
        let capabilities =
            serde_wasm_bindgen::from_value::<BrowserExecutionCapabilities>(capabilities_value)
                .map_err(|error| {
                    JsValue::from_str(&format!("invalid executionHost capabilities: {error}"))
                })?
                .validate()
                .map_err(|error| JsValue::from_str(&error))?;
        required_method(&target, "launch")?;
        required_method(&target, "cancel")?;
        Ok(Self {
            target,
            capabilities,
        })
    }

    pub(crate) fn capabilities(&self) -> BrowserExecutionCapabilities {
        self.capabilities
    }

    pub(crate) async fn launch(
        &self,
        task_id: &str,
        worker_id: &str,
        request: &ProgramExecutionRequest,
    ) -> Result<ProgramExecutionResponse, String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LaunchRequest<'a> {
            schema_version: u16,
            task_id: &'a str,
            worker_id: &'a str,
            program: &'a ProgramExecutionRequest,
        }
        let function = required_method(&self.target, "launch").map_err(js_error)?;
        let serializer =
            serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true);
        let payload = LaunchRequest {
            schema_version: BROWSER_EXECUTION_SUBMISSION_SCHEMA_V2,
            task_id,
            worker_id,
            program: request,
        }
        .serialize(&serializer)
        .map_err(|error| error.to_string())?;
        let result = function.call1(&self.target, &payload).map_err(js_error)?;
        let resolved = JsFuture::from(Promise::resolve(&result))
            .await
            .map_err(js_error)?;
        serde_wasm_bindgen::from_value(resolved).map_err(|error| error.to_string())
    }

    pub(crate) fn cancel(&self, task_id: &str) -> Result<(), String> {
        let function = required_method(&self.target, "cancel").map_err(js_error)?;
        function
            .call1(&self.target, &JsValue::from_str(task_id))
            .map_err(js_error)?;
        Ok(())
    }
}

pub(crate) fn execution_host_from_options(
    options: &JsValue,
) -> Result<Option<BrowserExecutionHost>, JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(None);
    }
    let value = Reflect::get(options, &JsValue::from_str("executionHost"))?;
    if value.is_null() || value.is_undefined() {
        Ok(None)
    } else {
        BrowserExecutionHost::new(value).map(Some)
    }
}

fn required_method(target: &JsValue, name: &str) -> Result<Function, JsValue> {
    Reflect::get(target, &JsValue::from_str(name))?
        .dyn_into::<Function>()
        .map_err(|_| JsValue::from_str(&format!("executionHost.{name} must be a function")))
}

fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| format!("browser execution host rejected: {value:?}"))
}
