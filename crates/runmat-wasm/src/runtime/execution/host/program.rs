//! Selected programs use Runtime retirement, never literal worker-exit flags.

use super::*;
use runmat_execution_artifact::{ProgramTaskCancellation, ProgramTaskTerminal};

impl BrowserExecutionHost {
    pub(crate) async fn launch_program(
        &self,
        task_id: &str,
        worker_id: &str,
        request: &BrowserWorkerRequest,
    ) -> Result<ProgramTaskTerminal, String> {
        let expected = crate::runtime::execution::controlled::assignment(request)?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LaunchRequest<'a> {
            schema_version: u16,
            task_id: &'a str,
            worker_id: &'a str,
            program: &'a BrowserWorkerRequest,
        }
        let payload = LaunchRequest {
            schema_version: BROWSER_EXECUTION_SUBMISSION_SCHEMA_V2,
            task_id,
            worker_id,
            program: request,
        }
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .map_err(|error| error.to_string())?;
        let terminal = self.program_call("launchProgram", &payload).await?;
        terminal
            .validate_assignment(expected)
            .map_err(|error| error.to_string())?;
        if terminal.response.is_none() {
            return Err("selected program launch omitted its execution response".into());
        }
        Ok(terminal)
    }

    pub(crate) async fn cancel_program(
        &self,
        cancellation: &ProgramTaskCancellation,
    ) -> Result<ProgramTaskTerminal, String> {
        let payload = cancellation
            .serialize(
                &serde_wasm_bindgen::Serializer::new()
                    .serialize_large_number_types_as_bigints(true),
            )
            .map_err(|error| error.to_string())?;
        let terminal = self.program_call("cancelProgram", &payload).await?;
        terminal
            .validate_assignment(&cancellation.assignment)
            .map_err(|error| error.to_string())?;
        Ok(terminal)
    }

    async fn program_call(
        &self,
        method: &str,
        payload: &JsValue,
    ) -> Result<ProgramTaskTerminal, String> {
        let function = required_method(&self.target, method).map_err(js_error)?;
        let promise = function
            .call1(&self.target, payload)
            .map_err(js_error)?
            .dyn_into::<Promise>()
            .map_err(|_| format!("executionHost.{method} must return a task terminal Promise"))?;
        let terminal = JsFuture::from(promise).await.map_err(js_error)?;
        serde_wasm_bindgen::from_value(terminal)
            .map_err(|error| format!("invalid selected program terminal: {error}"))
    }
}
