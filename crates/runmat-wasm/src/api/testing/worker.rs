use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::api::session::{install_wasm_interrupt, RunMatWasm};
use crate::wire::errors::js_error;

#[wasm_bindgen]
impl RunMatWasm {
    /// Execute one exact planned test in this session. Browser worker adapters
    /// use this narrow endpoint; scheduling remains in the portable Rust
    /// coordinator.
    // A RunMatWasm session is intentionally single-owner and non-reentrant.
    // The mutable borrow is the execution lease for the complete async attempt.
    #[allow(clippy::await_holding_refcell_ref)]
    #[wasm_bindgen(js_name = executeTestAttempt)]
    pub async fn execute_test_attempt_js(&self, input: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let input: TestAttemptInput = serde_wasm_bindgen::from_value(input)
            .map_err(|error| js_error(&format!("Test attempt input is invalid: {error}")))?;
        let cancellation = Arc::new(AtomicBool::new(false));
        let _interrupt = install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        let mut session = self.session.borrow_mut();
        let attempt = session
            .execute_planned_test(
                &input.snapshot,
                &input.plan,
                &input.test_id,
                input.attempt,
                cancellation,
            )
            .await
            .map_err(|error| js_error(&format!("Test attempt failed: {error}")))?;
        serde_wasm_bindgen::to_value(&SerializableAttempt::from(attempt))
            .map_err(|error| js_error(&format!("Test attempt serialization failed: {error}")))
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TestAttemptInput {
    snapshot: runmat_test::discovery::FrozenTestRunSnapshot,
    plan: runmat_test::plan::TestPlan,
    test_id: runmat_test::identity::TestId,
    attempt: u32,
}

#[derive(serde::Serialize)]
struct SerializableAttempt {
    result: runmat_test::result::AttemptResult,
    events: Vec<runmat_test::event::TestEvent>,
    coverage: Vec<runmat_test::coverage::CoverageFragment>,
}

impl From<runmat_core::testing::CoreTestAttempt> for SerializableAttempt {
    fn from(value: runmat_core::testing::CoreTestAttempt) -> Self {
        Self {
            result: value.result,
            events: value.events,
            coverage: value.coverage,
        }
    }
}
