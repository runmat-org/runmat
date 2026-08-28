use std::future::pending;

use js_sys::{Function, Promise, Reflect};
use runmat_test_runner::host::{CancellationPort, Clock, PortFuture};
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BrowserClock;

impl Clock for BrowserClock {
    fn now_ms(&self) -> u64 {
        js_sys::Date::now().max(0.0) as u64
    }

    fn sleep_until<'a>(&'a self, deadline_ms: u64) -> PortFuture<'a, ()> {
        let delay = deadline_ms
            .saturating_sub(self.now_ms())
            .min(i32::MAX as u64) as i32;
        Box::pin(async move {
            let promise = Promise::new(&mut |resolve, reject| {
                let global = js_sys::global();
                let timeout = Reflect::get(&global, &JsValue::from_str("setTimeout"))
                    .ok()
                    .and_then(|value| value.dyn_into::<Function>().ok());
                if let Some(timeout) = timeout {
                    if let Err(error) =
                        timeout.call2(&global, &resolve, &JsValue::from_f64(f64::from(delay)))
                    {
                        let _ = reject.call1(&JsValue::UNDEFINED, &error);
                    }
                } else {
                    let _ = reject.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("setTimeout is unavailable"),
                    );
                }
            });
            let _ = JsFuture::from(promise).await;
        })
    }
}

#[derive(Clone)]
pub(super) struct BrowserCancellation {
    backend: JsValue,
}

impl BrowserCancellation {
    pub fn new(backend: JsValue) -> Self {
        Self { backend }
    }
}

impl CancellationPort for BrowserCancellation {
    fn is_cancelled(&self) -> bool {
        sync_call(&self.backend, "isCancelled")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    }

    fn reason(&self) -> Option<String> {
        sync_call(&self.backend, "cancellationReason").and_then(|value| value.as_string())
    }

    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(async move {
            let Some(value) = sync_call(&self.backend, "waitForCancellation") else {
                return pending().await;
            };
            let promise = Promise::resolve(&value);
            JsFuture::from(promise)
                .await
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "browser run cancelled".into())
        })
    }
}

fn sync_call(target: &JsValue, name: &str) -> Option<JsValue> {
    let function = Reflect::get(target, &JsValue::from_str(name))
        .ok()?
        .dyn_into::<Function>()
        .ok()?;
    function.call0(target).ok()
}
