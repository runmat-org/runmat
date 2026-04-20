use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use runmat_core::{InputRequest, InputRequestKind, InputResponse};

use crate::runtime::logging::{
    ensure_runtime_log_forwarder_installed, ensure_stdout_forwarder_installed,
    ensure_trace_forwarder_installed, init_logging_once,
};
use crate::runtime::state::{
    js_stdin_handler, RUNTIME_LOG_NEXT_ID, RUNTIME_LOG_SUBSCRIBERS, STDOUT_NEXT_ID,
    STDOUT_SUBSCRIBERS, TRACE_NEXT_ID, TRACE_SUBSCRIBERS,
};
use crate::wire::errors::{js_error, js_value_to_string};

#[cfg(target_arch = "wasm32")]
pub(crate) async fn js_input_request(request: InputRequest) -> Result<InputResponse, String> {
    let handler = js_stdin_handler();
    let Some(handler) = handler else {
        return Err("stdin requested but no input handler is installed".to_string());
    };

    let js_request = js_sys::Object::new();
    js_sys::Reflect::set(
        &js_request,
        &JsValue::from_str("prompt"),
        &JsValue::from_str(&request.prompt),
    )
    .map_err(js_value_to_string)?;

    match request.kind {
        InputRequestKind::Line { echo } => {
            js_sys::Reflect::set(
                &js_request,
                &JsValue::from_str("kind"),
                &JsValue::from_str("line"),
            )
            .unwrap_or_default();
            js_sys::Reflect::set(
                &js_request,
                &JsValue::from_str("echo"),
                &JsValue::from_bool(echo),
            )
            .unwrap_or_default();
        }
        InputRequestKind::KeyPress => {
            js_sys::Reflect::set(
                &js_request,
                &JsValue::from_str("kind"),
                &JsValue::from_str("keyPress"),
            )
            .unwrap_or_default();
        }
    }

    let mut value = handler
        .call1(&JsValue::NULL, &js_request)
        .map_err(js_value_to_string)?;

    if value.is_instance_of::<js_sys::Promise>() {
        value = JsFuture::from(js_sys::Promise::from(value))
            .await
            .map_err(js_value_to_string)?;
    }

    if let Some(err) = extract_error_message(&value) {
        return Err(err);
    }

    match request.kind {
        InputRequestKind::Line { .. } => {
            if value.is_null() || value.is_undefined() {
                return Ok(InputResponse::Line(String::new()));
            }
            if let Some(text) = extract_line_value(&value) {
                return Ok(InputResponse::Line(text));
            }
            Err(
                "stdin handler must return a string (or Promise of a string) for line input"
                    .to_string(),
            )
        }
        InputRequestKind::KeyPress => Ok(InputResponse::KeyPress),
    }
}

#[cfg(target_arch = "wasm32")]
fn extract_error_message(value: &JsValue) -> Option<String> {
    if !value.is_object() {
        return None;
    }
    let obj = js_sys::Object::from(value.clone());
    js_sys::Reflect::get(&obj, &JsValue::from_str("error"))
        .ok()
        .and_then(|val| val.as_string())
}

#[cfg(target_arch = "wasm32")]
fn extract_line_value(value: &JsValue) -> Option<String> {
    if let Some(text) = value.as_string() {
        return Some(text);
    }
    if !value.is_object() {
        return None;
    }
    let obj = js_sys::Object::from(value.clone());
    if let Ok(raw) = js_sys::Reflect::get(&obj, &JsValue::from_str("value")) {
        if let Some(text) = raw.as_string() {
            return Some(text);
        }
    }
    if let Ok(raw) = js_sys::Reflect::get(&obj, &JsValue::from_str("line")) {
        if let Some(text) = raw.as_string() {
            return Some(text);
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = subscribeStdout)]
pub fn subscribe_stdout(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeStdout expects a Function"))?;
    ensure_stdout_forwarder_installed();
    let id = STDOUT_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STDOUT_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeStdout)]
pub fn subscribe_stdout(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeStdout is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeStdout)]
pub fn unsubscribe_stdout(id: u32) {
    let is_empty = STDOUT_SUBSCRIBERS.with(|cell| {
        let mut map = cell.borrow_mut();
        map.remove(&id);
        map.is_empty()
    });
    if is_empty {
        runmat_runtime::console::install_forwarder(None);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeStdout)]
pub fn unsubscribe_stdout(_id: u32) {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = subscribeRuntimeLog)]
pub fn subscribe_runtime_log(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeRuntimeLog expects a Function"))?;
    ensure_runtime_log_forwarder_installed();
    let id = RUNTIME_LOG_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeRuntimeLog)]
pub fn subscribe_runtime_log(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeRuntimeLog is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setLogFilter)]
pub fn set_log_filter(filter: &str) -> Result<(), JsValue> {
    init_logging_once();
    runmat_logging::update_log_filter(filter).map_err(|err| js_error(&err))
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = setLogFilter)]
pub fn set_log_filter(_filter: &str) -> Result<(), JsValue> {
    Err(js_error(
        "setLogFilter is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeRuntimeLog)]
pub fn unsubscribe_runtime_log(id: u32) {
    RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().remove(&id);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeRuntimeLog)]
pub fn unsubscribe_runtime_log(_id: u32) {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = subscribeTraceEvents)]
pub fn subscribe_trace_events(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeTraceEvents expects a Function"))?;
    ensure_trace_forwarder_installed();
    let id = TRACE_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TRACE_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeTraceEvents)]
pub fn subscribe_trace_events(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeTraceEvents is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeTraceEvents)]
pub fn unsubscribe_trace_events(id: u32) {
    TRACE_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().remove(&id);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeTraceEvents)]
pub fn unsubscribe_trace_events(_id: u32) {}
