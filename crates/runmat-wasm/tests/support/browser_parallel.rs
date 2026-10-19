use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::future_to_promise;

pub struct ExecutionHostClosures {
    _launch: Closure<dyn FnMut(JsValue) -> Promise>,
    _cancel: Closure<dyn FnMut(JsValue)>,
}

pub fn execution_host() -> (JsValue, ExecutionHostClosures) {
    let capabilities = Object::new();
    Reflect::set(
        &capabilities,
        &JsValue::from_str("topology"),
        &JsValue::from_str("flat"),
    )
    .unwrap();
    Reflect::set(
        &capabilities,
        &JsValue::from_str("maxWorkers"),
        &JsValue::from_f64(2.0),
    )
    .unwrap();
    let launch = Closure::<dyn FnMut(JsValue) -> Promise>::new(|request| {
        let program = Reflect::get(&request, &JsValue::from_str("program"))
            .expect("browser launch carries an exact program request");
        future_to_promise(async move { runmat_wasm::execute_program_artifact(program).await })
    });
    let cancel = Closure::<dyn FnMut(JsValue)>::new(|_| {});
    let host = Object::new();
    Reflect::set(&host, &JsValue::from_str("capabilities"), &capabilities).unwrap();
    Reflect::set(
        &host,
        &JsValue::from_str("launch"),
        launch.as_ref().unchecked_ref::<Function>(),
    )
    .unwrap();
    Reflect::set(
        &host,
        &JsValue::from_str("cancel"),
        cancel.as_ref().unchecked_ref::<Function>(),
    )
    .unwrap();
    (
        host.into(),
        ExecutionHostClosures {
            _launch: launch,
            _cancel: cancel,
        },
    )
}

pub fn source_request(name: &str, source: &str, requested_outputs: u32) -> JsValue {
    let source_payload = Object::new();
    Reflect::set(
        &source_payload,
        &JsValue::from_str("kind"),
        &JsValue::from_str("text"),
    )
    .unwrap();
    Reflect::set(
        &source_payload,
        &JsValue::from_str("name"),
        &JsValue::from_str(name),
    )
    .unwrap();
    Reflect::set(
        &source_payload,
        &JsValue::from_str("text"),
        &JsValue::from_str(source),
    )
    .unwrap();
    let request = Object::new();
    Reflect::set(&request, &JsValue::from_str("source"), &source_payload).unwrap();
    Reflect::set(
        &request,
        &JsValue::from_str("compatibility"),
        &JsValue::from_str("runmat"),
    )
    .unwrap();
    Reflect::set(
        &request,
        &JsValue::from_str("requestedOutputs"),
        &JsValue::from_f64(f64::from(requested_outputs)),
    )
    .unwrap();
    request.into()
}
