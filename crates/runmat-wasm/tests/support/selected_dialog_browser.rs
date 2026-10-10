use std::cell::{Cell, RefCell};
use std::rc::Rc;

use js_sys::{Object, Promise, Reflect};
use runmat_wasm::{init_runmat, subscribe_stdout, unsubscribe_stdout, RunMatWasm};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

use super::shared;

struct DialogProvider {
    bindings: JsValue,
    open_calls: Rc<Cell<usize>>,
    save_calls: Rc<Cell<usize>>,
    open_requests: Rc<RefCell<Vec<JsValue>>>,
    save_requests: Rc<RefCell<Vec<JsValue>>>,
    _open: Closure<dyn FnMut(JsValue) -> JsValue>,
    _save: Closure<dyn FnMut(JsValue) -> JsValue>,
}

struct DialogLifecycle {
    runtime: RunMatWasm,
    provider_a: DialogProvider,
    provider_b: Option<DialogProvider>,
    input_calls: Rc<Cell<usize>>,
    forwarded: Rc<RefCell<String>>,
    subscription: u32,
    _input: Closure<dyn FnMut(JsValue) -> JsValue>,
    _stdout: Closure<dyn FnMut(JsValue)>,
}

impl Drop for DialogLifecycle {
    fn drop(&mut self) {
        unsubscribe_stdout(self.subscription);
        let _ = self.runtime.set_input_handler(JsValue::NULL);
        self.runtime.dispose();
    }
}

impl DialogProvider {
    fn new(label: &'static str) -> Self {
        let bindings = shared::test_fs_provider();
        let open_calls = Rc::new(Cell::new(0));
        let save_calls = Rc::new(Cell::new(0));
        let open_requests = Rc::new(RefCell::new(Vec::new()));
        let save_requests = Rc::new(RefCell::new(Vec::new()));

        let open_count = open_calls.clone();
        let recorded_open_requests = open_requests.clone();
        let open = Closure::wrap(Box::new(move |request: JsValue| {
            open_count.set(open_count.get() + 1);
            recorded_open_requests.borrow_mut().push(request);
            let selection = Object::new();
            Reflect::set(
                &selection,
                &JsValue::from_str("paths"),
                &js_sys::Array::of1(&JsValue::from_str(&format!(
                    "/dialog/{label}/open_{label}.mat"
                ))),
            )
            .expect("set open paths");
            Reflect::set(
                &selection,
                &JsValue::from_str("filterIndex"),
                &JsValue::from_f64(1.0),
            )
            .expect("set open filter index");
            Promise::resolve::<JsValue>(selection.as_ref()).into()
        }) as Box<dyn FnMut(JsValue) -> JsValue>);

        let save_count = save_calls.clone();
        let recorded_save_requests = save_requests.clone();
        let save = Closure::wrap(Box::new(move |request: JsValue| {
            save_count.set(save_count.get() + 1);
            recorded_save_requests.borrow_mut().push(request);
            let selection = Object::new();
            Reflect::set(
                &selection,
                &JsValue::from_str("path"),
                &JsValue::from_str(&format!("/dialog/{label}/save_{label}.mat")),
            )
            .expect("set save path");
            Reflect::set(
                &selection,
                &JsValue::from_str("filterIndex"),
                &JsValue::from_f64(1.0),
            )
            .expect("set save filter index");
            Promise::resolve::<JsValue>(selection.as_ref()).into()
        }) as Box<dyn FnMut(JsValue) -> JsValue>);

        Reflect::set(
            &bindings,
            &JsValue::from_str("selectFileOpen"),
            open.as_ref(),
        )
        .expect("install open selector");
        Reflect::set(
            &bindings,
            &JsValue::from_str("selectFileSave"),
            save.as_ref(),
        )
        .expect("install save selector");

        Self {
            bindings,
            open_calls,
            save_calls,
            open_requests,
            save_requests,
            _open: open,
            _save: save,
        }
    }
}

fn init_options(provider: &DialogProvider) -> JsValue {
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("enableGpu"), &JsValue::FALSE)
        .expect("set enableGpu");
    Reflect::set(
        &options,
        &JsValue::from_str("fsProvider"),
        &provider.bindings,
    )
    .expect("set fsProvider");
    options.into()
}

fn string_property(value: &JsValue, name: &str) -> Option<String> {
    Reflect::get(value, &JsValue::from_str(name))
        .ok()
        .and_then(|property| property.as_string())
}

fn bool_property(value: &JsValue, name: &str) -> Option<bool> {
    Reflect::get(value, &JsValue::from_str(name))
        .ok()
        .and_then(|property| property.as_bool())
}

async fn execute_dialogs(runtime: &RunMatWasm, phase: &str) -> String {
    let script = format!(
        r#"
[open_name, open_path, open_filter] = uigetfile('*.mat', 'Open {phase}', 'open_default.mat');
[save_name, save_path, save_filter] = uiputfile('*.mat', 'Save {phase}', 'save_default.mat');
typed = input('value: ', 's');
fprintf('PHASE={phase}|OPEN=%s|%s|%.0f|SAVE=%s|%s|%.0f|INPUT=%s\n', ...
    open_name, open_path, open_filter, save_name, save_path, save_filter, typed);
"#
    );
    let payload = shared::execute_script_with_runtime(runtime, &script).await;
    assert_eq!(
        payload.error.as_ref().map(|error| error.message.as_str()),
        None
    );
    shared::stdout_text(&payload)
}

#[wasm_bindgen_test(async)]
async fn selected_dialog_provider_replacement_and_reset_preserve_exact_host_state() {
    let provider_a = DialogProvider::new("a");
    let runtime = init_runmat(init_options(&provider_a))
        .await
        .expect("initialize wasm runtime");

    let input_calls = Rc::new(Cell::new(0));
    let input_count = input_calls.clone();
    let input_handler = Closure::wrap(Box::new(move |_request: JsValue| {
        input_count.set(input_count.get() + 1);
        JsValue::from_str("retained-input")
    }) as Box<dyn FnMut(JsValue) -> JsValue>);
    runtime
        .set_input_handler(input_handler.as_ref().clone())
        .expect("install input handler");

    let forwarded = Rc::new(RefCell::new(String::new()));
    let forwarded_text = forwarded.clone();
    let stdout_handler = Closure::wrap(Box::new(move |entry: JsValue| {
        if let Some(text) = string_property(&entry, "text") {
            forwarded_text.borrow_mut().push_str(&text);
        }
    }) as Box<dyn FnMut(JsValue)>);
    let subscription = subscribe_stdout(stdout_handler.as_ref().clone())
        .expect("subscribe to stdout before reset");

    let mut lifecycle = DialogLifecycle {
        runtime,
        provider_a,
        provider_b: None,
        input_calls,
        forwarded,
        subscription,
        _input: input_handler,
        _stdout: stdout_handler,
    };

    let before = execute_dialogs(&lifecycle.runtime, "before").await;
    assert!(before.contains("PHASE=before|OPEN=open_a.mat|/dialog/a/|1"));
    assert!(before.contains("SAVE=save_a.mat|/dialog/a/|1|INPUT=retained-input"));
    assert_eq!(lifecycle.provider_a.open_calls.get(), 1);
    assert_eq!(lifecycle.provider_a.save_calls.get(), 1);
    let open_request = lifecycle.provider_a.open_requests.borrow()[0].clone();
    assert_eq!(
        string_property(&open_request, "title").as_deref(),
        Some("Open before")
    );
    assert_eq!(
        string_property(&open_request, "defaultPath").as_deref(),
        Some("/open_default.mat")
    );
    assert_eq!(bool_property(&open_request, "multiselect"), Some(false));
    let save_request = lifecycle.provider_a.save_requests.borrow()[0].clone();
    assert_eq!(
        string_property(&save_request, "title").as_deref(),
        Some("Save before")
    );
    assert_eq!(
        string_property(&save_request, "defaultPath").as_deref(),
        Some("/save_default.mat")
    );

    let provider_b = DialogProvider::new("b");
    lifecycle
        .runtime
        .set_fs_provider(provider_b.bindings.clone())
        .expect("replace selected filesystem/dialog provider");
    lifecycle.provider_b = Some(provider_b);
    let replaced = execute_dialogs(&lifecycle.runtime, "replaced").await;
    assert!(replaced.contains("PHASE=replaced|OPEN=open_b.mat|/dialog/b/|1"));
    assert!(replaced.contains("SAVE=save_b.mat|/dialog/b/|1|INPUT=retained-input"));
    assert_eq!(
        (
            lifecycle.provider_a.open_calls.get(),
            lifecycle.provider_a.save_calls.get()
        ),
        (1, 1)
    );
    let provider_b = lifecycle.provider_b.as_ref().expect("replacement provider");
    assert_eq!(
        (provider_b.open_calls.get(), provider_b.save_calls.get()),
        (1, 1)
    );

    lifecycle
        .runtime
        .reset_session()
        .expect("reset wasm session");
    let after_reset = execute_dialogs(&lifecycle.runtime, "reset").await;
    assert!(after_reset.contains("PHASE=reset|OPEN=open_b.mat|/dialog/b/|1"));
    assert!(after_reset.contains("SAVE=save_b.mat|/dialog/b/|1|INPUT=retained-input"));
    assert_eq!(
        (
            lifecycle.provider_a.open_calls.get(),
            lifecycle.provider_a.save_calls.get()
        ),
        (1, 1)
    );
    assert_eq!(
        (provider_b.open_calls.get(), provider_b.save_calls.get()),
        (2, 2)
    );
    assert_eq!(
        lifecycle.input_calls.get(),
        3,
        "reset must reinstall the exact input handler"
    );

    let forwarded = lifecycle.forwarded.borrow();
    assert!(forwarded.contains("PHASE=before"));
    assert!(forwarded.contains("PHASE=replaced"));
    assert!(forwarded.contains("PHASE=reset"));
    drop(forwarded);
}
