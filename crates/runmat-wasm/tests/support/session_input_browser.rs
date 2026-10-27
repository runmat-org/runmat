//! Input registrations belong to their session across replacement and reset.
use std::cell::Cell;
use std::rc::Rc;

use wasm_bindgen::{closure::Closure, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

use super::shared;

#[wasm_bindgen_test(async)]
async fn input_handlers_remain_with_their_session_when_peers_clear_dispose_and_reset() {
    let options = || {
        let options = js_sys::Object::new();
        js_sys::Reflect::set(&options, &"enableGpu".into(), &JsValue::FALSE).unwrap();
        js_sys::Reflect::set(&options, &"fsProvider".into(), &shared::test_fs_provider()).unwrap();
        options.into()
    };
    let old = runmat_wasm::init_runmat(options()).await.unwrap();
    let old_calls = Rc::new(Cell::new(0));
    let calls = Rc::clone(&old_calls);
    let old_handler = Closure::<dyn FnMut(JsValue) -> JsValue>::new(move |_| {
        calls.set(calls.get() + 1);
        js_sys::Promise::resolve(&JsValue::from_str("old")).into()
    });
    old.set_input_handler(old_handler.as_ref().clone()).unwrap();

    let next = runmat_wasm::init_runmat(options()).await.unwrap();
    let next_calls = Rc::new(Cell::new(0));
    let calls = Rc::clone(&next_calls);
    let next_handler = Closure::<dyn FnMut(JsValue) -> JsValue>::new(move |_| {
        calls.set(calls.get() + 1);
        js_sys::Promise::resolve(&JsValue::from_str("6*7")).into()
    });
    next.set_input_handler(next_handler.as_ref().clone())
        .unwrap();

    let result =
        shared::execute_script_with_runtime(&old, "assert(strcmp(input('Old: ','s'),'old'));")
            .await;
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(old_calls.get(), 1);
    assert_eq!(next_calls.get(), 0);

    old.set_input_handler(JsValue::NULL).unwrap();
    let result = shared::execute_script_with_runtime(&next, "assert(input('Number: ')==42);").await;
    assert!(result.error.is_none(), "{:?}", result.error);

    // Exact app sequence: replacement handler installed, old session disposed,
    // then a script reset before the following command-window execution.
    old.set_input_handler(old_handler.as_ref().clone()).unwrap();
    old.dispose();
    next.reset_session().unwrap();
    let result = shared::execute_script_with_runtime(
        &next,
        "assert(strcmp(input('String: ','s'),'6*7')); assert(input('Number: ')==42);",
    )
    .await;
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(next_calls.get(), 3);
    next.dispose();
}
