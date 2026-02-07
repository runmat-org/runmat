//! IndexedDB-backed persistence for native auto-offload calibration (wasm32 only).
//!
//! This is intentionally best-effort: failures must never prevent execution.

#![cfg(target_arch = "wasm32")]

use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    IdbDatabase, IdbFactory, IdbObjectStore, IdbOpenDbRequest, IdbRequest, IdbTransactionMode,
};

const DB_NAME: &str = "runmat";
const DB_VERSION: u32 = 1;
const STORE_NAME: &str = "auto_offload";

fn indexed_db_factory() -> Result<IdbFactory, JsValue> {
    let global = js_sys::global();
    let idb = Reflect::get(&global, &JsValue::from_str("indexedDB"))?;
    idb.dyn_into::<IdbFactory>()
}

fn request_promise(request: &IdbRequest) -> js_sys::Promise {
    let request = request.clone();
    js_sys::Promise::new(&mut |resolve, reject| {
        let request_success = request.clone();
        let resolve = resolve.clone();
        let onsuccess = Closure::once(move |_event: web_sys::Event| {
            // `result` can be null; still resolve it.
            let _ =
                resolve.call1(&JsValue::NULL, &request_success.result().unwrap_or(JsValue::NULL));
        });
        let request_error = request.clone();
        let reject = reject.clone();
        let onerror = Closure::once(move |_event: web_sys::Event| {
            let err = request_error
                .error()
                .map(JsValue::from)
                .unwrap_or_else(|_| JsValue::from_str("IndexedDB request failed"));
            let _ = reject.call1(&JsValue::NULL, &err);
        });
        request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onsuccess.forget();
        onerror.forget();
    })
}

async fn open_db() -> Result<IdbDatabase, JsValue> {
    let factory = indexed_db_factory()?;
    let open_req: IdbOpenDbRequest = factory.open_with_u32(DB_NAME, DB_VERSION)?;

    {
        // Create object store on first open / version bump.
        let upgrade_cb = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event
                .target()
                .and_then(|t| t.dyn_into::<IdbOpenDbRequest>().ok());
            let Some(req) = target else {
                return;
            };
            let Ok(result) = req.result() else {
                return;
            };
            let Ok(db) = result.dyn_into::<IdbDatabase>() else {
                return;
            };
            if db.object_store_names().contains(STORE_NAME) {
                return;
            }
            let _ = db.create_object_store(STORE_NAME);
        }) as Box<dyn FnMut(_)>);
        open_req.set_onupgradeneeded(Some(upgrade_cb.as_ref().unchecked_ref()));
        upgrade_cb.forget();
    }

    let value = JsFuture::from(request_promise(&open_req.unchecked_into::<IdbRequest>())).await?;
    value.dyn_into::<IdbDatabase>()
}

fn store_for(db: &IdbDatabase, mode: IdbTransactionMode) -> Result<IdbObjectStore, JsValue> {
    let tx = db.transaction_with_str_and_mode(STORE_NAME, mode)?;
    tx.object_store(STORE_NAME)
}

/// Load a cached calibration JSON payload by key.
pub async fn load(key: &str) -> Option<String> {
    let db = open_db().await.ok()?;
    let store = store_for(&db, IdbTransactionMode::Readonly).ok()?;
    let req = store.get(&JsValue::from_str(key)).ok()?;
    let value = JsFuture::from(request_promise(&req)).await.ok()?;
    if value.is_null() || value.is_undefined() {
        return None;
    }
    value.as_string()
}

/// Persist a calibration JSON payload by key (best-effort).
pub async fn save(key: &str, json: &str) -> Result<(), JsValue> {
    let db = open_db().await?;
    let store = store_for(&db, IdbTransactionMode::Readwrite)?;

    // Store a small object so we can extend schema later (TTL, timestamps, etc).
    let entry = Object::new();
    Reflect::set(&entry, &JsValue::from_str("key"), &JsValue::from_str(key))?;
    Reflect::set(&entry, &JsValue::from_str("json"), &JsValue::from_str(json))?;

    // Use the key as the primary key.
    let req = store.put_with_key(&entry, &JsValue::from_str(key))?;
    let _ = JsFuture::from(request_promise(&req)).await?;
    Ok(())
}

