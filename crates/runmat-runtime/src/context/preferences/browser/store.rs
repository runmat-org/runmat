//! One document per application-owned IndexedDB namespace.

use super::connection::{DOCUMENT_KEY, DOCUMENT_STORE};
use super::{browser_error, BrowserPreferenceError};
use crate::context::preferences::document::PreferenceDocument;
use futures::channel::oneshot;
use js_sys::Uint8Array;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Event, IdbDatabase, IdbRequest, IdbTransaction, IdbTransactionMode};

pub(super) async fn read_document(
    database: &IdbDatabase,
) -> Result<PreferenceDocument, BrowserPreferenceError> {
    let transaction = database
        .transaction_with_str_and_mode(DOCUMENT_STORE, IdbTransactionMode::Readonly)
        .map_err(browser_error)?;
    let store = transaction
        .object_store(DOCUMENT_STORE)
        .map_err(browser_error)?;
    let request = store
        .get(&JsValue::from_str(DOCUMENT_KEY))
        .map_err(browser_error)?;
    let (sender, receiver) = oneshot::channel();
    let sender = Rc::new(RefCell::new(Some(sender)));
    let success_request = request.clone();
    let success_sender = sender.clone();
    let on_success = Closure::<dyn FnMut(Event)>::new(move |_| {
        let result = success_request
            .result()
            .map_err(browser_error)
            .and_then(decode_document);
        if let Some(sender) = success_sender.borrow_mut().take() {
            let _ = sender.send(result);
        }
    });
    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));

    let failed_request = request.clone();
    let failed_sender = sender;
    let on_error = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = failed_sender.borrow_mut().take() {
            let _ = sender.send(Err(request_error(&failed_request)));
        }
    });
    request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    let owner = ReadHandlers {
        request,
        transaction,
        _success: on_success,
        _error: on_error,
    };
    let result = receiver
        .await
        .map_err(|_| BrowserPreferenceError::LostObservation)?;
    drop(owner);
    result
}

pub(super) fn decode_document(
    value: JsValue,
) -> Result<PreferenceDocument, BrowserPreferenceError> {
    if value.is_undefined() {
        return Ok(PreferenceDocument::empty());
    }
    if !value.is_instance_of::<Uint8Array>() {
        return Err(BrowserPreferenceError::InvalidStoredValue);
    }
    let bytes = Uint8Array::new(&value).to_vec();
    PreferenceDocument::decode(&bytes).map_err(BrowserPreferenceError::Document)
}

pub(super) fn request_error(request: &IdbRequest) -> BrowserPreferenceError {
    match request.error() {
        Ok(Some(source)) => browser_error(source.into()),
        Ok(None) => BrowserPreferenceError::LostObservation,
        Err(source) => browser_error(source),
    }
}

pub(super) fn transaction_error(transaction: &IdbTransaction) -> BrowserPreferenceError {
    transaction
        .error()
        .map_or(BrowserPreferenceError::LostObservation, |source| {
            browser_error(source.into())
        })
}

struct ReadHandlers {
    request: IdbRequest,
    transaction: IdbTransaction,
    _success: Closure<dyn FnMut(Event)>,
    _error: Closure<dyn FnMut(Event)>,
}

impl Drop for ReadHandlers {
    fn drop(&mut self) {
        self.request.set_onsuccess(None);
        self.request.set_onerror(None);
        let _ = self.transaction.abort();
    }
}
