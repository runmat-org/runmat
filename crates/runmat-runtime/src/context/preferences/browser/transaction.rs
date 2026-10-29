//! Browser-owned read-modify-write transaction; request success is not commit.

use super::connection::{OwnedDatabase, DOCUMENT_KEY, DOCUMENT_STORE};
use super::retirement::Observer;
use super::store::{decode_document, transaction_error};
use super::{browser_error, BrowserPreferenceError};
use crate::context::preferences::document::{
    apply_mutation, missing_default_indices, DocumentMutation,
};
use crate::context::preferences::input_roots::PreferenceInputRoots;
use crate::context::UserPathSnapshot;
use futures::channel::oneshot;
use futures::lock::OwnedMutexGuard;
use js_sys::Uint8Array;
use runmat_execution::profiling::CommandProfilingReceipt;
use runmat_execution::value::ValuePayload;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;
use web_sys::{Event, IdbObjectStore, IdbRequest, IdbTransaction, IdbTransactionMode};

pub(super) enum BrowserCheckedSource {
    Completed {
        values: Vec<ValuePayload>,
        user_path: Option<UserPathSnapshot>,
    },
    NeedsPreparation(Vec<usize>),
}

/// One immutable caller batch and root lease shared by preparation retries.
/// Only the proof bitset changes, after an actual selected-owner validation.
pub(super) struct SharedDefaults {
    values: Vec<Value>,
    _roots: PreferenceInputRoots,
    prepared: RefCell<Vec<Option<ValuePayload>>>,
}

impl SharedDefaults {
    pub(super) fn new(values: Vec<Value>, roots: PreferenceInputRoots) -> Self {
        let prepared = RefCell::new(vec![None; values.len()]);
        Self {
            values,
            _roots: roots,
            prepared,
        }
    }

    pub(super) fn value(&self, index: usize) -> &Value {
        &self.values[index]
    }

    pub(super) fn mark_prepared(&self, index: usize, payload: ValuePayload) {
        self.prepared.borrow_mut()[index] = Some(payload);
    }

    fn prepared(&self, index: usize) -> Option<ValuePayload> {
        self.prepared.borrow()[index].clone()
    }

    fn unproved(&self, indices: &[usize]) -> Vec<usize> {
        let prepared = self.prepared.borrow();
        indices
            .iter()
            .copied()
            .filter(|index| prepared[*index].is_none())
            .collect()
    }
}

pub(super) enum BrowserMutation {
    Prepared(DocumentMutation),
    ReadOrCreate {
        group: String,
        names: Vec<String>,
        defaults: Rc<SharedDefaults>,
    },
}

pub(super) async fn mutate(
    database: OwnedDatabase,
    mutation: BrowserMutation,
    gate: OwnedMutexGuard<()>,
    profiling: Option<&CommandProfilingReceipt>,
) -> Result<BrowserCheckedSource, BrowserPreferenceError> {
    let transaction = database
        .transaction_with_str_and_mode(DOCUMENT_STORE, IdbTransactionMode::Readwrite)
        .map_err(browser_error)?;
    let store = transaction
        .object_store(DOCUMENT_STORE)
        .map_err(browser_error)?;
    let request = store
        .get(&JsValue::from_str(DOCUMENT_KEY))
        .map_err(browser_error)?;
    let (completion_sender, completion_receiver) = oneshot::channel();
    let (caller_sender, caller_receiver) = oneshot::channel();
    let active = Rc::new(RefCell::new(ActiveWrite {
        _database: database,
        transaction,
        store,
        request,
        mutation: Some(mutation),
        prepared: None,
        completion_sender: Some(completion_sender),
        on_get: None,
        on_complete: None,
        on_abort: None,
        terminal: false,
        _gate: gate,
    }));
    install_callbacks(&active);

    // This host-owned observer outlives an abandoned caller and releases the
    // single active transaction only after complete/abort. It never reports a
    // successful publication to a dropped caller or treats put as completion.
    let observer = Observer::new(
        active,
        profiling.and_then(CommandProfilingReceipt::begin_deferred_work),
    );
    spawn_local(async move {
        let outcome = completion_receiver
            .await
            .unwrap_or(Err(BrowserPreferenceError::LostObservation));
        let terminal = {
            let mut state = observer.subject().borrow_mut();
            state.clear_handlers();
            state.terminal
        };
        let _ = caller_sender.send(outcome);
        observer.finish(terminal);
    });
    caller_receiver
        .await
        .map_err(|_| BrowserPreferenceError::LostObservation)?
}

fn install_callbacks(active: &Rc<RefCell<ActiveWrite>>) {
    let weak = Rc::downgrade(active);
    let on_get = Closure::<dyn FnMut(Event)>::new(move |_| on_get_success(&weak));
    let weak = Rc::downgrade(active);
    let on_complete = Closure::<dyn FnMut(Event)>::new(move |_| complete(&weak));
    let weak = Rc::downgrade(active);
    let on_abort = Closure::<dyn FnMut(Event)>::new(move |_| abort(&weak));
    let mut state = active.borrow_mut();
    state
        .request
        .set_onsuccess(Some(on_get.as_ref().unchecked_ref()));
    state
        .transaction
        .set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
    state
        .transaction
        .set_onabort(Some(on_abort.as_ref().unchecked_ref()));
    state.on_get = Some(on_get);
    state.on_complete = Some(on_complete);
    state.on_abort = Some(on_abort);
}

fn on_get_success(weak: &Weak<RefCell<ActiveWrite>>) {
    let Some(active) = weak.upgrade() else { return };
    let mut state = active.borrow_mut();
    let result = (|| {
        let value = state.request.result().map_err(browser_error)?;
        let document = decode_document(value)?;
        let mutation = state
            .mutation
            .take()
            .ok_or(BrowserPreferenceError::LostObservation)?;
        let mutation = match mutation {
            BrowserMutation::Prepared(mutation) => mutation,
            BrowserMutation::ReadOrCreate {
                group,
                names,
                defaults,
            } => {
                let missing = missing_default_indices(&document, &group, &names);
                let unproved = defaults.unproved(&missing);
                if !unproved.is_empty() {
                    return Ok(BrowserCheckedSource::NeedsPreparation(unproved));
                }
                let mut encoded = vec![None; names.len()];
                for index in missing {
                    encoded[index] =
                        Some(defaults.prepared(index).expect("selected default proof"));
                }
                DocumentMutation::ReadOrCreate {
                    group,
                    names,
                    defaults: Some(encoded),
                }
            }
        };
        let (document, prepared) =
            apply_mutation(document, mutation).map_err(BrowserPreferenceError::Mutation)?;
        if prepared.changed {
            let bytes = document
                .encode()
                .map_err(BrowserPreferenceError::Document)?;
            let bytes = Uint8Array::from(bytes.as_slice());
            state
                .store
                .put_with_key(&bytes.into(), &JsValue::from_str(DOCUMENT_KEY))
                .map_err(browser_error)?;
        }
        Ok(BrowserCheckedSource::Completed {
            values: prepared.values,
            user_path: prepared.user_path,
        })
    })();
    let abort_without_write =
        result.is_err() || matches!(&result, Ok(BrowserCheckedSource::NeedsPreparation(_)));
    state.prepared = Some(result);
    let transaction = state.transaction.clone();
    drop(state);
    if abort_without_write {
        let _ = transaction.abort();
    }
}

fn complete(weak: &Weak<RefCell<ActiveWrite>>) {
    let Some(active) = weak.upgrade() else { return };
    let mut state = active.borrow_mut();
    state.terminal = true;
    let outcome = state
        .prepared
        .take()
        .unwrap_or(Err(BrowserPreferenceError::LostObservation));
    if let Some(sender) = state.completion_sender.take() {
        let _ = sender.send(outcome);
    }
}

fn abort(weak: &Weak<RefCell<ActiveWrite>>) {
    let Some(active) = weak.upgrade() else { return };
    let mut state = active.borrow_mut();
    state.terminal = true;
    let outcome = match state.prepared.take() {
        Some(Ok(needs @ BrowserCheckedSource::NeedsPreparation(_))) => Ok(needs),
        Some(Err(error)) => Err(error),
        _ => Err(transaction_error(&state.transaction)),
    };
    if let Some(sender) = state.completion_sender.take() {
        let _ = sender.send(outcome);
    }
}

struct ActiveWrite {
    _database: OwnedDatabase,
    transaction: IdbTransaction,
    store: IdbObjectStore,
    request: IdbRequest,
    mutation: Option<BrowserMutation>,
    prepared: Option<Result<BrowserCheckedSource, BrowserPreferenceError>>,
    completion_sender:
        Option<oneshot::Sender<Result<BrowserCheckedSource, BrowserPreferenceError>>>,
    on_get: Option<Closure<dyn FnMut(Event)>>,
    on_complete: Option<Closure<dyn FnMut(Event)>>,
    on_abort: Option<Closure<dyn FnMut(Event)>>,
    terminal: bool,
    _gate: OwnedMutexGuard<()>,
}

impl ActiveWrite {
    fn clear_handlers(&mut self) {
        self.request.set_onsuccess(None);
        self.transaction.set_oncomplete(None);
        self.transaction.set_onabort(None);
        self.on_get = None;
        self.on_complete = None;
        self.on_abort = None;
    }
}

impl Drop for ActiveWrite {
    fn drop(&mut self) {
        self.clear_handlers();
        if !self.terminal {
            let _ = self.transaction.abort();
        }
    }
}
