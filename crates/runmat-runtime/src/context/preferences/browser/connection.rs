//! Origin-local IndexedDB connection with a late-success close owner.

use super::retirement::Observer;
use super::{browser_error, BrowserPreferenceError};
use futures::channel::oneshot;
use runmat_execution::profiling::CommandProfilingReceipt;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{Event, IdbDatabase, IdbFactory, IdbOpenDbRequest, WorkerGlobalScope};

pub(super) const DOCUMENT_STORE: &str = "settings";
pub(super) const DOCUMENT_KEY: &str = "preferences";

pub(super) struct OwnedDatabase(IdbDatabase);

impl Deref for OwnedDatabase {
    type Target = IdbDatabase;

    fn deref(&self) -> &IdbDatabase {
        &self.0
    }
}

impl Drop for OwnedDatabase {
    fn drop(&mut self) {
        self.0.close();
    }
}

pub(super) async fn open(
    namespace: &str,
    profiling: Option<&CommandProfilingReceipt>,
) -> Result<OwnedDatabase, BrowserPreferenceError> {
    open_version(namespace, 1, None, profiling).await
}

pub(super) async fn open_version(
    namespace: &str,
    version: u32,
    blocked: Option<oneshot::Sender<()>>,
    profiling: Option<&CommandProfilingReceipt>,
) -> Result<OwnedDatabase, BrowserPreferenceError> {
    let request = factory()?
        .open_with_u32(namespace, version)
        .map_err(browser_error)?;
    let (completion_sender, completion_receiver) = oneshot::channel();
    let (caller_sender, caller_receiver) = oneshot::channel();
    let owner = Rc::new(RefCell::new(OpenOwner {
        request,
        completion_sender: Some(completion_sender),
        blocked,
        upgrade_error: None,
        terminal: false,
        on_upgrade: None,
        on_success: None,
        on_error: None,
        on_blocked: None,
    }));
    install(&owner);
    // IndexedDB does not expose cancellation of an open request. A dropped
    // caller leaves this one host-owned observer to close any later success.
    let observer = Observer::new(
        owner,
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

fn install(owner: &Rc<RefCell<OpenOwner>>) {
    let weak = Rc::downgrade(owner);
    let on_upgrade = Closure::<dyn FnMut(Event)>::new(move |_| upgrade(&weak));
    let weak = Rc::downgrade(owner);
    let on_success = Closure::<dyn FnMut(Event)>::new(move |_| success(&weak));
    let weak = Rc::downgrade(owner);
    let on_error = Closure::<dyn FnMut(Event)>::new(move |_| failed(&weak));
    let weak = Rc::downgrade(owner);
    let on_blocked = Closure::<dyn FnMut(Event)>::new(move |_| blocked(&weak));
    let mut state = owner.borrow_mut();
    state
        .request
        .set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
    state
        .request
        .set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
    state
        .request
        .set_onerror(Some(on_error.as_ref().unchecked_ref()));
    state
        .request
        .set_onblocked(Some(on_blocked.as_ref().unchecked_ref()));
    state.on_upgrade = Some(on_upgrade);
    state.on_success = Some(on_success);
    state.on_error = Some(on_error);
    state.on_blocked = Some(on_blocked);
}

fn blocked(weak: &Weak<RefCell<OpenOwner>>) {
    if let Some(owner) = weak.upgrade() {
        if let Some(sender) = owner.borrow_mut().blocked.take() {
            let _ = sender.send(());
        }
    }
}

fn upgrade(weak: &Weak<RefCell<OpenOwner>>) {
    let Some(owner) = weak.upgrade() else { return };
    let request = owner.borrow().request.clone();
    let result = request
        .result()
        .map_err(browser_error)
        .and_then(|value| value.dyn_into::<IdbDatabase>().map_err(browser_error))
        .and_then(|database| {
            if !database.object_store_names().contains(DOCUMENT_STORE) {
                database
                    .create_object_store(DOCUMENT_STORE)
                    .map_err(browser_error)?;
            }
            Ok(())
        });
    if let Err(error) = result {
        owner.borrow_mut().upgrade_error = Some(error);
        if let Some(transaction) = request.transaction() {
            let _ = transaction.abort();
        }
    }
}

fn success(weak: &Weak<RefCell<OpenOwner>>) {
    let Some(owner) = weak.upgrade() else { return };
    let mut state = owner.borrow_mut();
    state.terminal = true;
    let result = state
        .request
        .result()
        .map_err(browser_error)
        .and_then(|value| value.dyn_into::<IdbDatabase>().map_err(browser_error))
        .map(OwnedDatabase);
    let outcome = if let Some(error) = state.upgrade_error.take() {
        drop(result); // OwnedDatabase closes even if upgrade partially succeeded.
        Err(error)
    } else {
        result
    };
    if let Some(sender) = state.completion_sender.take() {
        let _ = sender.send(outcome);
    }
}

fn failed(weak: &Weak<RefCell<OpenOwner>>) {
    let Some(owner) = weak.upgrade() else { return };
    let mut state = owner.borrow_mut();
    state.terminal = true;
    let error = if let Some(error) = state.upgrade_error.take() {
        error
    } else {
        request_error(&state.request)
    };
    if let Some(sender) = state.completion_sender.take() {
        let _ = sender.send(Err(error));
    }
}

pub(super) fn factory() -> Result<IdbFactory, BrowserPreferenceError> {
    if let Some(window) = web_sys::window() {
        return window
            .indexed_db()
            .map_err(browser_error)?
            .ok_or(BrowserPreferenceError::Unavailable);
    }
    let worker = js_sys::global()
        .dyn_into::<WorkerGlobalScope>()
        .map_err(|_| BrowserPreferenceError::Unavailable)?;
    worker
        .indexed_db()
        .map_err(browser_error)?
        .ok_or(BrowserPreferenceError::Unavailable)
}

fn request_error(request: &IdbOpenDbRequest) -> BrowserPreferenceError {
    match request.error() {
        Ok(Some(source)) => browser_error(source.into()),
        Ok(None) => BrowserPreferenceError::LostObservation,
        Err(source) => browser_error(source),
    }
}

struct OpenOwner {
    request: IdbOpenDbRequest,
    completion_sender: Option<oneshot::Sender<Result<OwnedDatabase, BrowserPreferenceError>>>,
    blocked: Option<oneshot::Sender<()>>,
    upgrade_error: Option<BrowserPreferenceError>,
    terminal: bool,
    on_upgrade: Option<Closure<dyn FnMut(Event)>>,
    on_success: Option<Closure<dyn FnMut(Event)>>,
    on_error: Option<Closure<dyn FnMut(Event)>>,
    on_blocked: Option<Closure<dyn FnMut(Event)>>,
}

impl OpenOwner {
    fn clear_handlers(&mut self) {
        self.request.set_onupgradeneeded(None);
        self.request.set_onsuccess(None);
        self.request.set_onerror(None);
        self.request.set_onblocked(None);
        self.on_upgrade = None;
        self.on_success = None;
        self.on_error = None;
        self.on_blocked = None;
    }
}

impl Drop for OpenOwner {
    fn drop(&mut self) {
        self.clear_handlers();
    }
}
