use std::cell::{Cell, RefCell};
use std::rc::Rc;

use js_sys::Promise;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{AbortController, Request, Window, WorkerGlobalScope};

use super::js_error;
use crate::http::{HttpRequest, TransportError, TransportErrorKind};

pub(super) enum HostScope {
    Window(Window),
    Worker(WorkerGlobalScope),
}

impl HostScope {
    pub(super) fn current(url: &url::Url) -> Result<Self, TransportError> {
        if let Some(window) = web_sys::window() {
            return Ok(Self::Window(window));
        }
        js_sys::global()
            .dyn_into::<WorkerGlobalScope>()
            .map(Self::Worker)
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::UnsupportedStreaming,
                    url,
                    "the host is neither Window nor WorkerGlobalScope",
                )
            })
    }

    pub(super) fn fetch(&self, request: &Request) -> Promise {
        match self {
            Self::Window(window) => window.fetch_with_request(request),
            Self::Worker(worker) => worker.fetch_with_request(request),
        }
    }

    fn set_timeout(&self, callback: &js_sys::Function, timeout_ms: i32) -> Result<i32, JsValue> {
        match self {
            Self::Window(window) => {
                window.set_timeout_with_callback_and_timeout_and_arguments_0(callback, timeout_ms)
            }
            Self::Worker(worker) => {
                worker.set_timeout_with_callback_and_timeout_and_arguments_0(callback, timeout_ms)
            }
        }
    }

    fn clear_timeout(&self, handle: i32) {
        match self {
            Self::Window(window) => window.clear_timeout_with_handle(handle),
            Self::Worker(worker) => worker.clear_timeout_with_handle(handle),
        }
    }
}

impl Clone for HostScope {
    fn clone(&self) -> Self {
        match self {
            Self::Window(window) => Self::Window(window.clone()),
            Self::Worker(worker) => Self::Worker(worker.clone()),
        }
    }
}

/// The active Fetch and timer have one owner before and after the response
/// head. Dropping a pending open or body aborts the same request generation.
pub(super) struct ActiveFetch {
    pub(super) scope: HostScope,
    pub(super) controller: AbortController,
    pub(super) timeout: Option<TimeoutLease>,
    pub(super) timed_out: Rc<Cell<bool>>,
    timeout_setup_error: Rc<RefCell<Option<JsValue>>>,
    completed: bool,
}

impl ActiveFetch {
    pub(super) fn new(scope: HostScope, url: &url::Url) -> Result<Self, TransportError> {
        let controller = AbortController::new().map_err(|source| js_error(url, source))?;
        Ok(Self {
            scope,
            controller,
            timeout: None,
            timed_out: Rc::new(Cell::new(false)),
            timeout_setup_error: Rc::new(RefCell::new(None)),
            completed: false,
        })
    }

    pub(super) fn arm_timeout(&mut self, request: &HttpRequest) -> Result<(), TransportError> {
        // Match the existing XHR behavior: zero means no browser timer. Clamp
        // at u32::MAX, then arm signed-i32-safe segments so long admitted
        // timeouts are not shortened to 24.8 days.
        let milliseconds = request.timeout.as_millis().min(u32::MAX as u128) as u64;
        if milliseconds == 0 {
            return Ok(());
        }
        self.timeout = Some(TimeoutLease::new(
            self.scope.clone(),
            self.controller.clone(),
            Rc::clone(&self.timed_out),
            Rc::clone(&self.timeout_setup_error),
            milliseconds,
            &request.url,
        )?);
        Ok(())
    }

    pub(super) fn complete(&mut self) {
        self.timeout.take();
        self.completed = true;
    }

    pub(super) fn abort(&mut self) {
        self.timeout.take();
        if !self.completed {
            self.controller.abort();
            self.completed = true;
        }
    }
}

impl Drop for ActiveFetch {
    fn drop(&mut self) {
        self.abort();
    }
}

pub(super) fn fetch_error(url: &url::Url, active: &ActiveFetch, source: JsValue) -> TransportError {
    if let Some(setup_error) = active.timeout_setup_error.borrow_mut().take() {
        return js_error(url, setup_error);
    }
    let kind = if active.timed_out.get() {
        TransportErrorKind::Timeout
    } else {
        TransportErrorKind::Connect
    };
    TransportError::new(kind, url, format!("{source:?}"))
        .with_source(super::super::BrowserJsError(source))
}

pub(super) fn active_failure(url: &url::Url, active: &ActiveFetch) -> Option<TransportError> {
    if let Some(source) = active.timeout_setup_error.borrow_mut().take() {
        return Some(js_error(url, source));
    }
    active.timed_out.get().then(|| {
        TransportError::new(
            TransportErrorKind::Timeout,
            url,
            "browser request timed out",
        )
    })
}

pub(super) struct TimeoutState {
    scope: HostScope,
    controller: AbortController,
    timed_out: Rc<Cell<bool>>,
    setup_error: Rc<RefCell<Option<JsValue>>>,
    pub(super) remaining_ms: u64,
    pub(super) current_segment_ms: u64,
    handle: Option<i32>,
}

/// Re-arm only when a signed-i32-safe timer segment expires. The closure
/// holds a weak self-reference; drop clears the handle without a retained cycle.
pub(super) struct TimeoutLease {
    pub(super) state: Rc<RefCell<TimeoutState>>,
    _callback: Rc<Closure<dyn FnMut()>>,
}

impl TimeoutLease {
    fn new(
        scope: HostScope,
        controller: AbortController,
        timed_out: Rc<Cell<bool>>,
        setup_error: Rc<RefCell<Option<JsValue>>>,
        milliseconds: u64,
        url: &url::Url,
    ) -> Result<Self, TransportError> {
        let state = Rc::new(RefCell::new(TimeoutState {
            scope,
            controller,
            timed_out,
            setup_error,
            remaining_ms: milliseconds,
            current_segment_ms: 0,
            handle: None,
        }));
        let callback_slot = Rc::new(RefCell::new(None::<std::rc::Weak<Closure<dyn FnMut()>>>));
        let callback_state = Rc::clone(&state);
        let callback_weak = Rc::clone(&callback_slot);
        let callback = Rc::new(Closure::<dyn FnMut()>::new(move || {
            let mut state = callback_state.borrow_mut();
            state.handle = None;
            state.remaining_ms -= state.current_segment_ms;
            if state.remaining_ms == 0 {
                state.timed_out.set(true);
                let controller = state.controller.clone();
                drop(state);
                controller.abort();
                return;
            }
            let Some(callback) = callback_weak
                .borrow()
                .as_ref()
                .and_then(std::rc::Weak::upgrade)
            else {
                return;
            };
            if let Err(source) = state.arm(callback.as_ref().as_ref().unchecked_ref()) {
                *state.setup_error.borrow_mut() = Some(source);
                let controller = state.controller.clone();
                drop(state);
                controller.abort();
            }
        }));
        *callback_slot.borrow_mut() = Some(Rc::downgrade(&callback));
        state
            .borrow_mut()
            .arm(callback.as_ref().as_ref().unchecked_ref())
            .map_err(|source| js_error(url, source))?;
        Ok(Self {
            state,
            _callback: callback,
        })
    }
}

impl TimeoutState {
    fn arm(&mut self, callback: &js_sys::Function) -> Result<(), JsValue> {
        let segment = self.remaining_ms.min(i32::MAX as u64);
        let handle = self.scope.set_timeout(callback, segment as i32)?;
        self.current_segment_ms = segment;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Drop for TimeoutLease {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        if let Some(handle) = state.handle.take() {
            state.scope.clear_timeout(handle);
        }
    }
}
