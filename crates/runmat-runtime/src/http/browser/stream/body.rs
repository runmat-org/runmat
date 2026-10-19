use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use js_sys::{Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::ReadableStreamDefaultReader;

use super::lifecycle::{active_failure, fetch_error, ActiveFetch};
use super::{js_error, limit_error};
use crate::http::{TransportError, TransportErrorKind};

pub(super) struct BrowserBody {
    active: ActiveFetch,
    reader: ReadableStreamDefaultReader,
    pending: Option<Pin<Box<JsFuture>>>,
    accepted: u64,
    accepted_limit: u64,
    url: url::Url,
    finished: bool,
}

impl BrowserBody {
    pub(super) fn new(
        active: ActiveFetch,
        reader: ReadableStreamDefaultReader,
        accepted_limit: u64,
        url: url::Url,
    ) -> Self {
        Self {
            active,
            reader,
            pending: None,
            accepted: 0,
            accepted_limit,
            url,
            finished: false,
        }
    }

    fn stop(&mut self) {
        self.finished = true;
        // Fetch abort errors the response stream. JsFuture already observes
        // rejection of a pending read; no unobserved reader.cancel() Promise
        // or detached cleanup task is created.
        self.active.abort();
        self.pending = None;
        self.reader.release_lock();
    }

    fn allocate_chunk(&mut self, chunk: &Uint8Array) -> Result<Vec<u8>, TransportError> {
        let length = usize::try_from(chunk.length()).map_err(|source| {
            TransportError::new(
                TransportErrorKind::Other,
                &self.url,
                "browser response chunk length overflows host size",
            )
            .with_source(source)
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|source| {
            TransportError::new(
                TransportErrorKind::Other,
                &self.url,
                format!("unable to allocate browser response chunk: {source}"),
            )
            .with_source(source)
        })?;
        bytes.resize(length, 0);
        chunk.copy_to(&mut bytes);
        Ok(bytes)
    }
}

impl Drop for BrowserBody {
    fn drop(&mut self) {
        if !self.finished {
            self.stop();
        }
    }
}

impl Stream for BrowserBody {
    type Item = Result<Vec<u8>, TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if let Some(error) = active_failure(&self.url, &self.active) {
            self.stop();
            return Poll::Ready(Some(Err(error)));
        }
        if self.pending.is_none() {
            self.pending = Some(Box::pin(JsFuture::from(self.reader.read())));
        }
        let result = self
            .pending
            .as_mut()
            .expect("pending read")
            .as_mut()
            .poll(cx);
        let value = match result {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(value)) => value,
            Poll::Ready(Err(source)) => {
                let error = fetch_error(&self.url, &self.active, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        };
        self.pending = None;
        if let Some(error) = active_failure(&self.url, &self.active) {
            self.stop();
            return Poll::Ready(Some(Err(error)));
        }
        let done = Reflect::get(&value, &JsValue::from_str("done"));
        match done {
            Ok(done) if done.as_bool() == Some(true) => {
                self.finished = true;
                self.active.complete();
                self.reader.release_lock();
                return Poll::Ready(None);
            }
            Ok(done) if done.as_bool() == Some(false) => {}
            Ok(source) | Err(source) => {
                let error = js_error(&self.url, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        }
        let chunk = match Reflect::get(&value, &JsValue::from_str("value")) {
            Ok(source) => match source.dyn_into::<Uint8Array>() {
                Ok(chunk) => chunk,
                Err(source) => {
                    let error = js_error(&self.url, source);
                    self.stop();
                    return Poll::Ready(Some(Err(error)));
                }
            },
            Err(source) => {
                let error = js_error(&self.url, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        };
        let next = self.accepted.checked_add(u64::from(chunk.length()));
        let Some(next) = next.filter(|next| *next <= self.accepted_limit) else {
            let error = limit_error(&self.url, self.accepted_limit);
            self.stop();
            return Poll::Ready(Some(Err(error)));
        };
        let bytes = match self.allocate_chunk(&chunk) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        };
        self.accepted = next;
        Poll::Ready(Some(Ok(bytes)))
    }
}
