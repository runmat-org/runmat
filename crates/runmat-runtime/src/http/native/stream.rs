use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::Stream;
use tokio::sync::mpsc;
use url::Url;

use super::{prepare_request, request_error, AbortOnDrop};
use crate::context::RuntimeServiceFuture;
use crate::execution::native_task;
use crate::http::{
    HttpBodyStream, HttpRequest, HttpResponseHead, TransportError, TransportErrorKind,
};
use runmat_execution::profiling::CommandProfilingReceipt;

pub(super) fn open_bounded(
    driver: tokio::runtime::Handle,
    clients: Arc<OnceLock<Result<reqwest::Client, Arc<reqwest::Error>>>>,
    request: HttpRequest,
    accepted_limit: u64,
    profiling: Option<CommandProfilingReceipt>,
) -> RuntimeServiceFuture<Result<HttpBodyStream, TransportError>> {
    Box::pin(async move {
        // Creating channels has no network effect. Submission begins only on
        // first poll, and the guard owns the task across head and body phases.
        let url = request.url.clone();
        let (head_tx, head_rx) = oneshot::channel();
        let (body_tx, body_rx) = mpsc::channel(1);
        let task = AbortOnDrop(native_task::spawn(
            &driver,
            open_impl(request, clients, accepted_limit, head_tx, body_tx),
            profiling,
        ));
        let head = match head_rx.await {
            Ok(Ok(head)) => head,
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                return match task.await {
                    Err(error) => {
                        Err(
                            TransportError::new(TransportErrorKind::Other, &url, error.to_string())
                                .with_source(error),
                        )
                    }
                    Ok(()) => Err(TransportError::new(
                        TransportErrorKind::Other,
                        &url,
                        "response head ended without a result",
                    )),
                };
            }
        };
        Ok(HttpBodyStream::new(
            head,
            Box::pin(NativeBody {
                receiver: body_rx,
                task: Some(task),
                url,
            }),
        ))
    })
}

async fn open_impl(
    mut request: HttpRequest,
    clients: Arc<OnceLock<Result<reqwest::Client, Arc<reqwest::Error>>>>,
    accepted_limit: u64,
    head_tx: oneshot::Sender<Result<HttpResponseHead, TransportError>>,
    body_tx: mpsc::Sender<Result<Vec<u8>, TransportError>>,
) {
    let builder = match prepare_request(&mut request, &clients) {
        Ok(builder) => builder,
        Err(error) => {
            let _ = head_tx.send(Err(error));
            return;
        }
    };
    let mut response = match builder.send().await {
        Ok(response) => response,
        Err(error) => {
            let _ = head_tx.send(Err(request_error(&request, error)));
            return;
        }
    };
    let status = response.status();
    if !status.is_success() {
        let _ = head_tx.send(Err(TransportError::new(
            TransportErrorKind::Status(status.as_u16()),
            &request.url,
            status.canonical_reason().unwrap_or("HTTP error"),
        )));
        return;
    }
    let content_length = response.content_length();
    if content_length.is_some_and(|length| length > accepted_limit) {
        let _ = head_tx.send(Err(limit_error(&request.url, accepted_limit)));
        return;
    }
    let head = HttpResponseHead {
        headers: response
            .headers()
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_string(),
                    value.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect(),
        content_length,
    };
    if head_tx.send(Ok(head)).is_err() {
        return;
    }

    let mut accepted = 0u64;
    loop {
        match response.chunk().await {
            Ok(Some(bytes)) => {
                let next = u64::try_from(bytes.len())
                    .ok()
                    .and_then(|length| accepted.checked_add(length));
                let Some(next) = next.filter(|next| *next <= accepted_limit) else {
                    let _ = body_tx
                        .send(Err(limit_error(&request.url, accepted_limit)))
                        .await;
                    return;
                };
                accepted = next;
                if body_tx.send(Ok(bytes.to_vec())).await.is_err() {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let _ = body_tx.send(Err(request_error(&request, error))).await;
                return;
            }
        }
    }
}

fn limit_error(url: &Url, accepted_limit: u64) -> TransportError {
    TransportError::new(
        TransportErrorKind::BodyLimitExceeded(accepted_limit),
        url,
        "compressed response body is too large",
    )
}

struct NativeBody {
    receiver: mpsc::Receiver<Result<Vec<u8>, TransportError>>,
    task: Option<AbortOnDrop<()>>,
    url: Url,
}

impl Stream for NativeBody {
    type Item = Result<Vec<u8>, TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(item)) => return Poll::Ready(Some(item)),
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => {}
        }
        let Some(task) = self.task.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(task).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(())) => {
                self.task.take();
                Poll::Ready(None)
            }
            Poll::Ready(Err(error)) => {
                self.task.take();
                Poll::Ready(Some(Err(TransportError::new(
                    TransportErrorKind::Other,
                    &self.url,
                    error.to_string(),
                )
                .with_source(error))))
            }
        }
    }
}

#[cfg(test)]
#[path = "stream/tests.rs"]
mod tests;
