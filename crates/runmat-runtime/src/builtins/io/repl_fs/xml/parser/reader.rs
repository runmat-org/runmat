//! Bounded source exposure to quick-xml's asynchronous reader.

use super::super::cooperation::{checkpoint_owner, XmlBudgetOwner};
use crate::execution::ExecutionServiceError;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncBufRead, AsyncRead, ReadBuf};

const SOURCE_CHUNK: usize = 4096;

pub(super) struct CooperativeXmlBuf<'a> {
    source: &'a [u8],
    position: usize,
    exposed_end: usize,
    completed: u64,
    budget: Option<XmlBudgetOwner>,
    pending: Option<Pin<Box<dyn Future<Output = Result<(), ExecutionServiceError>>>>>,
}

impl<'a> CooperativeXmlBuf<'a> {
    pub(super) fn new(source: &'a [u8], budget: Option<XmlBudgetOwner>) -> Self {
        Self {
            source,
            position: 0,
            exposed_end: 0,
            completed: 0,
            budget,
            pending: None,
        }
    }
}

impl AsyncBufRead for CooperativeXmlBuf<'_> {
    fn poll_fill_buf(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
        if self.position == self.exposed_end {
            if let Some(owner) = self.budget.clone() {
                if self.pending.is_none() {
                    self.pending = Some(checkpoint_owner(owner, self.completed));
                    self.completed = 0;
                }
                if let Some(future) = self.pending.as_mut() {
                    match future.as_mut().poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(error)) => {
                            self.pending = None;
                            return Poll::Ready(Err(io::Error::other(error)));
                        }
                        Poll::Ready(Ok(())) => self.pending = None,
                    }
                }
            }
            self.exposed_end = self
                .position
                .saturating_add(SOURCE_CHUNK)
                .min(self.source.len());
        }
        Poll::Ready(Ok(&self.source[self.position..self.exposed_end]))
    }

    fn consume(mut self: Pin<&mut Self>, amount: usize) {
        let used = amount.min(self.exposed_end - self.position);
        self.position += used;
        self.completed += used as u64;
    }
}

impl AsyncRead for CooperativeXmlBuf<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let available = match self.as_mut().poll_fill_buf(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            Poll::Ready(Ok(available)) => available,
        };
        let copied = available.len().min(buf.remaining());
        buf.put_slice(&available[..copied]);
        self.consume(copied);
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;
    use std::task::Poll;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn dropping_a_pending_refill_returns_the_one_budget_to_its_owner() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || {
                Box::pin(async {
                    futures::future::pending::<Result<(), ExecutionServiceError>>().await
                })
            },
        );
        let owner = std::rc::Rc::new(std::cell::RefCell::new(Some(budget)));
        let bytes = vec![b'x'; SOURCE_CHUNK * 2];
        let mut reader = CooperativeXmlBuf::new(&bytes, Some(owner.clone()));
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let first = Pin::new(&mut reader).poll_fill_buf(&mut cx);
        assert!(matches!(first, Poll::Ready(Ok(chunk)) if chunk.len() == SOURCE_CHUNK));
        Pin::new(&mut reader).consume(SOURCE_CHUNK);
        assert!(matches!(
            Pin::new(&mut reader).poll_fill_buf(&mut cx),
            Poll::Pending
        ));
        assert!(owner.borrow().is_none(), "pending turn owns the budget");
        drop(reader);
        assert!(owner.borrow().is_some(), "drop restores the budget");
    }
}
