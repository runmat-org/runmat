use crate::bytecode::Bytecode;
use miette::{SourceOffset, SourceSpan};
use runmat_runtime::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

runmat_thread_local! {
    static CURRENT_PC: Cell<usize> = const { Cell::new(0) };
}

#[inline]
pub fn set_vm_pc(pc: usize) {
    CURRENT_PC.with(|cell| cell.set(pc));
}

#[inline]
pub fn current_vm_pc() -> usize {
    CURRENT_PC.with(|cell| cell.get())
}

/// Retain the existing diagnostic PC as an invocation-local poll projection.
/// Sibling VM continuations may execute on the same thread between polls.
pub(crate) fn scope_vm_pc<F: Future>(future: F) -> VmPcFuture<F> {
    VmPcFuture {
        pc: 0,
        future: Some(Box::pin(future)),
    }
}

pub(crate) struct VmPcFuture<F: Future> {
    pc: usize,
    future: Option<Pin<Box<F>>>,
}

struct PcActivation<'a> {
    frame: &'a mut usize,
    previous: usize,
}

impl<'a> PcActivation<'a> {
    fn enter(frame: &'a mut usize) -> Self {
        let previous = CURRENT_PC.with(|cell| cell.replace(*frame));
        Self { frame, previous }
    }
}

impl Drop for PcActivation<'_> {
    fn drop(&mut self) {
        *self.frame = CURRENT_PC.with(|cell| cell.replace(self.previous));
    }
}

impl<F: Future> Future for VmPcFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _activation = PcActivation::enter(&mut this.pc);
        let result = this
            .future
            .as_mut()
            .expect("VM PC future polled after completion")
            .as_mut()
            .poll(context);
        if result.is_ready() {
            drop(this.future.take());
        }
        result
    }
}

impl<F: Future> Drop for VmPcFuture<F> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let _activation = PcActivation::enter(&mut self.pc);
            drop(future);
        }
    }
}

impl<F: Future> Unpin for VmPcFuture<F> {}

pub fn attach_span_at(bytecode: &Bytecode, pc: usize, mut err: RuntimeError) -> RuntimeError {
    if err.span.is_none() {
        if let Some(span) = bytecode.instr_spans.get(pc) {
            let len = span.end.saturating_sub(span.start).max(1);
            err.span = Some(SourceSpan::new(SourceOffset::from(span.start), len));
        }
    }
    err
}

pub fn attach_span_from_pc(bytecode: &Bytecode, err: RuntimeError) -> RuntimeError {
    let pc = current_vm_pc();
    attach_span_at(bytecode, pc, err)
}

#[inline]
pub fn mex(id: &str, msg: &str) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(id, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Instr;
    use futures::task::noop_waker;

    #[test]
    fn suspended_frames_restore_their_own_diagnostic_pc() {
        let mut bytecode = Bytecode::with_instructions(vec![Instr::Return, Instr::Return], 0);
        bytecode.instr_spans = vec![
            runmat_hir::Span { start: 5, end: 6 },
            runmat_hir::Span { start: 20, end: 21 },
        ];
        let child_bytecode = bytecode.clone();
        let (parent_tx, parent_rx) = futures::channel::oneshot::channel();
        let (child_tx, child_rx) = futures::channel::oneshot::channel();
        let previous = current_vm_pc();
        let mut parent = Box::pin(scope_vm_pc(async move {
            set_vm_pc(1);
            parent_rx.await.expect("resume parent");
            attach_span_from_pc(&bytecode, mex("ParentFailure", "parent"))
                .span
                .expect("parent source span")
                .offset()
        }));
        let mut child = Box::pin(scope_vm_pc(async move {
            set_vm_pc(0);
            child_rx.await.expect("resume child");
            attach_span_from_pc(&child_bytecode, mex("ChildFailure", "child"))
                .span
                .expect("child source span")
                .offset()
        }));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(parent.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(current_vm_pc(), previous);
        assert!(matches!(child.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(current_vm_pc(), previous);
        parent_tx.send(()).expect("parent awaits resume");
        assert!(matches!(
            parent.as_mut().poll(&mut context),
            Poll::Ready(20)
        ));
        assert_eq!(current_vm_pc(), previous);
        child_tx.send(()).expect("child awaits resume");
        assert!(matches!(child.as_mut().poll(&mut context), Poll::Ready(5)));
        assert_eq!(current_vm_pc(), previous);
    }
}
