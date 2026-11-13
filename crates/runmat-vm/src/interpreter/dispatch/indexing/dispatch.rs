//! Concrete indexing-family selection at the existing instruction future owner.
use super::{
    scalar_read_and_cell, scalar_store, slice_read, slice_store, InterpreterVars, RuntimeError,
};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pin_project_lite::pin_project! {
    #[project = IndexingFutureProjection]
    enum IndexingFuture<Read, Store, SliceRead, SliceStore> {
        Read { #[pin] future: Read },
        Store { #[pin] future: Store },
        SliceRead { #[pin] future: SliceRead },
        SliceStore { #[pin] future: SliceStore },
        NotApplicable { #[pin] future: std::future::Ready<Result<bool, RuntimeError>> },
    }
}

impl<A, B, C, D> Future for IndexingFuture<A, B, C, D>
where
    A: Future<Output = Result<bool, RuntimeError>>,
    B: Future<Output = Result<bool, RuntimeError>>,
    C: Future<Output = Result<bool, RuntimeError>>,
    D: Future<Output = Result<bool, RuntimeError>>,
{
    type Output = Result<bool, RuntimeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.project() {
            IndexingFutureProjection::Read { future } => future.poll(context),
            IndexingFutureProjection::Store { future } => future.poll(context),
            IndexingFutureProjection::SliceRead { future } => future.poll(context),
            IndexingFutureProjection::SliceStore { future } => future.poll(context),
            IndexingFutureProjection::NotApplicable { future } => future.poll(context),
        }
    }
}

pub fn dispatch_indexing<'a>(
    instr: &'a crate::bytecode::Instr,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &'a mut InterpreterVars,
    function_registry: &'a crate::bytecode::FunctionRegistry,
    pc: usize,
    runtime: &'a runmat_runtime::context::RuntimeContext,
    sequence_register: &'a mut super::super::SequenceState,
) -> impl std::future::Future<Output = Result<bool, RuntimeError>> + 'a {
    // The immutable instruction selects a concrete child before polling. The
    // existing instruction owner still pins and admits this same future; dormant
    // store/slice states do not add an async forwarding frame to a read call.
    match instr {
        crate::bytecode::Instr::Index(_)
        | crate::bytecode::Instr::IndexCell { .. }
        | crate::bytecode::Instr::IndexCellExpand { .. }
        | crate::bytecode::Instr::IndexCellList { .. }
        | crate::bytecode::Instr::StoreIndexCell { .. }
        | crate::bytecode::Instr::StoreIndexCellDelete { .. } => IndexingFuture::Read {
            future: scalar_read_and_cell::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            ),
        },
        crate::bytecode::Instr::StoreIndex(_) | crate::bytecode::Instr::StoreIndexDelete(_) => {
            IndexingFuture::Store {
                future: scalar_store::dispatch(
                    instr,
                    stack,
                    vars,
                    function_registry,
                    pc,
                    runtime,
                    sequence_register,
                ),
            }
        }
        crate::bytecode::Instr::IndexSlice(..) => IndexingFuture::SliceRead {
            future: slice_read::dispatch(
                instr,
                stack,
                vars,
                function_registry,
                pc,
                runtime,
                sequence_register,
            ),
        },
        crate::bytecode::Instr::StoreSlice(..) | crate::bytecode::Instr::StoreSliceDelete(..) => {
            IndexingFuture::SliceStore {
                future: slice_store::dispatch(
                    instr,
                    stack,
                    vars,
                    function_registry,
                    pc,
                    runtime,
                    sequence_register,
                ),
            }
        }
        _ => IndexingFuture::NotApplicable {
            future: std::future::ready(Ok(false)),
        },
    }
}
