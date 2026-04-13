use crate::bytecode::Bytecode;
use miette::{SourceOffset, SourceSpan};
use runmat_runtime::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use std::cell::Cell;

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
