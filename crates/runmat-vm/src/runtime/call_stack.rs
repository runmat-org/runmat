use crate::bytecode::Bytecode;
use miette::SourceSpan;
use runmat_runtime::{CallFrame, RuntimeError};
use runmat_thread_local::runmat_thread_local;
use std::cell::{Cell, RefCell};

pub const DEFAULT_CALLSTACK_LIMIT: usize = 200;
pub const DEFAULT_ERROR_NAMESPACE: &str = "RunMat";

#[derive(Default, Clone)]
struct CallStackState {
    frames: Vec<CallFrame>,
    depth: usize,
}

pub struct CallFrameGuard;

impl Drop for CallFrameGuard {
    fn drop(&mut self) {
        pop_call_frame();
    }
}

runmat_thread_local! {
    static CALL_STACK: RefCell<CallStackState> = const {
        RefCell::new(CallStackState {
            frames: Vec::new(),
            depth: 0,
        })
    };
    static CALL_STACK_LIMIT: Cell<usize> = const { Cell::new(DEFAULT_CALLSTACK_LIMIT) };
    static ERROR_NAMESPACE: RefCell<String> = const {
        RefCell::new(String::new())
    };
}

pub fn callstack_limit() -> usize {
    CALL_STACK_LIMIT.with(|limit| limit.get())
}

pub fn error_namespace() -> String {
    let ns = ERROR_NAMESPACE.with(|ns| ns.borrow().clone());
    if ns.trim().is_empty() {
        DEFAULT_ERROR_NAMESPACE.to_string()
    } else {
        ns
    }
}

pub fn set_error_namespace(namespace: &str) {
    let namespace = if namespace.trim().is_empty() {
        DEFAULT_ERROR_NAMESPACE
    } else {
        namespace
    };
    ERROR_NAMESPACE.with(|ns| {
        *ns.borrow_mut() = namespace.to_string();
    });
}

pub fn set_call_stack_limit(limit: usize) {
    CALL_STACK_LIMIT.with(|cell| cell.set(limit));
    CALL_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if limit == 0 {
            stack.frames.clear();
        } else if stack.frames.len() > limit {
            while stack.frames.len() > limit {
                stack.frames.remove(0);
            }
        }
    });
}

pub fn push_call_frame(name: &str, bytecode: &Bytecode, pc: usize) -> CallFrameGuard {
    let span = bytecode
        .instr_spans
        .get(pc)
        .map(|span| (span.start, span.end));
    let frame = CallFrame {
        function: name.to_string(),
        source_id: bytecode.source_id.map(|id| id.0),
        span,
    };
    CALL_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.depth = stack.depth.saturating_add(1);
        let limit = callstack_limit();
        if limit == 0 {
            return;
        }
        if stack.frames.len() == limit {
            stack.frames.remove(0);
        }
        stack.frames.push(frame);
    });
    CallFrameGuard
}

pub fn pop_call_frame() {
    CALL_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.depth > 0 {
            stack.depth -= 1;
        }
        if !stack.frames.is_empty() {
            stack.frames.pop();
        }
    });
}

pub fn attach_call_frames(
    bytecode: &Bytecode,
    current_function_name: &str,
    mut err: RuntimeError,
) -> RuntimeError {
    if !err.context.call_frames.is_empty() || !err.context.call_stack.is_empty() {
        return err;
    }
    let (mut frames, depth) = CALL_STACK.with(|stack| {
        let stack = stack.borrow();
        let frames = stack.frames.clone();
        (frames, stack.depth)
    });
    let limit = callstack_limit();
    if frames.is_empty() {
        if limit == 0 {
            return err;
        }
        let span = err.span.as_ref().map(|span: &SourceSpan| {
            let start = span.offset();
            let end = start + span.len();
            (start, end)
        });
        if span.is_some() || !current_function_name.is_empty() {
            frames.push(CallFrame {
                function: current_function_name.to_string(),
                source_id: bytecode.source_id.map(|id| id.0),
                span,
            });
        }
    }
    let elided = if frames.is_empty() {
        0
    } else {
        depth.saturating_sub(frames.len())
    };
    err.context.call_frames = frames;
    err.context.call_frames_elided = elided;
    err
}
