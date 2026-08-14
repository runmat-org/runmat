use runmat_runtime::native::{
    NativeCall, NativeCallKind, NativeDeoptReason, NativeEntryPoint, NativeExit, NativeExitKind,
    NativeFrame, NativeHostStatus, NativeResumeKind, NativeResumeState, NativeValueRef,
};

use crate::{JitError, JitResult};

use super::executor::GenericExecution;
use super::state::HostState;

/// Invocation-owned state that survives every generated-code exit.
///
/// ABI records borrow this object only while one synchronous machine-code
/// entry is active. Rust values, futures, and executor objects remain outside
/// the native ABI, while locals, roots, exact resume identity, and speculative
/// result slots persist for later continuation/deoptimization cohorts.
pub struct GenericInvocation {
    state: HostState,
    entrypoint: NativeEntryPoint,
    argument_refs: Vec<NativeValueRef>,
    results: Vec<NativeValueRef>,
    requested_outputs: u32,
    resume: NativeResumeState,
    resume_pending: bool,
    terminal: bool,
}

#[derive(Debug, PartialEq)]
pub enum GenericInvocationStep {
    Completed(GenericExecution),
    Suspended {
        continuation: u64,
        generation: u64,
    },
    Deoptimized {
        reason: NativeDeoptReason,
        target: NativeResumeKind,
        guard: u64,
    },
}

impl GenericInvocation {
    pub(super) fn new(
        state: HostState,
        entrypoint: NativeEntryPoint,
        argument_refs: Vec<NativeValueRef>,
        requested_outputs: u32,
        resume: NativeResumeState,
    ) -> Self {
        Self {
            state,
            entrypoint,
            argument_refs,
            results: vec![NativeValueRef::NULL; requested_outputs as usize],
            requested_outputs,
            resume,
            resume_pending: false,
            terminal: false,
        }
    }

    pub fn resume_state(&self) -> NativeResumeState {
        self.resume
    }

    pub fn advance(&mut self) -> JitResult<GenericInvocationStep> {
        if self.terminal {
            return Err(JitError::Host(
                "completed native invocation cannot be entered again".into(),
            ));
        }
        if self.resume_pending {
            self.state.prepare_resume(self.resume)?;
            self.resume_pending = false;
        }
        self.results.fill(NativeValueRef::NULL);
        let host = super::callbacks::table(&mut self.state);
        host.validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        let roots = self.state.refresh_roots();
        let mut frame = NativeFrame {
            locals: slice_mut_pointer(&mut self.state.locals),
            local_count: self.state.locals.len(),
            roots,
            resume: &mut self.resume,
            ..NativeFrame::default()
        };
        let mut call = NativeCall {
            kind: NativeCallKind::DIRECT,
            requested_outputs: self.requested_outputs,
            host: &host,
            frame: &mut frame,
            arguments: slice_pointer(&self.argument_refs),
            argument_count: self.argument_refs.len(),
            results: slice_mut_pointer(&mut self.results),
            result_capacity: self.results.len(),
            ..NativeCall::default()
        };
        call.validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        let mut exit = NativeExit::completed(0);
        // SAFETY: every ABI record borrows invocation-owned storage that stays
        // live and stable until this synchronous machine-code entry returns.
        let _runtime_guard = self.state.runtime.enter();
        let status = unsafe { (self.entrypoint)(&mut call, &mut exit) };
        if status != NativeHostStatus::OK {
            return Err(self.state.host_failure.take().unwrap_or_else(|| {
                JitError::Host(format!("native host returned status {}", status.0))
            }));
        }
        call.validate_exit(&exit)
            .map_err(|error| JitError::Host(error.to_string()))?;
        match exit.kind {
            NativeExitKind::COMPLETED => {
                self.terminal = true;
                let outputs = self
                    .results
                    .iter()
                    .take(exit.produced_outputs as usize)
                    .map(|reference| self.state.arena.get(*reference).cloned())
                    .collect::<JitResult<Vec<_>>>()?;
                Ok(GenericInvocationStep::Completed(GenericExecution {
                    outputs,
                }))
            }
            NativeExitKind::EXCEPTION => {
                let error = self.state.last_error.take().unwrap_or_else(|| {
                    runmat_runtime::runtime_error::semantic_error(
                        "NativeException",
                        "native execution returned an exception without host error state",
                    )
                });
                Err(JitError::from(self.state.annotate_error(error)))
            }
            NativeExitKind::CANCELLED => Err(JitError::Cancelled),
            NativeExitKind::SUSPENDED => {
                self.resume_pending = true;
                Ok(GenericInvocationStep::Suspended {
                    continuation: exit.suspension.continuation,
                    generation: exit.suspension.generation,
                })
            }
            NativeExitKind::DEOPTIMIZED => {
                self.resume_pending = true;
                Ok(GenericInvocationStep::Deoptimized {
                    reason: exit.deoptimization.reason,
                    target: exit.deoptimization.target,
                    guard: exit.deoptimization.guard,
                })
            }
            other => Err(JitError::UnsupportedExit(other.0)),
        }
    }
}

fn slice_pointer<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr()
    }
}

fn slice_mut_pointer<T>(slice: &mut [T]) -> *mut T {
    if slice.is_empty() {
        std::ptr::null_mut()
    } else {
        slice.as_mut_ptr()
    }
}
