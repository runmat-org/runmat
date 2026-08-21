use runmat_runtime::native::{
    NativeCall, NativeCallKind, NativeDeoptReason, NativeEntryPoint, NativeExit, NativeExitKind,
    NativeFrame, NativeHostStatus, NativeResumeKind, NativeResumeState, NativeValueRef,
};

use crate::deopt::MaterializedFrame;
use crate::osr::OsrTarget;
use crate::{NativeExecutorError, NativeExecutorResult};

use super::executor::NativeExecution;
use super::state::HostState;

/// Invocation-owned state that survives every generated-code exit.
///
/// ABI records borrow this object only while one synchronous machine-code
/// entry is active. Rust values, futures, and executor objects remain outside
/// the native ABI, while locals, roots, exact resume identity, and speculative
/// result slots persist for later continuation/deoptimization cohorts.
pub struct NativeInvocation {
    state: HostState,
    entrypoint: NativeEntryPoint,
    argument_refs: Vec<NativeValueRef>,
    results: Vec<NativeValueRef>,
    requested_outputs: u32,
    resume: NativeResumeState,
    resume_pending: bool,
    terminal: bool,
    deoptimization_target: Option<NativeResumeKind>,
    osr_target: Option<(OsrTarget, NativeEntryPoint)>,
    _active_osr_target: Option<OsrTarget>,
}

#[derive(Debug, PartialEq)]
pub enum NativeInvocationStep {
    Completed(NativeExecution),
    Suspended {
        continuation: u64,
        generation: u64,
    },
    Deoptimized {
        reason: NativeDeoptReason,
        target: NativeResumeKind,
        guard: u64,
        frame: MaterializedFrame,
    },
}

impl NativeInvocation {
    pub(super) fn new(
        state: HostState,
        entrypoint: NativeEntryPoint,
        argument_refs: Vec<NativeValueRef>,
        requested_outputs: u32,
        resume: NativeResumeState,
        osr_target: Option<(OsrTarget, NativeEntryPoint)>,
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
            deoptimization_target: None,
            osr_target,
            _active_osr_target: None,
        }
    }

    pub fn resume_state(&self) -> NativeResumeState {
        self.resume
    }

    pub async fn resume_suspension(
        &mut self,
        continuation: u64,
        generation: u64,
    ) -> NativeExecutorResult<()> {
        let completion =
            match super::awaiting::complete(&mut self.state, continuation, generation).await {
                Ok(completion) => completion,
                Err(NativeExecutorError::Runtime(error)) => {
                    let error = *error;
                    let exception = self.state.arena.insert(runmat_value::Value::MException(
                        runmat_runtime::runtime_error::exception_from_error(&error),
                    ));
                    let native_exception = runmat_runtime::native::NativeException {
                        handle: exception.handle,
                        generation: exception.generation,
                        source: self.state.current_source,
                    };
                    if let Some(target) =
                        super::site::redirect_exception(&mut self.state, native_exception)?
                    {
                        self.install_resume_target(target)?;
                        self.resume_pending = false;
                        return Ok(());
                    }
                    return Err(NativeExecutorError::from(self.state.annotate_error(error)));
                }
                Err(error) => return Err(error),
            };
        let target = super::site::resume_await(&mut self.state, completion)?;
        self.install_resume_target(target)?;
        self.resume_pending = false;
        Ok(())
    }

    /// Resume a guard failure in the generic native version at the exact site.
    pub fn resume_deoptimization(&mut self) -> NativeExecutorResult<()> {
        if self.deoptimization_target != Some(NativeResumeKind::GENERIC_NATIVE) {
            return Err(NativeExecutorError::Host(
                "native deoptimization is not targeted at generic native code".into(),
            ));
        }
        self.state.prepare_resume(self.resume)?;
        self.resume_pending = false;
        self.deoptimization_target = None;
        Ok(())
    }

    /// Transfer the live generic-native frame to the session-admitted
    /// optimized entrypoint at the exact materialized loop header.
    pub fn resume_optimization(&mut self) -> NativeExecutorResult<()> {
        if self.deoptimization_target != Some(NativeResumeKind::OPTIMIZED_NATIVE) {
            return Err(NativeExecutorError::Host(
                "native continuation is not targeted at optimized code".into(),
            ));
        }
        let (target, entrypoint) = self.osr_target.take().ok_or_else(|| {
            NativeExecutorError::Host(
                "optimized native continuation has no retained OSR target".into(),
            )
        })?;
        self.state.prepare_resume(self.resume)?;
        self.entrypoint = entrypoint;
        self._active_osr_target = Some(target);
        self.resume_pending = false;
        self.deoptimization_target = None;
        Ok(())
    }

    pub fn advance(&mut self) -> NativeExecutorResult<NativeInvocationStep> {
        if self.terminal {
            return Err(NativeExecutorError::Host(
                "completed native invocation cannot be entered again".into(),
            ));
        }
        if self.resume_pending {
            self.state.prepare_resume(self.resume)?;
            self.resume_pending = false;
        }
        loop {
            let exit = self.enter_once()?;
            match exit.kind {
                NativeExitKind::COMPLETED => {
                    self.terminal = true;
                    let outputs = self
                        .results
                        .iter()
                        .take(exit.produced_outputs as usize)
                        .map(|reference| self.state.arena.get(*reference).cloned())
                        .collect::<NativeExecutorResult<Vec<_>>>()?;
                    return Ok(NativeInvocationStep::Completed(NativeExecution {
                        outputs,
                        captures: self.state.capture_results()?,
                        loop_backedges: self.state.loop_backedges(),
                        osr_entry: self.state.osr_entry(),
                        vectorized_regions: self.state.vectorized_regions(),
                        workspace: self.state.workspace_snapshot(),
                        expression: self.state.expression_result()?,
                    }));
                }
                NativeExitKind::EXCEPTION => {
                    let error = self.state.last_error.take().unwrap_or_else(|| {
                        runmat_runtime::runtime_error::semantic_error(
                            "NativeException",
                            "native execution returned an exception without host error state",
                        )
                    });
                    if let Some(target) =
                        super::site::redirect_exception(&mut self.state, exit.exception)?
                    {
                        self.install_resume_target(target)?;
                        continue;
                    }
                    return Err(NativeExecutorError::from(self.state.annotate_error(error)));
                }
                NativeExitKind::CANCELLED => return Err(NativeExecutorError::Cancelled),
                NativeExitKind::SUSPENDED => {
                    self.resume_pending = true;
                    return Ok(NativeInvocationStep::Suspended {
                        continuation: exit.suspension.continuation,
                        generation: exit.suspension.generation,
                    });
                }
                NativeExitKind::DEOPTIMIZED => {
                    self.resume_pending = true;
                    self.deoptimization_target = Some(exit.deoptimization.target);
                    return Ok(NativeInvocationStep::Deoptimized {
                        reason: exit.deoptimization.reason,
                        target: exit.deoptimization.target,
                        guard: exit.deoptimization.guard,
                        frame: self.state.take_deoptimization()?,
                    });
                }
                other => return Err(NativeExecutorError::UnsupportedExit(other.0)),
            }
        }
    }

    fn install_resume_target(
        &mut self,
        target: runmat_runtime::native::NativeSiteRequest,
    ) -> NativeExecutorResult<()> {
        self.resume.function = target.function;
        self.resume.block = target.block;
        self.resume.position = target.position;
        self.resume.phase = target.phase.0;
        self.resume.ordinal = target.ordinal;
        self.state.prepare_resume(self.resume)
    }

    fn enter_once(&mut self) -> NativeExecutorResult<NativeExit> {
        self.results.fill(NativeValueRef::NULL);
        let host = super::callbacks::table(&mut self.state);
        host.validate()
            .map_err(|error| NativeExecutorError::Host(error.to_string()))?;
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
            .map_err(|error| NativeExecutorError::Host(error.to_string()))?;
        let mut exit = NativeExit::completed(0);
        // SAFETY: every ABI record borrows invocation-owned storage that stays
        // live and stable until this synchronous machine-code entry returns.
        let _runtime_guard = self.state.runtime.enter();
        let status = unsafe { (self.entrypoint)(&mut call, &mut exit) };
        if status != NativeHostStatus::OK {
            return Err(self.state.host_failure.take().unwrap_or_else(|| {
                NativeExecutorError::Host(format!("native host returned status {}", status.0))
            }));
        }
        call.validate_exit(&exit)
            .map_err(|error| NativeExecutorError::Host(error.to_string()))?;
        Ok(exit)
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
