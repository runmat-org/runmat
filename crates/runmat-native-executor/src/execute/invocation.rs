//! Public continuation ownership. A retained terminal invocation owns no frame.
use runmat_runtime::native::{
    NativeDeoptReason, NativeEntryPoint, NativeResumeKind, NativeResumeState, NativeValueRef,
};

use super::executor::RootedNativeExecution;
use super::state::HostState;
use crate::deopt::MaterializedFrame;
use crate::osr::OsrTarget;
use crate::{NativeExecutorError, NativeExecutorResult};

mod active;
#[cfg(test)]
mod cancellation_tests;
mod lifecycle;
#[cfg(test)]
mod observation_tests;

use active::ActiveInvocation;
use lifecycle::ExecutionLease;

/// One generated invocation. Suspensions/deoptimizations retain the active
/// frame; terminal execution outcomes release it even if this object is kept.
pub struct NativeInvocation {
    active: Option<ActiveInvocation>,
    last_resume: NativeResumeState,
}

#[derive(Debug, PartialEq)]
pub enum NativeInvocationStep {
    Completed(RootedNativeExecution),
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
    pub(super) fn cleanup_runtime(
        &self,
    ) -> NativeExecutorResult<runmat_runtime::context::RuntimeContext> {
        Ok(self.active()?.state.runtime.clone())
    }

    pub(super) fn new(
        state: HostState,
        entrypoint: NativeEntryPoint,
        argument_refs: Vec<NativeValueRef>,
        requested_outputs: u32,
        resume: NativeResumeState,
        osr_target: Option<(OsrTarget, NativeEntryPoint)>,
    ) -> Self {
        Self {
            active: Some(ActiveInvocation::new(
                state,
                entrypoint,
                argument_refs,
                requested_outputs,
                resume,
                osr_target,
            )),
            last_resume: resume,
        }
    }

    pub fn inject_deoptimization_fault(&mut self, fault: crate::deopt::FaultInjection) {
        if let Some(active) = &mut self.active {
            active.inject_deoptimization_fault(fault);
        }
    }

    pub fn resume_state(&self) -> NativeResumeState {
        self.active
            .as_ref()
            .map_or(self.last_resume, ActiveInvocation::resume_state)
    }

    /// The activation that owns the suspended native frame's shared lexical
    /// bindings. It can differ from the parent activation supplied at entry.
    pub fn deoptimized_lexical_environment(
        &self,
    ) -> NativeExecutorResult<Option<runmat_value::LexicalEnvironment>> {
        let active = self.active()?;
        if active.deoptimization_target != Some(NativeResumeKind::INTERPRETER) {
            return Err(NativeExecutorError::Host(
                "native invocation is not awaiting interpreter deoptimization".into(),
            ));
        }
        Ok(active.state.lexical_environment.clone())
    }

    /// Invalid continuation identities leave the pending frame untouched.
    /// After the first poll accepts the identity, dropping this future abandons
    /// the invocation; an unpolled future has made no transition.
    pub async fn resume_suspension(
        &mut self,
        continuation: u64,
        generation: u64,
    ) -> NativeExecutorResult<()> {
        let state = &self.active()?.state;
        if state.pending_call.is_some() {
            super::call_suspension::validate_pending(state, continuation, generation)?;
        } else {
            super::awaiting::validate_pending(state, continuation, generation)?;
        }
        let mut lease = ExecutionLease::take(self)?;
        lease.check_cancellation()?;
        let runtime = lease.active().state.runtime.clone();
        let token = runtime.cancellation();
        let result = {
            let cancelled = std::pin::pin!(token.cancelled());
            let operation = std::pin::pin!(runtime.scope(
                lease
                    .active_mut()
                    .resume_suspension(continuation, generation)
            ));
            match futures::future::select(cancelled, operation).await {
                futures::future::Either::Left((reason, _)) => {
                    Err(super::control::cancellation_error(reason))
                }
                futures::future::Either::Right((result, _)) => result,
            }
        };
        // The operation and cancellation listener are dropped before ownership
        // is restored or terminal cleanup runs.
        result?;
        lease.restore();
        Ok(())
    }

    pub fn resume_deoptimization(&mut self) -> NativeExecutorResult<()> {
        if self.active()?.deoptimization_target != Some(NativeResumeKind::GENERIC_NATIVE) {
            return Err(NativeExecutorError::Host(
                "native deoptimization is not targeted at generic native code".into(),
            ));
        }
        let mut lease = ExecutionLease::take(self)?;
        lease.check_cancellation()?;
        lease.active_mut().resume_deoptimization()?;
        lease.restore();
        Ok(())
    }

    pub fn resume_optimization(&mut self) -> NativeExecutorResult<()> {
        if self.active()?.deoptimization_target != Some(NativeResumeKind::OPTIMIZED_NATIVE) {
            return Err(NativeExecutorError::Host(
                "native continuation is not targeted at optimized code".into(),
            ));
        }
        let mut lease = ExecutionLease::take(self)?;
        lease.check_cancellation()?;
        lease.active_mut().resume_optimization()?;
        lease.restore();
        Ok(())
    }

    /// Errors after entering execution are terminal. Request-validation errors
    /// on the resume APIs above remain retryable. Caught language errors are
    /// redirected inside the active executor and are not terminal outcomes.
    pub fn advance(&mut self) -> NativeExecutorResult<NativeInvocationStep> {
        let mut lease = ExecutionLease::take(self)?;
        lease.check_cancellation()?;
        let step = lease.active_mut().advance()?;
        if !matches!(step, NativeInvocationStep::Completed(_)) {
            lease.restore();
        }
        Ok(step)
    }

    fn active(&self) -> NativeExecutorResult<&ActiveInvocation> {
        self.active.as_ref().ok_or_else(terminal_error)
    }
}

fn terminal_error() -> NativeExecutorError {
    NativeExecutorError::Host("completed native invocation cannot be entered again".into())
}

impl Drop for NativeInvocation {
    fn drop(&mut self) {
        if let Ok(lease) = ExecutionLease::take(self) {
            drop(lease);
        }
    }
}
