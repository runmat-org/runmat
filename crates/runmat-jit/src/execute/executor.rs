use runmat_runtime::context::RuntimeContext;
use runmat_runtime::native::*;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;

use crate::{CompiledExecutable, GenericCompiler, JitError, JitResult};

use super::state::HostState;

pub struct GenericExecutor {
    assembly: runmat_native_codegen::NativeAssembly,
    compiled: CompiledExecutable,
}

#[derive(Debug, PartialEq)]
pub struct GenericExecution {
    pub outputs: Vec<Value>,
}

impl GenericExecutor {
    pub fn compile(assembly: runmat_native_codegen::NativeAssembly) -> JitResult<Self> {
        let compiled = GenericCompiler::compile(&assembly)?;
        Ok(Self { assembly, compiled })
    }

    pub fn invoke(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        let function_ir = self
            .assembly
            .functions
            .iter()
            .find(|candidate| candidate.id == function)
            .cloned()
            .ok_or_else(|| {
                JitError::Host(format!("native function {} is unavailable", function.0))
            })?;
        let requested_outputs = u32::try_from(requested_outputs)
            .map_err(|_| JitError::Host("requested output count exceeds native ABI".into()))?;
        let (mut state, argument_refs) =
            HostState::new(function_ir, arguments, requested_outputs as usize, runtime)?;
        let host = super::callbacks::table(&mut state);
        host.validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        let mut results = vec![NativeValueRef::NULL; requested_outputs as usize];
        let mut resume = NativeResumeState {
            function: function.0,
            block: state.function.entry.0,
            local_count: state.locals.len() as u32,
            source: NativeSourceLocation {
                source: state.function.source.0,
                ..NativeSourceLocation::default()
            },
            ..NativeResumeState::default()
        };
        let roots = state.refresh_roots();
        let mut frame = NativeFrame {
            locals: slice_mut_pointer(&mut state.locals),
            local_count: state.locals.len(),
            roots,
            resume: &mut resume,
            ..NativeFrame::default()
        };
        let mut call = NativeCall {
            kind: NativeCallKind::DIRECT,
            requested_outputs,
            host: &host,
            frame: &mut frame,
            arguments: slice_pointer(&argument_refs),
            argument_count: argument_refs.len(),
            results: slice_mut_pointer(&mut results),
            result_capacity: results.len(),
            ..NativeCall::default()
        };
        call.validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        let mut exit = NativeExit::completed(0);
        let entrypoint = self.compiled.entrypoint(function)?;
        // SAFETY: all borrowed ABI records and their backing slices remain live
        // and stable until the synchronous generated entrypoint returns. The
        // explicit runtime guard makes direct session/global/persistent helpers
        // observe the same invocation-owned context as scoped async calls.
        let _runtime_guard = state.runtime.enter();
        let status = unsafe { entrypoint(&mut call, &mut exit) };
        if status != NativeHostStatus::OK {
            return Err(state.host_failure.take().unwrap_or_else(|| {
                JitError::Host(format!("native host returned status {}", status.0))
            }));
        }
        call.validate_exit(&exit)
            .map_err(|error| JitError::Host(error.to_string()))?;
        match exit.kind {
            NativeExitKind::COMPLETED => {
                let outputs = results
                    .into_iter()
                    .take(exit.produced_outputs as usize)
                    .map(|reference| state.arena.get(reference).cloned())
                    .collect::<JitResult<Vec<_>>>()?;
                Ok(GenericExecution { outputs })
            }
            NativeExitKind::EXCEPTION => {
                let error = state.last_error.take().unwrap_or_else(|| {
                    runmat_runtime::runtime_error::semantic_error(
                        "NativeException",
                        "native execution returned an exception without host error state",
                    )
                });
                Err(JitError::from(state.annotate_error(error)))
            }
            NativeExitKind::CANCELLED => Err(JitError::Cancelled),
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
