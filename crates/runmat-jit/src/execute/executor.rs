use runmat_runtime::context::RuntimeContext;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;

use crate::{CompiledExecutable, GenericCompiler, JitError, JitResult};

use super::invocation::{GenericInvocation, GenericInvocationStep};
use super::state::HostState;

pub struct GenericExecutor {
    assembly: runmat_native_codegen::NativeAssembly,
    compiled: CompiledExecutable,
    program_capture: Option<Vec<u8>>,
}

#[derive(Debug, PartialEq)]
pub struct GenericExecution {
    pub outputs: Vec<Value>,
}

impl GenericExecutor {
    pub fn compile(assembly: runmat_native_codegen::NativeAssembly) -> JitResult<Self> {
        Self::compile_with_program_capture(assembly, None)
    }

    pub fn compile_with_program_capture(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
    ) -> JitResult<Self> {
        let compiled = GenericCompiler::compile(&assembly)?;
        Ok(Self {
            assembly,
            compiled,
            program_capture,
        })
    }

    pub fn invoke(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        let mut invocation = self.begin(function, arguments, requested_outputs, runtime)?;
        match invocation.advance()? {
            GenericInvocationStep::Completed(execution) => Ok(execution),
            GenericInvocationStep::Suspended { .. } => Err(JitError::UnsupportedExit(
                runmat_runtime::native::NativeExitKind::SUSPENDED.0,
            )),
            GenericInvocationStep::Deoptimized { .. } => Err(JitError::UnsupportedExit(
                runmat_runtime::native::NativeExitKind::DEOPTIMIZED.0,
            )),
        }
    }

    pub async fn invoke_async(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        let mut invocation = self.begin(function, arguments, requested_outputs, runtime)?;
        loop {
            match invocation.advance()? {
                GenericInvocationStep::Completed(execution) => return Ok(execution),
                GenericInvocationStep::Suspended {
                    continuation,
                    generation,
                } => {
                    invocation
                        .resume_suspension(continuation, generation)
                        .await?;
                }
                GenericInvocationStep::Deoptimized { .. } => {
                    return Err(JitError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::DEOPTIMIZED.0,
                    ));
                }
            }
        }
    }

    pub fn begin(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericInvocation> {
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
        let (state, argument_refs) = HostState::new(
            function_ir,
            arguments,
            requested_outputs as usize,
            runtime,
            self.program_capture.clone(),
        )?;
        let resume = runmat_runtime::native::NativeResumeState {
            function: function.0,
            block: state.function.entry.0,
            phase: runmat_runtime::native::NativeSitePhase::RVALUE.0,
            local_count: state.locals.len() as u32,
            source: runmat_runtime::native::NativeSourceLocation {
                source: state.function.source.0,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
            ..runmat_runtime::native::NativeResumeState::default()
        };
        let entrypoint = self.compiled.entrypoint(function)?;
        Ok(GenericInvocation::new(
            state,
            entrypoint,
            argument_refs,
            requested_outputs,
            resume,
        ))
    }
}
