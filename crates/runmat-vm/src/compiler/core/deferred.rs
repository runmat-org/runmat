//! Deferred-call emission retains the target's original function-ID namespace.
use super::*;

impl Compiler {
    pub(super) fn compile_local_future(
        &mut self,
        function: FunctionId,
        args: &[MirCallArg],
        out_count: usize,
    ) -> Result<(), CompileError> {
        let (specs, has_expansion) = self.mir_call_arg_specs(args);
        for arg in args {
            self.compile_mir_call_arg(arg)?;
        }
        if has_expansion {
            self.emit(Instr::CreateSemanticFutureExpandMultiOutput(
                function, specs, out_count,
            ));
        } else {
            self.emit(Instr::CreateSemanticFuture(function, args.len(), out_count));
        }
        Ok(())
    }

    pub(super) fn compile_external_future(
        &mut self,
        function: FunctionId,
        args: &[MirCallArg],
        out_count: usize,
    ) -> Result<(), CompileError> {
        let (specs, _) = self.mir_call_arg_specs(args);
        for arg in args {
            self.compile_mir_call_arg(arg)?;
        }
        self.emit(Instr::CreateExternalFuture(function, specs, out_count));
        Ok(())
    }
}
