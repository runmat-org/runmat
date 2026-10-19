//! Deferred-call emission retains the target's original function-ID namespace.
use super::*;

impl Compiler {
    pub(super) fn compile_external_future(
        &mut self,
        function: FunctionId,
        args: &[MirCallArg],
        requested_outputs: RequestedOutputCount,
    ) -> Result<(), CompileError> {
        let (specs, _) = self.mir_call_arg_specs(args)?;
        for arg in args {
            self.compile_mir_call_arg(arg)?;
        }
        let out_count = requested_outputs.executor_carrier_count().ok_or_else(|| {
            self.compile_error(
                "future output cardinality cannot be derived from an assignment destination",
            )
        })?;
        self.emit(Instr::CreateExternalFuture(function, specs, out_count));
        Ok(())
    }
}
