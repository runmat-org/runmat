use runmat_mir::{MirCall, MirCallArg};

use crate::compiler::CompileError;
use crate::instr::Instr;

use super::core::{call_name, Compiler};

impl Compiler {
    pub(super) fn try_compile_parallel_call(
        &mut self,
        call: &MirCall,
    ) -> Result<bool, CompileError> {
        let Some(name) = call_name(call) else {
            return Ok(false);
        };
        match name {
            "parpool" => {
                self.compile_parallel_arguments(&call.args)?;
                self.emit(Instr::EnsurePool(call.args.len()));
            }
            "gcp" => {
                self.compile_parallel_arguments(&call.args)?;
                self.emit(Instr::CurrentPool(call.args.len()));
            }
            "parfeval" => {
                self.compile_scheduled_call(call, "parfeval", false)?;
            }
            "parfevalOnAll" => {
                self.compile_scheduled_call(call, "parfevalOnAll", true)?;
            }
            "fetchOutputs" => {
                if call.args.is_empty() || !(call.args.len() - 1).is_multiple_of(2) {
                    return Err(self
                        .compile_error(
                            "fetchOutputs requires a future followed by complete name-value pairs",
                        )
                        .with_identifier("RunMat:fetchOutputs:InvalidInput"));
                }
                self.compile_parallel_arguments(&call.args)?;
                let requested_outputs = call.requested_outputs.known_count().ok_or_else(|| {
                    self.compile_error(
                        "fetchOutputs output count cannot be derived from an assignment destination",
                    )
                })?;
                self.emit(Instr::FetchOutputs {
                    arg_count: call.args.len(),
                    requested_outputs,
                });
            }
            "fetchNext" => {
                let has_timeout = match call.args.len() {
                    1 => false,
                    2 => true,
                    _ => {
                        return Err(self
                            .compile_error("fetchNext requires a future array and optional timeout")
                            .with_identifier("RunMat:fetchNext:InvalidInput"));
                    }
                };
                self.compile_parallel_arguments(&call.args)?;
                let requested_outputs = call.requested_outputs.known_count().ok_or_else(|| {
                    self.compile_error(
                        "fetchNext output count cannot be derived from an assignment destination",
                    )
                })?;
                self.emit(Instr::FetchNext {
                    has_timeout,
                    requested_outputs,
                });
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn compile_scheduled_call(
        &mut self,
        call: &MirCall,
        builtin: &str,
        on_all: bool,
    ) -> Result<(), CompileError> {
        if call.args.len() < 2 {
            return Err(self
                .compile_error(format!(
                    "{builtin} requires a function handle and output count"
                ))
                .with_identifier(format!("RunMat:{builtin}:NotEnoughInputs")));
        }
        self.compile_parallel_arguments(&call.args)?;
        self.emit(Instr::ScheduleFeval {
            arg_count: call.args.len(),
            on_all,
        });
        Ok(())
    }

    fn compile_parallel_arguments(&mut self, arguments: &[MirCallArg]) -> Result<(), CompileError> {
        for argument in arguments {
            self.compile_mir_call_arg(argument)?;
        }
        Ok(())
    }
}
