use runmat_hir::FunctionId;
use runmat_value::Value;
use runmat_vm::{Bytecode, Instr};

use super::ExecutableUnit;

/// Bytecode-level call frame for invoking one exact semantic procedure through
/// the session's normal interpreter/JIT backend policy.
pub(crate) struct ProcedureProgram {
    pub bytecode: Bytecode,
    pub variables: Vec<Value>,
    pub result_slot: Option<usize>,
}

impl ExecutableUnit {
    pub(crate) fn procedure_program(
        &self,
        function: FunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Option<ProcedureProgram> {
        if requested_outputs > 1 {
            return None;
        }

        let result_slot = (requested_outputs == 1).then_some(arguments.len());
        let mut instructions = (0..arguments.len()).map(Instr::LoadVar).collect::<Vec<_>>();
        instructions.push(Instr::CallSemanticFunctionMulti(
            function,
            arguments.len(),
            requested_outputs,
        ));
        if let Some(slot) = result_slot {
            instructions.push(Instr::StoreVar(slot));
        }
        instructions.push(Instr::Return);

        let mut bytecode =
            Bytecode::with_instructions(instructions, arguments.len() + requested_outputs);
        bytecode.source_id = self.bytecode().source_id;
        bytecode.bound_functions = self.bytecode().bound_functions.clone();
        bytecode.function_registry = self.bytecode().function_registry.clone();

        let mut variables = arguments;
        if requested_outputs == 1 {
            variables.push(Value::Num(0.0));
        }
        Some(ProcedureProgram {
            bytecode,
            variables,
            result_slot,
        })
    }

    pub(crate) fn backend_cache_namespace(&self) -> String {
        format!(
            "{}|{}",
            self.revision().program_revision.canonical_identity(),
            self.revision().source_digest
        )
    }
}
