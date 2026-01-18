use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::HirProgram;
use runmat_runtime::RuntimeError;

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, RuntimeError> {
    block_on(runmat_ignition::execute(program))
}

#[allow(dead_code)]
pub fn interpret(bytecode: &runmat_ignition::Bytecode) -> Result<Vec<Value>, RuntimeError> {
    block_on(runmat_ignition::interpret(bytecode))
}
