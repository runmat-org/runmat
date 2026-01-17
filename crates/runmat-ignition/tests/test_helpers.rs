use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::HirProgram;

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, String> {
    block_on(runmat_ignition::execute(program))
}

pub fn interpret(bytecode: &runmat_ignition::Bytecode) -> Result<Vec<Value>, String> {
    block_on(runmat_ignition::interpret(bytecode))
}
