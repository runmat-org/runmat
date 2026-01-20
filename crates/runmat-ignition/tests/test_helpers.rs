use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::{HirProgram, LoweringContext, SemanticError};
use runmat_runtime::RuntimeError;

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, RuntimeError> {
    block_on(runmat_ignition::execute(program))
}

pub fn lower(program: &runmat_parser::Program) -> Result<HirProgram, SemanticError> {
    runmat_hir::lower(program, &LoweringContext::empty()).map(|result| result.hir)
}

#[allow(dead_code)]
pub fn interpret(bytecode: &runmat_ignition::Bytecode) -> Result<Vec<Value>, RuntimeError> {
    block_on(runmat_ignition::interpret(bytecode))
}
