#![allow(clippy::result_large_err)]

use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::{HirProgram, LoweringContext, SemanticError};
use runmat_runtime::RuntimeError;

const EXEC_STACK_BYTES: usize = 32 * 1024 * 1024;

fn run_with_stack<T>(f: impl FnOnce() -> T + Send + 'static) -> Result<T, RuntimeError>
where
    T: Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name("runmat-ignition-test".to_string())
        .stack_size(EXEC_STACK_BYTES)
        .spawn(f)
        .map_err(|err| RuntimeError::new(format!("failed to spawn test thread: {err}")))?;
    match handle.join() {
        Ok(result) => Ok(result),
        Err(_) => Err(RuntimeError::new("test thread panicked")),
    }
}

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, RuntimeError> {
    let program = program.clone();
    run_with_stack(move || block_on(runmat_ignition::execute(&program)))?
}

pub fn lower(program: &runmat_parser::Program) -> Result<HirProgram, SemanticError> {
    runmat_hir::lower(program, &LoweringContext::empty()).map(|result| result.hir)
}

#[allow(dead_code)]
pub fn interpret(bytecode: &runmat_ignition::Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let bytecode = bytecode.clone();
    run_with_stack(move || block_on(runmat_ignition::interpret(&bytecode)))?
}
