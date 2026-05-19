#![allow(clippy::result_large_err)]

use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::LoweringContext;
use runmat_runtime::RuntimeError;

const EXEC_STACK_BYTES: usize = 32 * 1024 * 1024;

fn run_with_stack<T>(f: impl FnOnce() -> T + Send + 'static) -> Result<T, RuntimeError>
where
    T: Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name("runmat-vm-test".to_string())
        .stack_size(EXEC_STACK_BYTES)
        .spawn(f)
        .map_err(|err| RuntimeError::new(format!("failed to spawn test thread: {err}")))?;
    match handle.join() {
        Ok(result) => Ok(result),
        Err(_) => Err(RuntimeError::new("test thread panicked")),
    }
}

pub fn compile_semantic_source(source: &str) -> Result<runmat_vm::Bytecode, RuntimeError> {
    let ast = runmat_parser::parse(source).map_err(|err| RuntimeError::new(err.to_string()))?;
    let hir = runmat_hir::lower(&ast, &LoweringContext::empty())
        .map_err(|err| RuntimeError::from(runmat_vm::CompileError::from(err)))?;
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
        .map_err(|err| RuntimeError::new(format!("{err:?}")))?;
    let entrypoint = hir.assembly.entrypoints[0].id;
    runmat_vm::compile(&hir.assembly, &mir, entrypoint).map_err(RuntimeError::from)
}

#[allow(dead_code)]
pub fn execute_semantic_source(source: &str) -> Result<Vec<Value>, RuntimeError> {
    let bc = compile_semantic_source(source)?;
    interpret(&bc)
}

#[allow(dead_code)]
pub fn interpret(bytecode: &runmat_vm::Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let bytecode = bytecode.clone();
    run_with_stack(move || block_on(runmat_vm::interpret(&bytecode)))?
}
