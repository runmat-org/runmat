#![allow(clippy::result_large_err)]

use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_hir::LoweringContext;
use runmat_runtime::RuntimeError;

const EXEC_STACK_BYTES: usize = 32 * 1024 * 1024;

struct VmThreadStateGuard;

impl VmThreadStateGuard {
    fn enter() -> Self {
        runmat_vm::reset_thread_state_for_tests();
        Self
    }
}

impl Drop for VmThreadStateGuard {
    fn drop(&mut self) {
        runmat_vm::reset_thread_state_for_tests();
    }
}

fn run_with_stack<T>(f: impl FnOnce() -> T) -> Result<T, RuntimeError> {
    let _state_guard = VmThreadStateGuard::enter();
    Ok(stacker::grow(EXEC_STACK_BYTES, f))
}

pub fn compile_source(source: &str) -> Result<runmat_vm::Bytecode, RuntimeError> {
    let ast = runmat_parser::parse(source).map_err(|err| RuntimeError::new(err.to_string()))?;
    let hir = runmat_hir::lower(&ast, &LoweringContext::empty())
        .map_err(|err| RuntimeError::from(runmat_vm::CompileError::from(err)))?;
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
        .map_err(|err| RuntimeError::new(format!("{err:?}")))?;
    let entrypoint = hir.assembly.entrypoints[0].id;
    runmat_vm::compile(&hir.assembly, &mir, entrypoint).map_err(RuntimeError::from)
}

#[allow(dead_code)]
pub fn execute_source(source: &str) -> Result<Vec<Value>, RuntimeError> {
    let bc = compile_source(source)?;
    interpret(&bc)
}

#[allow(dead_code)]
pub fn interpret(bytecode: &runmat_vm::Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let bytecode = bytecode.clone();
    run_with_stack(move || block_on(runmat_vm::interpret(&bytecode)))?
}
