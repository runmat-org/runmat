use std::collections::BTreeMap;

use cranelift_jit::JITModule;
use runmat_runtime::native::NativeEntryPoint;
use runmat_types::ProgramFunctionId;

use crate::{JitError, JitResult};

/// Finalized code plus the module that owns its executable memory.
pub struct CompiledExecutable {
    // Retained for exactly as long as any entry point may be called. This is
    // deliberately not exposed: executable-memory ownership is a JIT concern,
    // not part of the executor or native ABI.
    pub(crate) _module: JITModule,
    pub(crate) entrypoints: BTreeMap<ProgramFunctionId, *const u8>,
}

impl CompiledExecutable {
    pub fn entrypoint(&self, function: ProgramFunctionId) -> JitResult<NativeEntryPoint> {
        let address = *self.entrypoints.get(&function).ok_or_else(|| {
            JitError::Module(format!("compiled function {} is unavailable", function.0))
        })?;
        // SAFETY: GenericCompiler defines every exported function with the
        // runtime-owned NativeEntryPoint signature and retains its JITModule.
        Ok(unsafe { std::mem::transmute::<*const u8, NativeEntryPoint>(address) })
    }

    pub fn retained_function_count(&self) -> usize {
        self.entrypoints.len()
    }
}
