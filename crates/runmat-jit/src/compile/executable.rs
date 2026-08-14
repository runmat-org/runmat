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
    pub(crate) entrypoints: BTreeMap<ProgramFunctionId, NativeEntryPoint>,
    pub(crate) retained_code_bytes: u64,
}

impl CompiledExecutable {
    pub fn entrypoint(&self, function: ProgramFunctionId) -> JitResult<NativeEntryPoint> {
        self.entrypoints.get(&function).copied().ok_or_else(|| {
            JitError::Module(format!("compiled function {} is unavailable", function.0))
        })
    }

    pub fn retained_function_count(&self) -> usize {
        self.entrypoints.len()
    }

    /// Exact emitted function-body bytes retained by this executable. Module
    /// allocator metadata is intentionally excluded from the code budget.
    pub fn retained_code_bytes(&self) -> u64 {
        self.retained_code_bytes
    }
}
