use std::{any::Any, collections::BTreeMap};

use runmat_runtime::native::NativeEntryPoint;
use runmat_types::ProgramFunctionId;

use crate::{NativeExecutorError, NativeExecutorResult};

/// Entrypoints plus the owner that keeps their executable memory alive.
pub struct NativeExecutable {
    owner: Option<Box<dyn Any + Send>>,
    entrypoints: BTreeMap<ProgramFunctionId, NativeEntryPoint>,
    retained_code_bytes: u64,
}

impl NativeExecutable {
    /// Construct an executable backed by dynamically allocated code memory.
    pub fn owned(
        entrypoints: BTreeMap<ProgramFunctionId, NativeEntryPoint>,
        retained_code_bytes: u64,
        owner: impl Any + Send,
    ) -> NativeExecutorResult<Self> {
        Self::new(entrypoints, retained_code_bytes, Some(Box::new(owner)))
    }

    /// Construct an executable whose entrypoints are retained by the process image.
    pub fn linked(
        entrypoints: BTreeMap<ProgramFunctionId, NativeEntryPoint>,
    ) -> NativeExecutorResult<Self> {
        Self::new(entrypoints, 0, None)
    }

    fn new(
        entrypoints: BTreeMap<ProgramFunctionId, NativeEntryPoint>,
        retained_code_bytes: u64,
        owner: Option<Box<dyn Any + Send>>,
    ) -> NativeExecutorResult<Self> {
        if entrypoints.is_empty() {
            return Err(NativeExecutorError::Executable(
                "native executable has no entrypoints".into(),
            ));
        }
        Ok(Self {
            owner,
            entrypoints,
            retained_code_bytes,
        })
    }

    pub fn entrypoint(
        &self,
        function: ProgramFunctionId,
    ) -> NativeExecutorResult<NativeEntryPoint> {
        self.entrypoints.get(&function).copied().ok_or_else(|| {
            NativeExecutorError::Executable(format!(
                "compiled function {} is unavailable",
                function.0
            ))
        })
    }

    pub fn retained_function_count(&self) -> usize {
        self.entrypoints.len()
    }

    pub(crate) fn function_ids(&self) -> impl Iterator<Item = ProgramFunctionId> + '_ {
        self.entrypoints.keys().copied()
    }

    pub fn retained_code_bytes(&self) -> u64 {
        self.retained_code_bytes
    }

    pub fn owns_executable_memory(&self) -> bool {
        self.owner.is_some()
    }
}
