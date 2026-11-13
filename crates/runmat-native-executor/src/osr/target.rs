use std::rc::Rc;

use runmat_types::{FunctionId, ProgramPointId};

use crate::{NativeExecutor, NativeExecutorResult};

/// Session-admitted optimized code plus the exact loop header where it may
/// replace an active generic-native frame.
///
/// Retaining the executor here keeps its executable memory alive for the
/// complete invocation, even if the publication is retired concurrently.
#[derive(Clone)]
pub struct OsrTarget {
    executor: Rc<NativeExecutor>,
    function: FunctionId,
    point: ProgramPointId,
}

impl OsrTarget {
    pub fn new(
        executor: Rc<NativeExecutor>,
        function: FunctionId,
        point: ProgramPointId,
    ) -> Result<Self, &'static str> {
        if executor
            .function_metadata(function)
            .is_none_or(|body| point.function != body.portable_id)
        {
            return Err("OSR loop header must belong to the target function");
        }
        Ok(Self {
            executor,
            function,
            point,
        })
    }

    pub fn point(&self) -> ProgramPointId {
        self.point
    }

    pub(crate) fn entrypoint(
        &self,
    ) -> NativeExecutorResult<runmat_runtime::native::NativeEntryPoint> {
        self.executor.compiled_entrypoint(self.function)
    }

    pub(crate) fn executor(&self) -> &NativeExecutor {
        &self.executor
    }
}
