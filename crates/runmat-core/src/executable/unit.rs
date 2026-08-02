use std::rc::Rc;

use super::{CoveragePlan, ExecutableRevision, ExecutableSource, ExecutableSourceMap};

/// Immutable canonical compilation product reusable across invocations.
#[derive(Clone, Debug)]
pub struct ExecutableUnit {
    source: ExecutableSource,
    revision: ExecutableRevision,
    source_map: ExecutableSourceMap,
    coverage: CoveragePlan,
    bytecode: Rc<runmat_vm::Bytecode>,
    functions: Rc<runmat_vm::FunctionRegistry>,
}

impl ExecutableUnit {
    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        bytecode: runmat_vm::Bytecode,
    ) -> Self {
        let functions = Rc::new(bytecode.function_registry.clone());
        Self {
            source,
            revision,
            source_map,
            coverage: CoveragePlan::default(),
            bytecode: Rc::new(bytecode),
            functions,
        }
    }

    pub fn source(&self) -> &ExecutableSource {
        &self.source
    }

    pub fn revision(&self) -> &ExecutableRevision {
        &self.revision
    }

    pub fn source_map(&self) -> &ExecutableSourceMap {
        &self.source_map
    }

    pub fn coverage_plan(&self) -> &CoveragePlan {
        &self.coverage
    }

    pub fn procedure_names(&self) -> Vec<String> {
        let mut names = self.functions.names.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub(crate) fn procedure_input_count(&self, name: &str) -> Option<usize> {
        self.functions
            .resolve_name(name)
            .and_then(|id| self.functions.get(id))
            .map(|function| function.input_slots.len())
    }

    pub(crate) fn bytecode(&self) -> &runmat_vm::Bytecode {
        &self.bytecode
    }

    pub(crate) fn functions(&self) -> &runmat_vm::FunctionRegistry {
        &self.functions
    }
}
