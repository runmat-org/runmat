use std::rc::Rc;

use super::{CoveragePlan, ExecutableRevision, ExecutableSource, ExecutableSourceMap};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableExecutableKind {
    Function,
    Script,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortableExecutable {
    pub kind: PortableExecutableKind,
    pub entrypoint: String,
    pub function: usize,
    pub bytes: Vec<u8>,
}

/// Immutable canonical compilation product reusable across invocations.
#[derive(Clone, Debug)]
pub struct ExecutableUnit {
    source: ExecutableSource,
    revision: ExecutableRevision,
    source_map: ExecutableSourceMap,
    coverage: CoveragePlan,
    analysis: Rc<runmat_mir::analysis::AnalysisStore>,
    bytecode: Rc<runmat_vm::Bytecode>,
    functions: Rc<runmat_vm::FunctionRegistry>,
}

impl ExecutableUnit {
    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        analysis: runmat_mir::analysis::AnalysisStore,
        mut bytecode: runmat_vm::Bytecode,
    ) -> Self {
        let coverage = CoveragePlan::instrument(&source, &revision, &source_map, &mut bytecode);
        let functions = Rc::new(bytecode.function_registry.clone());
        Self {
            source,
            revision,
            source_map,
            coverage,
            analysis: Rc::new(analysis),
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

    /// Exact immutable semantic facts produced for this executable revision.
    pub fn analysis(&self) -> &runmat_mir::analysis::AnalysisStore {
        &self.analysis
    }

    pub fn procedure_names(&self) -> Vec<String> {
        let mut names = self.functions.names.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Materialize the exact interpreter product used by native, browser, and
    /// remote execution. Function sources preserve their semantic function
    /// registry; scripts preserve their top-level bytecode.
    pub fn portable_executable(
        &self,
        preferred_function: Option<&str>,
    ) -> Result<PortableExecutable, String> {
        let preferred_function = preferred_function.or_else(|| {
            std::path::Path::new(&self.source.relative_path)
                .file_stem()
                .and_then(|value| value.to_str())
        });
        if let Some(function) =
            preferred_function.and_then(|name| self.functions.resolve_name(name))
        {
            return Ok(PortableExecutable {
                kind: PortableExecutableKind::Function,
                entrypoint: function.0.to_string(),
                function: function.0,
                bytes: serde_json::to_vec(self.functions.as_ref())
                    .map_err(|error| error.to_string())?,
            });
        }
        if !self.bytecode.instructions.is_empty() {
            return Ok(PortableExecutable {
                kind: PortableExecutableKind::Script,
                entrypoint: "script".into(),
                function: 0,
                bytes: serde_json::to_vec(self.bytecode.as_ref())
                    .map_err(|error| error.to_string())?,
            });
        }
        Err(format!(
            "source does not define the requested function{}",
            preferred_function
                .map(|name| format!(" '{name}'"))
                .unwrap_or_default()
        ))
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
