use std::rc::Rc;

use super::{CoveragePlan, ExecutableRevision, ExecutableSource, ExecutableSourceMap};

/// Immutable canonical compilation product reusable across invocations.
#[derive(Clone, Debug)]
pub struct ExecutableUnit {
    source: ExecutableSource,
    revision: ExecutableRevision,
    source_map: ExecutableSourceMap,
    coverage: CoveragePlan,
    mir: Rc<runmat_mir::MirAssembly>,
    analysis: Rc<runmat_mir::analysis::AnalysisStore>,
    bytecode: Rc<runmat_vm::Bytecode>,
    layout: Rc<runmat_vm::VmAssemblyLayout>,
    functions: Rc<runmat_vm::FunctionRegistry>,
}

impl ExecutableUnit {
    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        mir: runmat_mir::MirAssembly,
        analysis: runmat_mir::analysis::AnalysisStore,
        mut bytecode: runmat_vm::Bytecode,
    ) -> Result<Self, String> {
        let coverage = CoveragePlan::instrument(&source, &revision, &source_map, &mut bytecode);
        let layout = bytecode
            .layout
            .clone()
            .ok_or_else(|| "canonical bytecode is missing its VM assembly layout".to_string())?;
        let functions = Rc::new(bytecode.function_registry.clone());
        Ok(Self {
            source,
            revision,
            source_map,
            coverage,
            mir: Rc::new(mir),
            analysis: Rc::new(analysis),
            bytecode: Rc::new(bytecode),
            layout: Rc::new(layout),
            functions,
        })
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

    /// Canonical analyzed MIR retained for native lowering and artifact production.
    pub fn mir(&self) -> &runmat_mir::MirAssembly {
        &self.mir
    }

    /// Exact VM layout shared by interpreter state materialization and native frames.
    pub fn vm_layout(&self) -> &runmat_vm::VmAssemblyLayout {
        &self.layout
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
