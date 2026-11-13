use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_language::{
    bind_program, check_program, logical_path_is_within, AnalysisOperationCatalog,
    ExecutionProfileFacts, ExecutionProfileIdentity, ExecutionProfileRequest,
    ExecutionProfileSource, FeaCompiler, FeaModuleGraph, FeaModuleSource, FunctionFacts,
    FunctionRequest, GeometryInspection, GeometryInspectionRequest, GeometryInspectionSource,
    LogicalSourcePath, ModuleGraphRequest, ModuleRequest, OperationDiagnostic,
    OperationResolutionRequest, ResolvedOperationFacts, RunMatFunctionSource, SourceAvailability,
    SourceDiagnostic, SourceModule, SourceSymbol,
};
use runmat_builtins::ComposedOperationCatalog;

use super::geometry_resolver::inspect_geometry_bytes;
use super::host::{diagnostic, fea_operation_catalog, resolve_operation};
use super::{source_digest, CheckedFeaSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaSourceSnapshotRequests {
    pub geometry: Vec<GeometryInspectionRequest>,
    pub execution_profiles: Vec<ExecutionProfileRequest>,
}

pub struct FeaSourceSnapshotBuilder {
    physical_root: PathBuf,
    logical_root: LogicalSourcePath,
    root: LogicalSourcePath,
    modules: BTreeMap<LogicalSourcePath, SourceModule>,
}

impl FeaSourceSnapshotBuilder {
    pub fn new(physical_root: impl Into<PathBuf>, root: SourceModule) -> Result<Self, String> {
        let logical_root = LogicalSourcePath::root();
        if !logical_path_is_within(&logical_root, &root.logical_path) {
            return Err(format!(
                "root FEA module `{}` is outside logical source root `{logical_root}`",
                root.logical_path
            ));
        }
        let root_path = root.logical_path.clone();
        let modules = BTreeMap::from([(root_path.clone(), root)]);
        Ok(Self {
            physical_root: physical_root.into(),
            logical_root,
            root: root_path,
            modules,
        })
    }

    pub fn source_module(
        logical_path: LogicalSourcePath,
        source: impl Into<Vec<u8>>,
    ) -> SourceModule {
        let source = source.into();
        SourceModule {
            logical_path,
            source_digest: source_digest(&source),
            source,
        }
    }

    pub fn add_module(&mut self, module: SourceModule) -> Result<(), String> {
        if !logical_path_is_within(&self.logical_root, &module.logical_path) {
            return Err(format!(
                "FEA module `{}` is outside logical source root `{}`",
                module.logical_path, self.logical_root
            ));
        }
        if let Some(existing) = self.modules.get(&module.logical_path) {
            if existing != &module {
                return Err(format!(
                    "FEA source snapshot contains conflicting bytes for `{}`",
                    module.logical_path
                ));
            }
            return Ok(());
        }
        self.modules.insert(module.logical_path.clone(), module);
        Ok(())
    }

    pub fn missing_module_paths(&self) -> Result<Vec<LogicalSourcePath>, String> {
        let geometry = BTreeMap::new();
        let host = SnapshotFeaHost::new(&self.modules, &geometry, Vec::new(), Vec::new())?;
        let _ = self.build_graph(&host)?;
        Ok(host.take_requests().modules)
    }

    pub fn prepare(self) -> Result<PreparedFeaSourceSnapshot, String> {
        let geometry = BTreeMap::new();
        let host = SnapshotFeaHost::new(&self.modules, &geometry, Vec::new(), Vec::new())?;
        let graph = self.build_graph(&host)?;
        Ok(PreparedFeaSourceSnapshot {
            physical_root: self.physical_root,
            graph,
            modules: self.modules,
            geometry: BTreeMap::new(),
        })
    }

    fn build_graph(&self, host: &SnapshotFeaHost<'_>) -> Result<FeaModuleGraph, String> {
        let root =
            self.modules.get(&self.root).cloned().ok_or_else(|| {
                format!("FEA source snapshot is missing root module `{}`", self.root)
            })?;
        Ok(
            FeaCompiler::new(host).build_module_graph(ModuleGraphRequest {
                source_root: self.logical_root.clone(),
                root,
            }),
        )
    }
}

pub struct PreparedFeaSourceSnapshot {
    physical_root: PathBuf,
    graph: FeaModuleGraph,
    modules: BTreeMap<LogicalSourcePath, SourceModule>,
    geometry: BTreeMap<LogicalSourcePath, Vec<u8>>,
}

impl PreparedFeaSourceSnapshot {
    pub fn source_root(&self) -> &Path {
        &self.physical_root
    }

    pub fn root_module(&self) -> &LogicalSourcePath {
        &self.graph.root
    }

    pub fn function_requests(&self) -> Result<Vec<FunctionRequest>, String> {
        self.graph
            .function_requests()
            .map_err(|error| error.to_string())
    }

    /// Includes unavailable import targets so removing a dependency still
    /// invalidates its consumers. Function imports use the compiler's resolver.
    pub fn source_dependencies(
        &self,
    ) -> Result<std::collections::BTreeSet<LogicalSourcePath>, String> {
        self.graph
            .source_dependencies()
            .map_err(|error| error.to_string())
    }

    pub fn requests(
        &self,
        functions: impl IntoIterator<Item = FunctionFacts>,
        profiles: impl IntoIterator<Item = ExecutionProfileFacts>,
    ) -> Result<FeaSourceSnapshotRequests, String> {
        let host = SnapshotFeaHost::new(&self.modules, &self.geometry, functions, profiles)?;
        let bound = bind_program(&host, self.graph.clone());
        let _ = check_program(&host, bound);
        let requests = host.take_requests();
        Ok(FeaSourceSnapshotRequests {
            geometry: requests.geometry,
            execution_profiles: requests.execution_profiles,
        })
    }

    pub fn add_geometry(
        &mut self,
        logical_path: LogicalSourcePath,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), String> {
        let logical_root = LogicalSourcePath::root();
        if !logical_path_is_within(&logical_root, &logical_path) {
            return Err(format!(
                "geometry input `{logical_path}` is outside logical source root `{logical_root}`"
            ));
        }
        let bytes = bytes.into();
        if let Some(existing) = self.geometry.get(&logical_path) {
            if existing != &bytes {
                return Err(format!(
                    "FEA source snapshot contains conflicting geometry bytes for `{logical_path}`"
                ));
            }
            return Ok(());
        }
        self.geometry.insert(logical_path, bytes);
        Ok(())
    }

    pub fn check(
        self,
        functions: impl IntoIterator<Item = FunctionFacts>,
        profiles: impl IntoIterator<Item = ExecutionProfileFacts>,
    ) -> Result<CheckedFeaSource, String> {
        let mut profiles = profiles.into_iter().collect::<Vec<_>>();
        profiles.sort_by(|left, right| left.profile.cmp(&right.profile));
        let host =
            SnapshotFeaHost::new(&self.modules, &self.geometry, functions, profiles.clone())?;
        if self.graph.catalog_fingerprint != host.fingerprint() {
            return Err(
                "FEA operation catalog changed between source preparation and checking".into(),
            );
        }
        let root_module = self.graph.root.clone();
        let bound = bind_program(&host, self.graph);
        let mut program = check_program(&host, bound);
        super::admission::append_admission_diagnostics(&mut program, &root_module);
        Ok(CheckedFeaSource {
            program,
            root_module,
            source_root: self.physical_root,
            execution_profile_facts: profiles,
        })
    }
}

#[derive(Default)]
struct SnapshotRequestLog {
    modules: Vec<LogicalSourcePath>,
    geometry: Vec<GeometryInspectionRequest>,
    execution_profiles: Vec<ExecutionProfileRequest>,
}

struct SnapshotFeaHost<'a> {
    modules: &'a BTreeMap<LogicalSourcePath, SourceModule>,
    geometry: &'a BTreeMap<LogicalSourcePath, Vec<u8>>,
    functions: BTreeMap<(LogicalSourcePath, SourceSymbol), FunctionFacts>,
    profiles: BTreeMap<ExecutionProfileIdentity, ExecutionProfileFacts>,
    catalog: ComposedOperationCatalog,
    requests: RefCell<SnapshotRequestLog>,
}

impl<'a> SnapshotFeaHost<'a> {
    fn new(
        modules: &'a BTreeMap<LogicalSourcePath, SourceModule>,
        geometry: &'a BTreeMap<LogicalSourcePath, Vec<u8>>,
        functions: impl IntoIterator<Item = FunctionFacts>,
        profiles: impl IntoIterator<Item = ExecutionProfileFacts>,
    ) -> Result<Self, String> {
        let mut function_facts = BTreeMap::new();
        for facts in functions {
            let identity = (facts.module_path.clone(), facts.symbol.clone());
            if function_facts.insert(identity, facts).is_some() {
                return Err("FEA source snapshot contains duplicate callable facts".into());
            }
        }
        let mut profile_facts = BTreeMap::new();
        for facts in profiles {
            if profile_facts.insert(facts.profile.clone(), facts).is_some() {
                return Err(
                    "FEA source snapshot contains duplicate execution-profile facts".into(),
                );
            }
        }
        Ok(Self {
            modules,
            geometry,
            functions: function_facts,
            profiles: profile_facts,
            catalog: fea_operation_catalog()?,
            requests: RefCell::new(SnapshotRequestLog::default()),
        })
    }

    fn take_requests(&self) -> SnapshotRequestLog {
        std::mem::take(&mut *self.requests.borrow_mut())
    }
}

impl FeaModuleSource for SnapshotFeaHost<'_> {
    fn load(&self, request: &ModuleRequest) -> Result<SourceModule, SourceDiagnostic> {
        self.modules
            .get(&request.logical_path)
            .cloned()
            .ok_or_else(|| {
                push_unique(
                    &mut self.requests.borrow_mut().modules,
                    request.logical_path.clone(),
                );
                diagnostic(
                    "fea.host.module_unavailable",
                    format!(
                        "FEA module `{}` is absent from the immutable source snapshot",
                        request.logical_path
                    ),
                    SourceAvailability::Unavailable,
                )
            })
    }
}

impl RunMatFunctionSource for SnapshotFeaHost<'_> {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic> {
        self.functions
            .get(&(request.module_path.clone(), request.symbol.clone()))
            .cloned()
            .ok_or_else(|| {
                diagnostic(
                    "fea.host.function_analysis_unavailable",
                    format!(
                        "RunMat function `{}::{}` is absent from the immutable project facts",
                        request.module_path, request.symbol
                    ),
                    SourceAvailability::Unavailable,
                )
            })
    }
}

impl GeometryInspectionSource for SnapshotFeaHost<'_> {
    fn inspect(
        &self,
        request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic> {
        let Some(bytes) = self.geometry.get(&request.logical_path) else {
            push_unique(&mut self.requests.borrow_mut().geometry, request.clone());
            return Err(diagnostic(
                "fea.host.geometry_unavailable",
                format!(
                    "geometry `{}` is absent from the immutable source snapshot",
                    request.logical_path
                ),
                SourceAvailability::Unavailable,
            ));
        };
        inspect_geometry_bytes(bytes, request)
    }
}

impl ExecutionProfileSource for SnapshotFeaHost<'_> {
    fn resolve(
        &self,
        request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        self.profiles.get(&request.profile).cloned().ok_or_else(|| {
            push_unique(
                &mut self.requests.borrow_mut().execution_profiles,
                request.clone(),
            );
            diagnostic(
                "fea.host.execution_profile_unavailable",
                format!(
                    "execution profile `{}` is absent from the immutable project facts",
                    request.profile
                ),
                SourceAvailability::Unavailable,
            )
        })
    }
}

impl AnalysisOperationCatalog for SnapshotFeaHost<'_> {
    fn fingerprint(&self) -> [u8; 32] {
        self.catalog.fingerprint()
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        resolve_operation(&self.catalog, request)
    }
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}
