use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use runmat_analysis_language::{
    geometry_operation_extension, output_operation_extension, AnalysisOperationCatalog,
    ExecutionProfileFacts, ExecutionProfileRequest, ExecutionProfileSource, FeaModuleSource,
    FunctionFacts, FunctionRequest, GeometryInspection, GeometryInspectionRequest,
    GeometryInspectionSource, LogicalSourcePath, ModuleRequest, OperationDiagnostic,
    OperationResolutionRequest, ResolvedOperationFacts, RunMatFunctionSource, SourceAvailability,
    SourceDiagnostic, SourceDigest, SourceModule,
};
use runmat_builtins::{compose_operation_catalog, ComposedOperationCatalog};
use sha2::{Digest, Sha256};

use super::geometry_resolver::inspect_geometry;

const LOGICAL_ROOT: &str = "project";

pub struct FilesystemFeaHost {
    physical_root: PathBuf,
    catalog: ComposedOperationCatalog,
    source_overlays: BTreeMap<LogicalSourcePath, Vec<u8>>,
}

impl FilesystemFeaHost {
    pub fn new(physical_root: &Path) -> Result<Self, String> {
        let physical_root = normalized_root(physical_root)?;
        let catalog = compose_operation_catalog(&[
            geometry_operation_extension(),
            output_operation_extension(),
        ])
        .map_err(|error| format!("cannot compose FEA operation catalog: {error}"))?;
        Ok(Self {
            physical_root,
            catalog,
            source_overlays: BTreeMap::new(),
        })
    }

    /// Replace the source bytes for one module while retaining filesystem resolution for its
    /// imports and referenced geometry. Editor hosts use this for unsaved buffers so diagnostics
    /// come from the production compiler rather than a second parser or a temporary file.
    pub fn with_source_overlay(
        mut self,
        physical_path: &Path,
        source: impl Into<Vec<u8>>,
    ) -> Result<Self, String> {
        let logical_path = self.logical_path(physical_path)?;
        self.source_overlays.insert(logical_path, source.into());
        Ok(self)
    }

    pub fn logical_root(&self) -> &'static str {
        LOGICAL_ROOT
    }

    pub fn load_physical_module(&self, path: &Path) -> Result<SourceModule, String> {
        let logical_path = self.logical_path(path)?;
        self.read_module(logical_path)
            .map_err(|diagnostic| diagnostic.to_string())
    }

    pub(super) fn logical_path(&self, path: &Path) -> Result<LogicalSourcePath, String> {
        let relative = path.strip_prefix(&self.physical_root).map_err(|_| {
            format!(
                "FEA source `{}` is outside source root `{}`",
                path.display(),
                self.physical_root.display()
            )
        })?;
        if relative.as_os_str().is_empty() {
            return Err("FEA source path must name a file below the source root".to_string());
        }
        let relative = logical_components(relative)?;
        LogicalSourcePath::new(format!("{LOGICAL_ROOT}/{relative}"))
            .map_err(|error| error.to_string())
    }

    fn physical_path(&self, logical_path: &LogicalSourcePath) -> Result<PathBuf, SourceDiagnostic> {
        let prefix = format!("{LOGICAL_ROOT}/");
        let relative = logical_path.as_str().strip_prefix(&prefix).ok_or_else(|| {
            diagnostic(
                "fea.host.path_outside_root",
                format!("logical source `{logical_path}` is outside `{LOGICAL_ROOT}`"),
                SourceAvailability::Available,
            )
        })?;
        if relative.is_empty()
            || relative
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(diagnostic(
                "fea.host.invalid_logical_path",
                format!("logical source `{logical_path}` is not normalized"),
                SourceAvailability::Available,
            ));
        }
        Ok(relative
            .split('/')
            .fold(self.physical_root.clone(), |path, part| path.join(part)))
    }

    fn read_module(
        &self,
        logical_path: LogicalSourcePath,
    ) -> Result<SourceModule, SourceDiagnostic> {
        if let Some(source) = self.source_overlays.get(&logical_path) {
            return Ok(SourceModule {
                logical_path,
                source_digest: source_digest(source),
                source: source.clone(),
            });
        }
        let path = self.physical_path(&logical_path)?;
        let source = runmat_filesystem::read(&path).map_err(|error| {
            diagnostic(
                "fea.host.module_unavailable",
                format!("cannot read FEA module {}: {error}", path.display()),
                SourceAvailability::Unavailable,
            )
        })?;
        Ok(SourceModule {
            logical_path,
            source_digest: source_digest(&source),
            source,
        })
    }
}

impl FeaModuleSource for FilesystemFeaHost {
    fn load(&self, request: &ModuleRequest) -> Result<SourceModule, SourceDiagnostic> {
        self.read_module(request.logical_path.clone())
    }
}

impl RunMatFunctionSource for FilesystemFeaHost {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic> {
        Err(diagnostic(
            "fea.host.function_analysis_unavailable",
            format!(
                "RunMat function `{}::{}` requires the project callable-analysis host",
                request.module_path, request.symbol
            ),
            SourceAvailability::Unavailable,
        ))
    }
}

impl GeometryInspectionSource for FilesystemFeaHost {
    fn inspect(
        &self,
        request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic> {
        let path = self.physical_path(&request.logical_path)?;
        inspect_geometry(&path, request)
    }
}

impl ExecutionProfileSource for FilesystemFeaHost {
    fn resolve(
        &self,
        request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        Err(diagnostic(
            "fea.host.execution_profile_unavailable",
            format!(
                "execution profile `{}` requires project execution-policy resolution",
                request.profile
            ),
            SourceAvailability::Unavailable,
        ))
    }
}

impl AnalysisOperationCatalog for FilesystemFeaHost {
    fn fingerprint(&self) -> [u8; 32] {
        self.catalog.fingerprint()
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        self.catalog
            .operation(&request.operation)
            .cloned()
            .map(|operation| ResolvedOperationFacts { operation })
            .ok_or_else(|| OperationDiagnostic {
                code: "fea.operation.unknown".to_string(),
                message: format!("unknown analysis operation `{}`", request.operation),
            })
    }
}

pub fn source_digest(bytes: &[u8]) -> SourceDigest {
    SourceDigest(Sha256::digest(bytes).into())
}

pub(super) fn diagnostic(
    code: &str,
    message: String,
    availability: SourceAvailability,
) -> SourceDiagnostic {
    SourceDiagnostic {
        code: code.to_string(),
        message,
        availability,
    }
}

fn normalized_root(path: &Path) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err("FEA source root must not be empty".to_string());
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "FEA source root `{}` must be lexically normalized",
            path.display()
        ));
    }
    Ok(path.to_path_buf())
}

fn logical_components(path: &Path) -> Result<String, String> {
    path.components()
        .map(|component| match component {
            Component::Normal(part) => part
                .to_str()
                .map(str::to_string)
                .ok_or_else(|| "FEA source paths must be valid UTF-8".to_string()),
            _ => Err(format!(
                "FEA source path `{}` must be relative and normalized",
                path.display()
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("/"))
}
