use std::collections::BTreeMap;
use std::path::Path;

use runmat_analysis_language::{
    AnalysisOperationCatalog, ExecutionProfileFacts, ExecutionProfileIdentity,
    ExecutionProfileRequest, ExecutionProfileSource, FeaModuleSource, FunctionFacts,
    FunctionRequest, GeometryInspection, GeometryInspectionRequest, GeometryInspectionSource,
    LogicalSourcePath, ModuleRequest, OperationDiagnostic, OperationResolutionRequest,
    ResolvedOperationFacts, RunMatFunctionSource, SourceAvailability, SourceDiagnostic,
    SourceModule, SourceSymbol,
};

use super::host::{diagnostic, FilesystemFeaHost};

/// Compiler host for one frozen project plus its immutable analyzed external facts.
///
/// Filesystem access remains confined to [`FilesystemFeaHost`]. This wrapper adds only facts
/// derived from the same frozen project revision; it never performs callback execution or owns a
/// second source graph.
pub struct ProjectFeaHost {
    files: FilesystemFeaHost,
    functions: BTreeMap<(LogicalSourcePath, SourceSymbol), FunctionFacts>,
    profiles: BTreeMap<ExecutionProfileIdentity, ExecutionProfileFacts>,
}

impl ProjectFeaHost {
    pub fn new(
        physical_root: &Path,
        functions: impl IntoIterator<Item = FunctionFacts>,
        profiles: impl IntoIterator<Item = ExecutionProfileFacts>,
    ) -> Result<Self, String> {
        let mut host = Self {
            files: FilesystemFeaHost::new(physical_root)?,
            functions: BTreeMap::new(),
            profiles: BTreeMap::new(),
        };
        for facts in functions {
            let key = (facts.module_path.clone(), facts.symbol.clone());
            if host.functions.insert(key, facts).is_some() {
                return Err("project FEA function facts contain a duplicate identity".to_string());
            }
        }
        for facts in profiles {
            if host.profiles.insert(facts.profile.clone(), facts).is_some() {
                return Err("project FEA execution profiles contain a duplicate identity".into());
            }
        }
        Ok(host)
    }
}

impl FeaModuleSource for ProjectFeaHost {
    fn load(&self, request: &ModuleRequest) -> Result<SourceModule, SourceDiagnostic> {
        self.files.load(request)
    }
}

impl RunMatFunctionSource for ProjectFeaHost {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic> {
        self.functions
            .get(&(request.module_path.clone(), request.symbol.clone()))
            .cloned()
            .ok_or_else(|| {
                diagnostic(
                    "fea.host.function_analysis_unavailable",
                    format!(
                        "RunMat function `{}::{}` is not present in the frozen project facts",
                        request.module_path, request.symbol
                    ),
                    SourceAvailability::Unavailable,
                )
            })
    }
}

impl GeometryInspectionSource for ProjectFeaHost {
    fn inspect(
        &self,
        request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic> {
        self.files.inspect(request)
    }
}

impl ExecutionProfileSource for ProjectFeaHost {
    fn resolve(
        &self,
        request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        self.profiles.get(&request.profile).cloned().ok_or_else(|| {
            diagnostic(
                "fea.host.execution_profile_unavailable",
                format!(
                    "execution profile `{}` is not present in the frozen project facts",
                    request.profile
                ),
                SourceAvailability::Unavailable,
            )
        })
    }
}

impl AnalysisOperationCatalog for ProjectFeaHost {
    fn fingerprint(&self) -> [u8; 32] {
        self.files.fingerprint()
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        AnalysisOperationCatalog::resolve(&self.files, request)
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_language::{
        ExecutionPlacement, FunctionRequest, RunMatFunctionSource, SourceDigest,
    };
    use runmat_types::{CapabilityRequirement, CapabilitySet, EffectSet};

    use super::*;

    #[test]
    fn project_host_resolves_only_exact_frozen_facts() {
        let temporary = tempfile::tempdir().unwrap();
        let module_path = LogicalSourcePath::new("design_curve.m").unwrap();
        let symbol = SourceSymbol::new("designCurve").unwrap();
        let function = FunctionFacts {
            module_path: module_path.clone(),
            symbol: symbol.clone(),
            source_digest: SourceDigest([7; 32]),
            declaration_span: runmat_types::Span::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
            cooperative_suspension: false,
            capabilities: CapabilitySet::default(),
        };
        let profile_id = ExecutionProfileIdentity::new("engineering").unwrap();
        let profile = ExecutionProfileFacts {
            profile: profile_id.clone(),
            placement: ExecutionPlacement::Remote,
            capabilities: CapabilitySet([CapabilityRequirement::DistributedRuntime].into()),
        };
        let host =
            ProjectFeaHost::new(temporary.path(), [function.clone()], [profile.clone()]).unwrap();

        assert_eq!(
            RunMatFunctionSource::resolve(
                &host,
                &FunctionRequest {
                    module_path,
                    symbol,
                },
            )
            .unwrap(),
            function
        );
        assert_eq!(
            ExecutionProfileSource::resolve(
                &host,
                &ExecutionProfileRequest {
                    profile: profile_id,
                },
            )
            .unwrap(),
            profile
        );
        assert!(RunMatFunctionSource::resolve(
            &host,
            &FunctionRequest {
                module_path: LogicalSourcePath::new("other.m").unwrap(),
                symbol: SourceSymbol::new("other").unwrap(),
            },
        )
        .is_err());
    }
}
