use runmat_analysis_language::*;
use runmat_types::{CapabilitySet, EffectSet};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct TestHost {
    modules: BTreeMap<String, SourceModule>,
    geometries: BTreeMap<String, GeometryInspection>,
}

impl TestHost {
    pub fn with_module(mut self, path: &str, source: &str, digest: u8) -> Self {
        self.modules
            .insert(path.to_string(), source_module(path, source, digest));
        self
    }

    pub fn with_geometry(mut self, path: &str, digest: u8) -> Self {
        self.geometries.insert(
            path.to_string(),
            GeometryInspection {
                content_digest: SourceDigest([digest; 32]),
                topology_digest: SourceDigest([digest.wrapping_add(1); 32]),
                capabilities: vec![
                    GeometryInspectionCapability::Topology,
                    GeometryInspectionCapability::PersistentIds,
                ],
            },
        );
        self
    }
}

impl FeaModuleSource for TestHost {
    fn load(&self, request: &ModuleRequest) -> Result<SourceModule, SourceDiagnostic> {
        self.modules
            .get(request.logical_path.as_str())
            .cloned()
            .ok_or_else(|| SourceDiagnostic {
                code: "fea.test.module_missing".to_string(),
                message: format!("missing `{}`", request.logical_path),
                availability: SourceAvailability::Available,
            })
    }
}

impl RunMatFunctionSource for TestHost {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic> {
        Ok(FunctionFacts {
            module_path: request.module_path.clone(),
            symbol: request.symbol.clone(),
            source_digest: SourceDigest([9; 32]),
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
        })
    }
}

impl GeometryInspectionSource for TestHost {
    fn inspect(
        &self,
        request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic> {
        self.geometries
            .get(request.logical_path.as_str())
            .cloned()
            .ok_or_else(|| SourceDiagnostic {
                code: "fea.test.geometry_unavailable".to_string(),
                message: "geometry inspection is unavailable in this test host".to_string(),
                availability: SourceAvailability::Unavailable,
            })
    }
}

impl ExecutionProfileSource for TestHost {
    fn resolve(
        &self,
        _request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        unreachable!("semantic-only tests do not resolve execution profiles")
    }
}

impl AnalysisOperationCatalog for TestHost {
    fn fingerprint(&self) -> [u8; 32] {
        [7; 32]
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        Err(OperationDiagnostic {
            code: "fea.test.operation_missing".to_string(),
            message: format!("missing `{}`", request.operation),
        })
    }
}

pub fn source_module(path: &str, source: &str, digest: u8) -> SourceModule {
    SourceModule {
        logical_path: LogicalSourcePath::new(path).unwrap(),
        source_digest: SourceDigest([digest; 32]),
        source: source.as_bytes().to_vec(),
    }
}
