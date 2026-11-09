use runmat_analysis_language::*;
use runmat_types::{CapabilitySet, EffectSet};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct TestHost {
    modules: BTreeMap<String, SourceModule>,
    geometries: BTreeMap<String, GeometryInspection>,
    functions: BTreeMap<(String, String), FunctionFacts>,
    pub profiles: BTreeMap<String, ExecutionProfileFacts>,
}

impl TestHost {
    pub fn with_module(mut self, path: &str, source: &str, digest: u8) -> Self {
        self.modules
            .insert(path.to_string(), source_module(path, source, digest));
        self
    }

    #[allow(dead_code)]
    pub fn with_geometry(mut self, path: &str, digest: u8) -> Self {
        self = self.with_geometry_inspection(
            path,
            GeometryInspection {
                preview_extent_m: None,
                unit_evidence: Default::default(),
                content_digest: SourceDigest([digest; 32]),
                topology_digest: SourceDigest([digest.wrapping_add(1); 32]),
                capabilities: vec![
                    GeometryInspectionCapability::Topology,
                    GeometryInspectionCapability::PersistentIds,
                ],
                exchange: None,
                resolved_units: None,
                authored_names: Default::default(),
            },
        );
        self
    }

    /// A geometry whose exact inspection carries authored `(role, name)` pairs.
    #[allow(dead_code)]
    pub fn with_named_geometry(self, path: &str, digest: u8, names: &[(&str, &str)]) -> Self {
        let mut authored_names = BTreeMap::<String, Vec<String>>::new();
        for (role, name) in names {
            authored_names
                .entry((*role).to_string())
                .or_default()
                .push((*name).to_string());
        }
        self.with_geometry_inspection(
            path,
            GeometryInspection {
                preview_extent_m: None,
                content_digest: SourceDigest([digest; 32]),
                topology_digest: SourceDigest([digest.wrapping_add(1); 32]),
                capabilities: vec![
                    GeometryInspectionCapability::Topology,
                    GeometryInspectionCapability::PersistentIds,
                ],
                exchange: None,
                resolved_units: None,
                unit_evidence: Default::default(),
                authored_names,
            },
        )
    }

    pub fn with_geometry_inspection(mut self, path: &str, inspection: GeometryInspection) -> Self {
        self.geometries.insert(path.to_string(), inspection);
        self
    }

    pub fn with_function(
        self,
        path: &str,
        symbol: &str,
        inputs: Vec<runmat_types::OperationParameter>,
        outputs: Vec<runmat_types::OperationTypePattern>,
    ) -> Self {
        self.with_effectful_function(path, symbol, inputs, outputs, EffectSet::default())
    }

    pub fn with_effectful_function(
        mut self,
        path: &str,
        symbol: &str,
        inputs: Vec<runmat_types::OperationParameter>,
        outputs: Vec<runmat_types::OperationTypePattern>,
        effects: EffectSet,
    ) -> Self {
        self.functions.insert(
            (path.to_string(), symbol.to_string()),
            FunctionFacts {
                module_path: LogicalSourcePath::new(path).unwrap(),
                symbol: SourceSymbol::new(symbol).unwrap(),
                source_digest: SourceDigest([9; 32]),
                declaration_span: runmat_types::Span::default(),
                inputs,
                outputs,
                effects,
                cooperative_suspension: false,
                capabilities: CapabilitySet::default(),
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
        if let Some(facts) = self.functions.get(&(
            request.module_path.as_str().to_string(),
            request.symbol.as_str().to_string(),
        )) {
            return Ok(facts.clone());
        }
        Ok(FunctionFacts {
            module_path: request.module_path.clone(),
            symbol: request.symbol.clone(),
            source_digest: SourceDigest([9; 32]),
            declaration_span: runmat_types::Span::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
            cooperative_suspension: false,
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
        request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        self.profiles
            .get(request.profile.as_str())
            .cloned()
            .ok_or_else(|| SourceDiagnostic {
                code: "fea.test.execution_profile_missing".to_string(),
                message: format!("missing execution profile `{}`", request.profile),
                availability: SourceAvailability::Available,
            })
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
        if let Some(operation) = geometry_operation_extension()
            .operations
            .into_iter()
            .chain(output_operation_extension().operations)
            .find(|operation| operation.identity == request.operation)
        {
            return Ok(ResolvedOperationFacts { operation });
        }
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
