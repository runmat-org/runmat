use runmat_analysis_language::*;
use runmat_types::{CapabilitySet, EffectSet, OperationParameter, OperationTypePattern};
use std::collections::BTreeMap;

#[derive(Default)]
struct TestHost {
    modules: BTreeMap<String, SourceModule>,
}

impl TestHost {
    fn with_module(mut self, path: &str, source: &str, digest: u8) -> Self {
        self.modules
            .insert(path.to_string(), source_module(path, source, digest));
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
            inputs: Vec::<OperationParameter>::new(),
            outputs: Vec::<OperationTypePattern>::new(),
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
        })
    }
}

impl GeometryInspectionSource for TestHost {
    fn inspect(
        &self,
        _request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic> {
        unreachable!("binding does not inspect geometry")
    }
}

impl ExecutionProfileSource for TestHost {
    fn resolve(
        &self,
        _request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic> {
        unreachable!("binding does not resolve execution profiles")
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

#[test]
fn compiler_loads_transitive_modules_and_binds_explicit_symbols() {
    let host = TestHost::default()
        .with_module(
            "project/models/bracket.fea",
            "import material steel from \"../materials/steel.fea\"\nmodel bracket {\n}\n",
            2,
        )
        .with_module(
            "project/materials/steel.fea",
            "material steel {\nname = \"steel\"\n}\n",
            3,
        );
    let root = source_module(
        "project/main.fea",
        "import model bracket from \"./models/bracket.fea\"\nimport function pulse from \"./pulse.m\"\nstudy main {\n}\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.graph.modules.len(), 3);
    assert_eq!(program.graph.edges.len(), 2);
    assert_eq!(program.graph.catalog_fingerprint, [7; 32]);
    let root_symbols = &program.symbols[&LogicalSourcePath::new("project/main.fea").unwrap()];
    assert!(root_symbols.contains_key(&FeaSymbolKey {
        kind: FeaSymbolKind::Model,
        name: "bracket".to_string(),
    }));
    assert!(root_symbols.contains_key(&FeaSymbolKey {
        kind: FeaSymbolKind::Function,
        name: "pulse".to_string(),
    }));
    assert_eq!(program.functions.len(), 1);
}

#[test]
fn compiler_reports_cycles_duplicates_and_missing_exports() {
    let host = TestHost::default()
        .with_module(
            "project/a.fea",
            "import model b from \"./b.fea\"\nmodel a {\n}\n",
            2,
        )
        .with_module(
            "project/b.fea",
            "import model a from \"./a.fea\"\nmodel b {\n}\n",
            3,
        )
        .with_module("project/materials.fea", "material aluminum {\n}\n", 4);
    let root = source_module(
        "project/main.fea",
        "import model a from \"./a.fea\"\nimport material steel from \"./materials.fea\"\nmaterial steel {\n}\nmaterial steel {\n}\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });

    let codes = program
        .graph
        .diagnostics
        .iter()
        .chain(&program.diagnostics)
        .map(|diagnostic| diagnostic.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"fea.module.import_cycle"));
    assert!(codes.contains(&"fea.binding.missing_export"));
    assert!(codes.contains(&"fea.binding.duplicate_symbol"));
    assert!(!program.is_valid());
}

#[test]
fn compiler_rejects_imports_outside_the_declared_source_root() {
    let host = TestHost::default();
    let root = source_module(
        "project/main.fea",
        "import material steel from \"../steel.fea\"\n",
        1,
    );
    let program = FeaCompiler::new(&host).check_bindings(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root,
        },
    });
    assert!(program
        .graph
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.module.invalid_path"));
    assert!(!program.is_valid());
}

fn source_module(path: &str, source: &str, digest: u8) -> SourceModule {
    SourceModule {
        logical_path: LogicalSourcePath::new(path).unwrap(),
        source_digest: SourceDigest([digest; 32]),
        source: source.as_bytes().to_vec(),
    }
}
