use crate::{
    bind_program, build_module_graph, AnalysisOperationCatalog, BoundFeaProgram, CheckRequest,
    ExecutionProfileSource, FeaDiagnostic, FeaModuleGraph, FeaModuleSource,
    GeometryInspectionSource, ModuleGraphRequest, ParsedFeaModule, RunMatFunctionSource,
    SourceModule,
};
use runmat_design_syntax::parse_fea;

pub trait FeaCompilerHost:
    FeaModuleSource
    + RunMatFunctionSource
    + GeometryInspectionSource
    + ExecutionProfileSource
    + AnalysisOperationCatalog
{
}

impl<T> FeaCompilerHost for T where
    T: FeaModuleSource
        + RunMatFunctionSource
        + GeometryInspectionSource
        + ExecutionProfileSource
        + AnalysisOperationCatalog
{
}

pub struct FeaCompiler<'a, H: FeaCompilerHost + ?Sized> {
    host: &'a H,
}

impl<'a, H: FeaCompilerHost + ?Sized> FeaCompiler<'a, H> {
    pub fn new(host: &'a H) -> Self {
        Self { host }
    }

    pub fn parse(&self, source: SourceModule) -> ParsedFeaModule {
        ParsedFeaModule::from_source(source)
    }

    pub fn build_module_graph(&self, request: ModuleGraphRequest) -> FeaModuleGraph {
        build_module_graph(self.host, request)
    }

    pub fn check_bindings(&self, request: CheckRequest) -> BoundFeaProgram {
        let graph = self.build_module_graph(request.modules);
        bind_program(self.host, graph)
    }

    pub fn check(&self, request: CheckRequest) -> crate::CheckedFeaProgram {
        let bound = self.check_bindings(request);
        crate::check_program(self.host, bound)
    }
}

impl ParsedFeaModule {
    pub fn from_source(source: SourceModule) -> Self {
        match String::from_utf8(source.source.clone()) {
            Ok(text) => {
                let syntax = parse_fea(text);
                let diagnostics = crate::module_graph::syntax_diagnostics(
                    &source.logical_path,
                    &syntax.diagnostics,
                );
                Self {
                    source,
                    syntax: Some(syntax),
                    diagnostics,
                }
            }
            Err(error) => Self {
                diagnostics: vec![FeaDiagnostic::error(
                    "fea.source.encoding",
                    format!("FEA source is not valid UTF-8: {error}"),
                    crate::FeaDiagnosticPhase::Syntax,
                    None,
                )],
                source,
                syntax: None,
            },
        }
    }
}
