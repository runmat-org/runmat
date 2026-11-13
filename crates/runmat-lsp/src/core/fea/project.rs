use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_analysis_language::{
    operation_pattern_from_value_fact, remote_execution_profile_facts, ExecutionProfileFacts,
    FunctionFacts, FunctionRequest, LogicalSourcePath,
};
use runmat_types::{OperationParameter, OperationParameterArity};

use super::super::analysis::{
    analyze_document_with_compat_and_source, semantic_document_facts, CompatMode,
};
use super::super::project::ProjectContext;

pub(crate) struct FeaProjectFacts {
    pub functions: Vec<FunctionFacts>,
    pub profiles: Vec<ExecutionProfileFacts>,
    pub function_sources: BTreeMap<LogicalSourcePath, Arc<str>>,
}

pub(crate) fn resolve_project_facts(
    source_root: &Path,
    requests: &[FunctionRequest],
    entry_path: &Path,
    compat: CompatMode,
    source_overlays: &BTreeMap<PathBuf, String>,
) -> Result<FeaProjectFacts, String> {
    let project = ProjectContext::discover_from_source_name(entry_path.to_str());
    if !requests.is_empty() && project.is_none() {
        return Err("imported `.m` functions require an enclosing frozen RunMat project".into());
    }
    let profiles = match &project {
        Some(project) => {
            let manifest =
                runmat_config::project::load_project_manifest(project.manifest_path())
                    .map_err(|error| format!("cannot load project execution profiles: {error}"))?;
            remote_execution_profile_facts(
                manifest
                    .execution_profiles
                    .iter()
                    .map(|(name, profile)| (name.clone(), profile.accelerator.is_some())),
            )
            .map_err(|error| error.to_string())?
        }
        None => Vec::new(),
    };

    let mut functions = Vec::with_capacity(requests.len());
    let mut function_sources = BTreeMap::new();
    for request in requests {
        let path = physical_path(source_root, &request.module_path)?;
        let project = project.as_ref().ok_or_else(|| {
            "imported `.m` functions require an enclosing frozen RunMat project".to_string()
        })?;
        if !project.contains_visible_source(&path) {
            return Err(format!(
                "imported function {} is not visible in the frozen RunMat project",
                path.display()
            ));
        }
        let source = match source_overlays.get(&path) {
            Some(source) => source.clone(),
            None => runmat_filesystem::read_to_string(&path).map_err(|error| {
                format!("cannot read imported function {}: {error}", path.display())
            })?,
        };
        let document = analyze_document_with_compat_and_source(&source, compat, path.to_str());
        functions.push(function_facts(request, &source, &document)?);
        function_sources.insert(request.module_path.clone(), Arc::<str>::from(source));
    }
    Ok(FeaProjectFacts {
        functions,
        profiles,
        function_sources,
    })
}

fn function_facts(
    request: &FunctionRequest,
    source: &str,
    document: &super::super::analysis::DocumentAnalysis,
) -> Result<FunctionFacts, String> {
    if let Some(diagnostic) = document.diagnostics.first() {
        return Err(format!(
            "cannot analyze imported function `{}::{}`: {}",
            request.module_path, request.symbol, diagnostic.message
        ));
    }
    let lowering = document.lowering.as_ref().ok_or_else(|| {
        format!(
            "imported function `{}::{}` has no lowered program",
            request.module_path, request.symbol
        )
    })?;
    let function = lowering
        .assembly
        .functions
        .iter()
        .find(|function| {
            function.parent.is_none()
                && function.name.0 == request.symbol.as_str()
                && matches!(function.kind, runmat_hir::FunctionKind::Named)
        })
        .ok_or_else(|| {
            format!(
                "imported source does not define function `{}`",
                request.symbol
            )
        })?;
    let semantic = semantic_document_facts(document)
        .and_then(|facts| {
            facts
                .functions
                .iter()
                .find(|facts| facts.name == request.symbol.as_str())
        })
        .and_then(|facts| facts.analysis.as_ref())
        .ok_or_else(|| {
            format!(
                "imported function `{}` has no static function facts",
                request.symbol
            )
        })?;
    if !semantic.callable.parameters_complete
        || !semantic.callable.outputs_complete
        || semantic.callable.parameters.len() != function.params.len()
    {
        return Err(format!(
            "static callable facts for imported function `{}` are incomplete",
            request.symbol
        ));
    }
    let inputs = function
        .params
        .iter()
        .zip(&semantic.callable.parameters)
        .map(|(binding, fact)| {
            let binding_facts = lowering
                .assembly
                .bindings
                .get(binding.0)
                .ok_or_else(|| "imported function parameter binding is missing".to_string())?;
            let arity = if function.abi.varargin == Some(*binding) {
                OperationParameterArity::Variadic
            } else if function.argument_validations.iter().any(|validation| {
                validation.binding == *binding && validation.default_value.is_some()
            }) {
                OperationParameterArity::Optional
            } else {
                OperationParameterArity::Required
            };
            Ok(OperationParameter {
                name: binding_facts.name.0.clone(),
                value: operation_pattern_from_value_fact(fact),
                arity,
                default: None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(FunctionFacts {
        module_path: request.module_path.clone(),
        symbol: request.symbol.clone(),
        source_digest: runmat_runtime::analysis::source_digest(source.as_bytes()),
        declaration_span: function.span,
        inputs,
        outputs: semantic
            .outputs
            .iter()
            .map(operation_pattern_from_value_fact)
            .collect(),
        effects: semantic.effects.clone(),
        cooperative_suspension: semantic.cooperative_suspension,
        capabilities: semantic.capabilities.clone(),
    })
}

fn physical_path(root: &Path, logical: &LogicalSourcePath) -> Result<PathBuf, String> {
    runmat_runtime::analysis::physical_source_path(root, logical).map_err(|error| error.to_string())
}
