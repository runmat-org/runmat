use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_language::{
    operation_pattern_from_value_fact, FunctionFacts, FunctionRequest, LogicalSourcePath,
    SourceDigest, SourceSymbol,
};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_types::OperationParameter;

use super::compile_project_definitions;

pub(super) struct CallableParameter {
    pub name: String,
    pub arity: runmat_types::OperationParameterArity,
}

pub(crate) async fn compile_function_facts(
    document_path: &Path,
    source_root: &Path,
    requests: &[FunctionRequest],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<FunctionFacts>> {
    let mut definitions = Vec::with_capacity(requests.len());
    for request in requests {
        let expected_path = expected_source_path(source_root, &request.module_path)?;
        let definition = runmat_package::resolve_visible_function_definition(
            frozen,
            document_path,
            &expected_path,
            request.symbol.as_str(),
        )
        .map_err(anyhow::Error::msg)?;
        definitions.push((
            request_key(&request.module_path, &request.symbol),
            definition.qualified_name.clone(),
            definition,
        ));
    }
    let (_, compiled) = compile_project_definitions(definitions, frozen, config).await?;
    requests
        .iter()
        .map(|request| {
            let module_path = &request.module_path;
            let symbol = &request.symbol;
            let callable = compiled
                .get(&request_key(module_path, symbol))
                .with_context(|| {
                    format!("frozen project did not compile `{module_path}::{symbol}`")
                })?;
            if callable.parameters.len() != callable.function.callable.parameters.len() {
                bail!("static parameter facts for `{module_path}::{symbol}` are incomplete");
            }
            Ok(FunctionFacts {
                module_path: module_path.clone(),
                symbol: symbol.clone(),
                source_digest: SourceDigest(*callable.callable.source_digest.bytes()),
                declaration_span: callable.declaration_span,
                inputs: callable
                    .parameters
                    .iter()
                    .zip(&callable.function.callable.parameters)
                    .map(|(parameter, fact)| OperationParameter {
                        name: parameter.name.clone(),
                        value: operation_pattern_from_value_fact(fact),
                        arity: parameter.arity,
                        default: None,
                    })
                    .collect(),
                outputs: callable
                    .function
                    .outputs
                    .iter()
                    .map(operation_pattern_from_value_fact)
                    .collect(),
                effects: callable.function.effects.clone(),
                capabilities: callable.function.capabilities.clone(),
            })
        })
        .collect()
}

pub(super) fn expected_source_path(
    source_root: &Path,
    module_path: &LogicalSourcePath,
) -> Result<std::path::PathBuf> {
    let relative = module_path
        .as_str()
        .strip_prefix("project/")
        .context("imported function is outside the project logical root")?;
    Ok(relative
        .split('/')
        .fold(source_root.to_path_buf(), |path, part| path.join(part)))
}

fn request_key(module_path: &LogicalSourcePath, symbol: &SourceSymbol) -> String {
    format!("{}::{symbol}", module_path.as_str())
}
