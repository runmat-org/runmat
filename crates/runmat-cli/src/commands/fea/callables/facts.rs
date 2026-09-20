use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_language::{
    operation_pattern_from_value_fact, FunctionFacts, FunctionRequest, LogicalSourcePath,
    SourceDigest, SourceSymbol,
};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_package::ProjectSymbolDefinition;
use runmat_types::OperationParameter;

use super::compile_project_definitions;

pub(crate) async fn compile_function_facts(
    document_path: &Path,
    source_root: &Path,
    requests: &[FunctionRequest],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<FunctionFacts>> {
    let symbols = runmat_package::source_symbols_from_frozen(frozen, document_path);
    let mut definitions = Vec::with_capacity(requests.len());
    for request in requests {
        let expected_path = expected_source_path(source_root, &request.module_path)?;
        let definition = exact_definition(&symbols.definitions, &expected_path, &request.symbol)?;
        definitions.push((
            request_key(&request.module_path, &request.symbol),
            definition.qualified_name.clone(),
            definition.clone(),
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

pub(super) fn exact_definition<'a>(
    definitions: &'a [ProjectSymbolDefinition],
    expected_path: &Path,
    symbol: &SourceSymbol,
) -> Result<&'a ProjectSymbolDefinition> {
    let matches = definitions
        .iter()
        .filter(|definition| {
            !definition.is_private
                && definition.source_path == expected_path
                && definition
                    .qualified_name
                    .rsplit('.')
                    .next()
                    .is_some_and(|name| name == symbol.as_str())
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [definition] => Ok(*definition),
        [] => bail!(
            "imported function `{symbol}` is not defined by `{}`",
            expected_path.display()
        ),
        _ => bail!(
            "imported function `{symbol}` has multiple visible definitions in `{}`",
            expected_path.display()
        ),
    }
}

fn request_key(module_path: &LogicalSourcePath, symbol: &SourceSymbol) -> String {
    format!("{}::{symbol}", module_path.as_str())
}
