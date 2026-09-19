use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_language::{FunctionFacts, LogicalSourcePath, SourceDigest, SourceSymbol};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_package::ProjectSymbolDefinition;
use runmat_types::OperationParameter;

use super::{compile_project_definitions, operation_pattern};

pub(crate) async fn compile_function_facts(
    document_path: &Path,
    source_root: &Path,
    requests: &[(LogicalSourcePath, SourceSymbol)],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<FunctionFacts>> {
    let symbols = runmat_package::source_symbols_from_frozen(frozen, document_path);
    let mut definitions = Vec::with_capacity(requests.len());
    for (module_path, symbol) in requests {
        let expected_path = expected_source_path(source_root, module_path)?;
        let definition = exact_definition(&symbols.definitions, &expected_path, symbol)?;
        definitions.push((
            request_key(module_path, symbol),
            definition.qualified_name.clone(),
            definition.clone(),
        ));
    }
    let (_, compiled) = compile_project_definitions(definitions, frozen, config).await?;
    requests
        .iter()
        .map(|(module_path, symbol)| {
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
                inputs: callable
                    .parameters
                    .iter()
                    .zip(&callable.function.callable.parameters)
                    .map(|(parameter, fact)| OperationParameter {
                        name: parameter.name.clone(),
                        value: operation_pattern(fact),
                        arity: parameter.arity,
                        default: None,
                    })
                    .collect(),
                outputs: callable
                    .function
                    .outputs
                    .iter()
                    .map(operation_pattern)
                    .collect(),
                effects: callable.function.effects.clone(),
                capabilities: callable.function.capabilities.clone(),
            })
        })
        .collect()
}

fn expected_source_path(
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
