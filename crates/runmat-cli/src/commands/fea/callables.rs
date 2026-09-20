use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_package::ProjectSymbolDefinition;

use crate::commands::session::create_session;

struct CompiledCallable {
    callable: BundleCallable,
    function: runmat_mir::analysis::FunctionAnalysis,
    executable: runmat_execution::ExecutableUnitEnvelope,
}

async fn compile_project_definitions(
    requested: Vec<(String, String, ProjectSymbolDefinition)>,
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<(ProgramRevision, BTreeMap<String, CompiledCallable>)> {
    let project = frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest.bytes()),
        Digest::from_bytes(*project.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let mut session = create_session(
        false,
        false,
        config,
        "create FEA callable compilation session",
    )?;
    session.install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))?;
    let mut compiled = BTreeMap::new();
    for (key, qualified_function, definition) in requested {
        let source_id = definition.source_id.as_ref().with_context(|| {
            format!("FEA function `{qualified_function}` has no frozen source identity")
        })?;
        let source_bytes = runmat_filesystem::read_async(&definition.source_path)
            .await
            .with_context(|| {
                format!(
                    "read frozen FEA function source {}",
                    definition.source_path.display()
                )
            })?;
        let source_digest = Digest::sha256(&source_bytes);
        if source_digest.bytes() != source_id.content_digest.bytes() {
            bail!(
                "FEA function `{qualified_function}` changed after its project source graph was frozen"
            );
        }
        let source_text = String::from_utf8(source_bytes).with_context(|| {
            format!(
                "FEA function source {} is not UTF-8 MATLAB text",
                definition.source_path.display()
            )
        })?;
        let owner_identity = source_id.package_instance.to_string();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::frozen(
                    owner_identity.clone(),
                    source_id.relative_path.as_str(),
                    definition.source_path.to_string_lossy(),
                    source_text,
                ),
                Some(revision.clone()),
            )
            .await
            .with_context(|| format!("compile FEA function `{qualified_function}`"))?;
        let entrypoint = entrypoint_name(&qualified_function)?;
        let executable = unit
            .portable_envelope_for(Some(entrypoint))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("freeze FEA function `{qualified_function}`"))?;
        let function = unit
            .function_analysis(executable.manifest.identity.entrypoint_function)
            .cloned()
            .with_context(|| {
                format!("FEA function `{qualified_function}` has no static function facts")
            })?;
        compiled.insert(
            key,
            CompiledCallable {
                callable: BundleCallable {
                    owner_identity,
                    qualified_name: qualified_function,
                    source_digest,
                },
                function,
                executable,
            },
        );
    }
    Ok((revision, compiled))
}

pub(super) async fn compile_geometry_generator(
    document_path: &std::path::Path,
    source_root: &std::path::Path,
    definition: &runmat_geometry_core::GeometryDefinition,
    bindings: &std::collections::BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<runmat_analysis_execution::PreparedGeometryGenerator> {
    let runmat_geometry_core::GeometryDefinitionSource::Generated(generated) = &definition.source
    else {
        bail!("geometry generator compilation requires a generated geometry definition");
    };
    let module = runmat_analysis_language::LogicalSourcePath::new(generated.module_path.clone())?;
    let symbol = runmat_analysis_language::SourceSymbol::new(generated.function.clone())?;
    let expected_path = runmat_runtime::analysis::physical_source_path(source_root, &module)
        .map_err(anyhow::Error::msg)?;
    let project_definition = runmat_package::resolve_visible_function_definition(
        frozen,
        document_path,
        &expected_path,
        symbol.as_str(),
    )
    .map_err(anyhow::Error::msg)?;
    let key = format!("{}::{symbol}", module.as_str());
    let (revision, mut compiled) = compile_project_definitions(
        vec![(
            key.clone(),
            project_definition.qualified_name.clone(),
            project_definition,
        )],
        frozen,
        config,
    )
    .await?;
    let compiled = compiled
        .remove(&key)
        .context("frozen geometry generator was not compiled")?;
    let request =
        runmat_analysis_execution::resolve_geometry_generator_request(definition, bindings)?;
    let admitted = runmat_analysis_execution::admit_named_geometry_generator(
        request,
        revision,
        compiled.callable,
        compiled.executable.manifest.identity.source_digest,
        &compiled.function,
        &std::collections::BTreeSet::new(),
    )?;
    runmat_analysis_execution::prepare_geometry_generator(admitted, compiled.executable)
        .map_err(anyhow::Error::new)
}

fn entrypoint_name(qualified_function: &str) -> Result<&str> {
    qualified_function
        .rsplit('.')
        .next()
        .filter(|name| !name.is_empty())
        .context("FEA function has an empty function name")
}
