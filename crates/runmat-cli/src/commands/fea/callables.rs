use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{PhysicalSignal, SignalId};
use runmat_analysis_execution::{admit_named_signal_generator, AdmittedSignalGenerator};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ExecutableUnitEnvelope, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_package::ProjectSymbolDefinition;
use runmat_types::CapabilityRequirement;

use crate::cli::Cli;
use crate::commands::session::create_session;

#[derive(Clone, Debug)]
pub(super) struct CompiledSignalGenerator {
    pub signal_id: SignalId,
    pub admission: AdmittedSignalGenerator,
    pub executable: ExecutableUnitEnvelope,
}

impl CompiledSignalGenerator {
    pub(super) fn validate(&self) -> Result<()> {
        self.executable.validate().with_context(|| {
            format!(
                "validate portable executable for FEA signal `{}`",
                self.signal_id.0
            )
        })?;
        let identity = &self.executable.manifest.identity;
        if identity.program != self.admission.program_revision
            || identity.root_package != self.admission.callable.owner_identity
            || identity.entrypoint_function != self.admission.function
            || identity.source_digest != self.admission.executable_source_digest
        {
            bail!(
                "portable executable for FEA signal `{}` differs from its admitted callable",
                self.signal_id.0
            );
        }
        Ok(())
    }
}

struct CompiledCallable {
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: runmat_mir::analysis::FunctionAnalysis,
    executable: ExecutableUnitEnvelope,
}

pub(super) async fn compile_signal_generators(
    document_path: &Path,
    signals: &[PhysicalSignal],
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> Result<Vec<CompiledSignalGenerator>> {
    let requested = signals
        .iter()
        .filter_map(|signal| {
            signal
                .generator()
                .map(|generator| (signal, generator.qualified_function.as_str()))
        })
        .collect::<Vec<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let resolved = crate::commands::package::resolve_for_source(document_path, cli)
        .await?
        .context("callable-backed FEA signals require an enclosing frozen RunMat project")?;
    let frozen = &resolved.resolved.frozen;
    let project = frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest.bytes()),
        Digest::from_bytes(*project.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let symbols = runmat_package::source_symbols_from_frozen(frozen, document_path);
    let mut session = create_session(
        false,
        false,
        config,
        "create FEA callable compilation session",
    )?;
    session.install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))?;

    let mut compiled = BTreeMap::new();
    for (_, qualified_function) in &requested {
        if compiled.contains_key(*qualified_function) {
            continue;
        }
        let definition = unique_definition(&symbols.definitions, qualified_function)?;
        let source_id = definition.source_id.as_ref().with_context(|| {
            format!("FEA generator `{qualified_function}` has no frozen source identity")
        })?;
        let source_bytes = runmat_filesystem::read_async(&definition.source_path)
            .await
            .with_context(|| {
                format!(
                    "read frozen FEA generator source {}",
                    definition.source_path.display()
                )
            })?;
        let source_digest = Digest::sha256(&source_bytes);
        if source_digest.bytes() != source_id.content_digest.bytes() {
            bail!(
                "FEA generator `{qualified_function}` changed after its project source graph was frozen"
            );
        }
        let source_text = String::from_utf8(source_bytes).with_context(|| {
            format!(
                "FEA generator source {} is not UTF-8 MATLAB text",
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
            .with_context(|| format!("compile FEA generator `{qualified_function}`"))?;
        let entrypoint = definition
            .qualified_name
            .rsplit('.')
            .next()
            .filter(|name| !name.is_empty())
            .context("FEA generator has an empty function name")?;
        let executable = unit
            .portable_envelope_for(Some(entrypoint))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("freeze FEA generator `{qualified_function}`"))?;
        let function = unit
            .function_analysis(executable.manifest.identity.entrypoint_function)
            .cloned()
            .with_context(|| {
                format!("FEA generator `{qualified_function}` has no static function facts")
            })?;
        let executable_source_digest = unit
            .revision()
            .source_digest
            .parse::<Digest>()
            .with_context(|| {
                format!("FEA generator `{qualified_function}` has an invalid executable digest")
            })?;
        compiled.insert(
            (*qualified_function).to_string(),
            CompiledCallable {
                callable: BundleCallable {
                    owner_identity,
                    qualified_name: (*qualified_function).to_string(),
                    source_digest,
                },
                executable_source_digest,
                function,
                executable,
            },
        );
    }

    requested
        .into_iter()
        .map(|(signal, qualified_function)| {
            let compiled = compiled
                .get(qualified_function)
                .expect("every requested callable was compiled");
            let admission = admit_named_signal_generator(
                signal,
                revision.clone(),
                compiled.callable.clone(),
                compiled.executable_source_digest,
                &compiled.function,
                available_capabilities,
            )
            .with_context(|| format!("admit FEA generator `{qualified_function}`"))?;
            Ok(CompiledSignalGenerator {
                signal_id: signal.signal_id().clone(),
                admission,
                executable: compiled.executable.clone(),
            })
        })
        .collect()
}

pub(crate) async fn validate_signal_generators(
    document_path: &Path,
    signals: &[PhysicalSignal],
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    for compiled in
        compile_signal_generators(document_path, signals, cli, config, &BTreeSet::new()).await?
    {
        compiled.validate()?;
    }
    Ok(())
}

fn unique_definition<'a>(
    definitions: &'a [ProjectSymbolDefinition],
    qualified_function: &str,
) -> Result<&'a ProjectSymbolDefinition> {
    let matches = definitions
        .iter()
        .filter(|definition| definition.name == qualified_function && !definition.is_private)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [definition] => Ok(*definition),
        [] => bail!(
            "FEA generator `{qualified_function}` does not resolve to a visible named project function"
        ),
        _ => bail!("FEA generator `{qualified_function}` is ambiguous in the frozen project graph"),
    }
}
