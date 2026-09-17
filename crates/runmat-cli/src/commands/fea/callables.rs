use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{
    GeneratorCallable, PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum, SignalId,
};
use runmat_analysis_execution::{
    admit_named_random_spectrum_generator, admit_named_response_spectrum_generator,
    admit_named_signal_generator, AdmittedNumericalGenerator, AdmittedSignalGenerator,
};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NumericalGeneratorTarget {
    ResponseSpectrum(usize),
    RandomSpectrum(usize),
}

#[derive(Clone, Debug)]
pub(super) struct CompiledNumericalGenerator {
    pub target: NumericalGeneratorTarget,
    pub admission: AdmittedNumericalGenerator,
    pub executable: ExecutableUnitEnvelope,
}

impl CompiledNumericalGenerator {
    pub(super) fn validate(&self) -> Result<()> {
        self.executable
            .validate()
            .context("validate portable executable for FEA numerical generator")?;
        let identity = &self.executable.manifest.identity;
        let admission = &self.admission.invocation;
        if identity.program != admission.program_revision
            || identity.root_package != admission.callable.owner_identity
            || identity.entrypoint_function != admission.function
            || identity.source_digest != admission.executable_source_digest
        {
            bail!("portable executable for FEA numerical input differs from its admission");
        }
        Ok(())
    }
}

pub(super) async fn compile_signal_generators(
    document_path: &Path,
    signals: &[PhysicalSignal],
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> Result<Vec<CompiledSignalGenerator>> {
    let requested = project_generators(signals)?;
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let resolved = crate::commands::package::resolve_for_source(document_path, cli)
        .await?
        .context("callable-backed FEA signals require an enclosing frozen RunMat project")?;
    compile_signal_generators_in_frozen(
        document_path,
        signals,
        &resolved.resolved.frozen,
        config,
        available_capabilities,
    )
    .await
}

pub(super) async fn compile_numerical_generators(
    document_path: &Path,
    response_spectra: &[ResponseSpectrum],
    random_spectra: &[RandomExcitationSpectrum],
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> Result<Vec<CompiledNumericalGenerator>> {
    let requested = numerical_project_generators(response_spectra, random_spectra)?;
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let resolved = crate::commands::package::resolve_for_source(document_path, cli)
        .await?
        .context("callable-backed FEA spectra require an enclosing frozen RunMat project")?;
    compile_numerical_generators_in_frozen(
        document_path,
        response_spectra,
        random_spectra,
        &resolved.resolved.frozen,
        config,
        available_capabilities,
    )
    .await
}

pub(super) async fn compile_signal_generators_in_frozen(
    document_path: &Path,
    signals: &[PhysicalSignal],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> Result<Vec<CompiledSignalGenerator>> {
    let requested = project_generators(signals)?;
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let names = requested.iter().map(|(_, name)| *name).collect::<Vec<_>>();
    let (revision, compiled) =
        compile_project_callables(document_path, &names, frozen, config).await?;

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

pub(super) async fn compile_numerical_generators_in_frozen(
    document_path: &Path,
    response_spectra: &[ResponseSpectrum],
    random_spectra: &[RandomExcitationSpectrum],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> Result<Vec<CompiledNumericalGenerator>> {
    let requested = numerical_project_generators(response_spectra, random_spectra)?;
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let names = requested.iter().map(|(_, name)| *name).collect::<Vec<_>>();
    let (revision, compiled) =
        compile_project_callables(document_path, &names, frozen, config).await?;
    requested
        .into_iter()
        .map(|(target, qualified_function)| {
            let compiled = compiled
                .get(qualified_function)
                .expect("every requested callable was compiled");
            let admission = match target {
                NumericalGeneratorTarget::ResponseSpectrum(index) => {
                    admit_named_response_spectrum_generator(
                        &response_spectra[index],
                        revision.clone(),
                        compiled.callable.clone(),
                        compiled.executable_source_digest,
                        &compiled.function,
                        available_capabilities,
                    )
                }
                NumericalGeneratorTarget::RandomSpectrum(index) => {
                    admit_named_random_spectrum_generator(
                        &random_spectra[index],
                        revision.clone(),
                        compiled.callable.clone(),
                        compiled.executable_source_digest,
                        &compiled.function,
                        available_capabilities,
                    )
                }
            }
            .with_context(|| format!("admit FEA generator `{qualified_function}`"))?;
            Ok(CompiledNumericalGenerator {
                target,
                admission,
                executable: compiled.executable.clone(),
            })
        })
        .collect()
}

async fn compile_project_callables(
    document_path: &Path,
    qualified_functions: &[&str],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<(ProgramRevision, BTreeMap<String, CompiledCallable>)> {
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
    for qualified_function in qualified_functions {
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
    Ok((revision, compiled))
}

fn numerical_project_generators<'a>(
    response_spectra: &'a [ResponseSpectrum],
    random_spectra: &'a [RandomExcitationSpectrum],
) -> Result<Vec<(NumericalGeneratorTarget, &'a str)>> {
    let mut requested = Vec::new();
    for (index, spectrum) in response_spectra.iter().enumerate() {
        if let Some(invocation) = spectrum.ordinates.generator() {
            requested.push((
                NumericalGeneratorTarget::ResponseSpectrum(index),
                project_function_name(invocation, "response spectrum")?,
            ));
        }
    }
    for (index, spectrum) in random_spectra.iter().enumerate() {
        if let Some(invocation) = spectrum.spectral_density.generator() {
            requested.push((
                NumericalGeneratorTarget::RandomSpectrum(index),
                project_function_name(invocation, "random spectrum")?,
            ));
        }
    }
    Ok(requested)
}

fn project_function_name<'a>(
    invocation: &'a runmat_analysis_core::GeneratorInvocation,
    object: &str,
) -> Result<&'a str> {
    match &invocation.callable {
        GeneratorCallable::ImportedFunction(identity) => Ok(&identity.symbol),
        GeneratorCallable::ProjectFunction { qualified_name } => Ok(qualified_name),
        GeneratorCallable::BoundFunction { .. } => bail!(
            "{object} contains a runtime-bound callable that cannot be loaded from a .fea document"
        ),
    }
}

fn project_generators(signals: &[PhysicalSignal]) -> Result<Vec<(&PhysicalSignal, &str)>> {
    let mut requested = Vec::new();
    for signal in signals {
        let Some(generator) = signal.generator() else {
            continue;
        };
        match &generator.invocation.callable {
            GeneratorCallable::ImportedFunction(identity) => {
                requested.push((signal, identity.symbol.as_str()));
            }
            GeneratorCallable::ProjectFunction { qualified_name } => {
                requested.push((signal, qualified_name.as_str()));
            }
            GeneratorCallable::BoundFunction { .. } => {
                bail!(
                    "signal `{}` contains a runtime-bound callable that cannot be loaded from a .fea document",
                    signal.signal_id().0
                );
            }
        }
    }
    Ok(requested)
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

pub(crate) async fn validate_numerical_generators(
    document_path: &Path,
    response_spectra: &[ResponseSpectrum],
    random_spectra: &[RandomExcitationSpectrum],
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    for compiled in compile_numerical_generators(
        document_path,
        response_spectra,
        random_spectra,
        cli,
        config,
        &BTreeSet::new(),
    )
    .await?
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
