use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_package::{ContentDigest, FrozenProject};
use runmat_python::{
    discover_python, PythonArtifactBundle, PythonArtifactBundleEntry, PythonDiscoveryRequest,
    PythonEnvironmentIdentity, PythonExecutionMode, PYTHON_ADAPTER_ID, PYTHON_ADAPTER_VERSION,
    PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE,
};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAdapterRequirement, InteropManifest,
    INTEROP_MANIFEST_SCHEMA_VERSION,
};

pub(super) struct PreparedPythonArtifacts {
    pub(super) interop: InteropManifest,
    pub(super) objects: Vec<LogicalObject>,
    pub(super) bundle: PythonArtifactBundle,
}

pub(super) fn prepare(
    project: Option<&FrozenProject>,
    config: &runmat_config::runtime::PythonConfig,
    required: bool,
) -> Result<PreparedPythonArtifacts> {
    let project_artifacts = project
        .map(|project| project.python_artifacts.as_slice())
        .unwrap_or_default();
    if !required && project_artifacts.is_empty() {
        return Ok(PreparedPythonArtifacts {
            interop: InteropManifest::empty(),
            objects: Vec::new(),
            bundle: PythonArtifactBundle::empty(),
        });
    }
    let request = discovery_request(config)?;
    let installation = discover_python(&request)
        .map_err(|error| anyhow::anyhow!("resolve Python environment: {error}"))?;
    let execution_mode = match config.execution_mode {
        runmat_config::runtime::PythonExecutionModeConfig::InProcess => {
            PythonExecutionMode::InProcess
        }
        runmat_config::runtime::PythonExecutionModeConfig::OutOfProcess => {
            PythonExecutionMode::OutOfProcess
        }
    };
    let environment = PythonEnvironmentIdentity::from_installation(&installation, execution_mode);
    let mut declarations = BTreeMap::new();
    for declaration in project_artifacts {
        if declarations
            .insert(declaration.name.clone(), declaration)
            .is_some()
        {
            bail!(
                "Python artifact name `{}` is declared by more than one package",
                declaration.name
            );
        }
    }
    let mut entries = Vec::with_capacity(declarations.len());
    for (logical_name, declaration) in declarations {
        let bytes = std::fs::read(&declaration.path).with_context(|| {
            format!(
                "read Python artifact `{logical_name}` at `{}`",
                declaration.path.display()
            )
        })?;
        let digest = ContentDigest::sha256(&bytes);
        if digest != declaration.digest {
            bail!(
                "Python artifact `{logical_name}` changed after project resolution: expected {}, found {}",
                declaration.digest,
                digest
            );
        }
        let filename = declaration
            .path
            .file_name()
            .and_then(|value| value.to_str())
            .context("Python wheel filename is not valid Unicode")?
            .to_owned();
        entries.push(PythonArtifactBundleEntry::wheel(
            logical_name.clone(),
            declaration.module.clone(),
            filename.clone(),
            bytes.clone(),
        ));
    }
    let bundle = PythonArtifactBundle::new(environment, entries)?;
    let objects = vec![LogicalObject::new(
        ObjectNamespace::ForeignArtifact,
        "python/artifacts.json",
        PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE,
        bundle.canonical_bytes()?,
    )?];
    let interop = InteropManifest {
        schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
        foreign_types: Vec::new(),
        adapters: vec![ForeignAdapterRequirement {
            adapter: PYTHON_ADAPTER_ID.into(),
            minimum_version: PYTHON_ADAPTER_VERSION,
            capabilities: CapabilitySet(BTreeSet::from([CapabilityRequirement::ForeignRuntime])),
            artifact_identities: bundle.artifact_identities().into_iter().collect(),
        }],
    };
    interop
        .validate()
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    Ok(PreparedPythonArtifacts {
        interop,
        objects,
        bundle,
    })
}

fn discovery_request(
    config: &runmat_config::runtime::PythonConfig,
) -> Result<PythonDiscoveryRequest> {
    Ok(PythonDiscoveryRequest {
        executable: config.executable.clone(),
        version: config.version.as_deref().map(parse_version).transpose()?,
        minimum_version: parse_version(&config.minimum_version)?,
        maximum_version: config
            .maximum_version
            .as_deref()
            .map(parse_version)
            .transpose()?,
        exact_version: None,
        required_abi_tag: None,
        required_platform_tag: None,
    })
}

fn parse_version(value: &str) -> Result<(u16, u16)> {
    let mut parts = value.split('.');
    let major = parts.next().and_then(|value| value.parse().ok());
    let minor = parts.next().and_then(|value| value.parse().ok());
    if parts.next().is_some() || major.is_none() || minor.is_none() {
        bail!("Python versions must use major.minor form");
    }
    Ok((major.expect("checked"), minor.expect("checked")))
}
