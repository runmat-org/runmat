//! Captured standalone policy admission before foreign installation effects.

use std::collections::BTreeSet;

use runmat_config::runtime::{
    ForeignConfig, NativeFfiIsolation, PythonExecutionModeConfig, UnmanifestedMexPolicy,
};
use runmat_runtime::foreign::{
    JavaRuntimeConfiguration, MexIsolationPolicy, NativeFfiIsolationPolicy,
    PythonRuntimeConfiguration,
};
use runmat_types::{ForeignAdapterId, InteropManifest};

use crate::AotProcessInput;

pub(super) struct AdmittedForeignPolicy {
    config: Option<ForeignConfig>,
    python: Option<PythonRuntimeConfiguration>,
    native_required: bool,
    mex_required: bool,
    java_required: bool,
    pub(super) native_bundle: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub(super) mex_bundle: runmat_mex::MexArtifactBundle,
    pub(super) java_bundle: runmat_java::JavaArtifactBundle,
    pub(super) python_bundle: runmat_python::PythonArtifactBundle,
}

impl AdmittedForeignPolicy {
    pub(super) fn admit(
        object: Option<&runmat_execution_artifact::LogicalObject>,
        manifest: &InteropManifest,
        input: &AotProcessInput,
        native: &ForeignAdapterId,
        mex: &ForeignAdapterId,
        java: &ForeignAdapterId,
        python: &ForeignAdapterId,
    ) -> Result<Self, String> {
        let nonempty = !manifest.foreign_types.is_empty()
            || !manifest.adapters.is_empty()
            || !manifest.adapter_contracts.is_empty();
        if nonempty && object.is_none() {
            return Err("standalone foreign program has no captured policy".into());
        }
        let config: Option<ForeignConfig> = object
            .map(runmat_execution_artifact::decode_typed_foreign_policy)
            .transpose()
            .map_err(|error| format!("standalone foreign policy is invalid: {error}"))?;
        let required =
            |id: &ForeignAdapterId| manifest.adapters.iter().any(|item| &item.adapter == id);
        let native_required = required(native);
        let mex_required = required(mex);
        let java_required = required(java);
        let python_required = required(python);
        let native_bundle = runmat_native_ffi::NativeInterfaceArtifactBundle::from_canonical_bytes(
            &input.native_interfaces,
        )
        .map_err(|error| format!("standalone native bundle is invalid: {error}"))?;
        let mex_bundle = runmat_mex::MexArtifactBundle::from_canonical_bytes(&input.mex_artifacts)
            .map_err(|error| format!("standalone MEX bundle is invalid: {error}"))?;
        let java_bundle =
            runmat_java::JavaArtifactBundle::from_canonical_bytes(&input.java_artifacts)
                .map_err(|error| format!("standalone Java bundle is invalid: {error}"))?;
        let python_bundle =
            runmat_python::PythonArtifactBundle::from_canonical_bytes(&input.python_artifacts)
                .map_err(|error| format!("standalone Python bundle is invalid: {error}"))?;
        if object.is_none()
            && (!native_bundle.interfaces.is_empty()
                || !mex_bundle.artifacts.is_empty()
                || !java_bundle.artifacts.is_empty()
                || python_bundle.environment.is_some()
                || !python_bundle.artifacts.is_empty())
        {
            return Err("standalone pure program carries foreign artifacts without policy".into());
        }
        preflight_required_identities(
            manifest,
            ForeignBundleAdmission {
                native: (native, &native_bundle),
                mex: (mex, &mex_bundle),
                java: (java, &java_bundle),
                python: (python, &python_bundle),
            },
        )?;
        let python = if python_required {
            let config = config
                .as_ref()
                .ok_or("standalone Python policy is absent")?;
            Some(preflight_python(config, &python_bundle)?)
        } else {
            None
        };
        if java_required {
            preflight_java(config.as_ref().ok_or("standalone Java policy is absent")?)?;
        }
        crate::mex_artifacts::ensure_required_providers(&mex_bundle)?;
        Ok(Self {
            config,
            python,
            native_required,
            mex_required,
            java_required,
            native_bundle,
            mex_bundle,
            java_bundle,
            python_bundle,
        })
    }

    pub(super) fn apply(
        &mut self,
        native: &runmat_runtime::foreign::NativeFfiAdapter,
        mex: &runmat_runtime::foreign::MexRuntimeSession,
        java: &runmat_runtime::foreign::JavaAdapter,
        python: &runmat_runtime::foreign::PythonAdapter,
    ) -> Result<(), String> {
        let Some(config) = self.config.as_ref() else {
            return Ok(());
        };
        if self.native_required {
            native
                .set_isolation_policy(NativeFfiIsolationPolicy {
                    isolate: config.native.isolation == NativeFfiIsolation::Process,
                    invocation_timeout: config
                        .native
                        .timeout
                        .map(|value| value.duration())
                        .or_else(|| {
                            config
                                .native
                                .timeout_ms
                                .map(|value| std::time::Duration::from_millis(value.get()))
                        }),
                })
                .map_err(|error| format!("apply standalone native policy: {error}"))?;
        }
        if self.mex_required {
            mex.set_isolation_policy(MexIsolationPolicy {
                unmanifested: match config.mex.unmanifested {
                    UnmanifestedMexPolicy::Isolate => {
                        runmat_runtime::foreign::UnmanifestedMexPolicy::Isolate
                    }
                    UnmanifestedMexPolicy::Deny => {
                        runmat_runtime::foreign::UnmanifestedMexPolicy::Deny
                    }
                },
                invocation_timeout: config
                    .mex
                    .timeout_ms
                    .map(|value| std::time::Duration::from_millis(value.get())),
            });
        }
        if self.java_required {
            java.configure(JavaRuntimeConfiguration {
                home: config.java.home.clone(),
                minimum_version: config.java.minimum_version.get(),
                maximum_version: config.java.maximum_version.map(|value| value.get()),
                classpath: config.java.classpath.clone(),
                options: config.java.options.clone(),
            })
            .map_err(|error| format!("apply standalone Java policy: {error}"))?;
        }
        if let Some(configured) = self.python.take() {
            python
                .configure(configured)
                .map_err(|error| format!("apply standalone Python policy: {error}"))?;
        }
        Ok(())
    }
}

/// Each archived family stays paired with its admitted adapter identity.
struct ForeignBundleAdmission<'a> {
    native: (
        &'a ForeignAdapterId,
        &'a runmat_native_ffi::NativeInterfaceArtifactBundle,
    ),
    mex: (&'a ForeignAdapterId, &'a runmat_mex::MexArtifactBundle),
    java: (&'a ForeignAdapterId, &'a runmat_java::JavaArtifactBundle),
    python: (
        &'a ForeignAdapterId,
        &'a runmat_python::PythonArtifactBundle,
    ),
}

fn preflight_required_identities(
    manifest: &InteropManifest,
    archived: ForeignBundleAdmission<'_>,
) -> Result<(), String> {
    let ForeignBundleAdmission {
        native: (native, native_bundle),
        mex: (mex, mex_bundle),
        java: (java, java_bundle),
        python: (python, python_bundle),
    } = archived;
    if manifest
        .adapters
        .iter()
        .any(|item| ![native, mex, java, python].contains(&&item.adapter))
    {
        return Err("standalone interop names an unsupported foreign adapter".into());
    }
    let native_manifests = native_bundle
        .manifests()
        .map_err(|error| format!("standalone native manifests are invalid: {error}"))?;
    for (manifest, entry) in native_manifests.iter().zip(&native_bundle.interfaces) {
        manifest
            .validate_current_library(&entry.library)
            .map_err(|error| format!("standalone native library is invalid: {error}"))?;
    }
    let native_ids = native_manifests
        .into_iter()
        .map(|item| item.identity.to_string())
        .collect::<BTreeSet<_>>();
    let mex_ids = mex_bundle
        .artifacts
        .iter()
        .map(|item| -> Result<String, String> {
            let manifest = runmat_mex::MexArtifactManifest::from_canonical_bytes(&item.manifest)
                .map_err(|error| format!("standalone MEX artifact is invalid: {error}"))?;
            manifest
                .validate_current_module(&item.module)
                .map_err(|error| format!("standalone MEX module is invalid: {error}"))?;
            Ok(manifest.identity.to_string())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let java_ids = java_bundle
        .artifacts
        .iter()
        .map(|item| item.identity.to_string())
        .collect::<BTreeSet<_>>();
    let python_ids = python_bundle.artifact_identities();
    for (adapter, ids) in [
        (native, &native_ids),
        (mex, &mex_ids),
        (java, &java_ids),
        (python, &python_ids),
    ] {
        let requirement = manifest
            .adapters
            .iter()
            .find(|item| &item.adapter == adapter);
        if let Some(requirement) = requirement {
            let declared = requirement
                .artifact_identities
                .iter()
                .map(|identity| identity.as_str().to_owned())
                .collect::<BTreeSet<_>>();
            if &declared != ids {
                return Err(format!(
                    "standalone adapter {} archived artifacts differ from its exact grant",
                    adapter
                ));
            }
        } else if !ids.is_empty() {
            return Err(format!(
                "standalone adapter {} has unselected archived artifacts",
                adapter
            ));
        }
    }
    Ok(())
}

fn preflight_java(policy: &ForeignConfig) -> Result<(), String> {
    if !policy.java.classpath.is_empty() {
        return Err("standalone Java classpath contains unarchived local paths".into());
    }
    if let Some(home) = &policy.java.home {
        let installation = runmat_java::discover_jvm(&runmat_java::JavaDiscoveryRequest {
            explicit_home: Some(home.clone()),
            environment_home: None,
            java_executable: None,
            search_system: false,
            required_major: None,
        })
        .map_err(|error| format!("standalone Java home is unavailable: {error}"))?;
        if installation.version.major < policy.java.minimum_version.get()
            || policy
                .java
                .maximum_version
                .is_some_and(|value| installation.version.major > value.get())
        {
            return Err("standalone Java home violates captured version bounds".into());
        }
    }
    Ok(())
}

fn preflight_python(
    policy: &ForeignConfig,
    bundle: &runmat_python::PythonArtifactBundle,
) -> Result<PythonRuntimeConfiguration, String> {
    let version = |value: &str| parse_python_version(value);
    let mut configured = PythonRuntimeConfiguration {
        executable: policy.python.executable.clone(),
        version: policy.python.version.as_deref().map(version).transpose()?,
        minimum_version: version(&policy.python.minimum_version)?,
        maximum_version: policy
            .python
            .maximum_version
            .as_deref()
            .map(version)
            .transpose()?,
        exact_version: policy
            .python
            .exact_version
            .map(|value| runmat_python::PythonVersion {
                major: value.major,
                minor: value.minor,
                patch: value.patch,
            }),
        required_abi_tag: policy.python.required_abi_tag.clone(),
        required_platform_tag: policy.python.required_platform_tag.clone(),
        execution_mode: match policy.python.execution_mode {
            PythonExecutionModeConfig::InProcess => runmat_python::PythonExecutionMode::InProcess,
            PythonExecutionModeConfig::OutOfProcess => {
                runmat_python::PythonExecutionMode::OutOfProcess
            }
        },
        module_paths: policy.python.module_paths.clone(),
        artifact_identities: policy.python.artifact_identities.clone(),
    };
    if let Some(environment) = &bundle.environment {
        let pair = (environment.version.major, environment.version.minor);
        if configured.execution_mode != environment.execution_mode
            || configured
                .exact_version
                .is_some_and(|value| value != environment.version)
            || configured
                .required_abi_tag
                .as_ref()
                .is_some_and(|value| value != &environment.abi_tag)
            || configured
                .required_platform_tag
                .as_ref()
                .is_some_and(|value| value != &environment.platform_tag)
            || configured.version.is_some_and(|value| value != pair)
            || pair < configured.minimum_version
            || configured.maximum_version.is_some_and(|value| pair > value)
        {
            return Err("standalone Python policy differs from archived environment".into());
        }
        if configured.executable.is_some() {
            let installation =
                runmat_python::discover_python(&runmat_python::PythonDiscoveryRequest {
                    executable: configured.executable.clone(),
                    version: configured.version,
                    minimum_version: configured.minimum_version,
                    maximum_version: configured.maximum_version,
                    exact_version: Some(environment.version),
                    required_abi_tag: Some(environment.abi_tag.clone()),
                    required_platform_tag: Some(environment.platform_tag.clone()),
                })
                .map_err(|error| {
                    format!("standalone Python executable differs from archive: {error}")
                })?;
            if installation.implementation != environment.implementation {
                return Err("standalone Python implementation differs from archive".into());
            }
        }
    } else if configured.executable.is_some() {
        return Err("standalone Python executable has no archived environment".into());
    }
    let bundle_ids = bundle.artifact_identities();
    if !configured.artifact_identities.is_subset(&bundle_ids) {
        return Err("standalone Python policy pins unavailable artifacts".into());
    }
    match (
        &policy.python.installed_module_paths,
        configured.module_paths.is_empty(),
    ) {
        (None, true) => {}
        (Some(source), false)
            if source.artifact_identities == bundle_ids && !bundle.artifacts.is_empty() =>
        {
            configured.module_paths.clear();
        }
        _ => return Err("standalone Python module paths lack archived provenance".into()),
    }
    Ok(configured)
}

fn parse_python_version(value: &str) -> Result<(u16, u16), String> {
    let mut parts = value.split('.');
    let pair = (
        parts.next().and_then(|value| value.parse().ok()),
        parts.next().and_then(|value| value.parse().ok()),
    );
    match pair {
        (Some(major), Some(minor)) if parts.next().is_none() => Ok((major, minor)),
        _ => Err("standalone Python version must use major.minor form".into()),
    }
}
