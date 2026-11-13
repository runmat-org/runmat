//! Selected-policy and host-resolution checks before adapter mutation.

use std::collections::BTreeSet;

use runmat_config::runtime::ForeignConfig;
use runmat_runtime::foreign::PythonRuntimeConfiguration;
use runmat_types::InteropManifest;

use crate::foreign_unit_host::RegisteredForeignAdapterIds;

use super::install::{MaterializedForeignView, SelectedForeignInstallError};

pub(super) struct SelectedPolicyAdmission {
    pub python: Option<PythonRuntimeConfiguration>,
    pub native_required: bool,
    pub mex_required: bool,
    pub java_required: bool,
}

pub(super) fn admit(
    policy: &ForeignConfig,
    manifest: &InteropManifest,
    materialized: &dyn MaterializedForeignView,
    ids: &RegisteredForeignAdapterIds,
) -> Result<SelectedPolicyAdmission, SelectedForeignInstallError> {
    let requires = |adapter: &runmat_types::ForeignAdapterId| {
        manifest
            .adapters
            .iter()
            .any(|item| &item.adapter == adapter)
    };
    let java_required = requires(&ids.java);
    let python_required = requires(&ids.python);
    if java_required {
        preflight_java(policy)?;
    }
    let python = if python_required {
        Some(preflight_python(policy, materialized)?)
    } else {
        None
    };
    Ok(SelectedPolicyAdmission {
        python,
        native_required: requires(&ids.native_ffi),
        mex_required: requires(&ids.mex),
        java_required,
    })
}

fn preflight_java(policy: &ForeignConfig) -> Result<(), SelectedForeignInstallError> {
    if !policy.java.classpath.is_empty() {
        return Err(SelectedForeignInstallError::Validation(
            "selected Java classpath contains submitter-local paths without authenticated archive materialization".into(),
        ));
    }
    if let Some(home) = &policy.java.home {
        let installation = runmat_java::discover_jvm(&runmat_java::JavaDiscoveryRequest {
            explicit_home: Some(home.clone()),
            environment_home: None,
            java_executable: None,
            search_system: false,
            required_major: None,
        })
        .map_err(|error| {
            SelectedForeignInstallError::Validation(format!(
                "selected Java home is unavailable on this host: {error}"
            ))
        })?;
        if installation.version.major < policy.java.minimum_version.get()
            || policy
                .java
                .maximum_version
                .is_some_and(|maximum| installation.version.major > maximum.get())
        {
            return Err(SelectedForeignInstallError::Validation(
                "selected Java home does not satisfy captured version bounds".into(),
            ));
        }
    }
    Ok(())
}

fn preflight_python(
    policy: &ForeignConfig,
    materialized: &dyn MaterializedForeignView,
) -> Result<PythonRuntimeConfiguration, SelectedForeignInstallError> {
    let bundle = materialized.python_bundle().ok_or_else(|| {
        SelectedForeignInstallError::Validation("selected Python artifact bundle is absent".into())
    })?;
    let mut configured = super::policy::python_runtime(&policy.python)
        .map_err(SelectedForeignInstallError::Validation)?;
    if let Some(environment) = &bundle.environment {
        let pair = (environment.version.major, environment.version.minor);
        if configured.execution_mode != environment.execution_mode
            || configured
                .exact_version
                .is_some_and(|version| version != environment.version)
            || configured
                .required_abi_tag
                .as_ref()
                .is_some_and(|tag| tag != &environment.abi_tag)
            || configured
                .required_platform_tag
                .as_ref()
                .is_some_and(|tag| tag != &environment.platform_tag)
            || configured.version.is_some_and(|version| version != pair)
            || pair < configured.minimum_version
            || configured
                .maximum_version
                .is_some_and(|maximum| pair > maximum)
        {
            return Err(SelectedForeignInstallError::Validation(
                "selected Python policy differs from authenticated archived environment".into(),
            ));
        }
        if configured.executable.is_some() {
            let installation = runmat_python::discover_python(&runmat_python::PythonDiscoveryRequest {
                executable: configured.executable.clone(),
                version: configured.version,
                minimum_version: configured.minimum_version,
                maximum_version: configured.maximum_version,
                exact_version: Some(environment.version),
                required_abi_tag: Some(environment.abi_tag.clone()),
                required_platform_tag: Some(environment.platform_tag.clone()),
            }).map_err(|error| SelectedForeignInstallError::Validation(format!(
                "selected Python executable is unavailable or differs from archived environment: {error}"
            )))?;
            if installation.implementation != environment.implementation {
                return Err(SelectedForeignInstallError::Validation(
                    "selected Python executable implementation differs from archived environment"
                        .into(),
                ));
            }
        }
    } else if configured.executable.is_some() {
        return Err(SelectedForeignInstallError::Validation(
            "selected Python executable has no archived environment identity".into(),
        ));
    }
    let bundle_ids = bundle.artifact_identities();
    if !configured.artifact_identities.is_subset(&bundle_ids) {
        return Err(SelectedForeignInstallError::Validation(
            "selected Python policy pins unavailable artifact identities".into(),
        ));
    }
    let wheel_ids = bundle.wheel_identities().map(ToString::to_string).collect();
    admit_installed_module_paths(
        &policy.python,
        &mut configured,
        &wheel_ids,
        bundle.artifacts.len(),
    )?;
    Ok(configured)
}

pub(super) fn admit_installed_module_paths(
    policy: &runmat_config::runtime::PythonConfig,
    configured: &mut PythonRuntimeConfiguration,
    wheel_ids: &BTreeSet<String>,
    installed_root_count: usize,
) -> Result<(), SelectedForeignInstallError> {
    match &policy.installed_module_paths {
        None if configured.module_paths.is_empty() => {}
        Some(source) if &source.artifact_identities == wheel_ids && installed_root_count != 0 => {
            if source.installed_path_indices.is_empty() {
                return Err(SelectedForeignInstallError::Validation(
                    "selected Python installed-root certificate lacks indexed path provenance; recapture the source policy".into(),
                ));
            }
            if source.installed_path_indices.len() != installed_root_count {
                return Err(SelectedForeignInstallError::Validation(
                    "selected Python installed-root count differs from authenticated bundle".into(),
                ));
            }
            let mut previous = None;
            for &index in &source.installed_path_indices {
                let index = usize::try_from(index).map_err(|_| {
                    SelectedForeignInstallError::Validation(
                        "selected Python installed-root index exceeds host address space".into(),
                    )
                })?;
                if index >= configured.module_paths.len()
                    || previous.is_some_and(|previous| index <= previous)
                {
                    return Err(SelectedForeignInstallError::Validation(
                        "selected Python installed-root indices must be strictly increasing and in bounds".into(),
                    ));
                }
                previous = Some(index);
            }
            if configured.module_paths.len() != source.installed_path_indices.len() {
                return Err(SelectedForeignInstallError::Validation(
                    "selected Python policy contains explicit module paths without authenticated materialization".into(),
                ));
            }
            configured.module_paths.clear();
        }
        _ => {
            return Err(SelectedForeignInstallError::Validation(
                "selected Python module paths lack matching installed-bundle provenance".into(),
            ))
        }
    }
    Ok(())
}
