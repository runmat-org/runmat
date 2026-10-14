//! The Config-owned foreign schema at the authenticated Artifact boundary.

use runmat_config::runtime::ForeignConfig;
use runmat_execution_artifact::{
    capture_typed_foreign_policy, decode_typed_foreign_policy, LogicalObject,
};

/// Issue one bounded canonical policy object from the existing Config schema.
pub fn capture_foreign_policy_object(config: &ForeignConfig) -> Result<LogicalObject, String> {
    capture_typed_foreign_policy(config).map_err(|error| error.to_string())
}

/// Validate Artifact framing and losslessly decode the admitted Config policy.
pub fn decode_admitted_policy(object: &LogicalObject) -> Result<ForeignConfig, String> {
    decode_typed_foreign_policy(object).map_err(|error| error.to_string())
}

// Captured policies are projected only at their owning adapter boundary.
pub(crate) fn native_policy(
    config: &ForeignConfig,
) -> runmat_runtime::foreign::NativeFfiIsolationPolicy {
    runmat_runtime::foreign::NativeFfiIsolationPolicy {
        isolate: config.native.isolation == runmat_config::runtime::NativeFfiIsolation::Process,
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
    }
}

pub(crate) fn mex_policy(config: &ForeignConfig) -> runmat_runtime::foreign::MexIsolationPolicy {
    runmat_runtime::foreign::MexIsolationPolicy {
        unmanifested: match config.mex.unmanifested {
            runmat_config::runtime::UnmanifestedMexPolicy::Isolate => {
                runmat_runtime::foreign::UnmanifestedMexPolicy::Isolate
            }
            runmat_config::runtime::UnmanifestedMexPolicy::Deny => {
                runmat_runtime::foreign::UnmanifestedMexPolicy::Deny
            }
        },
        invocation_timeout: config
            .mex
            .timeout_ms
            .map(|value| std::time::Duration::from_millis(value.get())),
    }
}

pub(crate) fn java_runtime(
    config: &ForeignConfig,
) -> runmat_runtime::foreign::JavaRuntimeConfiguration {
    runmat_runtime::foreign::JavaRuntimeConfiguration {
        home: config.java.home.clone(),
        minimum_version: config.java.minimum_version.get(),
        maximum_version: config.java.maximum_version.map(|value| value.get()),
        classpath: config.java.classpath.clone(),
        options: config.java.options.clone(),
    }
}

pub(super) fn python_runtime(
    config: &runmat_config::runtime::PythonConfig,
) -> Result<runmat_runtime::foreign::PythonRuntimeConfiguration, String> {
    let version = |value: &str| {
        super::python_artifacts::parse_version(value).map_err(|error| error.to_string())
    };
    Ok(runmat_runtime::foreign::PythonRuntimeConfiguration {
        executable: config.executable.clone(),
        version: config.version.as_deref().map(version).transpose()?,
        minimum_version: version(&config.minimum_version)?,
        maximum_version: config.maximum_version.as_deref().map(version).transpose()?,
        exact_version: config
            .exact_version
            .map(|version| runmat_python::PythonVersion {
                major: version.major,
                minor: version.minor,
                patch: version.patch,
            }),
        required_abi_tag: config.required_abi_tag.clone(),
        required_platform_tag: config.required_platform_tag.clone(),
        execution_mode: match config.execution_mode {
            runmat_config::runtime::PythonExecutionModeConfig::InProcess => {
                runmat_python::PythonExecutionMode::InProcess
            }
            runmat_config::runtime::PythonExecutionModeConfig::OutOfProcess => {
                runmat_python::PythonExecutionMode::OutOfProcess
            }
        },
        module_paths: config.module_paths.clone(),
        artifact_identities: config.artifact_identities.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_canonical_json_cannot_smuggle_unknown_config_fields() {
        let policy = capture_foreign_policy_object(&ForeignConfig::default()).unwrap();
        assert!(decode_admitted_policy(&policy).is_ok());
        let mut value: serde_json::Value = serde_json::from_slice(&policy.bytes).unwrap();
        value["native"]["unexpected"] = serde_json::json!(true);
        let unknown =
            runmat_execution_artifact::foreign_policy_object_from_json_value(value).unwrap();
        assert!(decode_admitted_policy(&unknown).is_err());
    }

    #[test]
    fn selected_python_certificate_preserves_explicit_paths_and_rejects_legacy_indices() {
        use runmat_config::runtime::PythonInstalledModulePaths;
        use std::collections::BTreeSet;
        use std::path::PathBuf;

        let identities = BTreeSet::from(["wheel-id".to_owned()]);
        let mut policy = runmat_config::runtime::PythonConfig {
            module_paths: vec![PathBuf::from("explicit"), PathBuf::from("installed")],
            installed_module_paths: Some(PythonInstalledModulePaths {
                artifact_identities: identities.clone(),
                installed_path_indices: vec![1],
            }),
            ..Default::default()
        };
        let mut runtime = python_runtime(&policy).unwrap();
        let error = super::super::policy_admission::admit_installed_module_paths(
            &policy,
            &mut runtime,
            &identities,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("explicit module paths"));
        assert_eq!(runtime.module_paths, policy.module_paths);
        assert_eq!(policy.module_paths.len(), 2);

        policy.module_paths.remove(0);
        let source = policy.installed_module_paths.as_mut().unwrap();
        source.installed_path_indices = vec![0];
        let mut runtime = python_runtime(&policy).unwrap();
        super::super::policy_admission::admit_installed_module_paths(
            &policy,
            &mut runtime,
            &identities,
            1,
        )
        .unwrap();
        assert!(runtime.module_paths.is_empty());

        let source = policy.installed_module_paths.as_mut().unwrap();
        source.installed_path_indices.clear();
        let mut runtime = python_runtime(&policy).unwrap();
        let error = super::super::policy_admission::admit_installed_module_paths(
            &policy,
            &mut runtime,
            &identities,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("recapture"));
        assert_eq!(runtime.module_paths, policy.module_paths);

        policy
            .installed_module_paths
            .as_mut()
            .unwrap()
            .installed_path_indices = vec![1];
        let mut runtime = python_runtime(&policy).unwrap();
        assert!(
            super::super::policy_admission::admit_installed_module_paths(
                &policy,
                &mut runtime,
                &identities,
                1,
            )
            .is_err()
        );
        assert_eq!(runtime.module_paths, policy.module_paths);
    }
}
