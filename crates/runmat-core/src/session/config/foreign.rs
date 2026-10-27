use super::*;

#[cfg(not(target_arch = "wasm32"))]
fn python_config_from_runtime(
    configuration: &runmat_runtime::foreign::PythonRuntimeConfiguration,
) -> runmat_config::runtime::PythonConfig {
    use runmat_config::runtime::{PythonConfig, PythonExactVersion, PythonExecutionModeConfig};
    PythonConfig {
        executable: configuration.executable.clone(),
        version: configuration
            .version
            .map(|(major, minor)| format!("{major}.{minor}")),
        minimum_version: format!(
            "{}.{}",
            configuration.minimum_version.0, configuration.minimum_version.1
        ),
        maximum_version: configuration
            .maximum_version
            .map(|(major, minor)| format!("{major}.{minor}")),
        execution_mode: match configuration.execution_mode {
            runmat_python::PythonExecutionMode::InProcess => PythonExecutionModeConfig::InProcess,
            runmat_python::PythonExecutionMode::OutOfProcess => {
                PythonExecutionModeConfig::OutOfProcess
            }
        },
        exact_version: configuration
            .exact_version
            .map(|version| PythonExactVersion {
                major: version.major,
                minor: version.minor,
                patch: version.patch,
            }),
        required_abi_tag: configuration.required_abi_tag.clone(),
        required_platform_tag: configuration.required_platform_tag.clone(),
        module_paths: configuration.module_paths.clone(),
        installed_module_paths: None,
        artifact_identities: configuration.artifact_identities.clone(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn python_config_from_adapter(
    adapter: &runmat_runtime::foreign::PythonAdapter,
) -> runmat_config::runtime::PythonConfig {
    let mut config = python_config_from_runtime(&adapter.configuration());
    if let Some(installed) = adapter.installed_path_provenance() {
        config.installed_module_paths = certified_installed_module_paths(&config, &installed);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn certified_installed_module_paths(
    config: &runmat_config::runtime::PythonConfig,
    installed: &runmat_runtime::foreign::PythonInstalledPathProvenance<'_>,
) -> Option<runmat_config::runtime::PythonInstalledModulePaths> {
    let mut effective = config.module_paths.iter().enumerate();
    let installed_path_indices = installed
        .module_paths()
        .iter()
        .map(|root| {
            effective
                .find(|(_, path)| *path == root)
                .and_then(|(index, _)| u64::try_from(index).ok())
        })
        .collect::<Option<Vec<_>>>()?;
    Some(runmat_config::runtime::PythonInstalledModulePaths {
        artifact_identities: installed
            .wheel_identities()
            .iter()
            .map(ToString::to_string)
            .collect(),
        installed_path_indices,
    })
}

impl RunMatSession {
    #[cfg(not(target_arch = "wasm32"))]
    fn update_selected_foreign_config(
        &mut self,
        update: impl FnOnce(&mut runmat_config::runtime::ForeignConfig),
    ) {
        let mut config = self
            .selected_foreign_config
            .take()
            .expect("native sessions capture their initial foreign policy");
        update(&mut Arc::make_mut(&mut config).foreign);
        self.set_selected_foreign_config(config);
    }

    /// Reconcile successful adapter-local pyenv/artifact mutations before a
    /// selected policy is frozen. The adapter's typed snapshot retains pins
    /// that cannot be represented by pyenv's two name-value options alone.
    #[cfg(not(target_arch = "wasm32"))]
    pub(in crate::session) fn refresh_selected_python_config(&mut self) {
        let effective = python_config_from_adapter(&self.python_adapter);
        let changed = self
            .selected_foreign_config
            .as_ref()
            .is_some_and(|config| config.foreign.python != effective);
        if changed {
            self.update_selected_foreign_config(|foreign| foreign.python = effective);
        }
    }

    /// Apply the resolved foreign policy before installing project artifacts.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_foreign_config(
        &mut self,
        config: &runmat_config::runtime::ForeignConfig,
    ) -> Result<(), RuntimeError> {
        self.set_mex_config(&config.mex);
        self.set_native_ffi_config(&config.native)?;
        self.set_java_config(&config.java)?;
        self.set_python_config(&config.python)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_mex_config(&mut self, config: &runmat_config::runtime::MexConfig) {
        let unmanifested = match config.unmanifested {
            runmat_config::runtime::UnmanifestedMexPolicy::Isolate => {
                runmat_runtime::foreign::UnmanifestedMexPolicy::Isolate
            }
            runmat_config::runtime::UnmanifestedMexPolicy::Deny => {
                runmat_runtime::foreign::UnmanifestedMexPolicy::Deny
            }
        };
        self.mex_runtime
            .set_isolation_policy(runmat_runtime::foreign::MexIsolationPolicy {
                unmanifested,
                invocation_timeout: config
                    .timeout_ms
                    .map(|milliseconds| std::time::Duration::from_millis(milliseconds.get())),
            });
        self.update_selected_foreign_config(|foreign| foreign.mex = config.clone());
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_native_ffi_config(
        &mut self,
        config: &runmat_config::runtime::NativeFfiConfig,
    ) -> Result<(), RuntimeError> {
        if config.timeout.is_some() && config.timeout_ms.is_some() {
            return Err(build_runtime_error(
                "native timeout and timeout_ms are mutually exclusive",
            )
            .with_identifier("RunMat:NativeFfi:InvalidTimeout")
            .build());
        }
        let timeout = config
            .timeout
            .map(|timeout| timeout.duration())
            .or_else(|| {
                config
                    .timeout_ms
                    .map(|milliseconds| std::time::Duration::from_millis(milliseconds.get()))
            });
        let policy = match config.isolation {
            runmat_config::runtime::NativeFfiIsolation::Process => {
                runmat_runtime::foreign::NativeFfiIsolationPolicy::isolated(timeout)
            }
            runmat_config::runtime::NativeFfiIsolation::InProcess => {
                runmat_runtime::foreign::NativeFfiIsolationPolicy {
                    isolate: false,
                    invocation_timeout: timeout,
                }
            }
        };
        self.native_ffi_adapter.set_isolation_policy(policy)?;
        self.update_selected_foreign_config(|foreign| foreign.native = config.clone());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_native_ffi_isolation_policy(
        &mut self,
        policy: runmat_runtime::foreign::NativeFfiIsolationPolicy,
    ) -> Result<(), RuntimeError> {
        if policy
            .invocation_timeout
            .is_some_and(|timeout| timeout.is_zero())
        {
            return Err(build_runtime_error("native timeout must be nonzero")
                .with_identifier("RunMat:NativeFfi:InvalidTimeout")
                .build());
        }
        self.native_ffi_adapter.set_isolation_policy(policy)?;
        let timeout = policy
            .invocation_timeout
            .and_then(runmat_config::runtime::NativeFfiTimeout::new);
        self.update_selected_foreign_config(|foreign| {
            foreign.native = runmat_config::runtime::NativeFfiConfig {
                isolation: if policy.isolate {
                    runmat_config::runtime::NativeFfiIsolation::Process
                } else {
                    runmat_config::runtime::NativeFfiIsolation::InProcess
                },
                timeout_ms: None,
                timeout,
            }
        });
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_java_config(
        &mut self,
        config: &runmat_config::runtime::JavaConfig,
    ) -> Result<(), RuntimeError> {
        self.java_adapter
            .configure(runmat_runtime::foreign::JavaRuntimeConfiguration {
                home: config.home.clone(),
                minimum_version: config.minimum_version.get(),
                maximum_version: config.maximum_version.map(|value| value.get()),
                classpath: config.classpath.clone(),
                options: config.options.clone(),
            })?;
        self.update_selected_foreign_config(|foreign| foreign.java = config.clone());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn configure_python_runtime(
        &mut self,
        configuration: runmat_runtime::foreign::PythonRuntimeConfiguration,
    ) -> Result<(), RuntimeError> {
        self.python_adapter.configure(configuration)?;
        let captured = python_config_from_adapter(&self.python_adapter);
        self.update_selected_foreign_config(|foreign| foreign.python = captured);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_python_config(
        &mut self,
        config: &runmat_config::runtime::PythonConfig,
    ) -> Result<(), RuntimeError> {
        fn version(value: &str, label: &str) -> Result<(u16, u16), RuntimeError> {
            let mut parts = value.split('.');
            let major = parts.next().and_then(|value| value.parse().ok());
            let minor = parts.next().and_then(|value| value.parse().ok());
            if parts.next().is_some() || major.is_none() || minor.is_none() {
                return Err(
                    build_runtime_error(format!("{label} must use major.minor form"))
                        .with_builtin("python")
                        .with_identifier("RunMat:Python:InvalidVersion")
                        .build(),
                );
            }
            Ok((major.expect("checked"), minor.expect("checked")))
        }
        let minimum_version = version(&config.minimum_version, "Python minimum_version")?;
        let maximum_version = config
            .maximum_version
            .as_deref()
            .map(|value| version(value, "Python maximum_version"))
            .transpose()?;
        let required_version = config
            .version
            .as_deref()
            .map(|value| version(value, "Python version"))
            .transpose()?;
        let configuration = runmat_runtime::foreign::PythonRuntimeConfiguration {
            executable: config.executable.clone(),
            version: required_version,
            minimum_version,
            maximum_version,
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
        };
        self.python_adapter.configure(configuration)?;
        let mut captured = config.clone();
        captured.installed_module_paths = self
            .python_adapter
            .installed_path_provenance()
            .and_then(|installed| certified_installed_module_paths(&captured, &installed));
        self.update_selected_foreign_config(|foreign| foreign.python = captured);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_desktop_host_available(&mut self, available: bool) {
        self.java_adapter.set_desktop_available(available);
    }
}
