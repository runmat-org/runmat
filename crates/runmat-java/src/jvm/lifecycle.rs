use std::sync::{Arc, Mutex, OnceLock};

use jni::{InitArgsBuilder, JNIVersion, JavaVM};

use super::{JvmConfig, JvmError, JvmInstallation};

#[derive(Debug)]
struct ProcessJvmState {
    installation: JvmInstallation,
    launch_options: Vec<String>,
    vm: Arc<JavaVM>,
}

fn process_state() -> &'static Mutex<Option<Arc<ProcessJvmState>>> {
    static STATE: OnceLock<Mutex<Option<Arc<ProcessJvmState>>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

#[derive(Debug, Clone)]
pub struct JvmProcess {
    state: Arc<ProcessJvmState>,
}

impl JvmProcess {
    pub fn launch(installation: JvmInstallation, config: &JvmConfig) -> Result<Self, JvmError> {
        config.validate()?;
        if !config.accepts(&installation.version) {
            let maximum = config
                .maximum_major
                .map(|value| value.to_string())
                .unwrap_or_else(|| "newest available".into());
            return Err(JvmError::UnsupportedVersion {
                found: installation.version.raw.clone(),
                required: format!("{} through {maximum}", config.minimum_major),
            });
        }
        let launch_options = config.launch_options();
        let mut state = process_state()
            .lock()
            .map_err(|_| JvmError::Startup("process JVM lock was poisoned".into()))?;
        if let Some(existing) = state.as_ref() {
            if existing.installation.library != installation.library
                || existing.launch_options != launch_options
            {
                return Err(JvmError::ConfigurationConflict(
                    "JVM library and bootstrap options are immutable after startup".into(),
                ));
            }
            return Ok(Self {
                state: Arc::clone(existing),
            });
        }

        let mut builder = InitArgsBuilder::new().version(JNIVersion::V8);
        for option in &launch_options {
            builder = builder.option(option);
        }
        let arguments = builder
            .build()
            .map_err(|error| JvmError::InvalidConfiguration(error.to_string()))?;
        let library = installation.library.clone();
        let vm = JavaVM::with_libjvm(arguments, || Ok(library.as_os_str()))
            .map_err(|error| JvmError::Startup(error.to_string()))?;
        let process = Arc::new(ProcessJvmState {
            installation,
            launch_options,
            vm: Arc::new(vm),
        });
        *state = Some(Arc::clone(&process));
        Ok(Self { state: process })
    }

    pub fn installation(&self) -> &JvmInstallation {
        &self.state.installation
    }

    pub fn with_attached<R, E>(
        &self,
        operation: impl FnOnce(&mut jni::JNIEnv<'_>) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<JvmError>,
    {
        let mut environment = self
            .state
            .vm
            .attach_current_thread()
            .map_err(|error| JvmError::ThreadAttachment(error.to_string()))
            .map_err(E::from)?;
        operation(&mut environment)
    }
}
