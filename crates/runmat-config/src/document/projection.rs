use super::{RunmatConfigDocumentError, RunmatConfigFormat};
use crate::desktop::DesktopConfig;
use crate::runtime::{
    AccelerateConfig, FeaConfig, GcConfig, JitConfig, LanguageConfig, LoggingConfig,
    PlottingConfig, RunMatRuntimeConfig, TelemetryConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Default)]
pub(super) struct ConfigProjection {
    #[serde(default)]
    pub desktop: DesktopConfig,
    #[serde(default)]
    runtime: RuntimeSection,
    #[serde(flatten)]
    _other: BTreeMap<String, serde::de::IgnoredAny>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct RuntimeSection {
    callstack_limit: Option<usize>,
    error_namespace: Option<String>,
    verbose: Option<bool>,
    language: Option<LanguageConfig>,
    logging: Option<LoggingConfig>,
    telemetry: Option<TelemetryConfig>,
    jit: Option<JitConfig>,
    gc: Option<GcConfig>,
    accelerate: Option<AccelerateConfig>,
    plotting: Option<PlottingConfig>,
    fea: Option<FeaConfig>,
}

impl From<&RunMatRuntimeConfig> for RuntimeSection {
    fn from(value: &RunMatRuntimeConfig) -> Self {
        Self {
            callstack_limit: Some(value.runtime.callstack_limit),
            error_namespace: Some(value.runtime.error_namespace.clone()),
            verbose: Some(value.runtime.verbose),
            language: Some(value.language.clone()),
            logging: Some(value.logging.clone()),
            telemetry: Some(value.telemetry.clone()),
            jit: Some(value.jit.clone()),
            gc: Some(value.gc.clone()),
            accelerate: Some(value.accelerate.clone()),
            plotting: Some(value.plotting.clone()),
            fea: Some(value.fea.clone()),
        }
    }
}

impl RuntimeSection {
    fn resolve(self) -> RunMatRuntimeConfig {
        let mut config = RunMatRuntimeConfig::default();
        if let Some(value) = self.callstack_limit {
            config.runtime.callstack_limit = value;
        }
        if let Some(value) = self.error_namespace {
            config.runtime.error_namespace = value;
        }
        if let Some(value) = self.verbose {
            config.runtime.verbose = value;
        }
        if let Some(value) = self.language {
            config.language = value;
        }
        if let Some(value) = self.logging {
            config.logging = value;
        }
        if let Some(value) = self.telemetry {
            config.telemetry = value;
        }
        if let Some(value) = self.jit {
            config.jit = value;
        }
        if let Some(value) = self.gc {
            config.gc = value;
        }
        if let Some(value) = self.accelerate {
            config.accelerate = value;
        }
        if let Some(value) = self.plotting {
            config.plotting = value;
        }
        if let Some(value) = self.fea {
            config.fea = value;
        }
        config
    }
}

pub(super) struct ResolvedProjection {
    pub desktop: DesktopConfig,
    pub runtime: RunMatRuntimeConfig,
}

pub(super) fn parse_projection(
    source: &str,
    format: RunmatConfigFormat,
) -> Result<ResolvedProjection, RunmatConfigDocumentError> {
    let projection: ConfigProjection = match format {
        RunmatConfigFormat::Toml => toml_edit::de::from_str(source)
            .map_err(|error| RunmatConfigDocumentError::TomlParse(error.to_string()))?,
        RunmatConfigFormat::Json => serde_json::from_str(source)?,
    };
    Ok(ResolvedProjection {
        desktop: projection.desktop,
        runtime: projection.runtime.resolve(),
    })
}
