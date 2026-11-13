//! Portable, project-only execution policy. Collector/product telemetry and
//! resolved process credentials are intentionally not part of this wire schema.
use super::*;
use crate::document::RunmatConfigFormat;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const CLUSTER_RUNTIME_POLICY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterRuntimePolicy {
    pub schema_version: u16,
    pub runtime: RuntimeConfig,
    pub accelerate: AccelerateConfig,
    pub language: LanguageConfig,
    pub foreign: ForeignConfig,
    pub jit: JitConfig,
    pub gc: GcConfig,
    pub plotting: PlottingConfig,
    pub fea: FeaConfig,
}

impl ClusterRuntimePolicy {
    pub fn from_project_document(source: &str, format: RunmatConfigFormat) -> Result<Self> {
        let config = HostRuntimeSettings {
            project: Some(HostConfigDocument {
                source: source.into(),
                format,
            }),
            policy: ConfigurationPolicy::ClusterWorker,
            ..Default::default()
        }
        .resolve()?
        .config;
        let policy = Self::from_project_config(config);
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_project_file(path: &std::path::Path) -> Result<Self> {
        Self::from_project_document(
            &std::fs::read_to_string(path)?,
            RunmatConfigFormat::from_path(path)?,
        )
    }

    /// Built-ins are captured explicitly so a worker never fills settings from
    /// its own home or ambient environment. Do not pass a local effective config.
    pub fn built_in() -> Self {
        Self::from_project_config(RunMatRuntimeConfig::default())
    }

    fn from_project_config(config: RunMatRuntimeConfig) -> Self {
        Self {
            schema_version: CLUSTER_RUNTIME_POLICY_SCHEMA_VERSION,
            runtime: config.runtime,
            accelerate: config.accelerate,
            language: config.language,
            foreign: config.foreign,
            jit: config.jit,
            gc: config.gc,
            plotting: config.plotting,
            fea: config.fea,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CLUSTER_RUNTIME_POLICY_SCHEMA_VERSION {
            bail!("unsupported cluster runtime policy schema");
        }
        if self.jit.threshold == 0 {
            bail!("JIT threshold must be positive");
        }
        if self.gc.threads == Some(0) || self.gc.young_size_mb == Some(0) {
            bail!("GC threads and young-generation size must be positive");
        }
        // Reapply typed cross-field validation even for directly constructed Rust values.
        let _: Self = serde_json::from_value(serde_json::to_value(self)?)?;
        Ok(())
    }

    pub fn runtime_config(&self) -> Result<RunMatRuntimeConfig> {
        self.validate()?;
        let mut config = RunMatRuntimeConfig {
            runtime: self.runtime.clone(),
            accelerate: self.accelerate.clone(),
            language: self.language.clone(),
            foreign: self.foreign.clone(),
            jit: self.jit.clone(),
            gc: self.gc.clone(),
            plotting: self.plotting.clone(),
            fea: self.fea.clone(),
            ..Default::default()
        };
        // Task payloads never turn on a collector or product telemetry. Worker
        // admin/process configuration owns these independently of task policy.
        config.telemetry.enabled = false;
        Ok(config)
    }
}
