use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories_next::UserDirs;
use serde::Deserialize;
use tracing::debug;

use crate::cli::CliArgs;

const DEFAULT_ARTIFACTS_DIR: &str = "artifacts/runmatfunc";
const DEFAULT_DOCS_DIR: &str = "docs/generated";
const DEFAULT_GENERATION_PLAN: &str = "crates/runmat-runtime/generation-plan-2.md";
const DEFAULT_FUSION_DESIGN: &str = "docs/fusion-runtime-design.md";

/// Fully materialised application configuration after merging defaults, config files,
/// and environment overrides.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub default_model: Option<String>,
    pub artifacts_dir: PathBuf,
    pub docs_output_dir: PathBuf,
    pub snippet_includes: Vec<String>,
    pub snippet_excludes: Vec<String>,
    pub generation_plan: Option<PathBuf>,
    pub fusion_design_doc: Option<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            artifacts_dir: PathBuf::from(DEFAULT_ARTIFACTS_DIR),
            docs_output_dir: PathBuf::from(DEFAULT_DOCS_DIR),
            snippet_includes: vec![
                "crates/runmat-runtime/src/builtins/**/*.rs".to_string(),
                "crates/runmat-runtime/src/**/tests/**/*.rs".to_string(),
            ],
            snippet_excludes: vec![
                "crates/runmat-runtime/src/legacy/**".to_string(),
                "crates/runmat-runtime/src/old_builtins/**".to_string(),
            ],
            generation_plan: Some(PathBuf::from(DEFAULT_GENERATION_PLAN)),
            fusion_design_doc: Some(PathBuf::from(DEFAULT_FUSION_DESIGN)),
        }
    }
}

impl AppConfig {
    pub fn load(args: &CliArgs) -> Result<Self> {
        let mut config = Self::default();

        if let Some(path) = args.config_path.as_ref() {
            let raw = Self::read_raw_config(path)?;
            config.merge_raw(raw);
        } else if let Some(path) = std::env::var_os("RUNMATFUNC_CONFIG") {
            let path = PathBuf::from(path);
            if let Some(raw) = Self::maybe_read_raw_config(&path)? {
                config.merge_raw(raw);
            }
        } else if let Some(path) = default_config_path() {
            if let Some(raw) = Self::maybe_read_raw_config(&path)? {
                config.merge_raw(raw);
            }
        }

        config.apply_env_overrides();
        config.deduplicate_patterns();

        Ok(config)
    }

    pub fn docs_output_dir(&self) -> &Path {
        &self.docs_output_dir
    }

    pub fn artifacts_dir(&self) -> &Path {
        &self.artifacts_dir
    }

    pub fn generation_plan_path(&self) -> Option<&Path> {
        self.generation_plan.as_deref()
    }

    pub fn fusion_design_doc(&self) -> Option<&Path> {
        self.fusion_design_doc.as_deref()
    }

    pub fn queue_file_path(&self) -> PathBuf {
        self.artifacts_dir.join("queue.json")
    }

    pub fn transcripts_dir(&self) -> PathBuf {
        self.artifacts_dir.join("transcripts")
    }

    fn merge_raw(&mut self, raw: RawAppConfig) {
        if let Some(model) = raw.default_model {
            self.default_model = Some(model);
        }
        if let Some(dir) = raw.artifacts_dir {
            self.artifacts_dir = dir;
        }
        if let Some(dir) = raw.docs_output_dir {
            self.docs_output_dir = dir;
        }
        if let Some(mut includes) = raw.snippet_includes {
            self.merge_pattern_list(&mut includes, true);
        }
        if let Some(mut excludes) = raw.snippet_excludes {
            self.merge_pattern_list(&mut excludes, false);
        }
        if let Some(path) = raw.generation_plan {
            self.generation_plan = Some(path);
        }
        if let Some(path) = raw.fusion_design_doc {
            self.fusion_design_doc = Some(path);
        }
    }

    fn merge_pattern_list(&mut self, patterns: &mut Vec<String>, includes: bool) {
        patterns.retain(|p| !p.trim().is_empty());
        if includes {
            for pattern in patterns.drain(..) {
                if !self.snippet_includes.contains(&pattern) {
                    self.snippet_includes.push(pattern);
                }
            }
        } else {
            for pattern in patterns.drain(..) {
                if !self.snippet_excludes.contains(&pattern) {
                    self.snippet_excludes.push(pattern);
                }
            }
        }
    }

    fn deduplicate_patterns(&mut self) {
        dedup_vec(&mut self.snippet_includes);
        dedup_vec(&mut self.snippet_excludes);
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(model) = std::env::var("RUNMATFUNC_DEFAULT_MODEL") {
            if !model.trim().is_empty() {
                self.default_model = Some(model);
            }
        }
        if let Ok(dir) = std::env::var("RUNMATFUNC_ARTIFACTS_DIR") {
            if !dir.trim().is_empty() {
                self.artifacts_dir = PathBuf::from(dir);
            }
        }
        if let Ok(dir) = std::env::var("RUNMATFUNC_DOCS_OUTPUT_DIR") {
            if !dir.trim().is_empty() {
                self.docs_output_dir = PathBuf::from(dir);
            }
        }
        if let Ok(plan) = std::env::var("RUNMATFUNC_GENERATION_PLAN") {
            if !plan.trim().is_empty() {
                self.generation_plan = Some(PathBuf::from(plan));
            }
        }
        if let Ok(doc) = std::env::var("RUNMATFUNC_FUSION_DESIGN_DOC") {
            if !doc.trim().is_empty() {
                self.fusion_design_doc = Some(PathBuf::from(doc));
            }
        }
        if let Ok(extra) = std::env::var("RUNMATFUNC_SNIPPET_INCLUDE") {
            let mut parsed = parse_list_env(&extra);
            self.merge_pattern_list(&mut parsed, true);
        }
        if let Ok(extra) = std::env::var("RUNMATFUNC_SNIPPET_EXCLUDE") {
            let mut parsed = parse_list_env(&extra);
            self.merge_pattern_list(&mut parsed, false);
        }
    }

    fn read_raw_config(path: &Path) -> Result<RawAppConfig> {
        debug!("loading config from {}", path.display());
        let content = fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let raw: RawAppConfig = toml::from_str(&content)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(raw)
    }

    fn maybe_read_raw_config(path: &Path) -> Result<Option<RawAppConfig>> {
        if path.exists() {
            Self::read_raw_config(path).map(Some)
        } else {
            debug!("config file {} not found; using defaults", path.display());
            Ok(None)
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct RawAppConfig {
    pub default_model: Option<String>,
    pub artifacts_dir: Option<PathBuf>,
    pub docs_output_dir: Option<PathBuf>,
    pub snippet_includes: Option<Vec<String>>,
    pub snippet_excludes: Option<Vec<String>>,
    pub generation_plan: Option<PathBuf>,
    pub fusion_design_doc: Option<PathBuf>,
}

fn default_config_path() -> Option<PathBuf> {
    UserDirs::new().map(|dirs| dirs.home_dir().join(".runmatfunc").join("config.toml"))
}

fn parse_list_env(value: &str) -> Vec<String> {
    value
        .split(|c| matches!(c, ';' | ',' | ' '))
        .filter_map(|item| {
            let trimmed = item.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect()
}

fn dedup_vec(vec: &mut Vec<String>) {
    let mut seen = Vec::new();
    vec.retain(|item| {
        if seen.contains(item) {
            false
        } else {
            seen.push(item.clone());
            true
        }
    });
}
