use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::document::{migrate_legacy_desktop_config, RunmatConfigDocument, RunmatConfigFormat};
use crate::runtime::RunMatRuntimeConfig;

/// Source of a leaf value. Paths in `sources` use the canonical document spelling,
/// e.g. `runtime.jit.enabled`; arrays are single replacement-valued fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "layer", rename_all = "snake_case")]
pub enum ConfigSource {
    BuiltIn,
    User { path: Option<PathBuf> },
    Project { path: Option<PathBuf> },
    Explicit { path: PathBuf },
    Environment { variable: String },
    CommandLine { argument: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedRuntimeConfig {
    pub config: RunMatRuntimeConfig,
    pub sources: BTreeMap<String, ConfigSource>,
}

impl ResolvedRuntimeConfig {
    pub fn source(&self, field: &str) -> Option<&ConfigSource> {
        self.sources.get(field)
    }

    /// Unambiguous lookup for dictionary keys containing dots or backslashes.
    pub fn source_segments(&self, field: &[&str]) -> Option<&ConfigSource> {
        self.sources.get(
            &field
                .iter()
                .map(|part| escape_segment(part))
                .collect::<Vec<_>>()
                .join("."),
        )
    }
}

#[derive(Debug, Clone, Default)]
pub enum UserSettingsSource {
    #[default]
    NativeDefault,
    Disabled,
    /// Missing host settings files are treated as an empty user layer.
    File(PathBuf),
    /// Canonical JSON or TOML document supplied by a filesystem-free host.
    Document {
        source: String,
        format: RunmatConfigFormat,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ConfigurationPolicy {
    #[default]
    Local,
    /// Only built-ins plus the explicitly selected/discovered project document.
    /// No user settings, RUNMAT_CONFIG, or process environment are consulted.
    ClusterWorker,
}

/// Filesystem-free, serializable settings boundary shared by browser hosts and SDKs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct HostRuntimeSettings {
    pub user_settings_json: Option<String>,
    pub project: Option<HostConfigDocument>,
    pub policy: ConfigurationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostConfigDocument {
    pub source: String,
    pub format: RunmatConfigFormat,
}

impl HostRuntimeSettings {
    pub fn resolve(&self) -> Result<ResolvedRuntimeConfig> {
        resolve(&ConfigLoadOptions {
            user: self
                .user_settings_json
                .as_ref()
                .map(|source| UserSettingsSource::Document {
                    source: source.clone(),
                    format: RunmatConfigFormat::Json,
                })
                .unwrap_or(UserSettingsSource::Disabled),
            project_document: self
                .project
                .as_ref()
                .map(|project| (project.source.clone(), project.format)),
            policy: self.policy,
            ..ConfigLoadOptions::default()
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConfigLoadOptions {
    pub user: UserSettingsSource,
    /// Project directory to search. None disables project discovery (browser hosts).
    pub project_directory: Option<PathBuf>,
    /// Already-read project document, for browser/VFS hosts. Takes precedence over discovery.
    pub project_document: Option<(String, RunmatConfigFormat)>,
    pub policy: ConfigurationPolicy,
    /// Hosts can disable process environment while retaining local user layering.
    pub environment: bool,
    /// Host-selected explicit file (e.g. CLI --config), above RUNMAT_CONFIG
    /// and below individual environment overrides. Ignored by cluster policy.
    pub explicit_file: Option<PathBuf>,
}

impl ConfigLoadOptions {
    pub fn native() -> Self {
        Self {
            project_directory: std::env::current_dir().ok(),
            environment: true,
            ..Self::default()
        }
    }

    pub fn cluster_worker(project_directory: PathBuf) -> Self {
        Self {
            project_directory: Some(project_directory),
            policy: ConfigurationPolicy::ClusterWorker,
            ..Self::default()
        }
    }
}

pub(super) fn resolve(options: &ConfigLoadOptions) -> Result<ResolvedRuntimeConfig> {
    let mut value = canonical_value(&RunMatRuntimeConfig::default())?;
    let mut sources = BTreeMap::new();
    record_sources(&value, "runtime", &ConfigSource::BuiltIn, &mut sources);
    if options.policy == ConfigurationPolicy::Local {
        match &options.user {
            UserSettingsSource::Disabled => {}
            UserSettingsSource::Document { source, format } => {
                let layer = user_runtime_value(source, *format)?;
                merge(
                    &mut value,
                    layer,
                    "runtime",
                    &ConfigSource::User { path: None },
                    &mut sources,
                );
            }
            UserSettingsSource::File(path) => apply_user_file(&mut value, &mut sources, path)?,
            UserSettingsSource::NativeDefault => {
                let explicit_user = options
                    .environment
                    .then(|| std::env::var_os("RUNMAT_USER_SETTINGS"))
                    .flatten()
                    .map(PathBuf::from);
                if let Some(path) = explicit_user.or_else(|| {
                    super::discovery::user_config_candidates()
                        .into_iter()
                        .find(|p| p.is_file())
                }) {
                    apply_user_file(&mut value, &mut sources, &path)?;
                }
            }
        }
    }
    if let Some((source, format)) = &options.project_document {
        let layer = runtime_value(source, *format)?;
        merge(
            &mut value,
            layer,
            "runtime",
            &ConfigSource::Project { path: None },
            &mut sources,
        );
    } else if let Some(path) = options
        .project_directory
        .as_ref()
        .and_then(|p| super::discovery::discover_project_config_path_from(p))
    {
        let layer = read_runtime(&path)?;
        merge(
            &mut value,
            layer,
            "runtime",
            &ConfigSource::Project { path: Some(path) },
            &mut sources,
        );
    }
    if options.policy == ConfigurationPolicy::Local && options.environment {
        if let Some(path) = std::env::var_os("RUNMAT_CONFIG") {
            let path = PathBuf::from(path);
            if !path.is_file() {
                bail!(
                    "RUNMAT_CONFIG points to a {}: {}",
                    if path.is_dir() {
                        "directory, expected a file"
                    } else {
                        "missing file"
                    },
                    path.display()
                );
            }
            let layer = read_runtime(&path)?;
            merge(
                &mut value,
                layer,
                "runtime",
                &ConfigSource::Explicit { path },
                &mut sources,
            );
        }
    }
    if options.policy == ConfigurationPolicy::Local {
        if let Some(path) = &options.explicit_file {
            let layer = read_runtime(path)?;
            merge(
                &mut value,
                layer,
                "runtime",
                &ConfigSource::Explicit { path: path.clone() },
                &mut sources,
            );
        }
        if options.environment {
            for (variable, layer) in super::env::environment_layers()? {
                merge(
                    &mut value,
                    layer,
                    "runtime",
                    &ConfigSource::Environment { variable },
                    &mut sources,
                );
            }
        }
    }
    let config = decode_runtime(value)?;
    // Keep only effective leaf paths: replacing an optional object must not leave
    // provenance for the old null parent or for displaced descendants.
    let mut effective = BTreeMap::new();
    record_sources(
        &canonical_value(&config)?,
        "runtime",
        &ConfigSource::BuiltIn,
        &mut effective,
    );
    sources.retain(|path, _| effective.contains_key(path));
    Ok(ResolvedRuntimeConfig { config, sources })
}

fn apply_user_file(
    value: &mut Value,
    sources: &mut BTreeMap<String, ConfigSource>,
    path: &Path,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading user settings {}", path.display()))?;
    let layer = user_runtime_value(&source, RunmatConfigFormat::from_path(path)?)?;
    merge(
        value,
        layer,
        "runtime",
        &ConfigSource::User {
            path: Some(path.to_owned()),
        },
        sources,
    );
    Ok(())
}

fn read_runtime(path: &Path) -> Result<Value> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading configuration {}", path.display()))?;
    runtime_value(&source, RunmatConfigFormat::from_path(path)?)
}

pub(super) fn parse_value(source: &str, format: RunmatConfigFormat) -> Result<Value> {
    let value = match format {
        RunmatConfigFormat::Json => serde_json::from_str(source)?,
        RunmatConfigFormat::Toml => serde_json::to_value(toml::from_str::<toml::Value>(source)?)?,
    };
    if !value.is_object() {
        bail!("settings document must be an object");
    }
    Ok(value)
}

pub(super) fn runtime_value(source: &str, format: RunmatConfigFormat) -> Result<Value> {
    let migrated = migrate_legacy_desktop_config(source, format)?;
    // Preserve canonical strict validation for project and explicit files.
    RunmatConfigDocument::parse(migrated.source.clone(), format)?;
    Ok(parse_value(&migrated.source, format)?
        .get("runtime")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new())))
}

/// User files are forward compatible. Unknown values stay in the source document
/// but do not become runtime policy until this runtime understands their schema.
pub(super) fn user_runtime_value(source: &str, format: RunmatConfigFormat) -> Result<Value> {
    let document = parse_value(source, format)?;
    let mut runtime = document
        .get("runtime")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if !runtime.is_object() {
        bail!("runtime must be an object");
    }
    retain_user_fields(&mut runtime)?;
    // Validate only after defaults and all fields in this user layer are combined.
    let mut effective = canonical_value(&RunMatRuntimeConfig::default())?;
    merge(
        &mut effective,
        runtime.clone(),
        "runtime",
        &ConfigSource::BuiltIn,
        &mut BTreeMap::new(),
    );
    decode_runtime(effective)?;
    Ok(runtime)
}

pub(super) fn retain_user_fields(runtime: &mut Value) -> Result<()> {
    let mut schema = canonical_value(&RunMatRuntimeConfig::default())?;
    schema["foreign"]["python"]["exact_version"] =
        serde_json::json!({"major":0,"minor":0,"patch":0});
    schema["foreign"]["python"]["installed_module_paths"] =
        serde_json::json!({"artifact_identities":[],"installed_path_indices":[]});
    schema["foreign"]["native"]["timeout"] = serde_json::json!({"secs":0,"nanos":0});
    retain_known(runtime, &schema, "");
    Ok(())
}

fn retain_known(value: &mut Value, schema: &Value, path: &str) {
    if let (Some(object), Some(known)) = (value.as_object_mut(), schema.as_object()) {
        // This table is an explicitly open string dictionary, not schema fields.
        if path == "logging.otel.resource_attributes" {
            return;
        }
        object.retain(|key, child| {
            if let Some(template) = known.get(key) {
                let path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                retain_known(child, template, &path);
                true
            } else {
                false
            }
        });
    }
}

pub(super) fn canonical_value(config: &RunMatRuntimeConfig) -> Result<Value> {
    let mut value = serde_json::to_value(config)?;
    let execution = value
        .as_object_mut()
        .expect("runtime serializes as object")
        .remove("runtime")
        .expect("execution fields");
    for (key, child) in execution
        .as_object()
        .expect("execution serializes as object")
    {
        value
            .as_object_mut()
            .expect("runtime object")
            .insert(key.clone(), child.clone());
    }
    Ok(value)
}

pub(super) fn decode_runtime(mut value: Value) -> Result<RunMatRuntimeConfig> {
    let object = value.as_object_mut().context("runtime must be an object")?;
    let mut execution = Map::new();
    for field in ["callstack_limit", "error_namespace", "verbose"] {
        if let Some(value) = object.remove(field) {
            execution.insert(field.into(), value);
        }
    }
    object.insert("runtime".into(), Value::Object(execution));
    Ok(serde_json::from_value(value)?)
}

pub(super) fn merge(
    target: &mut Value,
    layer: Value,
    path: &str,
    source: &ConfigSource,
    sources: &mut BTreeMap<String, ConfigSource>,
) {
    if let (Some(target), Value::Object(layer)) = (target.as_object_mut(), &layer) {
        for (key, value) in layer {
            let path = format!("{path}.{}", escape_segment(key));
            merge(
                target.entry(key.clone()).or_insert(Value::Null),
                value.clone(),
                &path,
                source,
                sources,
            );
        }
    } else {
        sources.retain(|key, _| key != path && !key.starts_with(&format!("{path}.")));
        *target = layer;
        record_sources(target, path, source, sources);
    }
}

fn record_sources(
    value: &Value,
    path: &str,
    source: &ConfigSource,
    sources: &mut BTreeMap<String, ConfigSource>,
) {
    if let Value::Object(object) = value {
        if object.is_empty() {
            sources.insert(path.into(), source.clone());
        }
        for (key, value) in object {
            record_sources(
                value,
                &format!("{path}.{}", escape_segment(key)),
                source,
                sources,
            );
        }
    } else {
        sources.insert(path.into(), source.clone());
    }
}

fn escape_segment(segment: &str) -> String {
    segment.replace('\\', "\\\\").replace('.', "\\.")
}
