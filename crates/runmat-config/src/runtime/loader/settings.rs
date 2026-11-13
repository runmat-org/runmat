use anyhow::{bail, Context, Result};
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::{self, OpenOptions};
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use super::layers::{
    canonical_value, decode_runtime, merge, parse_value, retain_user_fields, user_runtime_value,
    ConfigSource,
};
use crate::document::RunmatConfigFormat;
use crate::runtime::RunMatRuntimeConfig;

/// Loss-preserving user-layer editor. It never saves an effective project or
/// environment configuration back into user preferences. Unknown user fields
/// survive edits; TOML comments and untouched formatting survive as well.
#[derive(Debug, Clone)]
pub struct UserSettingsDocument {
    source: String,
    format: RunmatConfigFormat,
    origin: Option<PathBuf>,
    original_source: Option<String>,
}

impl UserSettingsDocument {
    pub fn parse(source: impl Into<String>, format: RunmatConfigFormat) -> Result<Self> {
        let source = source.into();
        user_runtime_value(&source, format)?;
        Ok(Self {
            source,
            format,
            origin: None,
            original_source: None,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let format = RunmatConfigFormat::from_path(path)?;
        let original_source = match std::fs::read_to_string(path) {
            Ok(source) => Some(source),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading user settings {}", path.display()))
            }
        };
        let source = original_source.clone().unwrap_or_else(|| match format {
            RunmatConfigFormat::Toml => String::new(),
            RunmatConfigFormat::Json => "{}\n".into(),
        });
        let mut document = Self::parse(source, format)?;
        document.origin = Some(path.to_owned());
        document.original_source = original_source;
        Ok(document)
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn format(&self) -> RunmatConfigFormat {
        self.format
    }

    /// User values over built-ins, never project/environment values.
    pub fn runtime(&self) -> Result<RunMatRuntimeConfig> {
        let mut value = canonical_value(&RunMatRuntimeConfig::default())?;
        merge(
            &mut value,
            user_runtime_value(&self.source, self.format)?,
            "runtime",
            &ConfigSource::BuiltIn,
            &mut Default::default(),
        );
        decode_runtime(value)
    }

    /// Edit strongly typed runtime fields. Only changed fields are persisted;
    /// unrelated fields and unknown settings are never reconstructed.
    pub fn edit(&mut self, edit: impl FnOnce(&mut RunMatRuntimeConfig)) -> Result<()> {
        let mut runtime = self.runtime()?;
        let before = canonical_value(&runtime)?;
        edit(&mut runtime);
        let after = canonical_value(&runtime)?;
        // Validate public types that can have cross-field invariants.
        decode_runtime(after.clone())?;
        let mut changes = Vec::new();
        changed_fields(&before, &after, &mut vec!["runtime".into()], &mut changes);
        self.apply_changes(changes)
    }

    /// Explicitly set a canonical field even when it equals its built-in default.
    /// `path` is relative to runtime, e.g. ["jit", "enabled"]. Segment arrays
    /// also permit literal dotted resource-attribute keys without ambiguity.
    pub fn set<T: serde::Serialize>(&mut self, path: &[&str], value: T) -> Result<()> {
        if path.is_empty() || path.iter().any(|part| part.is_empty()) {
            bail!("empty user setting path");
        }
        let new = serde_json::to_value(value)?;
        let mut layer = new.clone();
        for segment in path.iter().rev() {
            layer = serde_json::json!({*segment: layer});
        }
        let mut retained = layer.clone();
        retain_user_fields(&mut retained)?;
        if retained != layer {
            bail!("unknown user setting path");
        }
        let mut effective = canonical_value(&self.runtime()?)?;
        merge(
            &mut effective,
            layer,
            "runtime",
            &ConfigSource::BuiltIn,
            &mut Default::default(),
        );
        decode_runtime(effective)?;
        let mut full_path = vec!["runtime".into()];
        full_path.extend(path.iter().map(|s| (*s).to_owned()));
        let mut changes = Vec::new();
        explicit_fields(new, &mut full_path, &mut changes);
        self.apply_changes(changes)
    }

    /// Remove an override to inherit lower-precedence settings. Does not write
    /// a default value and does not discard any sibling field.
    pub fn reset(&mut self, path: &[&str]) -> Result<()> {
        if path.is_empty() || path.iter().any(|part| part.is_empty()) {
            bail!("empty user setting path");
        }
        let mut full_path = vec!["runtime".into()];
        full_path.extend(path.iter().map(|s| (*s).to_owned()));
        self.apply_changes(vec![(full_path, None)])
    }

    fn apply_changes(&mut self, changes: Vec<(Vec<String>, Option<Value>)>) -> Result<()> {
        let updated = match self.format {
            RunmatConfigFormat::Json => {
                let mut value = parse_value(&self.source, self.format)?;
                for (path, new) in changes {
                    update_json(&mut value, &path, new)?;
                }
                format!("{}\n", serde_json::to_string_pretty(&value)?)
            }
            RunmatConfigFormat::Toml => {
                let mut document = toml_edit::DocumentMut::from_str(&self.source)?;
                for (path, new) in changes {
                    let item = if new.as_ref().is_none_or(Value::is_null) {
                        if new == Some(Value::Null) {
                            let defaults = serde_json::json!({"runtime":canonical_value(&RunMatRuntimeConfig::default())?});
                            let inherited =
                                path.iter().try_fold(&defaults, |value, key| value.get(key));
                            if inherited.is_some_and(|value| !value.is_null()) {
                                bail!(
                                    "TOML cannot represent a null override for {}; use JSON",
                                    path.join(".")
                                );
                            }
                        }
                        None
                    } else {
                        let encoded = toml_edit::ser::to_document(
                            &serde_json::json!({"value":new.expect("non-null setting")}),
                        )?;
                        Some(encoded["value"].clone())
                    };
                    update_toml(document.as_item_mut(), &path, item)?;
                }
                document.to_string()
            }
        };
        user_runtime_value(&updated, self.format)?;
        self.source = updated;
        Ok(())
    }

    /// Save in the original format, rejecting concurrent external edits rather
    /// than silently overwriting them. A JSON browser host uses source() instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&mut self, path: &Path) -> Result<()> {
        if RunmatConfigFormat::from_path(path)? != self.format {
            bail!("user settings format does not match destination");
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let mut lock_name = path
            .file_name()
            .context("user settings destination must be a file")?
            .to_os_string();
        lock_name.push(".lock");
        // Share the preference service's application-document lock: neither
        // writer may publish a stale snapshot over the other's fields.
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.with_file_name(lock_name))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                bail!("user settings are being updated; retry after reloading")
            }
            Err(fs::TryLockError::Error(error)) => return Err(error.into()),
        }
        if self.origin.as_deref() == Some(path) {
            let current = match std::fs::read_to_string(path) {
                Ok(source) => Some(source),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.into()),
            };
            if current != self.original_source {
                bail!("user settings changed on disk; reload before saving");
            }
        }
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        match fs::metadata(path) {
            Ok(metadata) => temporary
                .as_file()
                .set_permissions(metadata.permissions())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        temporary.write_all(self.source.as_bytes())?;
        temporary.flush()?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .with_context(|| format!("saving user settings {}", path.display()))?;
        #[cfg(unix)]
        fs::File::open(parent)?
            .sync_all()
            .context("user settings saved but directory durability could not be confirmed")?;
        self.origin = Some(path.to_owned());
        self.original_source = Some(self.source.clone());
        Ok(())
    }
}

fn changed_fields(
    before: &Value,
    after: &Value,
    path: &mut Vec<String>,
    changes: &mut Vec<(Vec<String>, Option<Value>)>,
) {
    if before == after {
        return;
    }
    if let (Value::Object(before), Value::Object(after)) = (before, after) {
        for (key, value) in after {
            path.push(key.clone());
            changed_fields(
                before.get(key).unwrap_or(&Value::Null),
                value,
                path,
                changes,
            );
            path.pop();
        }
        for key in before.keys().filter(|key| !after.contains_key(*key)) {
            let mut removed = path.clone();
            removed.push(key.clone());
            changes.push((removed, None));
        }
    } else {
        changes.push((path.clone(), Some(after.clone())));
    }
}

fn explicit_fields(
    value: Value,
    path: &mut Vec<String>,
    changes: &mut Vec<(Vec<String>, Option<Value>)>,
) {
    if let Value::Object(object) = value {
        for (key, value) in object {
            path.push(key);
            explicit_fields(value, path, changes);
            path.pop();
        }
    } else {
        changes.push((path.clone(), Some(value)));
    }
}

fn update_json(target: &mut Value, path: &[String], new: Option<Value>) -> Result<()> {
    let (key, rest) = path.split_first().context("empty setting path")?;
    let object = target
        .as_object_mut()
        .context("user setting parent is not an object")?;
    if rest.is_empty() {
        if let Some(new) = new {
            object.insert(key.clone(), new);
        } else {
            object.remove(key);
        }
    } else if new.is_some() || object.contains_key(key) {
        update_json(
            object
                .entry(key.clone())
                .or_insert_with(|| serde_json::json!({})),
            rest,
            new,
        )?;
    }
    Ok(())
}

fn update_toml(
    target: &mut toml_edit::Item,
    path: &[String],
    new: Option<toml_edit::Item>,
) -> Result<()> {
    let (key, rest) = path.split_first().context("empty setting path")?;
    let inline = target
        .as_value()
        .is_some_and(|value| value.is_inline_table());
    let table = target
        .as_table_like_mut()
        .context("user setting parent is not a table")?;
    if rest.is_empty() {
        if let Some(mut new) = new {
            if let (Some(old), Some(value)) = (
                table.get(key).and_then(|v| v.as_value()),
                new.as_value_mut(),
            ) {
                *value.decor_mut() = old.decor().clone();
            }
            if let Some(existing) = table.get_mut(key) {
                // TableLike::insert reformats an existing key and discards its
                // leading comments. Replace only its item instead.
                *existing = new;
            } else {
                table.insert(key, new);
            }
        } else {
            table.remove(key);
        }
    } else {
        if table.get(key).is_none() {
            if new.is_none() {
                return Ok(());
            }
            table.insert(
                key,
                if inline {
                    toml_edit::Item::Value(toml_edit::Value::InlineTable(
                        toml_edit::InlineTable::new(),
                    ))
                } else {
                    toml_edit::Item::Table(toml_edit::Table::new())
                },
            );
        }
        update_toml(
            table.get_mut(key).context("missing settings parent")?,
            rest,
            new,
        )?;
    }
    Ok(())
}
