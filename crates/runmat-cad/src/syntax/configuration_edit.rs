//! Structured edits of a part's configurations, written back as the smallest text change: a
//! feature parameter in a scope of configurations, suppression in one configuration, a new
//! configuration, and a configuration's own part number and properties.

use super::edit::{edit_part, ConfigurationScope, EditFailure, SourceChange};
use crate::ids::{ConfigurationName, FeatureId};
use crate::model::configuration::{remove_patch_value, set_patch_value, Configuration};
use crate::model::{PartDocument, PropertyValue};
use serde_json::Value;

fn model(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

fn configuration_mut<'a>(
    document: &'a mut PartDocument,
    name: &str,
) -> Result<&'a mut Configuration, EditFailure> {
    document
        .configurations
        .iter_mut()
        .find(|c| c.name.as_str() == name)
        .ok_or_else(|| model(format!("the part has no configuration `{name}`")))
}

/// The configurations a scope names (`this`, `specify`); `None` for `all`. The default
/// configuration is the part as written, so `this` on it is `all`.
fn targets(scope: &ConfigurationScope) -> Option<Vec<String>> {
    match scope {
        ConfigurationScope::All => None,
        ConfigurationScope::This { configuration } if configuration == "default" => None,
        ConfigurationScope::This { configuration } => Some(vec![configuration.clone()]),
        ConfigurationScope::Specify { configurations } => Some(
            configurations
                .iter()
                .filter(|c| c.as_str() != "default")
                .cloned()
                .collect(),
        ),
    }
}

/// Sets the field at `path` of feature `feature`'s definition (`["extent", "distance"]`) to
/// `value` (the field's JSON, as the definition serializes): in the feature itself for `all`,
/// which clears every configuration's override of that field, or as an override in the named
/// configurations.
pub fn set_feature_value(
    source: &str,
    path: &str,
    feature: &str,
    field: &[&str],
    value: Value,
    scope: &ConfigurationScope,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let id = FeatureId::new(feature);
        let current = document
            .feature(&id)
            .ok_or_else(|| model(format!("the part has no feature `{feature}`")))?
            .clone();
        match targets(scope) {
            None => {
                let mut kind = match serde_json::to_value(&current.kind) {
                    Ok(Value::Object(map)) => map,
                    _ => return Err(model(format!("`{feature}` has no fields to set"))),
                };
                set_patch_value(&mut kind, field, value);
                let kind = serde_json::from_value(Value::Object(kind)).map_err(|e| {
                    model(format!(
                        "`{feature}` cannot take `{}`: {e}",
                        field.join(".")
                    ))
                })?;
                document.feature_mut(feature).expect("found above").kind = kind;
                for c in &mut document.configurations {
                    if let Some(patch) = c.features.get_mut(&id) {
                        if remove_patch_value(patch, field) {
                            c.features.remove(&id);
                        }
                    }
                }
            }
            Some(names) => {
                for name in names {
                    let c = configuration_mut(document, &name)?;
                    let patch = c.features.entry(id.clone()).or_default();
                    set_patch_value(patch, field, value.clone());
                    // The patch has to apply: a field the definition lacks is refused here.
                    let (_, problems) = c.apply_features(std::slice::from_ref(&current));
                    if let Some(problem) = problems.first() {
                        return Err(model(problem.message.clone()));
                    }
                }
            }
        }
        Ok(())
    })
}

/// Suppresses (`on`) or restores features: in the part itself for the default configuration,
/// else in configuration `configuration` only.
pub fn suppress_features(
    source: &str,
    path: &str,
    features: &[String],
    configuration: Option<&str>,
    on: bool,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        for id in features {
            if document.feature(&FeatureId::new(id)).is_none() {
                return Err(model(format!("the part has no feature `{id}`")));
            }
        }
        match configuration.filter(|c| *c != "default") {
            None => {
                for id in features {
                    document.feature_mut(id).expect("checked").suppressed = on;
                }
            }
            Some(name) => {
                let c = configuration_mut(document, name)?;
                for id in features {
                    if on {
                        c.suppress.insert(FeatureId::new(id));
                    } else {
                        c.suppress.remove(&FeatureId::new(id));
                    }
                }
            }
        }
        Ok(())
    })
}

/// Adds an empty configuration `name`, a copy of the part as written.
pub fn add_configuration(
    source: &str,
    path: &str,
    name: &str,
) -> Result<SourceChange, EditFailure> {
    if !runmat_design_syntax::json::is_identifier(name) || name == "default" {
        return Err(model(format!(
            "`{name}` can't name a configuration: use letters, digits and `_`, not `default`"
        )));
    }
    edit_part(source, path, |document| {
        let name = ConfigurationName::new(name);
        if document.configuration(&name).is_some() {
            return Err(model(format!(
                "the part already has a configuration `{name}`"
            )));
        }
        document.configurations.push(Configuration {
            name,
            ..Configuration::default()
        });
        Ok(())
    })
}

/// Sets configuration `configuration`'s own part number (`None` clears it, so the part's
/// number applies).
pub fn set_configuration_part_number(
    source: &str,
    path: &str,
    configuration: &str,
    part_number: Option<&str>,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        configuration_mut(document, configuration)?.part_number =
            part_number.filter(|p| !p.is_empty()).map(str::to_owned);
        Ok(())
    })
}

/// Sets (or, with `None`, clears) one of configuration `configuration`'s own properties
/// (`description`, `finish` …).
pub fn set_configuration_property(
    source: &str,
    path: &str,
    configuration: &str,
    property: &str,
    value: Option<PropertyValue>,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let c = configuration_mut(document, configuration)?;
        match value {
            Some(v) => {
                c.properties.insert(property.to_owned(), v);
            }
            None => {
                c.properties.remove(property);
            }
        }
        Ok(())
    })
}
