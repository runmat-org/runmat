//! Single documentation export owner for the CLI, build consumers and tooling.
use crate::{
    builtin_catalog_aliases, builtin_catalog_entries, builtin_constant_catalog_entries,
    public_builtin_catalog_entries, validate_builtin_documentation, BuiltinCatalogAudience,
    BuiltinDocumentationAuthority,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

mod directory;
mod legacy;
mod links;
mod projection;
#[cfg(test)]
mod tests;
#[cfg(test)]
use crate::{BuiltinConstantCatalogEntry, BuiltinDocumentation};
use legacy::annotate_legacy_document;
#[cfg(test)]
use legacy::{legacy_presentation_example_id, no_example_requirements};
use projection::{catalog_document, constant_document, validate_catalog_documentation};
pub const EXPORT_SCHEMA_VERSION: u32 = 2;
/// Explicit documentation ownership policy shared by the CLI and consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportMode {
    /// Preserve remaining authoritative sidecars; missing legacy owners may
    /// be reported during transition, but duplicate canonical owners reject.
    Normal { transition: bool },
    /// Review-only selected preview, never a final accepted consumer export.
    CatalogPreview,
}

/// Export from the explicitly supplied authoritative documentation directory.
///
/// Directory absence is an error, including in transition mode: consumers
/// must not silently replace missing package assets with incomplete docs.
pub fn export_from_directory(
    legacy_directory: &Path,
    mode: ExportMode,
    identities: &BTreeSet<String>,
) -> Result<Value, Box<dyn std::error::Error>> {
    build_export_with_legacy(
        directory::read_legacy_documents(legacy_directory)?,
        mode,
        identities,
    )
}

/// Project editor prose without asserting executable-example qualification.
///
/// Ownership, identity, link and slug validation is identical to the strict
/// export. Rows and their fixture metadata are retained, not repaired or
/// omitted. The extra envelope marker prevents use as qualification evidence.
pub fn export_presentation_from_directory(
    legacy_directory: &Path,
    transition: bool,
) -> Result<Value, Box<dyn std::error::Error>> {
    project_export(
        directory::read_legacy_documents(legacy_directory)?,
        ExportMode::Normal { transition },
        &BTreeSet::new(),
        ValidationPurpose::PresentationOnly,
    )
}

/// Project the complete canonical catalog for package-contained consumers.
///
/// This is an explicit source choice, not a fallback when directory assets are
/// unavailable. All strict ownership/identity/link checks and the presentation
/// marker remain; executable-example qualification is deliberately not claimed.
pub fn export_canonical_presentation() -> Result<Value, Box<dyn std::error::Error>> {
    project_export(
        BTreeMap::new(),
        ExportMode::Normal { transition: false },
        &BTreeSet::new(),
        ValidationPurpose::PresentationOnly,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ValidationPurpose {
    Qualified,
    PresentationOnly,
}

fn build_export_with_legacy(
    legacy: BTreeMap<String, Value>,
    mode: ExportMode,
    identities: &BTreeSet<String>,
) -> Result<Value, Box<dyn std::error::Error>> {
    project_export(legacy, mode, identities, ValidationPurpose::Qualified)
}

fn project_export(
    mut legacy: BTreeMap<String, Value>,
    mode: ExportMode,
    identities: &BTreeSet<String>,
    purpose: ValidationPurpose,
) -> Result<Value, Box<dyn std::error::Error>> {
    if mode == ExportMode::CatalogPreview && identities.is_empty() {
        return Err("catalog preview requires selected identities".into());
    }
    for entry in builtin_catalog_entries()
        .iter()
        .filter(|entry| entry.audience == BuiltinCatalogAudience::Fixture)
    {
        if legacy.contains_key(&entry.identity.name.to_ascii_lowercase()) {
            return Err(format!(
                "fixture {} must not have a public legacy documentation sidecar",
                entry.identity.name
            )
            .into());
        }
    }
    let mut catalog_entries = public_builtin_catalog_entries().to_vec();
    catalog_entries.sort_by(|left, right| {
        left.identity
            .name
            .to_ascii_lowercase()
            .cmp(&right.identity.name.to_ascii_lowercase())
    });

    let catalog_names = catalog_entries
        .iter()
        .map(|entry| entry.identity.name.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let complete_names = catalog_names
        .iter()
        .cloned()
        .chain(
            builtin_constant_catalog_entries()
                .iter()
                .map(|entry| entry.name.to_ascii_lowercase()),
        )
        .chain(legacy.keys().cloned())
        .collect::<BTreeSet<_>>();
    if let Some(name) = identities.difference(&complete_names).next() {
        return Err(format!("unknown documentation identity: {name}").into());
    }
    if mode == ExportMode::CatalogPreview {
        let canonical_names = catalog_entries
            .iter()
            .filter(|entry| entry.documentation.authority == BuiltinDocumentationAuthority::Catalog)
            .map(|entry| entry.identity.name.to_ascii_lowercase())
            .chain(
                builtin_constant_catalog_entries()
                    .iter()
                    .filter(|entry| entry.documentation.is_some())
                    .map(|entry| entry.name.to_ascii_lowercase()),
            )
            .collect::<BTreeSet<_>>();
        if let Some(name) = identities.difference(&canonical_names).next() {
            return Err(format!("catalog preview requires canonical documentation: {name}").into());
        }
    }
    // A scoped export remains strict about duplicate ownership and missing
    // documentation within its scope. Links resolve against the full catalog.
    // Unrelated unfinished cutovers cannot block field-conservation checks.
    if !identities.is_empty() {
        legacy.retain(|name, _| identities.contains(name));
        catalog_entries
            .retain(|entry| identities.contains(&entry.identity.name.to_ascii_lowercase()));
    }
    let legacy_sidecar_count = legacy.len();
    let selected_catalog_count = catalog_entries.len();
    // Report the complete selected link frontier before encoding documents.
    // A scoped export still resolves names against the full inventory.
    links::validate_scope(
        catalog_entries
            .iter()
            .filter(|entry| entry.documentation.authority == BuiltinDocumentationAuthority::Catalog)
            .map(|entry| (entry.identity.name, &entry.documentation))
            .chain(
                builtin_constant_catalog_entries()
                    .iter()
                    .filter_map(|entry| {
                        if !identities.is_empty()
                            && !identities.contains(&entry.name.to_ascii_lowercase())
                        {
                            return None;
                        }
                        entry
                            .documentation
                            .as_ref()
                            .map(|documentation| (entry.name, documentation))
                    }),
            ),
        &complete_names,
    )?;
    let mut documents = BTreeMap::new();
    let mut slugs = BTreeMap::new();
    let mut missing_legacy = Vec::new();

    for entry in catalog_entries {
        let key = entry.identity.name.to_ascii_lowercase();
        if !crate::catalog::valid_builtin_identity(entry.identity.name) {
            return Err(format!(
                "{} has invalid builtin identity syntax",
                entry.identity.name
            )
            .into());
        }
        match entry.documentation.authority {
            BuiltinDocumentationAuthority::Catalog => {
                if legacy.contains_key(&key) && mode != ExportMode::CatalogPreview {
                    return Err(format!(
                        "{} has canonical catalog documentation and a legacy sidecar",
                        entry.identity.name
                    )
                    .into());
                }
                if purpose == ValidationPurpose::Qualified {
                    validate_catalog_documentation(entry)?;
                }
                let document = catalog_document(entry);
                record_document(&mut documents, &mut slugs, key, document)?;
                if mode == ExportMode::CatalogPreview {
                    legacy.remove(&entry.identity.name.to_ascii_lowercase());
                }
            }
            BuiltinDocumentationAuthority::LegacySidecar => {
                let Some(mut document) = legacy.remove(&key) else {
                    missing_legacy.push(entry.identity.name);
                    continue;
                };
                annotate_legacy_document(&key, &mut document)?;
                record_document(&mut documents, &mut slugs, key, document)?;
            }
        }
    }

    if !missing_legacy.is_empty() && mode == (ExportMode::Normal { transition: false }) {
        return Err(format!(
            "catalog identities declare legacy documentation but have no sidecar: {}",
            missing_legacy.join(", ")
        )
        .into());
    }

    for entry in builtin_constant_catalog_entries() {
        if !identities.is_empty() && !identities.contains(&entry.name.to_ascii_lowercase()) {
            continue;
        }
        let Some(documentation) = &entry.documentation else {
            continue;
        };
        let key = entry.name.to_ascii_lowercase();
        if catalog_names.contains(&key)
            || builtin_catalog_aliases()
                .iter()
                .any(|alias| alias.alias.name.eq_ignore_ascii_case(entry.name))
        {
            return Err(format!(
                "{} declares separate constant and callable documentation",
                entry.name
            )
            .into());
        }
        if legacy.contains_key(&key) && mode != ExportMode::CatalogPreview {
            return Err(format!(
                "{} has canonical constant documentation and a legacy sidecar",
                entry.name
            )
            .into());
        }
        if purpose == ValidationPurpose::Qualified {
            let errors = validate_builtin_documentation(entry.name, documentation);
            if !errors.is_empty() {
                return Err(format!(
                    "{} has invalid canonical constant documentation: {errors:#?}",
                    entry.name
                )
                .into());
            }
        }
        record_document(
            &mut documents,
            &mut slugs,
            key,
            constant_document(entry, documentation),
        )?;
        if mode == ExportMode::CatalogPreview {
            legacy.remove(&entry.name.to_ascii_lowercase());
        }
    }

    for (key, mut document) in legacy {
        annotate_legacy_document(&key, &mut document)?;
        record_document(&mut documents, &mut slugs, key, document)?;
    }

    let mut export = json!({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "inventory": {
            "documents": documents.len(),
            "catalog_identities": selected_catalog_count,
            "legacy_sidecars": legacy_sidecar_count,
            "missing_catalog_documentation": missing_legacy,
        },
        "builtins": documents.into_values().collect::<Vec<_>>(),
    });
    if mode == ExportMode::CatalogPreview {
        export["preview"] = json!({
            "kind": "catalog",
            "final_acceptance": false,
            "sidecars_still_present": legacy_sidecar_count,
        });
    }
    if purpose == ValidationPurpose::PresentationOnly {
        export["presentation"] = json!({
            "kind": "editor_prose",
            "executable_qualification": false,
        });
    }
    Ok(export)
}

fn record_document(
    documents: &mut BTreeMap<String, Value>,
    slugs: &mut BTreeMap<String, String>,
    key: String,
    document: Value,
) -> Result<(), String> {
    let slug = document
        .get("slug")
        .and_then(Value::as_str)
        .or_else(|| document.get("title").and_then(Value::as_str))
        .unwrap_or(&key)
        .to_ascii_lowercase();
    if let Some(existing) = slugs.insert(slug.clone(), key.clone()) {
        return Err(format!(
            "duplicate builtin documentation slug {slug:?}: {existing} and {key}"
        ));
    }
    if documents.insert(key.clone(), document).is_some() {
        return Err(format!("duplicate builtin documentation identity: {key}"));
    }
    Ok(())
}
