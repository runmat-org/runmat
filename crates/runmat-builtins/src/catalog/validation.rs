use super::{
    builtin_catalog_aliases, BuiltinBindingAvailability, BuiltinCatalogEntry,
    BuiltinDocumentationAuthority, BuiltinExampleVerification,
};
use crate::BuiltinAsyncBehavior;
use runmat_types::EffectKind;
use std::collections::{BTreeMap, BTreeSet};

mod example;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinCatalogValidationError {
    pub identity: Option<&'static str>,
    pub message: String,
}

pub fn validate_builtin_catalog(
    entries: &[&'static BuiltinCatalogEntry],
) -> Vec<BuiltinCatalogValidationError> {
    validate_builtin_catalog_with_aliases(entries, &[])
}

pub fn validate_complete_builtin_catalog() -> Vec<BuiltinCatalogValidationError> {
    let entries = super::builtin_catalog_entries();
    let aliases = builtin_catalog_aliases();
    let mut errors = validate_builtin_catalog_with_aliases(entries, aliases);
    let primary_names = entries
        .iter()
        .map(|entry| entry.identity.name.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let legacy_names = crate::builtin_functions()
        .into_iter()
        .map(|function| function.name.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    for alias in aliases {
        let spelling = alias.alias.name.to_ascii_lowercase();
        if legacy_names.contains(&spelling) && !primary_names.contains(&spelling) {
            errors.push(error(
                Some(alias.alias.name),
                "builtin alias collides with a legacy runtime callable",
            ));
        }
    }
    errors
}

pub fn validate_builtin_catalog_with_aliases(
    entries: &[&'static BuiltinCatalogEntry],
    aliases: &[&'static super::BuiltinCatalogAlias],
) -> Vec<BuiltinCatalogValidationError> {
    let mut errors = Vec::new();
    let mut identities = BTreeSet::new();
    let mut bindings = BTreeMap::new();
    for entry in entries {
        let name = entry.identity.name;
        if !valid_builtin_identity(name) {
            errors.push(error(Some(name), "builtin identity has invalid syntax"));
            continue;
        }
        if !identities.insert(name.to_ascii_lowercase()) {
            errors.push(error(Some(name), "duplicate builtin catalog identity"));
        }
        if entry.category.is_empty() {
            errors.push(error(Some(name), "category must not be empty"));
        }
        if entry.documentation.summary.is_empty() {
            errors.push(error(Some(name), "documentation summary must not be empty"));
        }
        validate_documentation(entry, &mut errors);
        if entry.bindings.is_empty() {
            errors.push(error(
                Some(name),
                "catalog entry has no runtime binding declaration",
            ));
        }
        let declares_suspension = entry.contract.effects.contains(&EffectKind::MaySuspend);
        if matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ) == declares_suspension
        {
            errors.push(error(
                Some(name),
                "async behavior and MaySuspend effect disagree",
            ));
        }
        if !entry
            .contract
            .effects
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            errors.push(error(
                Some(name),
                "effect declarations must be sorted and unique",
            ));
        }
        if !entry
            .contract
            .capabilities
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            errors.push(error(
                Some(name),
                "capability declarations must be sorted and unique",
            ));
        }
        for binding in entry.bindings {
            if binding.variant.is_empty() {
                errors.push(error(Some(name), "binding variant must not be empty"));
            }
            if bindings
                .insert(entry.binding_identity(binding), name)
                .is_some()
            {
                errors.push(error(Some(name), "duplicate builtin binding identity"));
            }
            if matches!(binding.availability, BuiltinBindingAvailability::Required)
                && matches!(
                    entry.link.reachability,
                    super::BuiltinReachability::Feature(_)
                )
            {
                errors.push(error(
                    Some(name),
                    "feature-gated entry declares an unconditional required binding",
                ));
            }
        }
    }
    let primary_names = entries
        .iter()
        .map(|entry| {
            (
                entry.identity.name.to_ascii_lowercase(),
                entry.identity.name,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut public_names = primary_names.keys().cloned().collect::<BTreeSet<_>>();
    for alias in aliases {
        let spelling = alias.alias.name;
        let target = alias.canonical.name;
        if !valid_builtin_identity(spelling) || !valid_builtin_identity(target) {
            errors.push(error(
                Some(spelling),
                "builtin alias identity has invalid syntax",
            ));
            continue;
        }
        if spelling.eq_ignore_ascii_case(target) {
            errors.push(error(Some(spelling), "builtin alias cannot target itself"));
        }
        let canonical_target = primary_names.get(&target.to_ascii_lowercase());
        if canonical_target.is_none() {
            errors.push(error(
                Some(spelling),
                "builtin alias target is not a canonical catalog identity",
            ));
        } else if canonical_target.copied() != Some(target) {
            errors.push(error(
                Some(spelling),
                "builtin alias target must use the exact canonical spelling",
            ));
        }
        if !public_names.insert(spelling.to_ascii_lowercase()) {
            errors.push(error(Some(spelling), "duplicate public builtin spelling"));
        }
    }
    errors
}

fn valid_builtin_identity(value: &str) -> bool {
    let mut characters = value.chars();
    match (characters.next(), characters.next()) {
        (Some(first), second) if first.is_ascii_alphabetic() => second
            .into_iter()
            .chain(characters)
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '.')),
        (Some('_'), Some('_')) => characters
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '.')),
        _ => false,
    }
}

fn validate_documentation(
    entry: &'static BuiltinCatalogEntry,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    let name = entry.identity.name;
    let documentation = &entry.documentation;
    if documentation.authority != BuiltinDocumentationAuthority::Catalog {
        return;
    }
    if documentation.description.trim().is_empty() {
        errors.push(error(
            Some(name),
            "canonical documentation description must not be empty",
        ));
    }
    if documentation.keywords.is_empty() {
        errors.push(error(
            Some(name),
            "canonical documentation keywords must not be empty",
        ));
    }
    if documentation.examples.is_empty() == documentation.example_exemption.is_none() {
        errors.push(error(
            Some(name),
            "canonical documentation requires examples or one explicit exemption",
        ));
    }
    if documentation
        .example_exemption
        .is_some_and(|reason| reason.trim().is_empty())
    {
        errors.push(error(
            Some(name),
            "documentation example exemption must not be empty",
        ));
    }
    let mut example_ids = BTreeSet::new();
    for example in documentation.examples {
        if example.id.trim().is_empty() || !example_ids.insert(example.id) {
            errors.push(error(
                Some(name),
                "documentation example ids must be non-empty and unique",
            ));
        }
        if example.title.trim().is_empty() || example.program.trim().is_empty() {
            errors.push(error(
                Some(name),
                "documentation examples require a title and program",
            ));
        }
        let valid_verification = match example.verification {
            BuiltinExampleVerification::Succeeds => true,
            BuiltinExampleVerification::Assertions { source } => !source.trim().is_empty(),
            BuiltinExampleVerification::ExpectedError { identifier } => {
                !identifier.trim().is_empty()
            }
            BuiltinExampleVerification::Figure {
                minimum_figures,
                assertions,
            } => minimum_figures > 0 && !assertions.trim().is_empty(),
        };
        if !valid_verification {
            errors.push(error(
                Some(name),
                "documentation example verification must be complete",
            ));
        }
        example::validate(name, example, errors);
    }
}

fn error(
    identity: Option<&'static str>,
    message: impl Into<String>,
) -> BuiltinCatalogValidationError {
    BuiltinCatalogValidationError {
        identity,
        message: message.into(),
    }
}
