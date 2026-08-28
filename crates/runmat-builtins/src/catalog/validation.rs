use super::{BuiltinBindingAvailability, BuiltinCatalogEntry, BuiltinContractMaturity};
use crate::BuiltinAsyncBehavior;
use runmat_types::EffectKind;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinCatalogValidationError {
    pub identity: Option<&'static str>,
    pub message: String,
}

pub fn validate_builtin_catalog(
    entries: &[&'static BuiltinCatalogEntry],
) -> Vec<BuiltinCatalogValidationError> {
    let mut errors = Vec::new();
    let mut identities = BTreeSet::new();
    let mut bindings = BTreeMap::new();
    for entry in entries {
        let name = entry.identity.name;
        if name.is_empty() {
            errors.push(error(None, "builtin identity must not be empty"));
            continue;
        }
        if !identities.insert(name) {
            errors.push(error(Some(name), "duplicate builtin catalog identity"));
        }
        if entry.category.is_empty() {
            errors.push(error(Some(name), "category must not be empty"));
        }
        if entry.documentation.summary.is_empty() {
            errors.push(error(Some(name), "documentation summary must not be empty"));
        }
        if entry.bindings.is_empty() {
            errors.push(error(
                Some(name),
                "catalog entry has no runtime binding declaration",
            ));
        }
        if matches!(entry.contract.maturity, BuiltinContractMaturity::Complete)
            && entry.contract.inference_rule.0.is_empty()
        {
            errors.push(error(
                Some(name),
                "complete contract has no inference rule identity",
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
            if binding.identity.builtin != entry.identity {
                errors.push(error(
                    Some(name),
                    "binding declaration refers to a different builtin identity",
                ));
            }
            if binding.identity.variant.is_empty() {
                errors.push(error(Some(name), "binding variant must not be empty"));
            }
            if bindings.insert(binding.identity, name).is_some() {
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
    errors
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
