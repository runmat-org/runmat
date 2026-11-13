//! Explicit organization context. Absence and malformed context are distinct.
use anyhow::{Context as _, Result};
use uuid::Uuid;

use super::RemoteConfig;
use crate::ids::OrgId;

pub fn configured_public_org_id(
    config: &RemoteConfig,
    explicit: Option<Uuid>,
) -> Result<Option<OrgId>> {
    Ok(configured_org_uuid(config, explicit)?.map(OrgId::from_uuid))
}

pub(super) fn configured_org_uuid(
    config: &RemoteConfig,
    explicit: Option<Uuid>,
) -> Result<Option<Uuid>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    let environment = match std::env::var("RUNMAT_ORG_ID") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).context("Invalid RUNMAT_ORG_ID"),
    };
    select_context(config, explicit, environment.as_deref())
}

fn select_context(
    config: &RemoteConfig,
    explicit: Option<Uuid>,
    environment: Option<&str>,
) -> Result<Option<Uuid>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    match environment.filter(|value| !value.is_empty()) {
        Some(value) => super::parse_org_id_input(value)
            .map(Some)
            .context("Invalid RUNMAT_ORG_ID"),
        None => Ok(config.org_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_environment_and_saved_context_preserve_precedence_and_errors() {
        let explicit = Uuid::from_u128(1);
        let environment = Uuid::from_u128(2);
        let saved = Uuid::from_u128(3);
        let config = RemoteConfig {
            org_id: Some(saved),
            ..Default::default()
        };
        assert_eq!(
            select_context(&config, Some(explicit), Some("invalid")).unwrap(),
            Some(explicit)
        );
        assert_eq!(
            select_context(&config, None, Some(OrgId::from_uuid(environment).as_str())).unwrap(),
            Some(environment)
        );
        assert_eq!(
            select_context(&config, None, Some("")).unwrap(),
            Some(saved)
        );
        assert!(select_context(&config, None, Some("invalid")).is_err());
        assert_eq!(
            select_context(&RemoteConfig::default(), None, None).unwrap(),
            None
        );
    }
}
