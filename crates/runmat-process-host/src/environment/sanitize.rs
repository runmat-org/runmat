use std::collections::BTreeMap;

use tokio::process::Command;

use super::EnvironmentAllowlist;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum EnvironmentPolicy {
    Inherit,
    #[default]
    Clear,
    Allow(EnvironmentAllowlist),
}

pub fn apply_environment(
    command: &mut Command,
    policy: &EnvironmentPolicy,
    explicit: &BTreeMap<String, String>,
) {
    match policy {
        EnvironmentPolicy::Inherit => {}
        EnvironmentPolicy::Clear => {
            command.env_clear();
        }
        EnvironmentPolicy::Allow(allowlist) => {
            let allowed = std::env::vars_os()
                .filter(|(name, _)| allowlist.contains(&name.to_string_lossy()))
                .collect::<Vec<_>>();
            command.env_clear();
            command.envs(allowed);
        }
    }
    command.envs(explicit);
}
