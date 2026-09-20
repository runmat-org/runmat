use serde::{Deserialize, Serialize};

/// Project-owned routing facts for a named remote execution environment.
///
/// Analysis resource requirements remain in source. This profile binds a stable source-level
/// name to deployment routing and the capabilities that can be admitted before submission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ProjectExecutionProfile {
    /// Remote cluster identity understood by the configured RunMat server.
    pub cluster: String,
    /// Scheduler queue used when the caller does not provide a tighter override.
    #[serde(default)]
    pub queue: Option<String>,
    /// Pinned endpoint identity. Credentials remain in the credential store.
    #[serde(default, rename = "trust-identity")]
    pub trust_identity: Option<String>,
    /// Exact scheduler accelerator allocation admitted by this profile.
    #[serde(default)]
    pub accelerator: Option<ProjectAcceleratorProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ProjectAcceleratorProfile {
    /// Scheduler/provider class, for example `metal`, `cuda`, or `wgpu`.
    pub class: String,
    /// Guaranteed device memory for the single admitted accelerator.
    pub memory_bytes: u64,
}

impl ProjectExecutionProfile {
    pub(crate) fn validate(&self, name: &str) -> Vec<String> {
        let mut messages = Vec::new();
        validate_text(name, "execution profile name", &mut messages);
        validate_text(&self.cluster, "execution profile cluster", &mut messages);
        if let Some(queue) = self.queue.as_deref() {
            validate_text(queue, "execution profile queue", &mut messages);
        }
        if let Some(identity) = self.trust_identity.as_deref() {
            validate_text(identity, "execution profile trust identity", &mut messages);
        }
        if let Some(accelerator) = &self.accelerator {
            validate_text(
                &accelerator.class,
                "execution profile accelerator class",
                &mut messages,
            );
            if accelerator.memory_bytes == 0 {
                messages
                    .push("execution profile accelerator memory must be greater than zero".into());
            }
        }
        messages
    }
}

fn validate_text(value: &str, field: &str, messages: &mut Vec<String>) {
    if value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        messages.push(format!(
            "{field} must be non-empty, bounded, and contain no control characters"
        ));
    }
}
