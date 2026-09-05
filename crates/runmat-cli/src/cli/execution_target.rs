use anyhow::{bail, Result};
use clap::Args;
use uuid::Uuid;

#[derive(Args, Clone, Debug, Default)]
pub struct ExecutionTargetArgs {
    /// Run on this remote cluster instead of the local machine
    #[arg(long)]
    pub cluster: Option<String>,
    /// Remote project; defaults to the active project
    #[arg(long)]
    pub project: Option<Uuid>,
    /// Remote scheduler queue
    #[arg(long)]
    pub queue: Option<String>,
    /// Pinned endpoint identity fingerprint trusted for this run
    #[arg(long)]
    pub trust_identity: Option<String>,
    /// Stable submission identity for safe client retries
    #[arg(long)]
    pub idempotency_key: Option<String>,
    /// Number of remote worker allocations; zero retains automatic/driver-only behavior
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..=1024))]
    pub workers: Option<u32>,
    /// Return after remote submission instead of waiting for terminal output
    #[arg(long)]
    pub detach: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionTarget {
    Local,
    Remote(RemoteExecutionTarget),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteExecutionTarget {
    pub cluster: String,
    pub project: Option<Uuid>,
    pub queue: String,
    pub trust_identity: String,
    pub idempotency_key: Option<String>,
    pub workers: u32,
    pub detach: bool,
}

impl ExecutionTargetArgs {
    pub fn resolve(&self) -> Result<ExecutionTarget> {
        let Some(cluster) = self.cluster.as_deref() else {
            if self.project.is_some()
                || self.queue.is_some()
                || self.trust_identity.is_some()
                || self.idempotency_key.is_some()
                || self.workers.is_some()
                || self.detach
            {
                bail!("remote execution options require --cluster");
            }
            return Ok(ExecutionTarget::Local);
        };
        let cluster = bounded_required(cluster, "cluster")?;
        let trust_identity = bounded_required(
            self.trust_identity
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("remote execution requires --trust-identity"))?,
            "trust identity",
        )?;
        let queue = bounded_required(self.queue.as_deref().unwrap_or("default"), "queue")?;
        if self
            .idempotency_key
            .as_deref()
            .is_some_and(|value| bounded_required(value, "idempotency key").is_err())
        {
            bail!("idempotency key is empty, malformed, or exceeds its bound");
        }
        Ok(ExecutionTarget::Remote(RemoteExecutionTarget {
            cluster,
            project: self.project,
            queue,
            trust_identity,
            idempotency_key: self.idempotency_key.clone(),
            workers: self.workers.unwrap_or(0),
            detach: self.detach,
        }))
    }
}

fn bounded_required(value: &str, name: &str) -> Result<String> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        bail!("{name} is empty, malformed, or exceeds its bound");
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_target_rejects_remote_only_options() {
        let args = ExecutionTargetArgs {
            workers: Some(2),
            ..ExecutionTargetArgs::default()
        };
        assert!(args
            .resolve()
            .unwrap_err()
            .to_string()
            .contains("--cluster"));
    }

    #[test]
    fn remote_target_has_one_shared_default_queue() {
        let args = ExecutionTargetArgs {
            cluster: Some("engineering".into()),
            trust_identity: Some("sha256:trusted-node".into()),
            workers: Some(16),
            ..ExecutionTargetArgs::default()
        };
        assert_eq!(
            args.resolve().unwrap(),
            ExecutionTarget::Remote(RemoteExecutionTarget {
                cluster: "engineering".into(),
                project: None,
                queue: "default".into(),
                trust_identity: "sha256:trusted-node".into(),
                idempotency_key: None,
                workers: 16,
                detach: false,
            })
        );
    }
}
