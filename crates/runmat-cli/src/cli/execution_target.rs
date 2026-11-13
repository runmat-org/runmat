use anyhow::{bail, Result};
use clap::Args;
use runmat_server_client::execution::RemoteEndpointTrust;
use uuid::Uuid;

#[derive(Args, Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionTargetArgs {
    /// Organization for local FEA core-hours; defaults to the signed-in organization
    #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
    pub org: Option<Uuid>,
    /// Run on this remote cluster instead of the local machine
    #[arg(long)]
    pub cluster: Option<String>,
    /// Remote project; defaults to the active project
    #[arg(long, value_parser = runmat_server_client::auth::parse_project_id_input)]
    pub project: Option<Uuid>,
    /// Remote scheduler queue
    #[arg(long)]
    pub queue: Option<String>,
    /// Optional independent endpoint identity pin (otherwise trust authenticated Server admission)
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
    pub trust_identity: RemoteEndpointTrust,
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
        if self.org.is_some() {
            bail!("--org selects local FEA core-hours; remote execution uses the selected project's organization");
        }
        let cluster = bounded_required(cluster, "cluster")?;
        // Read remote defaults only after selecting a remote route. A trust
        // environment variable must not make ordinary local execution remote.
        let trust_identity = resolve_endpoint_trust(self.trust_identity.as_deref())?;
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

pub(crate) fn resolve_endpoint_trust(pin: Option<&str>) -> Result<RemoteEndpointTrust> {
    resolve_endpoint_trust_with_environment(pin, || std::env::var("RUNMAT_TRUST_IDENTITY"))
}

fn resolve_endpoint_trust_with_environment(
    pin: Option<&str>,
    environment: impl FnOnce() -> std::result::Result<String, std::env::VarError>,
) -> Result<RemoteEndpointTrust> {
    if let Some(pin) = pin {
        return RemoteEndpointTrust::from_optional_pin(Some(pin));
    }
    match environment() {
        Ok(pin) => RemoteEndpointTrust::from_optional_pin(Some(&pin)),
        Err(std::env::VarError::NotPresent) => Ok(RemoteEndpointTrust::AuthenticatedServer),
        Err(error) => Err(anyhow::anyhow!(error).context("invalid RUNMAT_TRUST_IDENTITY")),
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
    fn omitted_pin_uses_environment_or_authenticated_server() {
        assert_eq!(
            resolve_endpoint_trust_with_environment(None, || Err(std::env::VarError::NotPresent))
                .unwrap(),
            RemoteEndpointTrust::AuthenticatedServer
        );
        assert_eq!(
            resolve_endpoint_trust_with_environment(None, || Ok("sha256:environment".into()))
                .unwrap(),
            RemoteEndpointTrust::PinnedIdentity("sha256:environment".into())
        );
    }

    #[test]
    fn explicit_pin_takes_precedence_without_reading_environment() {
        assert_eq!(
            resolve_endpoint_trust_with_environment(Some("sha256:explicit"), || {
                panic!("explicit pins must not read the environment")
            })
            .unwrap(),
            RemoteEndpointTrust::PinnedIdentity("sha256:explicit".into())
        );
        for pin in ["", "pin other", "pin\nother", &"x".repeat(257)] {
            assert!(resolve_endpoint_trust_with_environment(Some(pin), || {
                panic!("invalid explicit pins must not fall back to the environment")
            })
            .is_err());
        }
    }

    #[test]
    fn malformed_environment_pin_fails_without_falling_back_to_server_trust() {
        for pin in ["", "pin other", "pin\nother", &"x".repeat(257)] {
            assert!(resolve_endpoint_trust_with_environment(None, || Ok(pin.into())).is_err());
        }
        let error = resolve_endpoint_trust_with_environment(None, || {
            Err(std::env::VarError::NotUnicode("invalid".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("invalid RUNMAT_TRUST_IDENTITY"));
    }

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
