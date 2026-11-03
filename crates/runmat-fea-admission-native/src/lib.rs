//! Lazy native FEA lease host shared by CLI and standalone native images.
//! Construction captures only a typed selector. Each admitted root owns its transport thread.
use std::sync::mpsc::{self, SyncSender};
use std::sync::Mutex;

use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::{
    FeaLocalOrganization, FeaRunAdmission, FeaRunBudget, FeaRunDenied, FeaRunKind, FeaRunOutcome,
    FeaRunPermit, FeaRunRequest,
};
use runmat_server_client::{auth, fea::*};
use uuid::Uuid;

pub struct NativeFeaAdmission {
    org: FeaLocalOrganization,
}
impl NativeFeaAdmission {
    /// A typed host selector; this constructor performs no auth or transport work.
    pub fn from_selection(org: FeaLocalOrganization) -> Self {
        Self { org }
    }
    pub fn new(org: Option<Uuid>) -> Self {
        // The existing selector parser reads only the explicit argument and ENV,
        // never credentials/config files. Preserve invalid selection until FEA.
        let org = match auth::configured_public_org_id(&auth::RemoteConfig::default(), org) {
            Ok(Some(org)) => org
                .as_uuid()
                .map(FeaLocalOrganization::Selected)
                .unwrap_or(FeaLocalOrganization::Invalid),
            Ok(None) => FeaLocalOrganization::Default,
            Err(_) => FeaLocalOrganization::Invalid,
        };
        Self { org }
    }
}

type Reply<T> = SyncSender<Result<T, FeaRunDenied>>;
enum Command {
    Heartbeat(FeaLeaseUsage, Reply<FeaRunBudget>),
    Settle(FeaLeaseOutcome, FeaLeaseUsage, Reply<()>),
}
struct Permit {
    commands: Mutex<tokio::sync::mpsc::Sender<Command>>,
    initial: FeaRunBudget,
}

impl Permit {
    fn request<T>(&self, command: impl FnOnce(Reply<T>) -> Command) -> Result<T, FeaRunDenied> {
        // One exchange owns the bounded command slot until its reply arrives.
        // The synchronous host may run inside Tokio, so blocking_send is unsuitable.
        let commands = self
            .commands
            .lock()
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        let (reply, result) = mpsc::sync_channel(1);
        commands
            .try_send(command(reply))
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        let result = result.recv().map_err(|_| FeaRunDenied::ServerUnreachable)?;
        drop(commands);
        result
    }
}

fn denial(error: FeaLeaseError) -> FeaRunDenied {
    match error {
        FeaLeaseError::SignInRequired => FeaRunDenied::SignInRequired,
        FeaLeaseError::LicenseInvalid => FeaRunDenied::LicenseInvalid,
        FeaLeaseError::AllowanceUsed {
            refills_at_unix_seconds,
        } => FeaRunDenied::AllowanceUsed {
            refills_at_unix_seconds,
        },
        FeaLeaseError::LeaseExpired => FeaRunDenied::LeaseExpired,
        FeaLeaseError::ServerUnreachable => FeaRunDenied::ServerUnreachable,
    }
}
fn budget(
    held: u64,
    expiry: chrono::DateTime<chrono::Utc>,
    interval: u64,
) -> Result<FeaRunBudget, FeaRunDenied> {
    let expires_at_unix_millis =
        u64::try_from(expiry.timestamp_millis()).map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let heartbeat_millis = interval
        .checked_mul(1000)
        .filter(|value| *value > 0 && *value <= 120_000)
        .ok_or(FeaRunDenied::LicenseInvalid)?;
    let budget = FeaRunBudget {
        held_core_millis: held,
        expires_at_unix_millis,
        heartbeat_millis,
    };
    budget.validate()?;
    Ok(budget)
}
fn usage(value: &ResourceUsage) -> FeaLeaseUsage {
    FeaLeaseUsage {
        cpu_core_millis: value.cpu_core_millis,
        wall_millis: value.wall_millis,
    }
}

impl FeaRunAdmission for NativeFeaAdmission {
    fn local_organization_hint(&self) -> FeaLocalOrganization {
        self.org
    }
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        if request.site != runmat_runtime::analysis::fea_run_admission::FeaRunSite::Local {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let request = request.clone();
        let org = match self.org {
            FeaLocalOrganization::Default => None,
            FeaLocalOrganization::Selected(org) => Some(org),
            FeaLocalOrganization::Invalid => return Err(FeaRunDenied::LicenseInvalid),
        };
        let (ready, initial) = mpsc::sync_channel(1);
        let (commands, mut receiver) = tokio::sync::mpsc::channel(1);
        // This runtime is owned and driven on a fresh OS thread, never the
        // caller's Tokio lane. A synchronous FEA root can safely await replies.
        std::thread::Builder::new()
            .name("runmat-fea-lease".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(_) => {
                        let _ = ready.send(Err(FeaRunDenied::ServerUnreachable));
                        return;
                    }
                };
                runtime.block_on(async move {
                    let opened = open(org, request).await;
                    let (client, lease, initial_budget) = match opened {
                        Ok(value) => value,
                        Err(error) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                    };
                    if ready.send(Ok(initial_budget)).is_err() {
                        return;
                    }
                    // Keep the reactor running between requests so idle connections
                    // observe peer closure before the next heartbeat uses the pool.
                    while let Some(command) = receiver.recv().await {
                        match command {
                            Command::Heartbeat(usage, reply) => {
                                let result = client
                                    .usage(&lease.lease_id, usage)
                                    .await
                                    .map_err(denial)
                                    .and_then(|hold| {
                                        budget(
                                            hold.held_core_millis,
                                            hold.expires_at,
                                            lease.heartbeat_seconds,
                                        )
                                    });
                                let _ = reply.send(result);
                            }
                            Command::Settle(outcome, usage, reply) => {
                                let result = client
                                    .settle(&lease.lease_id, outcome, usage)
                                    .await
                                    .map(|_| ())
                                    .map_err(denial);
                                let _ = reply.send(result);
                                break;
                            }
                        }
                    }
                });
            })
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        let initial = initial
            .recv()
            .map_err(|_| FeaRunDenied::ServerUnreachable)??;
        Ok(Box::new(Permit {
            commands: Mutex::new(commands),
            initial,
        }))
    }
}

async fn open(
    org: Option<Uuid>,
    request: FeaRunRequest,
) -> Result<(FeaLeaseClient, FeaLease, FeaRunBudget), FeaRunDenied> {
    let mut config = auth::RemoteConfig::load().map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let server =
        auth::resolve_server_url(&config, None).map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let token = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        auth::resolve_auth_token(&mut config, &server),
    )
    .await
    .map_err(|_| FeaRunDenied::ServerUnreachable)?
    .map_err(auth_denial)?;
    let client = FeaLeaseClient::new(&server, &token).map_err(denial)?;
    let org = match auth::configured_public_org_id(&config, org)
        .map_err(|_| FeaRunDenied::LicenseInvalid)?
    {
        Some(org) => org,
        None => client.sole_organization().await.map_err(denial)?,
    };
    let lease = client
        .start(
            &org,
            &StartFeaLease {
                run_id: request.run_id,
                kind: match request.kind {
                    FeaRunKind::Meshing => FeaLeaseKind::Meshing,
                    FeaRunKind::Solve => FeaLeaseKind::Solve,
                    FeaRunKind::MeshAndSolve => FeaLeaseKind::MeshAndSolve,
                },
                allocated_cpu_millicores: request.allocated_cpu_millicores,
                study_identity: request.study_identity,
                client: FeaLeaseClientKind::Cli,
            },
        )
        .await
        .map_err(denial)?;
    let initial = budget(
        lease.held_core_millis,
        lease.expires_at,
        lease.heartbeat_seconds,
    )?;
    Ok((client, lease, initial))
}

fn auth_denial(error: anyhow::Error) -> FeaRunDenied {
    if error.downcast_ref::<auth::SignInRequired>().is_some() {
        return FeaRunDenied::SignInRequired;
    }
    if let Some(error) = error.downcast_ref::<reqwest::Error>() {
        return match error.status().map(|status| status.as_u16()) {
            Some(401 | 403) => FeaRunDenied::SignInRequired,
            Some(400..=499) => FeaRunDenied::LicenseInvalid,
            _ => FeaRunDenied::ServerUnreachable,
        };
    }
    FeaRunDenied::LicenseInvalid
}

impl FeaRunPermit for Permit {
    fn budget(&self) -> FeaRunBudget {
        self.initial
    }
    fn heartbeat(&self, value: &ResourceUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.request(|reply| Command::Heartbeat(usage(value), reply))
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        value: &ResourceUsage,
    ) -> Result<(), FeaRunDenied> {
        let outcome = match outcome {
            FeaRunOutcome::Completed => FeaLeaseOutcome::Completed,
            FeaRunOutcome::Cancelled => FeaLeaseOutcome::Cancelled,
            FeaRunOutcome::Failed => FeaLeaseOutcome::Failed,
        };
        self.request(|reply| Command::Settle(outcome, usage(value), reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_hold_and_expiry_are_preserved_without_gpu_accounting() {
        let expiry = chrono::Utc::now() + chrono::Duration::seconds(60);
        let admitted = budget(1, expiry, 30).unwrap();
        assert_eq!(admitted.held_core_millis, 1);
        assert_eq!(admitted.heartbeat_millis, 30_000);
        assert!(budget(0, expiry, 30).is_err());
        assert!(budget(1, expiry, 0).is_err());
    }

    #[test]
    fn constructing_local_provider_does_not_initialize_transport_or_authenticate() {
        let _provider = NativeFeaAdmission::new(None);
        let selected = Uuid::from_u128(17);
        assert_eq!(
            NativeFeaAdmission::new(Some(selected)).local_organization_hint(),
            FeaLocalOrganization::Selected(selected)
        );
        assert_eq!(
            denial(FeaLeaseError::ServerUnreachable),
            FeaRunDenied::ServerUnreachable
        );
        assert_eq!(
            auth_denial(auth::SignInRequired.into()),
            FeaRunDenied::SignInRequired
        );
        let invalid = NativeFeaAdmission {
            org: FeaLocalOrganization::Invalid,
        };
        let request = FeaRunRequest::local(
            FeaRunKind::Solve,
            1000,
            runmat_execution::Digest::sha256(b"invalid-selector"),
        )
        .unwrap();
        assert!(matches!(
            invalid.admit(&request),
            Err(FeaRunDenied::LicenseInvalid)
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_permit_reply_does_not_depend_on_the_callers_tokio_lane() {
        let initial = budget(100, chrono::Utc::now() + chrono::Duration::seconds(60), 30).unwrap();
        let (commands, mut receiver) = tokio::sync::mpsc::channel(1);
        let owner = std::thread::spawn(move || {
            let Command::Heartbeat(usage, reply) = receiver.blocking_recv().unwrap() else {
                panic!("expected heartbeat")
            };
            assert_eq!((usage.cpu_core_millis, usage.wall_millis), (2, 1));
            reply.send(Err(FeaRunDenied::LeaseExpired)).unwrap();
            let Command::Settle(FeaLeaseOutcome::Cancelled, usage, reply) =
                receiver.blocking_recv().unwrap()
            else {
                panic!("expected own-run cancellation settlement")
            };
            assert_eq!((usage.cpu_core_millis, usage.wall_millis), (2, 1));
            reply.send(Ok(())).unwrap();
        });
        let permit = Box::new(Permit {
            commands: Mutex::new(commands),
            initial,
        });
        let usage = ResourceUsage {
            cpu_core_millis: 2,
            wall_millis: 1,
            ..Default::default()
        };
        assert_eq!(permit.heartbeat(&usage), Err(FeaRunDenied::LeaseExpired));
        permit.settle(FeaRunOutcome::Cancelled, &usage).unwrap();
        owner.join().unwrap();
    }
}
