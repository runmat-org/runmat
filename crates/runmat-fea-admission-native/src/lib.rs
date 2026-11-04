//! Lazy native FEA lease host shared by CLI and standalone native images.
//! Construction captures only a typed selector. Each admitted root owns its transport thread.
use std::sync::mpsc::{self, SyncSender};
use std::sync::Mutex;

use runmat_runtime::analysis::fea_run_admission::{
    FeaLocalOrganization, FeaRunAdmission, FeaRunBudget, FeaRunDenied, FeaRunKind, FeaRunOutcome,
    FeaRunPermit, FeaRunRequest, FeaRunUsage,
};
use runmat_server_client::ids::{ExecutionDriverLeaseId, OrgId};
use runmat_server_client::{auth, simulation::*};
use runmat_simulation_usage::{MeasurementVersion, OperationCount};
use uuid::Uuid;

pub struct NativeFeaAdmission {
    org: FeaLocalOrganization,
    client: SimulationClientKind,
    lower_cap: Option<OperationCount>,
    driver: Option<DriverLeaseAuthority>,
}

#[derive(Clone)]
struct DriverLeaseAuthority {
    server: String,
    org: OrgId,
    lease: ExecutionDriverLeaseId,
    fencing_token: u64,
    credential: String,
}
impl NativeFeaAdmission {
    /// A typed host selector; this constructor performs no auth or transport work.
    pub fn from_selection(org: FeaLocalOrganization) -> Self {
        Self {
            org,
            client: SimulationClientKind::Cli,
            lower_cap: None,
            driver: None,
        }
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
        Self::from_selection(org)
    }

    pub fn compiled(org: Option<Uuid>) -> Self {
        Self {
            client: SimulationClientKind::Compiled,
            ..Self::new(org)
        }
    }

    pub fn with_spending_cap(mut self, cap: OperationCount) -> Self {
        self.lower_cap = Some(cap);
        self
    }

    /// Trusted cluster driver authority. No personal token, organization discovery,
    /// or credential/config file is consulted on this route.
    pub fn for_driver(
        server: &str,
        org: OrgId,
        lease: ExecutionDriverLeaseId,
        fencing_token: u64,
        credential: &str,
    ) -> Result<Self, FeaRunDenied> {
        SimulationLeaseClient::for_driver(server, lease.clone(), fencing_token, credential)
            .map_err(denial)?;
        Ok(Self {
            org: FeaLocalOrganization::Default,
            client: SimulationClientKind::Cluster,
            lower_cap: None,
            driver: Some(DriverLeaseAuthority {
                server: server.to_owned(),
                org,
                lease,
                fencing_token,
                credential: credential.to_owned(),
            }),
        })
    }
}

type Reply<T> = SyncSender<Result<T, FeaRunDenied>>;
enum Command {
    Heartbeat(ReportSimulationUsage, Reply<FeaRunBudget>),
    Settle(SettleSimulationLease, Reply<()>),
}
struct Permit {
    commands: Mutex<tokio::sync::mpsc::Sender<Command>>,
    initial: FeaRunBudget,
    measurement_version: MeasurementVersion,
}

impl Permit {
    fn report(
        &self,
        value: &FeaRunUsage,
        target: Option<OperationCount>,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.request(|reply| {
            Command::Heartbeat(
                ReportSimulationUsage {
                    measurement_version: self.measurement_version.clone(),
                    used_ops: value.completed_ops,
                    grant_target_ops: target,
                    other_detail: None,
                },
                reply,
            )
        })
    }
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

fn denial(error: SimulationLeaseError) -> FeaRunDenied {
    match error {
        SimulationLeaseError::Rejected { status: 401, .. }
        | SimulationLeaseError::Rejected {
            code: SimulationErrorCode::SignInRequired,
            ..
        } => FeaRunDenied::SignInRequired,
        SimulationLeaseError::Rejected {
            code: SimulationErrorCode::AllowanceUsed,
            details,
            ..
        } => FeaRunDenied::AllowanceUsed {
            refills_at_unix_seconds: details.refills_at.map(|date| date.timestamp()),
        },
        SimulationLeaseError::Rejected {
            code: SimulationErrorCode::AllowanceUnpublished,
            ..
        } => FeaRunDenied::AllowanceUnpublished,
        SimulationLeaseError::Rejected {
            code: SimulationErrorCode::BudgetExceedsAvailable,
            details,
            ..
        } => FeaRunDenied::BudgetUnavailable {
            available_ops: details.available_ops,
        },
        SimulationLeaseError::Rejected {
            code: SimulationErrorCode::LeaseNotFound,
            ..
        } => FeaRunDenied::LeaseExpired,
        SimulationLeaseError::Transport(_) => FeaRunDenied::ServerUnreachable,
        _ => FeaRunDenied::LicenseInvalid,
    }
}
fn budget(
    cap: OperationCount,
    granted: OperationCount,
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
        budget_ops: cap,
        granted_ops: granted,
        expires_at_unix_millis,
        heartbeat_millis,
    };
    budget.validate()?;
    Ok(budget)
}
impl FeaRunAdmission for NativeFeaAdmission {
    fn local_organization_hint(&self) -> FeaLocalOrganization {
        self.org
    }
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        request.validate()?;
        let expected_site = if self.driver.is_some() {
            runmat_runtime::analysis::fea_run_admission::FeaRunSite::Dispatched
        } else {
            runmat_runtime::analysis::fea_run_admission::FeaRunSite::Local
        };
        if request.site != expected_site {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let request = request.clone();
        let measurement_version = request.measurement_version.clone();
        let client_kind = self.client;
        let lower_cap = self.lower_cap;
        let driver = self.driver.clone();
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
                    let opened = open(org, driver, request, client_kind, lower_cap).await;
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
                                    .usage(&lease.lease_id, &usage)
                                    .await
                                    .map_err(denial)
                                    .and_then(|hold| {
                                        budget(
                                            hold.budget_ops,
                                            hold.granted_ops,
                                            hold.expires_at,
                                            lease.heartbeat_seconds,
                                        )
                                    });
                                let _ = reply.send(result);
                            }
                            Command::Settle(request, reply) => {
                                let result = client
                                    .settle(&lease.lease_id, &request)
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
            measurement_version,
        }))
    }
}

async fn open(
    org: Option<Uuid>,
    driver: Option<DriverLeaseAuthority>,
    request: FeaRunRequest,
    client_kind: SimulationClientKind,
    lower_cap: Option<OperationCount>,
) -> Result<(SimulationLeaseClient, SimulationLease, FeaRunBudget), FeaRunDenied> {
    let (client, org) = if let Some(driver) = driver {
        (
            SimulationLeaseClient::for_driver(
                &driver.server,
                driver.lease,
                driver.fencing_token,
                &driver.credential,
            )
            .map_err(denial)?,
            driver.org,
        )
    } else {
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
        let client = SimulationLeaseClient::new(&server, &token).map_err(denial)?;
        let org = match auth::configured_public_org_id(&config, org)
            .map_err(|_| FeaRunDenied::LicenseInvalid)?
        {
            Some(org) => org,
            None => client.sole_organization().await.map_err(denial)?,
        };
        (client, org)
    };
    let lease = client
        .start(
            &org,
            &StartSimulationLease {
                run_id: runmat_server_client::ids::SimulationRunId::parse(&request.run_id)
                    .map_err(|_| FeaRunDenied::LicenseInvalid)?,
                attempt: 1,
                measurement_version: request.measurement_version,
                runmat_version: env!("CARGO_PKG_VERSION").to_owned(),
                kind: match request.kind {
                    FeaRunKind::Meshing => SimulationKind::Meshing,
                    FeaRunKind::Solve => SimulationKind::Solve,
                    FeaRunKind::MeshAndSolve => SimulationKind::MeshAndSolve,
                },
                study_identity: request
                    .study_identity
                    .parse()
                    .map_err(|_| FeaRunDenied::LicenseInvalid)?,
                client: client_kind,
                estimate_ops: None,
                budget_ops: lower_cap,
                initial_grant_ops: None,
            },
        )
        .await
        .map_err(denial)?;
    let initial = budget(
        lease.budget_ops,
        lease.granted_ops,
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
    fn heartbeat(&self, value: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.report(value, None)
    }
    fn request_grant(
        &self,
        value: &FeaRunUsage,
        required: OperationCount,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.report(value, Some(required))
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        value: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        let outcome = match outcome {
            FeaRunOutcome::Completed => SimulationOutcome::Completed,
            FeaRunOutcome::Cancelled => SimulationOutcome::Cancelled,
            FeaRunOutcome::Failed => SimulationOutcome::Failed,
        };
        self.request(|reply| {
            Command::Settle(
                SettleSimulationLease {
                    outcome,
                    measurement_version: self.measurement_version.clone(),
                    used_ops: value.completed_ops,
                    other_detail: None,
                },
                reply,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::resource::ResourceUsage;
    use runmat_simulation_usage::CompletedUsage;
    fn ops(value: u64) -> OperationCount {
        OperationCount::from_operations(value)
    }
    #[test]
    fn fixed_cap_partial_grant_and_expiry_preserve_operation_units() {
        let expiry = chrono::Utc::now() + chrono::Duration::seconds(60);
        let admitted = budget(ops(100), ops(1), expiry, 30).unwrap();
        assert_eq!(admitted.budget_ops, ops(100));
        assert_eq!(admitted.granted_ops, ops(1));
        assert_eq!(admitted.heartbeat_millis, 30_000);
        assert!(budget(ops(0), ops(0), expiry, 30).is_err());
        assert!(budget(ops(1), ops(2), expiry, 30).is_err());
        assert!(budget(ops(1), ops(1), expiry, 0).is_err());
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
            denial(SimulationLeaseError::Transport(
                runmat_server_client::ServerLeaseTransportError::ServerStatus(503)
            )),
            FeaRunDenied::ServerUnreachable
        );
        assert_eq!(
            auth_denial(auth::SignInRequired.into()),
            FeaRunDenied::SignInRequired
        );
        let invalid = NativeFeaAdmission::from_selection(FeaLocalOrganization::Invalid);
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
        let initial = budget(
            ops(100),
            ops(100),
            chrono::Utc::now() + chrono::Duration::seconds(60),
            30,
        )
        .unwrap();
        let (commands, mut receiver) = tokio::sync::mpsc::channel(1);
        let owner = std::thread::spawn(move || {
            let Command::Heartbeat(usage, reply) = receiver.blocking_recv().unwrap() else {
                panic!("expected pre-batch grant request")
            };
            assert_eq!(usage.grant_target_ops, Some(ops(75)));
            assert_eq!(usage.used_ops.solve, ops(3));
            reply.send(Ok(initial)).unwrap();
            let Command::Heartbeat(usage, reply) = receiver.blocking_recv().unwrap() else {
                panic!("expected heartbeat")
            };
            assert_eq!(usage.grant_target_ops, None);
            assert_eq!(usage.used_ops.solve, ops(3));
            reply.send(Err(FeaRunDenied::LeaseExpired)).unwrap();
            let Command::Settle(request, reply) = receiver.blocking_recv().unwrap() else {
                panic!("expected own-run cancellation settlement")
            };
            assert_eq!(request.outcome, SimulationOutcome::Cancelled);
            assert_eq!(request.used_ops.solve, ops(3));
            reply.send(Ok(())).unwrap();
        });
        let permit = Box::new(Permit {
            commands: Mutex::new(commands),
            initial,
            measurement_version: MeasurementVersion::reference_v1(),
        });
        let usage = FeaRunUsage {
            completed_ops: CompletedUsage {
                solve: ops(3),
                ..Default::default()
            },
            resources: ResourceUsage {
                cpu_core_millis: 2,
                wall_millis: 1,
                ..Default::default()
            },
        };
        assert_eq!(permit.request_grant(&usage, ops(75)).unwrap(), initial);
        assert_eq!(permit.heartbeat(&usage), Err(FeaRunDenied::LeaseExpired));
        permit.settle(FeaRunOutcome::Cancelled, &usage).unwrap();
        owner.join().unwrap();
    }
}
