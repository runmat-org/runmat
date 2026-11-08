//! Lazy native FEA lease host shared by CLI and standalone native images.
//! Construction captures only a typed selector. Each admitted root owns its transport thread.
use std::sync::mpsc::{self, SyncSender};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Grant from the latest server reply, for sizing the next request.
    granted: AtomicU64,
}

impl Permit {
    fn report(
        &self,
        value: &FeaRunUsage,
        target: Option<OperationCount>,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        let budget = self.request(|reply| {
            Command::Heartbeat(
                ReportSimulationUsage {
                    measurement_version: self.measurement_version.clone(),
                    used_ops: value.completed_ops,
                    grant_target_ops: target,
                    other_detail: None,
                },
                reply,
            )
        })?;
        self.granted
            .store(budget.granted_ops.operations(), Ordering::Release);
        Ok(budget)
    }
    /// The server adds what it can toward an absolute target within the
    /// budget and allowance; ask ahead by the shared doubling law.
    fn grant_target(&self, required: OperationCount) -> OperationCount {
        runmat_simulation_usage::grant_target(
            required,
            OperationCount::from_operations(self.granted.load(Ordering::Acquire)),
            self.initial.budget_ops,
        )
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
    if unauthorized(&error) {
        return FeaRunDenied::SignInRequired;
    }
    match error {
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
        let (commands, receiver) = tokio::sync::mpsc::channel(1);
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
                    let (session, lease, initial_budget) = match opened {
                        Ok(value) => value,
                        Err(error) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                    };
                    if ready.send(Ok(initial_budget)).is_err() {
                        return;
                    }
                    serve(session, lease, receiver).await;
                });
            })
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        let initial = initial
            .recv()
            .map_err(|_| FeaRunDenied::ServerUnreachable)??;
        Ok(Box::new(Permit {
            commands: Mutex::new(commands),
            granted: AtomicU64::new(initial.granted_ops.operations()),
            initial,
            measurement_version,
        }))
    }
}

/// Serve the permit's commands until it settles. Keeping the reactor running between requests
/// lets idle connections observe peer closure before the next heartbeat uses the pool.
async fn serve(
    mut session: LeaseSession,
    mut lease: SimulationLease,
    mut receiver: tokio::sync::mpsc::Receiver<Command>,
) {
    while let Some(command) = receiver.recv().await {
        match command {
            Command::Heartbeat(usage, reply) => {
                let result = session.usage(&mut lease, &usage).await;
                let _ = reply.send(result);
            }
            Command::Settle(request, reply) => {
                let result = session.settle(&lease, &request).await;
                let _ = reply.send(result);
                break;
            }
        }
    }
}

type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Where a person's lease gets its access token. A run outlives the token it starts with
/// (local sign-in tokens live minutes), so every lease request asks for the current one.
trait AccessTokens: Send {
    /// The token to send now: the stored token, refreshed first when it is about to expire.
    fn current(&mut self) -> BoxFuture<'_, Result<String, Failure>>;
    /// A new token from the stored refresh token, after the server rejected the current one.
    fn refreshed(&mut self) -> BoxFuture<'_, Result<String, Failure>>;
}

/// A failed lease exchange: the denial the run reports, and whether the server was only
/// unreachable (transport failure or 5xx), so the outage policy may try again.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Failure {
    denied: FeaRunDenied,
    transient: bool,
}
impl From<FeaRunDenied> for Failure {
    fn from(denied: FeaRunDenied) -> Self {
        Self {
            denied,
            transient: false,
        }
    }
}
impl From<Failure> for FeaRunDenied {
    fn from(failure: Failure) -> Self {
        failure.denied
    }
}
impl From<SimulationLeaseError> for Failure {
    fn from(error: SimulationLeaseError) -> Self {
        Self {
            transient: error.is_transient(),
            denied: denial(error),
        }
    }
}
fn unreachable() -> Failure {
    Failure {
        denied: FeaRunDenied::ServerUnreachable,
        transient: true,
    }
}

/// The signed-in person's stored credentials (`runmat login`), read again for every request so
/// a refresh by another RunMat process (which may rotate the refresh token) is seen here.
struct StoredSignIn;

impl StoredSignIn {
    async fn token(force: bool) -> Result<String, Failure> {
        let mut config = auth::RemoteConfig::load().map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let server =
            auth::resolve_server_url(&config, None).map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let timeout = std::time::Duration::from_secs(10);
        let token = if force {
            tokio::time::timeout(timeout, auth::refresh_auth_token(&mut config, &server)).await
        } else {
            tokio::time::timeout(timeout, auth::resolve_auth_token(&mut config, &server)).await
        };
        token.map_err(|_| unreachable())?.map_err(auth_failure)
    }
}

impl AccessTokens for StoredSignIn {
    fn current(&mut self) -> BoxFuture<'_, Result<String, Failure>> {
        Box::pin(Self::token(false))
    }
    fn refreshed(&mut self) -> BoxFuture<'_, Result<String, Failure>> {
        Box::pin(Self::token(true))
    }
}

/// One lease's server client. A person's client sends the current access token: it is rebuilt
/// when the stored token changes (refreshed before expiry) and, when the server answers 401,
/// once more after a forced refresh before the request is retried. A driver's run-scoped
/// credential never changes.
struct LeaseSession {
    server: String,
    token: String,
    client: SimulationLeaseClient,
    tokens: Option<Box<dyn AccessTokens>>,
}

/// One lease request with the session's token handling: refresh before sending, and on a 401
/// refresh and send once more. Lease requests carry absolute totals, so a repeat is safe.
/// Evaluates to `Result<_, Failure>`.
macro_rules! with_current_token {
    ($session:expr, |$client:ident| $call:expr) => {
        async {
            $session.prepare().await?;
            let first = {
                let $client = &$session.client;
                $call.await
            };
            match first {
                Err(error) if $session.tokens.is_some() && unauthorized(&error) => {
                    $session.after_rejection().await?;
                    let $client = &$session.client;
                    $call.await.map_err(Failure::from)
                }
                other => other.map_err(Failure::from),
            }
        }
        .await
    };
}

/// A renewal or settlement under the shared outage policy: while the server is unreachable
/// the request is resent with backoff inside the lease's outage window, then the original
/// denial is returned. Usage and settlement carry absolute totals, and the server accepts an
/// equal total again (and replays a settlement), so a resent request never counts twice.
macro_rules! through_outage {
    ($operation:literal, $lease:expr, $session:expr, |$client:ident| $call:expr) => {{
        let mut clock = runmat_server_client::lease_retry::LeaseRetry::for_lease(
            std::time::Duration::from_secs($lease.heartbeat_seconds),
            $lease.expires_at,
        )
        .begin($operation);
        loop {
            match with_current_token!($session, |$client| $call) {
                Err(failure) if failure.transient => {
                    if !clock.retry(&failure.denied).await {
                        break Err(failure.denied);
                    }
                }
                other => {
                    if other.is_ok() {
                        clock.recovered();
                    }
                    break other.map_err(FeaRunDenied::from);
                }
            }
        }
    }};
}

impl LeaseSession {
    async fn person(
        server: String,
        mut tokens: Box<dyn AccessTokens>,
    ) -> Result<Self, Failure> {
        let token = tokens.current().await?;
        let client = SimulationLeaseClient::new(&server, &token).map_err(denial)?;
        Ok(Self {
            server,
            token,
            client,
            tokens: Some(tokens),
        })
    }

    fn driver(driver: &DriverLeaseAuthority) -> Result<Self, FeaRunDenied> {
        Ok(Self {
            server: driver.server.clone(),
            token: String::new(),
            client: SimulationLeaseClient::for_driver(
                &driver.server,
                driver.lease.clone(),
                driver.fencing_token,
                &driver.credential,
            )
            .map_err(denial)?,
            tokens: None,
        })
    }

    fn use_token(&mut self, token: String) -> Result<(), FeaRunDenied> {
        if token != self.token {
            self.client = SimulationLeaseClient::new(&self.server, &token).map_err(denial)?;
            self.token = token;
        }
        Ok(())
    }

    async fn prepare(&mut self) -> Result<(), Failure> {
        let Some(tokens) = self.tokens.as_mut() else {
            return Ok(());
        };
        let token = tokens.current().await?;
        Ok(self.use_token(token)?)
    }

    async fn after_rejection(&mut self) -> Result<(), Failure> {
        let Some(tokens) = self.tokens.as_mut() else {
            return Ok(());
        };
        let token = tokens.refreshed().await?;
        Ok(self.use_token(token)?)
    }

    async fn usage(
        &mut self,
        lease: &mut SimulationLease,
        usage: &ReportSimulationUsage,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        let hold = through_outage!("simulation usage", lease, self, |client| client
            .usage(&lease.lease_id, usage))?;
        lease.expires_at = hold.expires_at;
        budget(
            hold.budget_ops,
            hold.granted_ops,
            hold.expires_at,
            lease.heartbeat_seconds,
        )
    }

    async fn settle(
        &mut self,
        lease: &SimulationLease,
        request: &SettleSimulationLease,
    ) -> Result<(), FeaRunDenied> {
        through_outage!("simulation settle", lease, self, |client| client
            .settle(&lease.lease_id, request))
        .map(|_| ())
    }
}

fn unauthorized(error: &SimulationLeaseError) -> bool {
    matches!(
        error,
        SimulationLeaseError::Rejected { status: 401, .. }
            | SimulationLeaseError::Rejected {
                code: SimulationErrorCode::SignInRequired,
                ..
            }
    )
}

async fn open(
    org: Option<Uuid>,
    driver: Option<DriverLeaseAuthority>,
    request: FeaRunRequest,
    client_kind: SimulationClientKind,
    lower_cap: Option<OperationCount>,
) -> Result<(LeaseSession, SimulationLease, FeaRunBudget), FeaRunDenied> {
    let (mut session, org) = if let Some(driver) = driver {
        (LeaseSession::driver(&driver)?, driver.org)
    } else {
        let config = auth::RemoteConfig::load().map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let server =
            auth::resolve_server_url(&config, None).map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let mut session = LeaseSession::person(server, Box::new(StoredSignIn)).await?;
        let org = match auth::configured_public_org_id(&config, org)
            .map_err(|_| FeaRunDenied::LicenseInvalid)?
        {
            Some(org) => org,
            None => with_current_token!(session, |client| client.sole_organization())?,
        };
        (session, org)
    };
    let start = StartSimulationLease {
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
    };
    let lease = with_current_token!(session, |client| client.start(&org, &start))?;
    let initial = budget(
        lease.budget_ops,
        lease.granted_ops,
        lease.expires_at,
        lease.heartbeat_seconds,
    )?;
    Ok((session, lease, initial))
}

/// A failed token refresh. Only a refresh that never got an answer, or got a 5xx, is transient.
fn auth_failure(error: anyhow::Error) -> Failure {
    if error.downcast_ref::<auth::SignInRequired>().is_some() {
        return FeaRunDenied::SignInRequired.into();
    }
    if let Some(error) = error.downcast_ref::<reqwest::Error>() {
        return match error.status().map(|status| status.as_u16()) {
            Some(401 | 403) => FeaRunDenied::SignInRequired.into(),
            Some(400..=499) => FeaRunDenied::LicenseInvalid.into(),
            _ => unreachable(),
        };
    }
    FeaRunDenied::LicenseInvalid.into()
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
        self.report(value, Some(self.grant_target(required)))
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
    fn grant_requests_ask_ahead_by_the_current_grant_within_the_budget() {
        let expiry = chrono::Utc::now() + chrono::Duration::seconds(60);
        let initial = budget(ops(10_000), ops(100), expiry, 30).unwrap();
        let (commands, _receiver) = tokio::sync::mpsc::channel(1);
        let permit = Permit {
            commands: Mutex::new(commands),
            granted: AtomicU64::new(initial.granted_ops.operations()),
            initial,
            measurement_version: MeasurementVersion::reference_v1(),
        };
        // Within the current grant: exactly what is required.
        assert_eq!(permit.grant_target(ops(80)), ops(80));
        // One element past the grant: ask for the grant again on top.
        assert_eq!(permit.grant_target(ops(110)), ops(210));
        // A large jump asks for twice the jump.
        assert_eq!(permit.grant_target(ops(1_000)), ops(1_900));
        // Never past the budget, and never below what is required.
        permit.granted.store(6_000, Ordering::Release);
        assert_eq!(permit.grant_target(ops(6_010)), ops(10_000));
        permit.granted.store(10_000, Ordering::Release);
        assert_eq!(permit.grant_target(ops(10_000)), ops(10_000));
        // 60 per-element requests of 10 ops starting from no grant.
        permit.granted.store(0, Ordering::Release);
        let mut renewals = 0;
        for element in 1..=60u64 {
            let required = element * 10;
            if required > permit.granted.load(Ordering::Acquire) {
                let target = permit.grant_target(ops(required));
                permit.granted.store(target.operations(), Ordering::Release);
                renewals += 1;
            }
        }
        assert!(renewals <= 7, "{renewals} renewals for 60 elements");
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
            auth_failure(auth::SignInRequired.into()).denied,
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
            granted: AtomicU64::new(initial.granted_ops.operations()),
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

/// A local run outlives the access token it starts with: the lease client must refresh it
/// before it expires, and after a 401 refresh and retry once.
#[cfg(test)]
mod token_refresh_tests {
    use super::*;
    use runmat_server_client::ids::SimulationLeaseId;
    use std::sync::atomic::AtomicI64;
    use std::sync::Arc;

    /// Tokens on a fake clock: `token-N` is valid until `expires_at` and the stored sign-in
    /// refreshes within 30 s of expiry, as `auth::resolve_auth_token` does.
    struct FakeTokens {
        clock: Arc<AtomicI64>,
        expires_at: i64,
        issued: u32,
        refreshes: Arc<AtomicU64>,
    }
    impl FakeTokens {
        fn refresh(&mut self) -> String {
            self.issued += 1;
            self.expires_at = self.clock.load(Ordering::SeqCst) + 300;
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            format!("token-{}", self.issued)
        }
    }
    impl AccessTokens for FakeTokens {
        fn current(&mut self) -> BoxFuture<'_, Result<String, Failure>> {
            Box::pin(async move {
                if self.clock.load(Ordering::SeqCst) + 30 >= self.expires_at {
                    return Ok(self.refresh());
                }
                Ok(format!("token-{}", self.issued))
            })
        }
        fn refreshed(&mut self) -> BoxFuture<'_, Result<String, Failure>> {
            Box::pin(async move { Ok(self.refresh()) })
        }
    }

    /// A lease server that accepts only the bearer tokens `accepted` returns true for, and
    /// records each request's path and bearer.
    async fn server(
        requests: usize,
        accepted: fn(&str) -> bool,
    ) -> (String, tokio::task::JoinHandle<Vec<(String, String)>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut seen = Vec::new();
            for _ in 0..requests {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let end = loop {
                    let mut chunk = [0; 1024];
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(read, 0);
                    bytes.extend_from_slice(&chunk[..read]);
                    if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        break index + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&bytes[..end]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|value| value.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                while bytes.len() < end + length {
                    let mut chunk = [0; 1024];
                    let read = socket.read(&mut chunk).await.unwrap();
                    bytes.extend_from_slice(&chunk[..read]);
                }
                let path = headers.lines().next().unwrap().to_string();
                let bearer = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("authorization: Bearer "))
                    .unwrap_or_default()
                    .to_string();
                let (status, body) = if !accepted(&bearer) {
                    ("401 Unauthorized", String::new())
                } else {
                    let used: serde_json::Value =
                        serde_json::from_slice(&bytes[end..end + length]).unwrap();
                    let total = used["usedOps"]["solve"].clone();
                    (
                        "200 OK",
                        serde_json::json!({"state":"active","grantedOps":"1000",
                            "budgetOps":"1000","totalOps":total,
                            "expiresAt":"2099-01-01T00:00:00Z"})
                        .to_string(),
                    )
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
                seen.push((path, bearer));
            }
            seen
        });
        (format!("http://{address}"), handle)
    }

    fn lease() -> SimulationLease {
        serde_json::from_value(serde_json::json!({
            "leaseId": SimulationLeaseId::from_uuid(Uuid::from_u128(2)),
            "state": "active", "grantedOps": "1000", "budgetOps": "1000",
            "expiresAt": "2099-01-01T00:00:00Z", "heartbeatSeconds": 30, "waived": false
        }))
        .unwrap()
    }

    fn usage(solve: u64) -> ReportSimulationUsage {
        ReportSimulationUsage {
            measurement_version: MeasurementVersion::parse("rsua-1").unwrap(),
            used_ops: runmat_simulation_usage::CompletedUsage {
                solve: OperationCount::from_operations(solve),
                ..Default::default()
            },
            grant_target_ops: None,
            other_detail: None,
        }
    }

    async fn session(
        endpoint: &str,
        clock: &Arc<AtomicI64>,
        refreshes: &Arc<AtomicU64>,
    ) -> LeaseSession {
        let tokens = FakeTokens {
            clock: clock.clone(),
            expires_at: 0,
            issued: 0,
            refreshes: refreshes.clone(),
        };
        LeaseSession::person(endpoint.to_string(), Box::new(tokens))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn heartbeats_past_the_token_lifetime_send_a_refreshed_token() {
        let (endpoint, server) = server(3, |bearer| !bearer.is_empty()).await;
        let clock = Arc::new(AtomicI64::new(1_000));
        let refreshes = Arc::new(AtomicU64::new(0));
        let mut session = session(&endpoint, &clock, &refreshes).await;
        let mut lease = lease();
        session.usage(&mut lease, &usage(1)).await.unwrap();
        // Four minutes in, the 5-minute token is still good.
        clock.store(1_240, Ordering::SeqCst);
        session.usage(&mut lease, &usage(2)).await.unwrap();
        // Ten minutes in: past expiry, so the session refreshes before sending.
        clock.store(1_600, Ordering::SeqCst);
        session.usage(&mut lease, &usage(3)).await.unwrap();
        let seen = server.await.unwrap();
        let bearers: Vec<_> = seen.iter().map(|(_, bearer)| bearer.as_str()).collect();
        assert_eq!(bearers, ["token-1", "token-1", "token-2"]);
        assert_eq!(refreshes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_rejected_token_is_refreshed_and_the_request_retried_once() {
        // The server has revoked token-1 although its stored expiry has not passed.
        let (endpoint, server) = server(2, |bearer| bearer == "token-2").await;
        let clock = Arc::new(AtomicI64::new(1_000));
        let refreshes = Arc::new(AtomicU64::new(0));
        let mut session = session(&endpoint, &clock, &refreshes).await;
        let budget = session.usage(&mut lease(), &usage(5)).await.unwrap();
        assert_eq!(budget.granted_ops, OperationCount::from_operations(1000));
        let seen = server.await.unwrap();
        assert!(seen.iter().all(|(path, _)| path.contains("/usage")));
        let bearers: Vec<_> = seen.iter().map(|(_, bearer)| bearer.as_str()).collect();
        assert_eq!(bearers, ["token-1", "token-2"]);
    }

    #[tokio::test]
    async fn a_token_rejected_after_refresh_asks_to_sign_in() {
        let (endpoint, server) = server(2, |_| false).await;
        let clock = Arc::new(AtomicI64::new(1_000));
        let refreshes = Arc::new(AtomicU64::new(0));
        let mut session = session(&endpoint, &clock, &refreshes).await;
        assert_eq!(
            session.usage(&mut lease(), &usage(5)).await,
            Err(FeaRunDenied::SignInRequired)
        );
        // One retry, not a loop.
        assert_eq!(server.await.unwrap().len(), 2);
    }
}

#[cfg(test)]
mod outage_tests;
