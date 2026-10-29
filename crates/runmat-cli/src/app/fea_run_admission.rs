//! The CLI owns network I/O; the runtime admission port stays synchronous and host-neutral.
use std::cell::Cell;
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::{
    replace_fea_run_admission, FeaRunAdmission, FeaRunAdmissionGuard, FeaRunDenied, FeaRunKind,
    FeaRunOutcome, FeaRunPermit, FeaRunRequest, FeaRunSite,
};
use runmat_server_client::usage::{
    AdmissionError, ClientSurface, CoreHourLeaseClient, CreateLeaseRequest, LeaseKind,
    LeaseOutcome, LeasePermit, UsageReport,
};

enum Work {
    Admit(
        CreateLeaseRequest,
        mpsc::SyncSender<std::result::Result<LeasePermit, FeaRunDenied>>,
    ),
    Heartbeat(LeasePermit, UsageReport),
    Settle(LeasePermit, LeaseOutcome, UsageReport),
    Shutdown,
}

struct CliAdmission(mpsc::Sender<Work>);

/// Restore the host port before stopping its worker, and drain all queued usage on normal exit.
pub(super) struct Installation {
    guard: Option<FeaRunAdmissionGuard>,
    sender: mpsc::Sender<Work>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for Installation {
    fn drop(&mut self) {
        drop(self.guard.take());
        let _ = self.sender.send(Work::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(super) fn install() -> Result<Installation> {
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("runmat-core-hour-leases".into())
        .spawn(move || {
            // Never block_on the caller's reactor: admission may run inside an async study.
            let reactor = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(reactor) => reactor,
                Err(error) => {
                    log::debug!("core-hour lease reactor unavailable: {error}");
                    return;
                }
            };
            work_loop(receiver, &reactor, None);
        })
        .context("failed to start core-hour lease worker")?;
    let mut installation = Installation {
        guard: None,
        sender,
        worker: Some(worker),
    };
    installation.guard = Some(
        replace_fea_run_admission(Some(Arc::new(CliAdmission(installation.sender.clone()))))
            .map_err(anyhow::Error::msg)?,
    );
    Ok(installation)
}

fn work_loop(
    receiver: mpsc::Receiver<Work>,
    reactor: &tokio::runtime::Runtime,
    mut client: Option<std::result::Result<CoreHourLeaseClient, AdmissionError>>,
) {
    for work in receiver {
        match work {
            Work::Admit(request, response) => {
                // Non-FEA commands must not refresh credentials, contact the
                // Server, or wait for a network timeout merely to shut down.
                if client.is_none() {
                    client = Some(reactor.block_on(CoreHourLeaseClient::from_environment()));
                }
                let result = match client.as_ref().expect("initialized on admission") {
                    Ok(client) => reactor.block_on(client.admit(request)).map_err(map_denial),
                    Err(error) => Err(map_denial_ref(error)),
                };
                let _ = response.send(result);
            }
            Work::Heartbeat(permit, usage) => {
                if let Some(Ok(client)) = &client {
                    if let Err(error) = reactor.block_on(client.heartbeat(&permit, usage)) {
                        log::debug!("core-hour heartbeat deferred: {error}");
                    }
                }
            }
            Work::Settle(permit, outcome, usage) => {
                if let Some(Ok(client)) = &client {
                    if let Err(error) = reactor.block_on(client.settle(&permit, outcome, usage)) {
                        log::debug!("core-hour settlement deferred: {error}");
                    }
                }
            }
            Work::Shutdown => break,
        }
    }
}

impl FeaRunAdmission for CliAdmission {
    fn admit(
        &self,
        request: &FeaRunRequest,
    ) -> std::result::Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        // Remote execution is authenticated and metered by the Server, never by local leases.
        if matches!(request.site, FeaRunSite::Dispatched) {
            return Ok(Box::new(DispatchedPermit));
        }
        let request = lease_request(request)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        self.0
            .send(Work::Admit(request, sender))
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        let lease = receiver
            .recv()
            .map_err(|_| FeaRunDenied::ServerUnreachable)??;
        Ok(Box::new(CliPermit {
            sender: self.0.clone(),
            lease: Some(lease),
            last_usage: Cell::new((0, 0)),
        }))
    }
}

fn lease_request(request: &FeaRunRequest) -> std::result::Result<CreateLeaseRequest, FeaRunDenied> {
    Ok(CreateLeaseRequest {
        idempotency_key: request.run_id.clone(),
        allocated_cpu_millicores: request
            .allocated_cpu_millicores
            .try_into()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?,
        estimated_wall_seconds: request.estimated_wall_seconds,
        kind: match request.kind {
            FeaRunKind::Meshing => LeaseKind::Meshing,
            FeaRunKind::Solve => LeaseKind::Solve,
            FeaRunKind::MeshAndSolve => LeaseKind::MeshAndSolve,
        },
        study_identity: Some(request.study_identity.clone()),
        client_surface: ClientSurface::Cli,
    })
}

fn usage_report(usage: &ResourceUsage) -> UsageReport {
    UsageReport {
        wall_millis: usage.wall_millis,
        cpu_core_millis: usage.cpu_core_millis,
    }
}

struct CliPermit {
    sender: mpsc::Sender<Work>,
    lease: Option<LeasePermit>,
    last_usage: Cell<(u64, u64)>,
}

impl FeaRunPermit for CliPermit {
    fn heartbeat(&self, usage: &ResourceUsage) {
        self.last_usage
            .set((usage.wall_millis, usage.cpu_core_millis));
        if let Some(lease) = &self.lease {
            let _ = self
                .sender
                .send(Work::Heartbeat(lease.clone(), usage_report(usage)));
        }
    }

    fn settle(mut self: Box<Self>, outcome: FeaRunOutcome, usage: &ResourceUsage) {
        if let Some(lease) = self.lease.take() {
            let outcome = match outcome {
                FeaRunOutcome::Completed => LeaseOutcome::Completed,
                FeaRunOutcome::Cancelled => LeaseOutcome::Cancelled,
                FeaRunOutcome::Failed => LeaseOutcome::Failed,
            };
            let _ = self
                .sender
                .send(Work::Settle(lease, outcome, usage_report(usage)));
        }
    }
}

impl Drop for CliPermit {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let (wall_millis, cpu_core_millis) = self.last_usage.get();
            let _ = self.sender.send(Work::Settle(
                lease,
                LeaseOutcome::Failed,
                UsageReport {
                    wall_millis,
                    cpu_core_millis,
                },
            ));
        }
    }
}

struct DispatchedPermit;
impl FeaRunPermit for DispatchedPermit {
    fn settle(self: Box<Self>, _: FeaRunOutcome, _: &ResourceUsage) {}
}

fn map_denial(error: AdmissionError) -> FeaRunDenied {
    map_denial_ref(&error)
}
fn map_denial_ref(error: &AdmissionError) -> FeaRunDenied {
    match error {
        AdmissionError::SignInRequired => FeaRunDenied::SignInRequired,
        AdmissionError::AllowanceUsed {
            refills_at_unix_seconds,
        } => FeaRunDenied::AllowanceUsed {
            refills_at_unix_seconds: *refills_at_unix_seconds,
        },
        AdmissionError::LeaseExpired => FeaRunDenied::LeaseExpired,
        AdmissionError::LicenseInvalid => FeaRunDenied::LicenseInvalid,
        AdmissionError::ServerUnreachable => FeaRunDenied::ServerUnreachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(site: FeaRunSite) -> FeaRunRequest {
        FeaRunRequest {
            run_id: "run_test".into(),
            kind: FeaRunKind::MeshAndSolve,
            site,
            allocated_cpu_millicores: 2000,
            estimated_wall_seconds: Some(60),
            study_identity: "sha256:test".into(),
        }
    }

    #[test]
    fn dispatched_runs_never_contact_local_worker_or_settle() {
        let (sender, receiver) = mpsc::channel();
        let permit = CliAdmission(sender)
            .admit(&request(FeaRunSite::Dispatched))
            .unwrap();
        permit.heartbeat(&ResourceUsage::default());
        permit.settle(FeaRunOutcome::Completed, &ResourceUsage::default());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn signed_out_denies_without_interactive_login() {
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let reactor = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            work_loop(
                receiver,
                &reactor,
                Some(Err(AdmissionError::SignInRequired)),
            );
        });
        let admission = CliAdmission(sender.clone());
        assert!(matches!(
            admission.admit(&request(FeaRunSite::Local)),
            Err(FeaRunDenied::SignInRequired)
        ));
        sender.send(Work::Shutdown).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn request_and_usage_keep_allocated_core_time() {
        let wire = lease_request(&request(FeaRunSite::Local)).unwrap();
        assert_eq!(wire.idempotency_key, "run_test");
        assert_eq!(wire.allocated_cpu_millicores, 2000);
        let report = usage_report(&ResourceUsage {
            wall_millis: 123,
            cpu_core_millis: 246,
            ..Default::default()
        });
        assert_eq!((report.wall_millis, report.cpu_core_millis), (123, 246));
    }
}
