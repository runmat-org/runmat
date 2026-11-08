use super::*;
use runmat_process_host::ipc::private_control::ControlFrameWriter;
use std::sync::{mpsc, Mutex};
use std::time::Duration;

struct Pending {
    sequence: u64,
    reply: mpsc::SyncSender<Reply>,
}
#[derive(Default)]
pub struct Replies {
    pending: Mutex<Option<Pending>>,
    invalid: std::sync::atomic::AtomicBool,
    reader_ready: std::sync::atomic::AtomicBool,
}
impl Replies {
    pub(crate) fn activate_reader(&self) {
        self.reader_ready
            .store(true, std::sync::atomic::Ordering::Release);
    }
    pub fn observe(&self, bytes: &[u8]) -> bool {
        if bytes.len() as u64 > MAX_FRAME_BYTES {
            return false;
        }
        #[derive(Deserialize)]
        #[serde(tag = "message", rename_all = "snake_case")]
        enum Probe {
            FeaAdmissionReply {},
        }
        if !matches!(
            serde_json::from_slice::<Probe>(bytes),
            Ok(Probe::FeaAdmissionReply {})
        ) {
            return false;
        }
        let parsed = serde_json::from_slice::<crate::protocol::WorkerDriverMessage>(bytes).ok();
        if let Some(pending) = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let reply = match parsed {
                Some(crate::protocol::WorkerDriverMessage::FeaAdmissionReply {
                    sequence,
                    reply,
                }) if sequence == pending.sequence => reply,
                _ => {
                    self.invalid
                        .store(true, std::sync::atomic::Ordering::Release);
                    Reply::Denied {
                        denied: FeaRunDenied::LicenseInvalid,
                    }
                }
            };
            let _ = pending.reply.try_send(reply);
        } else {
            self.invalid
                .store(true, std::sync::atomic::Ordering::Release);
        }
        true
    }
}
struct ClientState {
    sequence: u64,
    active: Option<u64>,
}
pub struct Client {
    assignment: ProgramExecutionAssignment,
    writer: ControlFrameWriter,
    replies: Arc<Replies>,
    calls: Mutex<ClientState>,
    cpu: u64,
}
impl Client {
    pub(crate) fn delegated_permit(
        self: &Arc<Self>,
        budget: FeaRunBudget,
    ) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        budget.validate()?;
        let mut state = self.calls.lock().unwrap_or_else(|error| error.into_inner());
        if state.sequence != 0 || state.active.is_some() {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.active = Some(1);
        Ok(Box::new(ProxyPermit {
            client: Arc::clone(self),
            root: 1,
            budget,
        }))
    }
    pub fn new(
        assignment: ProgramExecutionAssignment,
        cpu: u64,
        writer: ControlFrameWriter,
        replies: Arc<Replies>,
    ) -> Arc<Self> {
        Arc::new(Self {
            assignment,
            writer,
            replies,
            calls: Mutex::new(ClientState {
                sequence: 0,
                active: None,
            }),
            cpu,
        })
    }
    fn call_locked(
        &self,
        state: &mut ClientState,
        operation: Operation,
        timeout: Duration,
    ) -> Result<Reply, FeaRunDenied> {
        if !self
            .replies
            .reader_ready
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // A failed bootstrap can drop its attached permit before starting
            // the reply reader. Cleanup must not wait for an impossible reply.
            return Err(FeaRunDenied::ServerUnreachable);
        }
        if self
            .replies
            .invalid
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.sequence = state
            .sequence
            .checked_add(1)
            .ok_or(FeaRunDenied::LicenseInvalid)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        *self
            .replies
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Pending {
            sequence: state.sequence,
            reply: sender,
        });
        let call = Call {
            assignment: self.assignment.clone(),
            sequence: state.sequence,
            operation,
        };
        let encoded = runmat_canonical_codec::encode_json_bounded(
            &crate::protocol::WorkerProcessMessage::FeaAdmissionCall { call },
            MAX_FRAME_BYTES,
        )
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let result = self
            .writer
            .write_blocking(encoded)
            .map_err(|_| FeaRunDenied::ServerUnreachable)
            .and_then(|()| {
                receiver
                    .recv_timeout(timeout)
                    .map_err(|_| FeaRunDenied::ServerUnreachable)
            });
        self.replies
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        match result? {
            Reply::Denied { denied } => Err(denied),
            reply => Ok(reply),
        }
    }
}
struct ProxyPermit {
    client: Arc<Client>,
    root: u64,
    budget: FeaRunBudget,
}
pub struct WorkerAdmission(pub Arc<Client>);
impl FeaRunAdmission for WorkerAdmission {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        let mut state = self.0.calls.lock().unwrap_or_else(|e| e.into_inner());
        if state.active.is_some()
            || request.site != FeaRunSite::Local
            || request.allocated_cpu_millicores > self.0.cpu
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let reply = self.0.call_locked(
            &mut state,
            Operation::Start {
                request: request.clone(),
            },
            parent_reply_timeout(None),
        )?;
        let Reply::Budget { budget } = reply else {
            return Err(FeaRunDenied::LicenseInvalid);
        };
        budget.validate()?;
        let root = state.sequence;
        state.active = Some(root);
        Ok(Box::new(ProxyPermit {
            client: self.0.clone(),
            root,
            budget,
        }))
    }
    fn allocated_cpu_capacity_millicores(&self) -> Option<u64> {
        Some(self.0.cpu)
    }
}
impl FeaRunPermit for ProxyPermit {
    fn budget(&self) -> FeaRunBudget {
        self.budget
    }
    fn heartbeat(&self, usage: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.renew(Operation::Heartbeat {
            root: self.root,
            usage: usage.clone(),
        })
    }
    fn request_grant(
        &self,
        usage: &FeaRunUsage,
        required: runmat_simulation_usage::OperationCount,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.renew(Operation::RequestGrant {
            root: self.root,
            usage: usage.clone(),
            required,
        })
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        let mut state = self.client.calls.lock().unwrap_or_else(|e| e.into_inner());
        if state.active != Some(self.root) {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let reply = self.client.call_locked(
            &mut state,
            Operation::Settle {
                root: self.root,
                outcome,
                usage: usage.clone(),
            },
            parent_reply_timeout(Some(&self.budget)),
        );
        state.active = None;
        match reply? {
            Reply::Settled => Ok(()),
            _ => Err(FeaRunDenied::LicenseInvalid),
        }
    }
}
impl ProxyPermit {
    fn renew(&self, operation: Operation) -> Result<FeaRunBudget, FeaRunDenied> {
        let mut state = self.client.calls.lock().unwrap_or_else(|e| e.into_inner());
        if state.active != Some(self.root) {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        match self.client.call_locked(
            &mut state,
            operation,
            parent_reply_timeout(Some(&self.budget)),
        )? {
            Reply::Budget { budget } => Ok(budget),
            _ => Err(FeaRunDenied::LicenseInvalid),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_reply_fails_one_bounded_mailbox_without_touching_other_controls() {
        let replies = Replies::default();
        assert!(!replies.observe(br#"{"message":"collective_completion"}"#));
        assert!(!replies.observe(&vec![b' '; MAX_FRAME_BYTES as usize + 1]));
        let (sender, receiver) = mpsc::sync_channel(1);
        *replies.pending.lock().unwrap() = Some(Pending {
            sequence: 7,
            reply: sender,
        });
        let bytes = serde_json::to_vec(&crate::protocol::WorkerDriverMessage::FeaAdmissionReply {
            sequence: 6,
            reply: Reply::Settled,
        })
        .unwrap();
        assert!(replies.observe(&bytes));
        assert_eq!(
            receiver.try_recv().unwrap(),
            Reply::Denied {
                denied: FeaRunDenied::LicenseInvalid
            }
        );
        assert!(replies.invalid.load(std::sync::atomic::Ordering::Acquire));
        assert!(replies.pending.lock().unwrap().is_none());
    }
}
