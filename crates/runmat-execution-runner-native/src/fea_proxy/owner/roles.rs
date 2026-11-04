//! Role-bound construction and delegated-grant lifecycle on the private channel.
use super::*;

impl Parent {
    pub(super) fn with_role(
        provider: Provider,
        assignment: ProgramExecutionAssignment,
        resources: ResourceRequest,
        original_deadline: Option<u64>,
        invocation_bytes: u64,
        role: StorageRole,
    ) -> Result<Arc<Self>, String> {
        assignment.validate().map_err(|e| e.to_string())?;
        assignment
            .resources
            .validate_for_request(&resources)
            .map_err(|e| e.to_string())?;
        resources.validate().map_err(|e| e.to_string())?;
        if resources.memory_bytes < storage_bytes(0, role)? {
            return Err("FEA proxy metadata exceeds worker reservation".into());
        }
        let deadline = now()
            .checked_add(resources.max_wall_millis)
            .ok_or("FEA proxy deadline overflow")?;
        let deadline = original_deadline.map_or(deadline, |original| original.min(deadline));
        let root_minimum_bytes = storage_bytes(invocation_bytes, StorageRole::FirstRoot)?;
        Ok(Arc::new(Self {
            provider,
            assignment,
            resources,
            deadline,
            started: Instant::now(),
            root_minimum_bytes,
            state: Mutex::new(State {
                sequence: 0,
                active: None,
            }),
            evidence: Mutex::new(None),
            denied: Mutex::new(None),
            stopped: Mutex::new(None),
            delegated: None,
        }))
    }
    pub(crate) fn from_delegation(
        grant: &mut crate::fea_delegation::WorkerFeaGrant,
        owner: Arc<crate::fea_delegation::ChildGrant>,
        original_deadline: Option<u64>,
        invocation_bytes: u64,
    ) -> Result<Arc<Self>, String> {
        owner
            .validate_assignment(&grant.assignment)
            .map_err(|error| error.to_string())?;
        let mut parent = Self::with_role(
            Err(FeaRunDenied::LicenseInvalid),
            grant.assignment.clone(),
            grant.resources.clone(),
            original_deadline,
            invocation_bytes,
            StorageRole::Delegated,
        )?;
        let budget = parent
            .clamp(owner.budget().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        grant.parent.budget = budget;
        let inner = Arc::get_mut(&mut parent).expect("unpublished delegated parent");
        *inner
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner()) = State {
            sequence: 0,
            active: Some(Active {
                root: 1,
                cpu: u64::from(grant.resources.cpu_millicores),
                started: Instant::now(),
                last_wall: 0,
                budget,
                last_completed: CompletedUsage::default(),
                permit: Box::new(DelegatedPermit {
                    owner: Arc::clone(&owner),
                    initial: budget,
                }),
            }),
        };
        *inner
            .evidence
            .get_mut()
            .unwrap_or_else(|error| error.into_inner()) = Some(Evidence { budget });
        inner.delegated = Some(owner);
        Ok(parent)
    }
    pub(crate) fn delegated_budget(&self) -> Result<FeaRunBudget, FeaRunDenied> {
        self.check()?;
        self.clamp(
            self.delegated
                .as_ref()
                .ok_or(FeaRunDenied::LicenseInvalid)?
                .budget()?,
        )
    }
    pub(crate) fn may_execute(&self) -> Result<(), FeaRunDenied> {
        self.delegated
            .as_ref()
            .map_or(Ok(()), |owner| owner.may_execute())
    }
    pub(crate) fn retire_delegated(&self, successful: bool) -> Result<(), FeaRunDenied> {
        let Some(owner) = &self.delegated else {
            return Ok(());
        };
        if !successful {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(active) = state.active.take() {
                owner.settled(&FeaRunUsage {
                    completed_ops: active.last_completed,
                    resources: active.observed(),
                })?;
                *self
                    .evidence
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = None;
            }
        }
        owner.retire(successful)
    }
}

struct DelegatedPermit {
    owner: Arc<crate::fea_delegation::ChildGrant>,
    initial: FeaRunBudget,
}
impl FeaRunPermit for DelegatedPermit {
    fn budget(&self) -> FeaRunBudget {
        self.initial
    }
    fn heartbeat(&self, usage: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.owner.report(usage, None)
    }
    fn request_grant(
        &self,
        usage: &FeaRunUsage,
        required: OperationCount,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.owner.report(usage, Some(required))
    }
    fn settle(self: Box<Self>, _: FeaRunOutcome, usage: &FeaRunUsage) -> Result<(), FeaRunDenied> {
        // Reporting completion does not release capacity: the process owner
        // must establish full-tree retirement before closing the reservation.
        self.owner.settled(usage)
    }
}
