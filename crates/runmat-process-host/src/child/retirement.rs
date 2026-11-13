use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use super::{CapturedProcessOutput, ChildOutput, ChildProcess, ChildStdio, ProcessExit};
use crate::ProcessHostResult;

/// Private bytes that must survive when a child cannot be proved retired.
/// Persistence must retain the exact resource even if writing a recovery
/// marker fails.
pub trait ProcessRetirementLease: Send {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String>;
}

/// The process owner alone advances this observation after OS spawn and after
/// bounded retirement. A post-spawn containment error remains unverified.
#[derive(Clone, Default)]
pub struct ProcessSpawnEvidence(Arc<AtomicBool>);

impl ProcessSpawnEvidence {
    pub fn is_unverified(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub(crate) fn mark_child_created(&self) {
        self.0.store(true, Ordering::Release);
    }
    fn mark_retired(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Couples an owned process tree to its exact private-byte lease.
pub struct RetirementBoundChild<L: ProcessRetirementLease> {
    child: ChildProcess,
    lease: L,
    retired: bool,
    preserved: bool,
    retirement_unverified: Option<ProcessSpawnEvidence>,
}

impl<L: ProcessRetirementLease> RetirementBoundChild<L> {
    pub fn new(child: ChildProcess, lease: L) -> Result<Self, Box<(ChildProcess, L)>> {
        if !child.owns_process_tree() {
            return Err(Box::new((child, lease)));
        }
        Ok(Self::new_bound(child, lease))
    }

    fn new_bound(child: ChildProcess, lease: L) -> Self {
        Self {
            child,
            lease,
            retired: false,
            preserved: false,
            retirement_unverified: None,
        }
    }

    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }
    pub fn captured_stderr(&self) -> CapturedProcessOutput {
        self.child.captured_stderr()
    }
    pub fn take_stdio(&mut self) -> ProcessHostResult<ChildStdio> {
        self.child.take_stdio()
    }
    pub fn take_application_output(&mut self) -> ProcessHostResult<ChildOutput> {
        self.child.take_application_output()
    }
    pub fn lease(&self) -> &L {
        &self.lease
    }
    pub fn lease_mut(&mut self) -> &mut L {
        &mut self.lease
    }
    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        if self.child.owns_process_tree() {
            self.wait_and_retire_tree().await
        } else {
            // A joined member never retires its ancestor's tree.
            self.child.wait().await
        }
    }
    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        if !self.child.owns_process_tree() || self.retired {
            return self.child.try_wait();
        }
        if self.preserved {
            return Err(crate::ProcessHostError::Protocol(
                "child retirement is unverified; exact private bytes are preserved".into(),
            ));
        }
        match self.child.try_wait_unreaped() {
            Ok(exit) => Ok(exit),
            Err(error) => {
                let _ = self.persist_unverified("child exit observation could not be verified");
                Err(error)
            }
        }
    }

    pub async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        if self.retired {
            return Ok(());
        }
        if self.preserved {
            return Err(crate::ProcessHostError::Protocol(
                "child retirement is unverified; exact private bytes are preserved".into(),
            ));
        }
        self.child.terminate_tree().await?;
        self.retired = true;
        if let Some(flag) = &self.retirement_unverified {
            flag.mark_retired();
        }
        Ok(())
    }

    pub async fn wait_and_retire_tree(&mut self) -> ProcessHostResult<ProcessExit> {
        if self.retired {
            return self.child.wait().await;
        }
        if self.preserved {
            return Err(crate::ProcessHostError::Protocol(
                "child retirement is unverified; exact private bytes are preserved".into(),
            ));
        }
        let exit = match self.child.wait_and_retire_tree().await {
            Ok(exit) => exit,
            Err(error) => {
                // A failed observation may mean somebody already reaped the
                // leader. Do not retry numeric group signaling during Drop.
                let _ = self.persist_unverified("child completion could not be verified");
                return Err(error);
            }
        };
        self.retired = true;
        if let Some(flag) = &self.retirement_unverified {
            flag.mark_retired();
        }
        Ok(exit)
    }

    pub fn persist_unverified(&mut self, reason: &str) -> Result<(), String> {
        if !self.retired && !self.preserved {
            let result = self.lease.persist_unverified(self.child.id(), reason);
            self.preserved = true;
            return result;
        }
        Ok(())
    }
}

impl<L: ProcessRetirementLease> Drop for RetirementBoundChild<L> {
    fn drop(&mut self) {
        if self.retired || self.preserved {
            return;
        }
        match self.child.terminate_tree_blocking() {
            Ok(()) => {
                self.retired = true;
                if let Some(flag) = &self.retirement_unverified {
                    flag.mark_retired();
                }
            }
            Err(error) => {
                let persisted = self.lease.persist_unverified(
                    self.child.id(),
                    &format!("abandoned child process could not be retired: {error}"),
                );
                self.preserved = true;
                if let Err(marker_error) = persisted {
                    eprintln!("child retirement is unverified; exact private bytes were preserved but recovery marker failed: {marker_error}");
                }
            }
        }
    }
}

/// A joined child is only a member-retirement proof. Its lease must contain
/// member-owned bytes, never the ancestor's full-tree scratch lease.
pub struct MemberRetirementBoundChild<L: ProcessRetirementLease>(RetirementBoundChild<L>);

impl<L: ProcessRetirementLease> MemberRetirementBoundChild<L> {
    pub fn new(child: ChildProcess, lease: L) -> Result<Self, Box<(ChildProcess, L)>> {
        if child.owns_process_tree() {
            return Err(Box::new((child, lease)));
        }
        Ok(Self(RetirementBoundChild::new_bound(child, lease)))
    }
    pub fn id(&self) -> Option<u32> {
        self.0.id()
    }
    pub fn captured_stderr(&self) -> CapturedProcessOutput {
        self.0.captured_stderr()
    }
    pub fn take_stdio(&mut self) -> ProcessHostResult<ChildStdio> {
        self.0.take_stdio()
    }
    pub fn take_application_output(&mut self) -> ProcessHostResult<ChildOutput> {
        self.0.take_application_output()
    }
    pub fn lease(&self) -> &L {
        self.0.lease()
    }
    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        self.0.wait().await
    }
    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        self.0.try_wait()
    }
    pub async fn terminate_member(&mut self) -> ProcessHostResult<()> {
        self.0.terminate_tree().await
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContainedChildRetirement {
    OwnedTree,
    JoinedMember,
}

/// Bootstrap owner for either a standalone tree or a joined member. A joined
/// member's retirement does not certify retirement of its ancestor group.
pub enum BootstrapBoundChild<L: ProcessRetirementLease> {
    Tree(RetirementBoundChild<L>),
    Member(MemberRetirementBoundChild<L>),
}

impl<L: ProcessRetirementLease> BootstrapBoundChild<L> {
    pub fn new(child: ChildProcess, lease: L, retirement_unverified: ProcessSpawnEvidence) -> Self {
        if child.owns_process_tree() {
            let mut bound =
                RetirementBoundChild::new(child, lease).unwrap_or_else(|_| unreachable!());
            bound.retirement_unverified = Some(retirement_unverified);
            Self::Tree(bound)
        } else {
            let mut bound =
                MemberRetirementBoundChild::new(child, lease).unwrap_or_else(|_| unreachable!());
            bound.0.retirement_unverified = Some(retirement_unverified);
            Self::Member(bound)
        }
    }

    pub fn lease(&self) -> &L {
        match self {
            Self::Tree(child) => child.lease(),
            Self::Member(child) => child.lease(),
        }
    }

    pub fn id(&self) -> Option<u32> {
        match self {
            Self::Tree(child) => child.id(),
            Self::Member(child) => child.id(),
        }
    }

    pub fn captured_stderr(&self) -> CapturedProcessOutput {
        match self {
            Self::Tree(child) => child.captured_stderr(),
            Self::Member(child) => child.captured_stderr(),
        }
    }

    pub fn take_stdio(&mut self) -> ProcessHostResult<ChildStdio> {
        match self {
            Self::Tree(child) => child.take_stdio(),
            Self::Member(child) => child.take_stdio(),
        }
    }
    pub fn take_application_output(&mut self) -> ProcessHostResult<ChildOutput> {
        match self {
            Self::Tree(child) => child.take_application_output(),
            Self::Member(child) => child.take_application_output(),
        }
    }

    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        match self {
            Self::Tree(child) => child.wait().await,
            Self::Member(child) => child.wait().await,
        }
    }

    pub async fn wait_and_retire_tree(&mut self) -> ProcessHostResult<ProcessExit> {
        match self {
            Self::Tree(child) => child.wait_and_retire_tree().await,
            Self::Member(_) => Err(crate::ProcessHostError::Configuration(
                "joined child cannot retire its ancestor's process tree".into(),
            )),
        }
    }

    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        match self {
            Self::Tree(child) => child.try_wait(),
            Self::Member(child) => child.try_wait(),
        }
    }

    pub async fn terminate_contained_child(
        &mut self,
    ) -> ProcessHostResult<ContainedChildRetirement> {
        match self {
            Self::Tree(child) => child
                .terminate_tree()
                .await
                .map(|()| ContainedChildRetirement::OwnedTree),
            Self::Member(child) => child
                .terminate_member()
                .await
                .map(|()| ContainedChildRetirement::JoinedMember),
        }
    }
}
