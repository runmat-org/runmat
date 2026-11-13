//! Retryable project-child retirement without discarding the published catalog.

use super::*;

pub struct VerifiedNativeProjectRetirement<'a> {
    pub(super) owner: &'a NativeFfiAdapter,
    pub(super) authority: super::super::super::project_retirement::ProjectRetirementTicket<'a>,
    // Keep the former private bytes owned until publication or rollback.
    _catalog: NativeProjectCatalog,
    _lease: Option<runmat_process_host::private_bytes::PrivateByteLease>,
}

impl NativeFfiAdapter {
    pub(crate) async fn retire_project_session(
        &self,
        identity: &Rc<()>,
        runtime: Option<&RuntimeContext>,
    ) -> Result<VerifiedNativeProjectRetirement<'_>, RuntimeError> {
        if self.isolation_policy.get().isolate || self.isolated_client.borrow().is_some() {
            self.handles.admit_host_restart(&self.host_identity)?;
        }
        if self.isolated_call_active.get() {
            return Err(invalid_call(
                "native project host is busy during retirement",
            ));
        }
        if self.bootstrap_unverified.is_unverified() && self.isolated_client.borrow().is_none() {
            return Err(invalid_call(
                "native child spawn or containment remains unverified",
            ));
        }
        if self
            .isolated_client
            .borrow()
            .as_ref()
            .is_some_and(|client| !client.permits_project_retirement())
        {
            return Err(invalid_call(
                "native project restart conflicts with live or unverified user state in the isolated child",
            ));
        }
        if !self.project_catalog.borrow().is_empty()
            && self.project_catalog.borrow().lease.is_none()
        {
            return Err(invalid_call(
                "native project catalog lacks its private byte owner",
            ));
        }
        let catalog = self.project_catalog.borrow().clone();
        let lease = self.private_byte_lease.borrow().clone();
        let attempt = self
            .project_retirement
            .begin(identity)
            .ok_or_else(|| invalid_call("native project retirement authority changed"))?;
        match self.retire_isolated_client(false, runtime).await {
            NativeFfiRetirement::Retained(error) => Err(error),
            NativeFfiRetirement::Retired { scope, result } => {
                let valid_scope = matches!(
                    (self.process_group_placement.get(), scope),
                    (_, None)
                        | (
                            ProcessGroupPlacement::OwnGroup,
                            Some(ContainedChildRetirement::OwnedTree)
                        )
                        | (
                            ProcessGroupPlacement::JoinAncestorGroup,
                            Some(ContainedChildRetirement::JoinedMember)
                        )
                );
                if !valid_scope {
                    return Err(invalid_call(
                        "native project child retirement scope differs from placement",
                    ));
                }
                result?;
                Ok(VerifiedNativeProjectRetirement {
                    owner: self,
                    authority: attempt.verified(),
                    _catalog: catalog,
                    _lease: lease,
                })
            }
        }
    }
}
