//! Retryable retirement of the exact project classloader before publication.

use super::*;

/// Verified old loader closure. Dropping an unpublished ticket restores
/// invocation admission under the still-published old configuration.
pub struct VerifiedJavaProjectRetirement<'a> {
    pub(super) owner: &'a JavaAdapter,
    pub(super) authority: super::super::super::project_retirement::ProjectRetirementTicket<'a>,
    _old_session: Option<Rc<JavaLane>>,
}

impl JavaAdapter {
    /// An old session stays fenced and owned after an unverified close so a
    /// later attempt can retry that same classloader. No inventory is changed.
    pub(crate) fn retire_project_session(
        &self,
        identity: &Rc<()>,
    ) -> Result<VerifiedJavaProjectRetirement<'_>, RuntimeError> {
        self.handles.admit_host_restart(&self.host_identity)?;
        if self.retirement_requested.get() {
            return Err(invalid_conversion("Java adapter is terminally retiring"));
        }
        let authority = self.project_retirement.begin(identity).ok_or_else(|| {
            invalid_conversion("Java project replacement authority is stale or already active")
        })?;
        let old_session = self
            .session
            .try_borrow()
            .map_err(|_| invalid_conversion("Java session is busy during project retirement"))?
            .as_ref()
            .cloned();
        if let Some(session) = old_session.as_ref() {
            session
                .retire_classloader()
                .map_err(java_invocation_error)?;
            let mut slot = self.session.borrow_mut();
            if !slot
                .as_ref()
                .is_some_and(|current| Rc::ptr_eq(current, session))
            {
                return Err(invalid_conversion(
                    "Java project classloader changed during retirement",
                ));
            }
            slot.take();
        }
        self.resources.borrow_mut().clear();
        self.java_to_foreign.borrow_mut().clear();
        self.callbacks.borrow_mut().clear();
        if let Ok(mut released) = self.released.0.lock() {
            released.clear();
        }
        Ok(VerifiedJavaProjectRetirement {
            owner: self,
            authority: authority.verified(),
            _old_session: old_session,
        })
    }
}
