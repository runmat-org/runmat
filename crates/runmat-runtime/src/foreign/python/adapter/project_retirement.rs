//! Retryable interpreter retirement for an atomic project replacement.

use super::isolated::ProjectRetirementGuard;
use super::*;

/// A verified lane or child retirement retains the old owner until publication.
/// If publication is abandoned, the old configuration can start a fresh host.
pub struct VerifiedPythonProjectRetirement<'a> {
    pub(super) owner: &'a PythonAdapter,
    pub(super) authority: super::super::super::project_retirement::ProjectRetirementTicket<'a>,
    _old_session: Option<PythonSession>,
}

impl PythonAdapter {
    pub(crate) async fn retire_project_session(
        &self,
        identity: &Rc<()>,
        runtime: Option<&RuntimeContext>,
    ) -> Result<VerifiedPythonProjectRetirement<'_>, RuntimeError> {
        self.handles.admit_host_restart(&self.host_identity)?;
        if self.retirement_requested.get() {
            return Err(invalid_call("Python adapter is terminally retiring"));
        }
        if self.isolated_call_active.get() {
            return Err(invalid_call(
                "Python isolated call is active during project retirement",
            ));
        }
        let authority = self.project_retirement.begin(identity).ok_or_else(|| {
            invalid_call("Python project replacement authority is stale or already active")
        })?;
        if self.configuration.borrow().execution_mode == PythonExecutionMode::InProcess {
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let complete = {
                    let slot = self.session.borrow();
                    slot.as_ref().is_none_or(PythonSession::request_retirement)
                };
                if complete {
                    let old_session = self.session.borrow_mut().take();
                    if old_session.is_some() {
                        self.clear_project_retired_resources();
                        self.terminated.set(true);
                    }
                    return Ok(VerifiedPythonProjectRetirement {
                        owner: self,
                        authority: authority.verified(),
                        _old_session: old_session,
                    });
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(invalid_call(
                        "Python interpreter lane retirement could not be verified",
                    ));
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }

        super::isolated::admit_missing_client(
            &self.spawn_unverified,
            self.isolated_client.borrow().is_some(),
        )?;
        if self.isolated_client.borrow().is_none() {
            return Ok(VerifiedPythonProjectRetirement {
                owner: self,
                authority: authority.verified(),
                _old_session: None,
            });
        }
        let state = self.isolated_state();
        let mut guard = ProjectRetirementGuard::begin(state.clone());
        let released = take_releases(&state.released)?;
        let outcome = guard
            .client()
            .as_mut()
            .expect("project retirement guard owns exact isolated client")
            .shutdown_retirement(runtime, released)
            .await;
        match outcome {
            PythonClientRetirement::Retained {
                retirement,
                prior_protocol,
            } => {
                let error = super::super::isolation::runtime_error(retirement);
                if let Some(prior) = prior_protocol {
                    let prior = super::super::isolation::runtime_error(prior);
                    return Err(crate::runtime_error::complete_with_cleanup::<()>(
                        Err(prior),
                        Err(error),
                    )
                    .expect_err("unverified retirement supplies an error"));
                }
                Err(error)
            }
            PythonClientRetirement::Retired { scope, result } => {
                let matches_placement = matches!(
                    (self.process_group_placement.get(), scope),
                    (
                        ProcessGroupPlacement::OwnGroup,
                        runmat_process_host::ContainedChildRetirement::OwnedTree
                    ) | (
                        ProcessGroupPlacement::JoinAncestorGroup,
                        runmat_process_host::ContainedChildRetirement::JoinedMember
                    )
                );
                if !matches_placement {
                    return Err(invalid_call(
                        "Python child retirement scope differs from project placement",
                    ));
                }
                guard.finish_verified();
                self.clear_project_retired_resources();
                result.map_err(super::super::isolation::runtime_error)?;
                Ok(VerifiedPythonProjectRetirement {
                    owner: self,
                    authority: authority.verified(),
                    _old_session: None,
                })
            }
        }
    }

    fn clear_project_retired_resources(&self) {
        self.resources.borrow_mut().clear();
        self.python_to_foreign.borrow_mut().clear();
        if let Ok(mut released) = self.released.0.lock() {
            released.clear();
        }
    }
}
