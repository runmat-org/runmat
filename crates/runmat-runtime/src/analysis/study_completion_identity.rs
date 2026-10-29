//! Allocation fence for materializing completed studies; never numerical authority.
use runmat_analysis_core::{CoupledPhysicsFamily, PhysicsFamily, ProcedureStepId};
use runmat_execution::Digest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisStudyCompletionIdentity {
    server_run_id: String,
    /// One execution of the definition. Each execution publishes under its own attempt run id
    /// (`<definition run id>_<attempt>`), so a rerun is a new attempt and never collides with an
    /// earlier attempt's publication; the definition run id resolves to the latest attempt.
    attempt: Option<String>,
}

impl AnalysisStudyCompletionIdentity {
    pub fn new(server_run_id: impl Into<String>) -> Result<Self, String> {
        let server_run_id = server_run_id.into();
        if server_run_id.is_empty()
            || server_run_id.len() > 512
            || !server_run_id.is_ascii()
            || server_run_id.trim() != server_run_id
            || server_run_id.chars().any(char::is_control)
        {
            return Err("remote analysis study completion identity is invalid".into());
        }
        Ok(Self {
            server_run_id,
            attempt: None,
        })
    }

    /// A completion of one execution attempt. `attempt` is a lowercase alphanumeric token that
    /// orders attempts of the same definition chronologically when compared as strings.
    pub fn with_attempt(
        server_run_id: impl Into<String>,
        attempt: impl Into<String>,
    ) -> Result<Self, String> {
        let attempt = attempt.into();
        if attempt.is_empty()
            || attempt.len() > 40
            || !attempt
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        {
            return Err("analysis study attempt identity is invalid".into());
        }
        let mut identity = Self::new(server_run_id)?;
        identity.attempt = Some(attempt);
        Ok(identity)
    }

    /// A fresh, chronologically ordered attempt token for one execution in this process.
    pub fn fresh_attempt() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos() as u64)
            .unwrap_or(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed) as u32;
        format!(
            "a{nanos:016x}{:05x}{sequence:03x}",
            std::process::id() & 0xfffff
        )
    }

    pub(super) fn local_run_id(
        &self,
        family: PhysicsFamily,
        authored_study_digest: Digest,
        step_id: &ProcedureStepId,
    ) -> String {
        // Preserve the existing structural key domain. Other physics cannot collide with it.
        let domain: &[u8] = match family {
            PhysicsFamily::Structural => b"runmat-remote-structural-study-completion/v2\0",
            PhysicsFamily::Thermal => b"runmat-remote-thermal-study-completion/v2\0",
            PhysicsFamily::Electromagnetic => {
                b"runmat-remote-electromagnetic-study-completion/v1\0"
            }
            PhysicsFamily::Acoustic => b"runmat-remote-acoustic-study-completion/v1\0",
            PhysicsFamily::Fluid => b"runmat-remote-fluid-study-completion/v1\0",
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) => {
                b"runmat-remote-thermomechanical-study-completion/v1\0"
            }
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal) => {
                b"runmat-remote-electrothermal-study-completion/v1\0"
            }
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
                b"runmat-remote-conjugate-heat-transfer-study-completion/v1\0"
            }
            PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
                b"runmat-remote-fluid-structure-study-completion/v1\0"
            }
        };
        let digest = Digest::sha256(
            [
                domain,
                super::storage::ARTIFACT_SCHEMA_VERSION.as_bytes(),
                b"\0".as_slice(),
                self.server_run_id.as_bytes(),
                authored_study_digest.bytes(),
                step_id.as_str().as_bytes(),
            ]
            .concat(),
        );
        let base = format!(
            "run_remote_{}",
            digest.to_string().trim_start_matches("sha256:")
        );
        match &self.attempt {
            Some(attempt) => format!("{base}_{attempt}"),
            None => base,
        }
    }
}

/// The definition run id and attempt token of an attempt run id, if it is one.
pub fn split_attempt_run_id(run_id: &str) -> Option<(&str, &str)> {
    let (base, attempt) = run_id.rsplit_once('_')?;
    (base.starts_with("run_remote_")
        && base.len() == "run_remote_".len() + 64
        && attempt.starts_with('a')
        && attempt
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit()))
    .then_some((base, attempt))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_validation_and_physics_domains_are_distinct() {
        for invalid in ["", " leading", "trailing ", "line\nbreak", "nonasciié"] {
            assert!(AnalysisStudyCompletionIdentity::new(invalid).is_err());
        }
        assert!(AnalysisStudyCompletionIdentity::new("x".repeat(513)).is_err());
        let allocation = AnalysisStudyCompletionIdentity::new("server-run").unwrap();
        let definition = Digest::sha256(b"authored definition");
        let step = ProcedureStepId("step".into());
        let structural = allocation.local_run_id(PhysicsFamily::Structural, definition, &step);
        let thermal = allocation.local_run_id(PhysicsFamily::Thermal, definition, &step);
        assert_ne!(structural, thermal);
        assert_eq!(
            thermal,
            allocation.local_run_id(PhysicsFamily::Thermal, definition, &step)
        );
        assert_ne!(
            thermal,
            allocation.local_run_id(PhysicsFamily::Thermal, Digest::sha256(b"other"), &step)
        );
        assert_ne!(
            thermal,
            allocation.local_run_id(
                PhysicsFamily::Thermal,
                definition,
                &ProcedureStepId("other".into())
            )
        );
        let expected = Digest::sha256(
            [
                b"runmat-remote-structural-study-completion/v2\0".as_slice(),
                super::super::storage::ARTIFACT_SCHEMA_VERSION.as_bytes(),
                b"\0".as_slice(),
                b"server-run".as_slice(),
                definition.bytes(),
                step.as_str().as_bytes(),
            ]
            .concat(),
        );
        assert_eq!(
            structural,
            format!(
                "run_remote_{}",
                expected.to_string().trim_start_matches("sha256:")
            )
        );
    }
}
