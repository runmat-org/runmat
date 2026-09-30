//! Allocation fence for materializing completed studies; never numerical authority.
use runmat_analysis_core::{CoupledPhysicsFamily, PhysicsFamily, ProcedureStepId};
use runmat_execution::Digest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisStudyCompletionIdentity(String);

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
        Ok(Self(server_run_id))
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
                self.0.as_bytes(),
                authored_study_digest.bytes(),
                step_id.as_str().as_bytes(),
            ]
            .concat(),
        );
        format!(
            "run_remote_{}",
            digest.to_string().trim_start_matches("sha256:")
        )
    }
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
