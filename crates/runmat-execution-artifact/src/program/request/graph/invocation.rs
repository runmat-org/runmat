//! Captured invocation controls, separate from project execution policy.
use crate::{error::contract_error, ArtifactResult};
use runmat_execution::{console::ConsoleLimits, EnvironmentSnapshot};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphInvocationControls {
    pub environment: EnvironmentSnapshot,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_controls")]
    pub console: Option<ConsoleLimits>,
}

impl GraphInvocationControls {
    pub fn matches_original(
        controls: Option<&Self>,
        environment: &EnvironmentSnapshot,
        console: Option<ConsoleLimits>,
    ) -> bool {
        match controls {
            Some(controls) => controls.environment == *environment && controls.console == console,
            None => environment.entries().is_empty() && console.is_none(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.environment.entries().is_empty() && self.console.is_none()
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        self.environment.validate().map_err(contract_error)?;
        if let Some(console) = self.console {
            console.validate().map_err(contract_error)?;
        }
        Ok(())
    }

    pub fn minimum_memory_bytes(&self) -> ArtifactResult<u64> {
        runmat_execution::console::invocation_memory_bytes(&self.environment, self.console)
            .map_err(contract_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_controls_keep_exact_identity_bounds_and_explicit_console_absence() {
        let empty = EnvironmentSnapshot::default();
        let environment =
            EnvironmentSnapshot::new(vec![("FIXTURE".into(), "captured".into())]).unwrap();
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 1024,
        };
        let console = Some(limits);
        let controls = GraphInvocationControls {
            environment: environment.clone(),
            console,
        };
        controls.validate().unwrap();
        assert!(!controls.is_empty());
        assert!(GraphInvocationControls::matches_original(
            Some(&controls),
            &environment,
            console
        ));
        assert!(!GraphInvocationControls::matches_original(
            Some(&controls),
            &empty,
            console
        ));
        assert!(!GraphInvocationControls::matches_original(
            Some(&controls),
            &environment,
            None
        ));
        assert!(GraphInvocationControls::matches_original(
            None, &empty, None
        ));
        assert!(!GraphInvocationControls::matches_original(
            None,
            &environment,
            None
        ));
        assert!(!GraphInvocationControls::matches_original(
            None, &empty, console
        ));
        assert_eq!(
            controls.minimum_memory_bytes().unwrap(),
            environment
                .storage_profile()
                .unwrap()
                .minimum_memory_bytes()
                .unwrap()
                + limits.storage_profile().unwrap().minimum_memory_bytes
        );
        let bytes = serde_json::to_vec(&controls).unwrap();
        assert_eq!(
            serde_json::from_slice::<GraphInvocationControls>(&bytes).unwrap(),
            controls
        );
        assert!(serde_json::from_str::<GraphInvocationControls>(
            r#"{"environment":{"entries":[]}}"#
        )
        .is_err());
        assert!(serde_json::from_str::<GraphInvocationControls>(
            r#"{"environment":{"entries":[]},"console":{"max_records":0,"max_text_bytes":1}}"#
        )
        .is_err());
        let absent: GraphInvocationControls =
            serde_json::from_str(r#"{"environment":{"entries":[]},"console":null}"#).unwrap();
        assert!(absent.is_empty());
    }
}
