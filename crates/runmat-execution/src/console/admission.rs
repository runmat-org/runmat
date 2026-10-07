//! A shared additional component for source capture and bounded JSON transport.
//! Parent retention, allocator overhead and whole-execution RSS are separate.
use super::ConsoleLimits;
use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsoleStorageProfile {
    pub capture_bytes: u64,
    pub max_encoded_bytes: u64,
    pub minimum_memory_bytes: u64,
}

impl ConsoleLimits {
    pub fn storage_profile(self) -> Result<ConsoleStorageProfile, ContractError> {
        self.validate()?;
        let capture_bytes = self.capture_storage_bytes()?;
        // JSON text requires at most six bytes per UTF-8 byte. Fixed record
        // fields, u64 sequence, stream tags and framing fit 256 bytes/record;
        // controls and terminal metadata fit the additional 1024-byte bound.
        let max_encoded_bytes = self.max_text_bytes * 6 + u64::from(self.max_records) * 256 + 1024;
        // Original capture, an in-flight record inventory and encoding buffer.
        // Does not account for parent checkpoints, retained clones or allocator nodes.
        let minimum_memory_bytes = capture_bytes * 2 + max_encoded_bytes;
        Ok(ConsoleStorageProfile {
            capture_bytes,
            max_encoded_bytes,
            minimum_memory_bytes,
        })
    }

    pub fn validate_transport(self, max_message_bytes: u64) -> Result<(), ContractError> {
        self.validate()?;
        // One original record must fit a frame without splitting its write boundary.
        if self.max_text_bytes * 6 + 256 > max_message_bytes {
            return Err(ContractError::invalid(
                "console controls",
                "an original record can exceed the transport frame",
            ));
        }
        Ok(())
    }
}

/// Required optional field: omitted controls never silently become disabled.
pub fn deserialize_controls<'de, D>(deserializer: D) -> Result<Option<ConsoleLimits>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let limits: Option<ConsoleLimits> = serde::Deserialize::deserialize(deserializer)?;
    if let Some(limits) = limits {
        limits.validate().map_err(serde::de::Error::custom)?;
    }
    Ok(limits)
}

/// Shared lower bound for the original invocation components, excluding execution work.
pub fn invocation_memory_bytes(
    environment: &crate::EnvironmentSnapshot,
    console: Option<ConsoleLimits>,
) -> Result<u64, ContractError> {
    let environment = environment.storage_profile()?.minimum_memory_bytes()?;
    let console = console.map(|limits| limits.storage_profile()).transpose()?;
    environment
        .checked_add(console.map_or(0, |profile| profile.minimum_memory_bytes))
        .ok_or_else(|| ContractError::invalid("invocation context", "storage overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleRecord;

    #[derive(serde::Deserialize)]
    struct Controls {
        #[serde(deserialize_with = "deserialize_controls")]
        console: Option<ConsoleLimits>,
    }

    #[test]
    fn omitted_controls_reject_and_explicit_null_remains_not_requested() {
        assert!(serde_json::from_str::<Controls>("{}").is_err());
        assert!(serde_json::from_str::<Controls>(r#"{"console":null}"#)
            .unwrap()
            .console
            .is_none());
        assert!(serde_json::from_str::<Controls>(
            r#"{"console":{"max_records":0,"max_text_bytes":1}}"#
        )
        .is_err());
    }

    #[test]
    fn invocation_components_are_additive_and_do_not_replace_environment_storage() {
        let environment =
            crate::EnvironmentSnapshot::new(vec![("TEST".into(), "original".into())]).unwrap();
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 1024,
        };
        let baseline = invocation_memory_bytes(&environment, None).unwrap();
        assert_eq!(
            baseline,
            environment
                .storage_profile()
                .unwrap()
                .minimum_memory_bytes()
                .unwrap()
        );
        assert_eq!(
            invocation_memory_bytes(&environment, Some(limits)).unwrap(),
            baseline + limits.storage_profile().unwrap().minimum_memory_bytes
        );
    }

    #[test]
    fn escaping_controls_and_metadata_fit_the_shared_wire_component() {
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 16,
        };
        let mut capture = super::super::ConsoleCapture::new(limits).unwrap();
        for stream in [
            super::super::ConsoleStream::Stdout,
            super::super::ConsoleStream::Stderr,
        ] {
            capture.push(stream, "\0\n\u{1}\tλ".into()).unwrap();
        }
        let transcript = capture.finish(true);
        let profile = limits.storage_profile().unwrap();
        assert!(serde_json::to_vec(&transcript).unwrap().len() as u64 <= profile.max_encoded_bytes);
        assert_eq!(
            profile.capture_bytes,
            4 * std::mem::size_of::<ConsoleRecord>() as u64 + 16
        );
        assert_eq!(
            profile.minimum_memory_bytes,
            profile.capture_bytes * 2 + profile.max_encoded_bytes
        );
        assert!(limits.validate_transport(100).is_err());
        limits.validate_transport(1024).unwrap();
    }
}
