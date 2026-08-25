#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RunMatExtensionCapabilities(pub u64);

impl RunMatExtensionCapabilities {
    pub const NONE: Self = Self(0);
    pub const INVOKE: Self = Self(1 << 0);
    pub const READ_VALUE: Self = Self(1 << 1);
    pub const WRITE_VALUE: Self = Self(1 << 2);
    pub const CALLBACK: Self = Self(1 << 3);
    pub const TRANSFER: Self = Self(1 << 4);
    pub const SERIALIZE: Self = Self(1 << 5);
    pub const ZERO_COPY: Self = Self(1 << 6);

    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl core::ops::BitOr for RunMatExtensionCapabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunMatExtensionNegotiation {
    pub capabilities: RunMatExtensionCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMatExtensionNegotiationError {
    IncompatibleVersion,
    MissingHostCapability,
}

pub const fn negotiate_extension(
    host_version: crate::RunMatAbiVersion,
    host_capabilities: RunMatExtensionCapabilities,
    extension_version: crate::RunMatAbiVersion,
    required_host_capabilities: RunMatExtensionCapabilities,
    provided_capabilities: RunMatExtensionCapabilities,
) -> Result<RunMatExtensionNegotiation, RunMatExtensionNegotiationError> {
    if !host_version.is_compatible_with(extension_version) {
        return Err(RunMatExtensionNegotiationError::IncompatibleVersion);
    }
    if !host_capabilities.contains(required_host_capabilities) {
        return Err(RunMatExtensionNegotiationError::MissingHostCapability);
    }
    Ok(RunMatExtensionNegotiation {
        capabilities: host_capabilities.intersect(provided_capabilities),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunMatAbiVersion;

    #[test]
    fn negotiation_checks_version_and_required_host_capabilities() {
        let host_version = RunMatAbiVersion { major: 1, minor: 2 };
        let host = RunMatExtensionCapabilities::INVOKE | RunMatExtensionCapabilities::CALLBACK;
        let negotiated = negotiate_extension(
            host_version,
            host,
            RunMatAbiVersion { major: 1, minor: 1 },
            RunMatExtensionCapabilities::CALLBACK,
            RunMatExtensionCapabilities::INVOKE | RunMatExtensionCapabilities::ZERO_COPY,
        )
        .unwrap();
        assert_eq!(negotiated.capabilities, RunMatExtensionCapabilities::INVOKE);

        assert_eq!(
            negotiate_extension(
                host_version,
                host,
                RunMatAbiVersion { major: 2, minor: 0 },
                RunMatExtensionCapabilities::NONE,
                RunMatExtensionCapabilities::NONE,
            ),
            Err(RunMatExtensionNegotiationError::IncompatibleVersion)
        );
        assert_eq!(
            negotiate_extension(
                host_version,
                host,
                RunMatAbiVersion { major: 1, minor: 0 },
                RunMatExtensionCapabilities::WRITE_VALUE,
                RunMatExtensionCapabilities::NONE,
            ),
            Err(RunMatExtensionNegotiationError::MissingHostCapability)
        );
    }
}
