#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunMatAbiVersion {
    pub major: u16,
    pub minor: u16,
}

pub const RUNMAT_EXTENSION_ABI_VERSION: RunMatAbiVersion = RunMatAbiVersion { major: 1, minor: 0 };

impl RunMatAbiVersion {
    pub const fn is_compatible_with(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_requires_the_same_major_and_a_sufficient_minor() {
        assert!(RunMatAbiVersion { major: 1, minor: 3 }
            .is_compatible_with(RunMatAbiVersion { major: 1, minor: 2 }));
        assert!(!RunMatAbiVersion { major: 1, minor: 1 }
            .is_compatible_with(RunMatAbiVersion { major: 1, minor: 2 }));
        assert!(!RunMatAbiVersion { major: 2, minor: 0 }
            .is_compatible_with(RunMatAbiVersion { major: 1, minor: 0 }));
    }
}
