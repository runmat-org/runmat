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
