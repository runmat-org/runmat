//! Target-compiled acceleration backend inventory for a linked native runtime.
//!
//! These bits describe compiled support, not a discovered device, selected
//! provider, or permission for a particular program to use acceleration.

use crate::ContractError;

pub const NATIVE_ACCELERATOR_PROFILE_SYMBOL: &str = "RUNMAT_AOT_ACCELERATOR_PROFILE_V1";
pub const NATIVE_ACCELERATOR_PROFILE_BYTES: usize = 12;
const MAGIC: [u8; 8] = *b"RMACCPR1";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAcceleratorProfile {
    pub wgpu: bool,
    pub cuda: bool,
}

impl NativeAcceleratorProfile {
    pub const NONE: Self = Self {
        wgpu: false,
        cuda: false,
    };

    pub const fn has_backend(self) -> bool {
        self.wgpu || self.cuda
    }

    pub const fn canonical_bytes(self) -> [u8; NATIVE_ACCELERATOR_PROFILE_BYTES] {
        [
            MAGIC[0],
            MAGIC[1],
            MAGIC[2],
            MAGIC[3],
            MAGIC[4],
            MAGIC[5],
            MAGIC[6],
            MAGIC[7],
            1,
            (self.wgpu as u8) | ((self.cuda as u8) << 1),
            0,
            0,
        ]
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ContractError> {
        let invalid = || {
            ContractError::invalid(
                "native accelerator profile",
                "invalid target-built acceleration profile",
            )
        };
        if bytes.len() != NATIVE_ACCELERATOR_PROFILE_BYTES
            || bytes[..8] != MAGIC
            || bytes[8] != 1
            || bytes[9] & !0b11 != 0
            || bytes[10..] != [0, 0]
        {
            return Err(invalid());
        }
        let profile = Self {
            wgpu: bytes[9] & 1 != 0,
            cuda: bytes[9] & 2 != 0,
        };
        if profile.canonical_bytes() != bytes {
            return Err(invalid());
        }
        Ok(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::NativeAcceleratorProfile;

    #[test]
    fn profile_bytes_bind_only_compiled_backend_features() {
        for profile in [
            NativeAcceleratorProfile::NONE,
            NativeAcceleratorProfile {
                wgpu: true,
                cuda: false,
            },
            NativeAcceleratorProfile {
                wgpu: false,
                cuda: true,
            },
            NativeAcceleratorProfile {
                wgpu: true,
                cuda: true,
            },
        ] {
            assert_eq!(
                NativeAcceleratorProfile::from_canonical_bytes(&profile.canonical_bytes()).unwrap(),
                profile
            );
        }
        let mut invalid = NativeAcceleratorProfile::NONE.canonical_bytes();
        invalid[9] = 0b100;
        assert!(NativeAcceleratorProfile::from_canonical_bytes(&invalid).is_err());
    }
}
