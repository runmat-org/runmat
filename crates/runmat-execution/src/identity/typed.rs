use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::ContractError;

macro_rules! execution_id {
    ($name:ident, $prefix:literal, $domain:literal) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name([u8; 16]);

        impl $name {
            pub const fn from_bytes(bytes: [u8; 16]) -> Self {
                Self(bytes)
            }

            pub const fn bytes(&self) -> &[u8; 16] {
                &self.0
            }

            pub fn derive(parts: &[&[u8]]) -> Self {
                let mut hasher = Sha256::new();
                let domain = concat!("runmat-execution-id-v1:", $domain).as_bytes();
                hasher.update((domain.len() as u64).to_be_bytes());
                hasher.update(domain);
                for part in parts {
                    hasher.update((part.len() as u64).to_be_bytes());
                    hasher.update(part);
                }
                let digest = hasher.finalize();
                let mut bytes = [0_u8; 16];
                bytes.copy_from_slice(&digest[..16]);
                Self(bytes)
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str($prefix)?;
                for byte in self.0 {
                    write!(formatter, "{byte:02x}")?;
                }
                Ok(())
            }
        }

        impl FromStr for $name {
            type Err = ContractError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let encoded = value.strip_prefix($prefix).ok_or_else(|| {
                    ContractError::invalid(
                        stringify!($name),
                        concat!("missing `", $prefix, "` prefix"),
                    )
                })?;
                if encoded.len() != 32
                    || encoded
                        .bytes()
                        .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
                {
                    return Err(ContractError::invalid(
                        stringify!($name),
                        "expected 32 lowercase hexadecimal digits",
                    ));
                }
                let mut bytes = [0_u8; 16];
                for (index, byte) in bytes.iter_mut().enumerate() {
                    *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).map_err(
                        |_| ContractError::invalid(stringify!($name), "invalid hexadecimal"),
                    )?;
                }
                Ok(Self(bytes))
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&self.to_string())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                String::deserialize(deserializer)?
                    .parse()
                    .map_err(serde::de::Error::custom)
            }
        }
    };
}

execution_id!(RunId, "run_", "run");
execution_id!(ExecutionScopeId, "scope_", "scope");
execution_id!(FutureId, "future_", "future");
execution_id!(TaskId, "task_", "task");
execution_id!(AttemptId, "attempt_", "attempt");
execution_id!(ResultCommitId, "result_", "result-commit");
execution_id!(PoolId, "pool_", "pool");
execution_id!(JobId, "job_", "job");
execution_id!(DriverLeaseId, "driver_", "driver-lease");
execution_id!(NodeLeaseId, "lease_", "node-lease");
execution_id!(WorkerId, "worker_", "worker");
execution_id!(ArtifactId, "artifact_", "artifact");
execution_id!(ValueId, "value_", "value");
