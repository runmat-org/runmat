//! Execution-domain native policies, independent of a compiler or build host.
use crate::{ContractError, Digest};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOptimization {
    None,
    Speed,
    SpeedAndSize,
}

pub const NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NativeNumericalBehavior {
    PlatformAccurate,
    StrictReproducible { qualification: Digest },
    ToleranceBased { target_test_tolerances: Digest },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeNumericalPolicy {
    pub schema_version: u16,
    /// Exact selected backend implementation, not a per-builtin policy table.
    pub backend: Digest,
    pub behavior: NativeNumericalBehavior,
}

impl NativeNumericalPolicy {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
            });
        }
        let evidence = self.required_evidence();
        if self.backend.bytes().iter().all(|byte| *byte == 0)
            || evidence.is_some_and(|id| id.bytes().iter().all(|byte| *byte == 0))
        {
            return Err(ContractError::invalid(
                "native numerical policy",
                "backend/evidence identity is missing",
            ));
        }
        Ok(())
    }

    pub fn required_evidence(&self) -> Option<Digest> {
        match self.behavior {
            NativeNumericalBehavior::PlatformAccurate => None,
            NativeNumericalBehavior::StrictReproducible { qualification } => Some(qualification),
            NativeNumericalBehavior::ToleranceBased {
                target_test_tolerances,
            } => Some(target_test_tolerances),
        }
    }

    pub fn fingerprint(&self) -> Result<Digest, ContractError> {
        self.validate()?;
        Ok(
            Digest::sha256_serialized(&("runmat-native-numerical-policy-v1", self))
                .expect("native numerical policy serializes"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_optimization_wire_and_numerical_policy_identity_are_explicit() {
        for (policy, wire) in [
            (NativeOptimization::None, "\"none\""),
            (NativeOptimization::Speed, "\"speed\""),
            (NativeOptimization::SpeedAndSize, "\"speed_and_size\""),
        ] {
            assert_eq!(serde_json::to_string(&policy).unwrap(), wire);
        }
        let mut policy = NativeNumericalPolicy {
            schema_version: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
            backend: Digest::sha256(b"backend bytes"),
            behavior: NativeNumericalBehavior::PlatformAccurate,
        };
        let first = policy.fingerprint().unwrap();
        policy.behavior = NativeNumericalBehavior::StrictReproducible {
            qualification: Digest::sha256(b"qualification"),
        };
        assert_ne!(first, policy.fingerprint().unwrap());
        policy.schema_version += 1;
        assert!(policy.validate().is_err());
    }
}
