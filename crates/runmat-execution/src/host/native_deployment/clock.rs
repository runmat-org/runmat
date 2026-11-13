//! Explicit virtual-clock inputs, independent of service implementation trust.
use super::{
    NativeDeploymentServiceBehavior, NativeDeploymentServiceKind, NativeSelectedProductBinding,
};
use crate::ContractError;
use runmat_types::EffectKind;
use serde::{Deserialize, Serialize};

pub const NATIVE_CLOCK_INPUTS_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeClockInvocationIdentity {
    invocation: crate::Digest,
    inputs: Option<crate::Digest>,
}

impl NativeClockInvocationIdentity {
    pub fn new(
        invocation: crate::Digest,
        inputs: Option<&NativeClockInputs>,
    ) -> Result<Self, ContractError> {
        if invocation.bytes().iter().all(|byte| *byte == 0) {
            return Err(ContractError::invalid(
                "native.clock.invocation",
                "missing invocation identity",
            ));
        }
        Ok(Self {
            invocation,
            inputs: inputs.map(NativeClockInputs::fingerprint).transpose()?,
        })
    }
    pub fn inputs(&self) -> Option<crate::Digest> {
        self.inputs
    }
}

pub fn native_clock_service_abi() -> crate::Digest {
    crate::Digest::sha256(
        b"runmat-native-clock-v1-runtime-ptr-option-inputs-v1-ptr-result-owner-ptr",
    )
}

/// A logical CPU counter for language `cputime`, not measured profiling data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClockCpuInputs {
    pub initial_nanos: u64,
    pub read_step_nanos: u64,
}

/// Every time read advances the shared logical elapsed clock by `read_step`.
/// Sleep completion adds its requested duration; abandoned waits add nothing.
/// Calendar conversion uses a fixed offset, never a host timezone database.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClockInputs {
    pub schema_version: u16,
    pub initial_monotonic_nanos: u64,
    pub initial_wall_unix_seconds: u64,
    pub initial_wall_nanosecond: u32,
    pub read_step_nanos: u64,
    pub local_offset_seconds: i32,
    pub process_cpu: Option<NativeClockCpuInputs>,
}

impl NativeClockInputs {
    pub fn validate(&self) -> Result<(), ContractError> {
        const DAY_NANOS: u64 = 86_400_000_000_000;
        if self.schema_version != NATIVE_CLOCK_INPUTS_SCHEMA_VERSION
            || self.initial_wall_unix_seconds > 253_402_214_400
            || self.initial_wall_nanosecond >= 1_000_000_000
            || !(1..=DAY_NANOS).contains(&self.read_step_nanos)
            || !(-86_399..=86_399).contains(&self.local_offset_seconds)
            || self
                .process_cpu
                .as_ref()
                .is_some_and(|cpu| !(1..=DAY_NANOS).contains(&cpu.read_step_nanos))
        {
            return Err(ContractError::invalid(
                "native.clock.inputs",
                "unsupported schema or invalid clock/calendar/counter bounds",
            ));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<crate::Digest, ContractError> {
        self.validate()?;
        crate::Digest::sha256_serialized(&("runmat-native-clock-inputs-v1", self))
            .map_err(|error| ContractError::invalid("native.clock.inputs", error.to_string()))
    }
}

/// Structural policy validation only; decoded inputs never issue a service.
pub fn validate_native_clock_inputs(
    product: Option<&NativeSelectedProductBinding>,
    inputs: Option<&NativeClockInputs>,
) -> Result<(), ContractError> {
    let service = product.and_then(|product| {
        product
            .services
            .iter()
            .find(|service| service.kind == NativeDeploymentServiceKind::Effect(EffectKind::Clock))
    });
    match (service.map(|service| service.behavior), inputs) {
        (Some(NativeDeploymentServiceBehavior::DeterministicDeclared), Some(inputs)) => {
            inputs.validate()
        }
        (Some(NativeDeploymentServiceBehavior::HostedAmbient), None) | (None, None) => Ok(()),
        _ => Err(ContractError::invalid(
            "native.clock.inputs",
            "clock inputs do not match the selected service behavior",
        )),
    }
}
