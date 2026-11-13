//! Exact, invocation-owned canonical builtin work. This is not a program
//! artifact or an authority to resolve a user-visible name in a worker.

use serde::{Deserialize, Serialize};
use std::io::{self, Write};

use crate::identity::Digest;
use crate::resource::Capability;
use crate::resource::ResourceRequest;
use crate::value::{ElementType, InlineValue, ValueLimits, ValuePayload};
use crate::{ContractError, ProgramExecutionAssignment, ProgramRuntimeFailure};

pub const OWNED_BUILTIN_WORK_SCHEMA_VERSION: u16 = 4;
pub const MAX_OWNED_BUILTIN_ARGUMENTS: usize = 16;
pub const MAX_OWNED_BUILTIN_OUTPUTS: u16 = 16;
pub const OWNED_BUILTIN_VALUE_LIMITS: ValueLimits = ValueLimits {
    max_depth: 64,
    max_nodes: 1_000_000,
    max_inline_bytes: 64 * 1024 * 1024,
    max_elements: 100_000_000,
    max_fields: 100_000,
    max_text_bytes: 16 * 1024 * 1024,
};
/// The largest admitted failure presentation is one MiB of UTF-8 text,
/// escaped at at most six JSON bytes per byte, plus bounded variant metadata.
pub const OWNED_BUILTIN_FAILURE_WIRE_BYTES: u64 = 6 * 1024 * 1024 + 4096;
pub const OWNED_BUILTIN_MAX_RESPONSE_WIRE_BYTES: u64 =
    4 * OWNED_BUILTIN_VALUE_LIMITS.max_inline_bytes + OWNED_BUILTIN_FAILURE_WIRE_BYTES;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedBuiltinResponseContract {
    pub dense: Vec<OwnedBuiltinDenseOutputContract>,
    pub max_wire_bytes: u64,
}

impl OwnedBuiltinResponseContract {
    pub fn validate(&self, requested_outputs: u16) -> Result<(), ContractError> {
        OwnedBuiltinDenseOutputContract::validate_all(&self.dense, requested_outputs)?;
        if self.max_wire_bytes < OWNED_BUILTIN_FAILURE_WIRE_BYTES
            || self.max_wire_bytes > OWNED_BUILTIN_MAX_RESPONSE_WIRE_BYTES
        {
            return Err(ContractError::invalid(
                "owned builtin response",
                "response presentation is outside portable bounds",
            ));
        }
        Ok(())
    }

    pub fn validate_json_wire<T: Serialize>(&self, response: &T) -> Result<(), ContractError> {
        let mut counter = BoundedWireCounter {
            bytes: 0,
            limit: self.max_wire_bytes,
        };
        serde_json::to_writer(&mut counter, response).map_err(|_| {
            ContractError::invalid(
                "owned builtin response",
                "response presentation exceeds the admitted envelope",
            )
        })
    }
}

struct BoundedWireCounter {
    bytes: u64,
    limit: u64,
}

impl Write for BoundedWireCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let length = u64::try_from(bytes.len()).map_err(|_| io::Error::other("wire length"))?;
        self.bytes = self
            .bytes
            .checked_add(length)
            .filter(|total| *total <= self.limit)
            .ok_or_else(|| io::Error::other("response wire limit"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedBuiltinBinding {
    pub name: String,
    pub variant: String,
    /// The exact target-independent executable catalog contract.
    pub catalog: Digest,
}

/// One invocation's physical dense result, independent of builtin spelling.
/// The selected producer may choose any listed storage class, but shape and
/// aggregate byte geometry are fixed before worker or coordinator allocation.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedBuiltinDenseOutputContract {
    pub shape: Vec<u64>,
    pub allowed_element_types: Vec<ElementType>,
}

impl OwnedBuiltinDenseOutputContract {
    pub fn validate_all(contracts: &[Self], requested_outputs: u16) -> Result<(), ContractError> {
        if requested_outputs > MAX_OWNED_BUILTIN_OUTPUTS
            || contracts.len() != usize::from(requested_outputs)
        {
            return Err(ContractError::invalid(
                "owned builtin output",
                "dense output count differs from the requested count",
            ));
        }
        let total_bytes = contracts.iter().try_fold(0_u64, |total, contract| {
            total.checked_add(contract.validate()?).ok_or_else(|| {
                ContractError::invalid(
                    "owned builtin output",
                    "aggregate dense byte count overflow",
                )
            })
        })?;
        if total_bytes > OWNED_BUILTIN_VALUE_LIMITS.max_inline_bytes {
            return Err(ContractError::invalid(
                "owned builtin output",
                "aggregate dense outputs exceed portable limits",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<u64, ContractError> {
        if self.shape.len() > 16
            || self.shape.iter().any(|axis| *axis > u64::from(u32::MAX))
            || self.allowed_element_types.is_empty()
            || self.allowed_element_types.len() > 2
        {
            return Err(ContractError::invalid(
                "owned builtin output",
                "invalid dense output rank or storage-class alternatives",
            ));
        }
        let mut allowed = self.allowed_element_types.clone();
        allowed.sort_by_key(|element| *element as u8);
        allowed.dedup();
        if allowed.len() != self.allowed_element_types.len() {
            return Err(ContractError::invalid(
                "owned builtin output",
                "duplicate dense storage-class alternative",
            ));
        }
        let elements = self
            .shape
            .iter()
            .try_fold(1_u64, |count, axis| count.checked_mul(*axis))
            .ok_or_else(|| {
                ContractError::invalid("owned builtin output", "dense shape overflow")
            })?;
        if elements > OWNED_BUILTIN_VALUE_LIMITS.max_elements {
            return Err(ContractError::invalid(
                "owned builtin output",
                "dense shape exceeds portable element limits",
            ));
        }
        let maximum_width = self
            .allowed_element_types
            .iter()
            .map(|kind| kind.byte_width())
            .max()
            .unwrap_or(0);
        let bytes = elements.checked_mul(maximum_width).ok_or_else(|| {
            ContractError::invalid("owned builtin output", "dense byte count overflow")
        })?;
        if bytes > OWNED_BUILTIN_VALUE_LIMITS.max_inline_bytes {
            return Err(ContractError::invalid(
                "owned builtin output",
                "dense output exceeds portable inline limits",
            ));
        }
        Ok(bytes)
    }

    pub fn accepts(&self, value: &ValuePayload) -> bool {
        matches!(value, ValuePayload::Inline(inline)
            if matches!(inline.as_ref(), InlineValue::Dense(dense)
                if dense.shape == self.shape
                    && self.allowed_element_types.contains(&dense.element_type)))
    }
}

impl OwnedBuiltinBinding {
    pub fn executable_identity(&self) -> Result<Digest, ContractError> {
        Digest::sha256_serialized(&("runmat.owned-builtin-work.v1", self))
            .map_err(|error| ContractError::invalid("owned builtin binding", error.to_string()))
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.name.contains('\0')
            || self.variant.is_empty()
            || self.variant.len() > 128
            || self.variant.contains('\0')
        {
            return Err(ContractError::invalid(
                "owned builtin binding",
                "invalid exact binding locator",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedBuiltinWorkRequest {
    pub schema_version: u16,
    pub binding: OwnedBuiltinBinding,
    pub arguments: Vec<ValuePayload>,
    pub requested_outputs: u16,
    pub response: OwnedBuiltinResponseContract,
    pub resources: ResourceRequest,
    /// Conservative transport/decode/encode peak held throughout worker work.
    pub transport_scratch_bytes: u64,
    /// Exact target-compiled continuation allocation; the worker recomputes
    /// this from its selected binding before decoding or executing inputs.
    pub control_scratch_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment: Option<ProgramExecutionAssignment>,
}

impl OwnedBuiltinWorkRequest {
    pub fn validate_wire(&self) -> Result<(), ContractError> {
        if self.schema_version != OWNED_BUILTIN_WORK_SCHEMA_VERSION {
            return Err(ContractError::invalid(
                "owned builtin work",
                "unsupported request schema",
            ));
        }
        self.binding.validate()?;
        if self.arguments.len() > MAX_OWNED_BUILTIN_ARGUMENTS
            || self.requested_outputs > MAX_OWNED_BUILTIN_OUTPUTS
        {
            return Err(ContractError::invalid(
                "owned builtin work",
                "argument or output count exceeds the transport bound",
            ));
        }
        if self
            .arguments
            .iter()
            .any(|value| !matches!(value, ValuePayload::Inline(_)))
        {
            return Err(ContractError::invalid(
                "owned builtin work",
                "owned host CPU inputs must be self-contained inline values",
            ));
        }
        ValuePayload::validate_all(&self.arguments, OWNED_BUILTIN_VALUE_LIMITS)?;
        self.response.validate(self.requested_outputs)?;
        self.resources.validate()?;
        let required_scratch = self
            .transport_scratch_bytes
            .checked_add(self.control_scratch_bytes)
            .ok_or_else(|| ContractError::invalid("owned builtin work", "scratch overflow"))?;
        if self.transport_scratch_bytes == 0
            || self.control_scratch_bytes == 0
            || self.transport_scratch_bytes < self.response.max_wire_bytes
            || required_scratch > self.resources.scratch_bytes
            || required_scratch > self.resources.memory_bytes
        {
            return Err(ContractError::invalid(
                "owned builtin work",
                "transport scratch exceeds the admitted worker resources",
            ));
        }
        if !self.resources.accelerators.is_empty()
            || self.resources.required_capabilities.len() != 1
            || !self
                .resources
                .required_capabilities
                .contains(&Capability::BrowserWorker)
        {
            return Err(ContractError::invalid(
                "owned builtin work",
                "portable host CPU work cannot request accelerator or unrelated capabilities",
            ));
        }
        if let Some(assignment) = &self.assignment {
            assignment.validate()?;
            if !assignment.resources.accelerator_leases.is_empty() {
                return Err(ContractError::invalid(
                    "owned builtin work",
                    "CPU work cannot receive an accelerator lease",
                ));
            }
        }
        Ok(())
    }

    pub fn validate_response(
        &self,
        response: &OwnedBuiltinWorkResponse,
    ) -> Result<(), ContractError> {
        self.validate_wire()?;
        self.response.validate_json_wire(response)?;
        match response {
            OwnedBuiltinWorkResponse::Success { outputs } => self.validate_output_values(outputs),
            OwnedBuiltinWorkResponse::RuntimeFailure { failure } => failure.validate(),
            OwnedBuiltinWorkResponse::InfrastructureFailure { message } => {
                if message.is_empty() || message.len() > 1024 * 1024 || message.contains('\0') {
                    Err(ContractError::invalid(
                        "owned builtin response",
                        "invalid infrastructure failure",
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn validate_output_values(&self, outputs: &[ValuePayload]) -> Result<(), ContractError> {
        self.response.validate(self.requested_outputs)?;
        if outputs.len() != usize::from(self.requested_outputs) {
            return Err(ContractError::invalid(
                "owned builtin response",
                "output count differs from the admitted request",
            ));
        }
        if outputs
            .iter()
            .any(|value| !matches!(value, ValuePayload::Inline(_)))
        {
            return Err(ContractError::invalid(
                "owned builtin response",
                "owned host CPU outputs must be self-contained inline values",
            ));
        }
        ValuePayload::validate_all(outputs, OWNED_BUILTIN_VALUE_LIMITS)?;
        self.validate_dense_outputs(outputs)
    }

    pub fn validate_dense_outputs(&self, outputs: &[ValuePayload]) -> Result<(), ContractError> {
        if outputs.len() != self.response.dense.len()
            || !outputs
                .iter()
                .zip(&self.response.dense)
                .all(|(output, contract)| contract.accepts(output))
        {
            return Err(ContractError::invalid(
                "owned builtin response",
                "dense output shape or storage class differs from admitted request",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum OwnedBuiltinWorkResponse {
    Success { outputs: Vec<ValuePayload> },
    RuntimeFailure { failure: ProgramRuntimeFailure },
    InfrastructureFailure { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::Capability;
    use crate::value::{DenseValue, InlineValue};
    use std::collections::BTreeSet;

    fn request() -> OwnedBuiltinWorkRequest {
        OwnedBuiltinWorkRequest {
            schema_version: OWNED_BUILTIN_WORK_SCHEMA_VERSION,
            binding: OwnedBuiltinBinding {
                name: "qz".into(),
                variant: "default".into(),
                catalog: Digest::from_bytes([7; 32]),
            },
            arguments: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
                1f64.to_bits(),
            )))],
            requested_outputs: 1,
            response: OwnedBuiltinResponseContract {
                dense: vec![OwnedBuiltinDenseOutputContract {
                    shape: vec![1, 1],
                    allowed_element_types: vec![ElementType::F64],
                }],
                max_wire_bytes: OWNED_BUILTIN_FAILURE_WIRE_BYTES,
            },
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: 8 * 1024 * 1024 + 1,
                scratch_bytes: 8 * 1024 * 1024 + 1,
                max_wall_millis: 1000,
                max_artifact_bytes: 0,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::BrowserWorker]),
            },
            transport_scratch_bytes: 8 * 1024 * 1024,
            control_scratch_bytes: 1,
            assignment: None,
        }
    }

    #[test]
    fn owned_work_has_exact_executable_identity_and_bounded_output() {
        let mut first = request();
        first.validate_wire().unwrap();
        let identity = first.binding.executable_identity().unwrap();
        first.binding.variant = "alternate".into();
        assert_ne!(identity, first.binding.executable_identity().unwrap());
        first.binding.variant = "default".into();
        first
            .validate_output_values(&[ValuePayload::Inline(Box::new(InlineValue::Dense(
                DenseValue {
                    element_type: ElementType::F64,
                    shape: vec![1, 1],
                    little_endian_data: 1_f64.to_le_bytes().to_vec(),
                },
            )))])
            .unwrap();
        assert!(first
            .validate_response(&OwnedBuiltinWorkResponse::Success {
                outputs: Vec::new(),
            })
            .is_err());
        let mut zero_output = first.clone();
        zero_output.requested_outputs = 0;
        zero_output.response.dense.clear();
        zero_output.validate_wire().unwrap();
        first.resources.required_capabilities.clear();
        assert!(first.validate_wire().is_err());
    }
}
