//! Borrowed pre-encoding admission for self-contained owned CPU work.
//! This counts the existing inline codec's dense byte storage and a checked
//! upper bound for its JSON-compatible presentation, without first building a
//! `ValuePayload` or a second serializer.

use runmat_execution::OwnedBuiltinDenseOutputContract;
use runmat_value::Value;

use super::ValueCodecError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedInlineTransportPlan {
    input_data_bytes: u64,
    input_wire_bound: u64,
    output_data_bytes: u64,
    output_wire_bound: u64,
}

impl OwnedInlineTransportPlan {
    pub fn for_values(
        arguments: &[&Value],
        checked_output_bytes: u64,
        output_contracts: &[OwnedBuiltinDenseOutputContract],
    ) -> Result<Self, ValueCodecError> {
        let requested_outputs = u16::try_from(output_contracts.len())
            .map_err(|_| ValueCodecError::invalid("$", "owned CPU output count overflow"))?;
        OwnedBuiltinDenseOutputContract::validate_all(output_contracts, requested_outputs)
            .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
        let mut input_data_bytes = 0_u64;
        let mut input_text_bytes = 0_u64;
        let mut input_wire_bound = 4096_u64; // top-level request and field metadata
        for (index, value) in arguments.iter().enumerate() {
            let (data, text, wire) = value_footprint(value, &format!("$[{index}]"))?;
            input_data_bytes = add(input_data_bytes, data)?;
            input_text_bytes = add(input_text_bytes, text)?;
            input_wire_bound = add(input_wire_bound, wire)?;
        }
        let limits = runmat_execution::OWNED_BUILTIN_VALUE_LIMITS;
        if input_data_bytes > limits.max_inline_bytes {
            return Err(ValueCodecError::invalid(
                "$",
                "owned CPU input exceeds the inline transport bound",
            ));
        }
        if input_text_bytes > limits.max_text_bytes {
            return Err(ValueCodecError::invalid(
                "$",
                "owned CPU input exceeds the inline text bound",
            ));
        }
        let contracted_output_bytes =
            output_contracts.iter().try_fold(0_u64, |total, contract| {
                let bytes = contract
                    .validate()
                    .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
                add(total, bytes)
            })?;
        let output_bytes = checked_output_bytes.max(contracted_output_bytes);
        let output_count = len(output_contracts.len())?;
        let axes = output_contracts.iter().try_fold(0_u64, |total, contract| {
            add(total, len(contract.shape.len())?)
        })?;
        let types = output_contracts.iter().try_fold(0_u64, |total, contract| {
            add(total, len(contract.allowed_element_types.len())?)
        })?;
        input_wire_bound = add(
            input_wire_bound,
            add(
                mul(output_count, 512)?,
                add(mul(axes, 24)?, mul(types, 32)?)?,
            )?,
        )?;
        let success_wire_bound = add(mul(output_bytes, 4)?, add(mul(output_count, 512)?, 4096)?)?;
        let output_wire_bound =
            success_wire_bound.max(runmat_execution::OWNED_BUILTIN_FAILURE_WIRE_BYTES);
        if output_wire_bound > runmat_execution::OWNED_BUILTIN_MAX_RESPONSE_WIRE_BYTES {
            return Err(ValueCodecError::invalid(
                "$",
                "owned CPU response exceeds the portable wire bound",
            ));
        }
        Ok(Self {
            input_data_bytes,
            input_wire_bound,
            output_data_bytes: output_bytes,
            output_wire_bound,
        })
    }

    pub fn input_wire_bound(self) -> u64 {
        self.input_wire_bound
    }

    pub fn output_wire_bound(self) -> u64 {
        self.output_wire_bound
    }

    /// Concurrent coordinator allocations: inline payload, scheduler copy,
    /// JS request presentation, JS response presentation, and decoded output.
    pub fn coordinator_peak_bytes(self) -> Result<u64, ValueCodecError> {
        add(
            add(mul(self.input_data_bytes, 2)?, self.input_wire_bound)?,
            add(self.output_wire_bound, self.output_data_bytes)?,
        )
    }

    /// The worker keeps its incoming presentation, decoded inputs, encoded
    /// outputs, and outgoing presentation while its numerical plan is live.
    pub fn worker_peak_bytes(self) -> Result<u64, ValueCodecError> {
        add(
            add(self.input_wire_bound, self.input_data_bytes)?,
            add(self.output_data_bytes, self.output_wire_bound)?,
        )
    }
}

fn value_footprint(value: &Value, path: &str) -> Result<(u64, u64, u64), ValueCodecError> {
    let (data, text, shape) = match value {
        Value::Tensor(tensor) => (
            mul(len(tensor.len())?, len(tensor.numeric_dtype().byte_size())?)?,
            0,
            len(tensor.shape().len())?,
        ),
        Value::ComplexTensor(tensor) => (
            mul(
                mul(len(tensor.complex_storage().len())?, 2)?,
                len(tensor.numeric_dtype().byte_size())?,
            )?,
            0,
            len(tensor.shape().len())?,
        ),
        Value::LogicalArray(array) => (len(array.data.len())?, 0, len(array.shape.len())?),
        Value::CharArray(array) => (
            mul(len(array.data().len())?, 2)?,
            0,
            len(array.shape.len())?,
        ),
        Value::String(text) => (0, len(text.as_str().map_or(0, str::len))?, 0),
        Value::StringArray(array) => {
            let text = array.data().iter().try_fold(0_u64, |bytes, item| {
                add(bytes, len(item.as_str().map_or(0, str::len))?)
            })?;
            (
                0,
                add(text, mul(len(array.data().len())?, 64)?)?,
                len(array.shape.len())?,
            )
        }
        Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => (0, 0, 0),
        _ => {
            return Err(ValueCodecError::unsupported(
                path,
                "owned CPU work requires a scalar, dense numeric/logical, or text inline value",
            ))
        }
    };
    // A byte in the existing inline dense representation occupies at most
    // four JSON decimal-array bytes; text/control escaping at most six. The
    // bound also includes one inline variant/header and 24 bytes per axis.
    let wire = add(
        add(mul(data, 4)?, mul(text, 6)?)?,
        add(512, mul(shape, 24)?)?,
    )?;
    Ok((add(data, text)?, text, wire))
}

fn len(value: usize) -> Result<u64, ValueCodecError> {
    u64::try_from(value).map_err(|_| overflow())
}

fn add(left: u64, right: u64) -> Result<u64, ValueCodecError> {
    left.checked_add(right).ok_or_else(overflow)
}

fn mul(left: u64, right: u64) -> Result<u64, ValueCodecError> {
    left.checked_mul(right).ok_or_else(overflow)
}

fn overflow() -> ValueCodecError {
    ValueCodecError::invalid("$", "owned CPU transport byte count overflowed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ComplexTensor, LogicalArray, Tensor};

    #[test]
    fn borrowed_owned_work_plan_bounds_existing_inline_presentations() {
        let values = [
            Value::Tensor(Tensor::from_f32(vec![0.125, -0.25], vec![1, 2]).unwrap()),
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(0.5, -0.5)], vec![1, 1]).unwrap()),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::String("quoted \" text\n".into()),
        ];
        let borrowed = values.iter().collect::<Vec<_>>();
        let contracts = vec![
            OwnedBuiltinDenseOutputContract {
                shape: vec![1, 2],
                allowed_element_types: vec![runmat_execution::value::ElementType::F32],
            };
            2
        ];
        let plan = OwnedInlineTransportPlan::for_values(&borrowed, 128, &contracts).unwrap();
        let encoded = values
            .iter()
            .map(super::super::encode_inline_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let actual_wire = serde_json::to_vec(&encoded).unwrap();
        assert!(u64::try_from(actual_wire.len()).unwrap() <= plan.input_wire_bound());
        assert!(plan.coordinator_peak_bytes().unwrap() > plan.input_wire_bound());
        assert!(plan.worker_peak_bytes().unwrap() > plan.input_wire_bound());
    }
}
