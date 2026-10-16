//! Bounded JS metadata admission before the owned response crosses into Rust.
//! Exact payload and wire validation still belongs to the canonical request.

use js_sys::{Array, ArrayBuffer, JsString, Reflect, Uint8Array};
use runmat_execution::{
    value::ElementType, OwnedBuiltinWorkRequest, OWNED_BUILTIN_FAILURE_WIRE_BYTES,
    OWNED_BUILTIN_VALUE_LIMITS,
};
use wasm_bindgen::{JsCast, JsValue};

const FIELD_LIMIT: u32 = 16;
const TOKEN_LIMIT: u32 = 64;
const FAILURE_NODE_LIMIT: u64 = 100_000;

pub(super) fn preflight(
    response: &JsValue,
    request: &OwnedBuiltinWorkRequest,
) -> Result<(), String> {
    exact_fields(response, &["outcome", "outputs"])
        .or_else(|_| exact_fields(response, &["outcome", "failure"]))
        .or_else(|_| exact_fields(response, &["outcome", "message"]))?;
    let outcome = short_token(&field(response, "outcome")?)?;
    match outcome.as_str() {
        "success" => success(response, request),
        "failure" => {
            let message = field(response, "message")?;
            if string_units(&message)? > 1024 * 1024 {
                return Err("owned response failure text exceeds its predecode bound".into());
            }
            bounded_failure_tree(response)
        }
        "runtime_failure" => bounded_failure_tree(response),
        _ => Err("owned worker returned an unsupported response form".into()),
    }
}

pub(super) fn preflight_exit_ack(value: &JsValue, task_id: &str) -> Result<(), String> {
    exact_fields(value, &["taskId", "state"])?;
    if short_token(&field(value, "taskId")?)? != task_id {
        return Err("owned worker acknowledged a different task".into());
    }
    match short_token(&field(value, "state")?)?.as_str() {
        "terminated" | "notRunning" => Ok(()),
        _ => Err("owned worker has no physical-exit acknowledgment".into()),
    }
}

fn success(response: &JsValue, request: &OwnedBuiltinWorkRequest) -> Result<(), String> {
    exact_fields(response, &["outcome", "outputs"])?;
    let outputs = field(response, "outputs")?;
    if !Array::is_array(&outputs) {
        return Err("owned response outputs are not an array".into());
    }
    let outputs = outputs
        .dyn_ref::<Array>()
        .ok_or_else(|| "owned response outputs are not an array".to_string())?;
    if outputs.length() != u32::from(request.requested_outputs)
        || outputs.length() as usize != request.response.dense.len()
    {
        return Err("owned response output count differs from admission".into());
    }
    let mut aggregate_bytes = 0_u64;
    for (index, contract) in request.response.dense.iter().enumerate() {
        let output = outputs.get(index as u32);
        exact_fields(&output, &["form", "value"])?;
        if short_token(&field(&output, "form")?)? != "inline" {
            return Err("owned response output is not inline".into());
        }
        let inline = field(&output, "value")?;
        exact_fields(&inline, &["type", "value"])?;
        if short_token(&field(&inline, "type")?)? != "dense" {
            return Err("owned response output is not dense".into());
        }
        let dense = field(&inline, "value")?;
        exact_fields(&dense, &["element_type", "shape", "little_endian_data"])?;
        let shape_value = field(&dense, "shape")?;
        let shape_array = shape_value
            .dyn_ref::<Array>()
            .ok_or_else(|| "owned response shape differs from admission".to_string())?;
        if shape_array.length() != contract.shape.len() as u32 || shape_array.length() > FIELD_LIMIT
        {
            return Err("owned response shape differs from admission".into());
        }
        // Rank is at most 16, so this small metadata decode cannot copy a
        // numeric payload. The canonical validator checks it again afterward.
        let shape: Vec<u64> = serde_wasm_bindgen::from_value(shape_value)
            .map_err(|_| "owned response shape is invalid".to_string())?;
        if shape != contract.shape {
            return Err("owned response shape differs from admission".into());
        }
        let element_value = field(&dense, "element_type")?;
        short_token(&element_value)?;
        let element_type: ElementType = serde_wasm_bindgen::from_value(element_value)
            .map_err(|_| "owned response storage class is invalid".to_string())?;
        if !contract.allowed_element_types.contains(&element_type) {
            return Err("owned response storage class differs from admission".into());
        }
        let expected_bytes = shape
            .iter()
            .try_fold(1_u64, |count, axis| count.checked_mul(*axis))
            .and_then(|elements| elements.checked_mul(element_type.byte_width()))
            .ok_or_else(|| "owned response byte geometry overflow".to_string())?;
        let data = field(&dense, "little_endian_data")?;
        let actual_bytes = if Array::is_array(&data) {
            u64::from(
                data.dyn_ref::<Array>()
                    .ok_or_else(|| "owned response numeric payload is invalid".to_string())?
                    .length(),
            )
        } else if let Some(bytes) = data.dyn_ref::<Uint8Array>() {
            u64::from(bytes.length())
        } else {
            return Err("owned response numeric payload has no bounded byte view".into());
        };
        if actual_bytes != expected_bytes {
            return Err("owned response numeric byte length differs from admission".into());
        }
        aggregate_bytes = aggregate_bytes
            .checked_add(actual_bytes)
            .ok_or_else(|| "owned response aggregate byte geometry overflow".to_string())?;
    }
    if aggregate_bytes > OWNED_BUILTIN_VALUE_LIMITS.max_inline_bytes {
        return Err("owned response aggregate bytes exceed portable limits".into());
    }
    Ok(())
}

fn bounded_failure_tree(value: &JsValue) -> Result<(), String> {
    let mut budget = FailureBudget {
        bytes: OWNED_BUILTIN_FAILURE_WIRE_BYTES,
        nodes: FAILURE_NODE_LIMIT,
    };
    walk_failure(value, 0, &mut budget)
}

struct FailureBudget {
    bytes: u64,
    nodes: u64,
}

impl FailureBudget {
    fn charge(&mut self, bytes: u64) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or_else(|| "owned response failure exceeds its predecode bound".to_string())?;
        self.nodes = self
            .nodes
            .checked_sub(1)
            .ok_or_else(|| "owned response failure has too many fields".to_string())?;
        Ok(())
    }
}

fn walk_failure(value: &JsValue, depth: u32, budget: &mut FailureBudget) -> Result<(), String> {
    if depth > FIELD_LIMIT {
        return Err("owned response failure nesting exceeds its predecode bound".into());
    }
    if let Some(string) = value.dyn_ref::<JsString>() {
        // Six JSON bytes per UTF-16 code unit bounds escaping without
        // copying or stringifying the potentially large source text.
        return budget.charge(
            u64::from(string.length())
                .saturating_mul(6)
                .saturating_add(2),
        );
    }
    if value.is_null() || value.is_undefined() || !value.is_object() {
        return budget.charge(32);
    }
    if ArrayBuffer::is_view(value) {
        return Err("owned response failure cannot contain binary payloads".into());
    }
    if Array::is_array(value) {
        let array = value
            .dyn_ref::<Array>()
            .ok_or_else(|| "owned response failure array is invalid".to_string())?;
        if u64::from(array.length()) > budget.nodes {
            return Err("owned response failure array exceeds its predecode bound".into());
        }
        budget.charge(2)?;
        for index in 0..array.length() {
            walk_failure(&array.get(index), depth + 1, budget)?;
        }
    } else {
        let keys = Reflect::own_keys(value)
            .map_err(|_| "owned response failure keys are unreadable".to_string())?;
        if u64::from(keys.length()) > budget.nodes {
            return Err("owned response failure object exceeds its predecode bound".into());
        }
        budget.charge(2)?;
        for index in 0..keys.length() {
            let key = keys.get(index);
            walk_failure(&key, depth + 1, budget)?;
            let child = Reflect::get(value, &key)
                .map_err(|_| "owned response failure field is unreadable".to_string())?;
            walk_failure(&child, depth + 1, budget)?;
        }
    }
    Ok(())
}

fn exact_fields(value: &JsValue, expected: &[&str]) -> Result<(), String> {
    if !value.is_object() || Array::is_array(value) {
        return Err("owned response has invalid field geometry".into());
    }
    let keys =
        Reflect::own_keys(value).map_err(|_| "owned response keys are unreadable".to_string())?;
    if keys.length() as usize != expected.len() {
        return Err("owned response has unexpected fields".into());
    }
    for index in 0..keys.length() {
        let key = short_token(&keys.get(index))?;
        if !expected.contains(&key.as_str()) {
            return Err("owned response has unexpected fields".into());
        }
    }
    Ok(())
}

fn field(value: &JsValue, name: &str) -> Result<JsValue, String> {
    Reflect::get(value, &JsValue::from_str(name))
        .map_err(|_| "owned response field is unreadable".to_string())
}

fn short_token(value: &JsValue) -> Result<String, String> {
    if string_units(value)? > TOKEN_LIMIT {
        return Err("owned response token exceeds its predecode bound".into());
    }
    value
        .as_string()
        .ok_or_else(|| "owned response token is not text".to_string())
}

fn string_units(value: &JsValue) -> Result<u32, String> {
    value
        .dyn_ref::<JsString>()
        .map(JsString::length)
        .ok_or_else(|| "owned response field is not text".to_string())
}
