use super::*;

pub(super) fn encode_model(graph: &ExportGraph, opset_version: i64) -> Vec<u8> {
    let mut model = Vec::new();
    put_varint_field(&mut model, 1, 10);
    put_string_field(&mut model, 2, "runmat");
    put_string_field(&mut model, 3, env!("CARGO_PKG_VERSION"));
    put_varint_field(&mut model, 5, 1);
    put_message_field(&mut model, 7, &encode_graph(graph));
    put_message_field(&mut model, 8, &encode_opset("", opset_version));
    put_message_field(
        &mut model,
        14,
        &encode_string_pair("runmat.supported_execution", "sequential-feedforward"),
    );
    model
}

fn encode_graph(graph: &ExportGraph) -> Vec<u8> {
    let mut out = Vec::new();
    for node in &graph.nodes {
        put_message_field(&mut out, 1, &encode_node(node));
    }
    put_string_field(&mut out, 2, "RunMatExportedNetwork");
    for initializer in &graph.initializers {
        put_message_field(&mut out, 5, &encode_initializer(initializer, graph.dtype));
    }
    put_message_field(&mut out, 11, &encode_value_info(&graph.input, graph.dtype));
    put_message_field(&mut out, 12, &encode_value_info(&graph.output, graph.dtype));
    out
}

fn encode_node(node: &Node) -> Vec<u8> {
    let mut out = Vec::new();
    for input in &node.inputs {
        put_string_field(&mut out, 1, input);
    }
    for output in &node.outputs {
        put_string_field(&mut out, 2, output);
    }
    put_string_field(&mut out, 3, &node.name);
    put_string_field(&mut out, 4, &node.op_type);
    for attr in &node.attributes {
        put_message_field(&mut out, 5, &encode_attribute(attr));
    }
    out
}

fn encode_attribute(attribute: &Attribute) -> Vec<u8> {
    let mut out = Vec::new();
    match attribute {
        Attribute::Float { name, value } => {
            put_string_field(&mut out, 1, name);
            put_fixed32_field(&mut out, 2, value.to_bits());
            put_varint_field(&mut out, 20, 1);
        }
        Attribute::Int { name, value } => {
            put_string_field(&mut out, 1, name);
            put_varint_field(&mut out, 3, *value as u64);
            put_varint_field(&mut out, 20, 2);
        }
    }
    out
}

fn onnx_dtype(dtype: NumericDType) -> i32 {
    match dtype {
        NumericDType::F32 => ONNX_FLOAT,
        NumericDType::F64 => ONNX_DOUBLE,
        _ => unreachable!("ONNX export normalizes learned tensors to floating point"),
    }
}

pub(super) fn encode_initializer(initializer: &Initializer, dtype: NumericDType) -> Vec<u8> {
    let mut out = Vec::new();
    for dim in &initializer.dims {
        put_varint_field(&mut out, 1, *dim as u64);
    }
    put_varint_field(&mut out, 2, onnx_dtype(dtype) as u64);
    put_string_field(&mut out, 8, &initializer.name);
    let mut raw = Vec::with_capacity(initializer.data.len() * dtype.byte_size());
    match dtype {
        NumericDType::F64 => {
            for value in initializer.data.materialize_f64() {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        NumericDType::F32 => {
            for value in initializer.data.materialize_f32() {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => unreachable!("ONNX export normalizes learned tensors to floating point"),
    }
    put_bytes_field(&mut out, 9, &raw);
    out
}

fn encode_value_info(info: &ValueInfo, dtype: NumericDType) -> Vec<u8> {
    let mut out = Vec::new();
    put_string_field(&mut out, 1, &info.name);
    put_message_field(&mut out, 2, &encode_type(info, dtype));
    out
}

fn encode_type(info: &ValueInfo, dtype: NumericDType) -> Vec<u8> {
    let mut tensor = Vec::new();
    put_varint_field(&mut tensor, 1, onnx_dtype(dtype) as u64);
    put_message_field(&mut tensor, 2, &encode_shape(info));
    let mut out = Vec::new();
    put_message_field(&mut out, 1, &tensor);
    out
}

fn encode_shape(info: &ValueInfo) -> Vec<u8> {
    let mut out = Vec::new();
    put_message_field(
        &mut out,
        1,
        &encode_dimension(info.batch_size, Some("batch")),
    );
    put_message_field(&mut out, 1, &encode_dimension(Some(info.features), None));
    out
}

fn encode_dimension(value: Option<i64>, param: Option<&str>) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(value) = value {
        put_varint_field(&mut out, 1, value as u64);
    } else if let Some(param) = param {
        put_string_field(&mut out, 2, param);
    }
    out
}

fn encode_opset(domain: &str, version: i64) -> Vec<u8> {
    let mut out = Vec::new();
    if !domain.is_empty() {
        put_string_field(&mut out, 1, domain);
    }
    put_varint_field(&mut out, 2, version as u64);
    out
}

fn encode_string_pair(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    put_string_field(&mut out, 1, key);
    put_string_field(&mut out, 2, value);
    out
}

fn put_string_field(out: &mut Vec<u8>, field: u32, value: &str) {
    put_bytes_field(out, field, value.as_bytes());
}

fn put_message_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    put_bytes_field(out, field, value);
}

fn put_bytes_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    put_key(out, field, 2);
    put_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    put_key(out, field, 0);
    put_varint(out, value);
}

fn put_fixed32_field(out: &mut Vec<u8>, field: u32, value: u32) {
    put_key(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_key(out: &mut Vec<u8>, field: u32, wire_type: u8) {
    put_varint(out, ((field as u64) << 3) | wire_type as u64);
}

pub(super) fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}
