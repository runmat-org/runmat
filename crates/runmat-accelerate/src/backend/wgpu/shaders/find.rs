pub const FIND_SHADER_F64: &str = r#"
struct InputTensor {
    data: array<f64>,
};

struct OutputTensor {
    data: array<f64>,
};

struct MetaBuffer {
    data: array<u32>,
};

struct FindParams {
    len: u32,
    limit: u32,
    rows: u32,
    direction: u32,
    include_values: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: InputTensor;
@group(0) @binding(1) var<storage, read_write> OutIndices: OutputTensor;
@group(0) @binding(2) var<storage, read_write> OutRows: OutputTensor;
@group(0) @binding(3) var<storage, read_write> OutCols: OutputTensor;
@group(0) @binding(4) var<storage, read_write> OutValues: InputTensor;
@group(0) @binding(5) var<storage, read_write> Meta: MetaBuffer;
@group(0) @binding(6) var<uniform> params: FindParams;

fn write_result(slot: u32, linear_index: u32, value: f64, rows: u32) {
    let row = ((linear_index - 1u) % rows) + 1u;
    let col = ((linear_index - 1u) / rows) + 1u;
    OutIndices.data[slot] = f64(linear_index);
    OutRows.data[slot] = f64(row);
    OutCols.data[slot] = f64(col);
    if params.include_values != 0u {
        OutValues.data[slot] = value;
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }
    let len = params.len;
    let limit = params.limit;
    if len == 0u || limit == 0u {
        Meta.data[0] = 0u;
        return;
    }
    let rows = max(params.rows, 1u);
    var count: u32 = 0u;
    if params.direction == 0u {
        var idx: u32 = 0u;
        loop {
            if idx >= len {
                break;
            }
            let value = Input.data[idx];
            if value != 0.0 {
                if count < limit {
                    let linear = idx + 1u;
                    write_result(count, linear, value, rows);
                    count = count + 1u;
                    if count >= limit {
                        break;
                    }
                }
            }
            idx = idx + 1u;
        }
    } else {
        var idx: i32 = i32(len);
        loop {
            idx = idx - 1;
            if idx < 0 {
                break;
            }
            let value = Input.data[u32(idx)];
            if value != 0.0 {
                if count < limit {
                    let linear = u32(idx) + 1u;
                    write_result(count, linear, value, rows);
                    count = count + 1u;
                    if count >= limit {
                        break;
                    }
                }
            }
        }
    }
    Meta.data[0] = count;
}
"#;

pub const FIND_SHADER_F32: &str = r#"
struct InputTensor {
    data: array<f32>,
};

struct OutputTensor {
    data: array<f32>,
};

struct MetaBuffer {
    data: array<u32>,
};

struct FindParams {
    len: u32,
    limit: u32,
    rows: u32,
    direction: u32,
    include_values: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: InputTensor;
@group(0) @binding(1) var<storage, read_write> OutIndices: OutputTensor;
@group(0) @binding(2) var<storage, read_write> OutRows: OutputTensor;
@group(0) @binding(3) var<storage, read_write> OutCols: OutputTensor;
@group(0) @binding(4) var<storage, read_write> OutValues: InputTensor;
@group(0) @binding(5) var<storage, read_write> Meta: MetaBuffer;
@group(0) @binding(6) var<uniform> params: FindParams;

fn write_result(slot: u32, linear_index: u32, value: f32, rows: u32) {
    let row = ((linear_index - 1u) % rows) + 1u;
    let col = ((linear_index - 1u) / rows) + 1u;
    OutIndices.data[slot] = f32(linear_index);
    OutRows.data[slot] = f32(row);
    OutCols.data[slot] = f32(col);
    if params.include_values != 0u {
        OutValues.data[slot] = value;
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }
    let len = params.len;
    let limit = params.limit;
    if len == 0u || limit == 0u {
        Meta.data[0] = 0u;
        return;
    }
    let rows = max(params.rows, 1u);
    var count: u32 = 0u;
    if params.direction == 0u {
        var idx: u32 = 0u;
        loop {
            if idx >= len {
                break;
            }
            let value = Input.data[idx];
            if value != 0.0 {
                if count < limit {
                    let linear = idx + 1u;
                    write_result(count, linear, value, rows);
                    count = count + 1u;
                    if count >= limit {
                        break;
                    }
                }
            }
            idx = idx + 1u;
        }
    } else {
        var idx: i32 = i32(len);
        loop {
            idx = idx - 1;
            if idx < 0 {
                break;
            }
            let value = Input.data[u32(idx)];
            if value != 0.0 {
                if count < limit {
                    let linear = u32(idx) + 1u;
                    write_result(count, linear, value, rows);
                    count = count + 1u;
                    if count >= limit {
                        break;
                    }
                }
            }
        }
    }
    Meta.data[0] = count;
}
"#;
