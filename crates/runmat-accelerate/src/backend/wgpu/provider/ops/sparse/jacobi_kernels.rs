pub(super) fn prepare_shader(scalar: &str, finite_limit: &str, workgroup_size: u32) -> String {
    format!(
        r#"
struct IndexBuffer {{ data: array<u32>, }};
struct ScalarBuffer {{ data: array<{scalar}>, }};
struct StatusBuffer {{ data: array<atomic<u32>>, }};
@group(0) @binding(0) var<storage, read> RowOffsets: IndexBuffer;
@group(0) @binding(1) var<storage, read> ColumnIndices: IndexBuffer;
@group(0) @binding(2) var<storage, read> Values: ScalarBuffer;
@group(0) @binding(3) var<storage, read_write> InverseDiagonal: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Status: StatusBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let rows = arrayLength(&RowOffsets.data) - 1u;
    if (row >= rows) {{ return; }}
    var diagonal: {scalar} = 0.0;
    var found = false;
    for (var index = RowOffsets.data[row]; index < RowOffsets.data[row + 1u]; index = index + 1u) {{
        if (ColumnIndices.data[index] == row) {{
            diagonal = Values.data[index];
            found = true;
        }}
    }}
    let inverse = 1.0 / diagonal;
    if (!found || diagonal == 0.0 || inverse != inverse || abs(inverse) > {finite_limit}) {{
        atomicOr(&Status.data[0], 1u);
        InverseDiagonal.data[row] = 0.0;
        return;
    }}
    InverseDiagonal.data[row] = inverse;
}}
"#
    )
}

pub(super) fn apply_shader(scalar: &str, workgroup_size: u32) -> String {
    format!(
        r#"
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> InverseDiagonal: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Input: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> Output: ScalarBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < arrayLength(&Input.data)) {{
        Output.data[index] = InverseDiagonal.data[index] * Input.data[index];
    }}
}}
"#
    )
}
