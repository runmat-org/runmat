pub(super) fn sources(scalar: &str, finite_limit: &str, workgroup_size: u32) -> CgShaderSources {
    let state = state_declaration(scalar);
    CgShaderSources {
        spmv: format!(
            r#"{state}
struct IndexBuffer {{ data: array<u32>, }};
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> RowOffsets: IndexBuffer;
@group(0) @binding(1) var<storage, read> ColumnIndices: IndexBuffer;
@group(0) @binding(2) var<storage, read> Values: ScalarBuffer;
@group(0) @binding(3) var<storage, read> Input: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Output: ScalarBuffer;
@group(0) @binding(5) var<storage, read> State: SolverState;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let rows = arrayLength(&RowOffsets.data) - 1u;
    if (row >= rows || State.termination != 0u) {{ return; }}
    var sum: {scalar} = 0.0;
    for (var index = RowOffsets.data[row]; index < RowOffsets.data[row + 1u]; index = index + 1u) {{
        sum = sum + Values.data[index] * Input.data[ColumnIndices.data[index]];
    }}
    Output.data[row] = sum;
}}
"#
        ),
        initialize_vectors: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> Rhs: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Product: ScalarBuffer;
@group(0) @binding(2) var<storage, read> InverseDiagonal: ScalarBuffer;
@group(0) @binding(3) var<storage, read_write> Residual: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Preconditioned: ScalarBuffer;
@group(0) @binding(5) var<storage, read_write> Direction: ScalarBuffer;
@group(0) @binding(6) var<storage, read> State: SolverState;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index >= arrayLength(&Rhs.data) || State.termination != 0u) {{ return; }}
    let residual = Rhs.data[index] - Product.data[index];
    let preconditioned = InverseDiagonal.data[index] * residual;
    Residual.data[index] = residual;
    Preconditioned.data[index] = preconditioned;
    Direction.data[index] = preconditioned;
}}
"#
        ),
        dot_partials: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> Left: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Right: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> Partials: ScalarBuffer;
@group(0) @binding(3) var<storage, read> State: SolverState;
var<workgroup> Scratch: array<{scalar}, {workgroup_size}>;
@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {{
    let lane = local_id.x;
    var sum: {scalar} = 0.0;
    let index = global_id.x;
    if (State.termination == 0u && index < arrayLength(&Left.data)) {{
        sum = Left.data[index] * Right.data[index];
    }}
    Scratch[lane] = sum;
    workgroupBarrier();
    var stride = {workgroup_size}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lane < stride) {{ Scratch[lane] = Scratch[lane] + Scratch[lane + stride]; }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if (lane == 0u) {{ Partials.data[workgroup_id.x] = Scratch[0]; }}
}}
"#
        ),
        dot_finalize: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> Partials: ScalarBuffer;
@group(0) @binding(1) var<storage, read_write> Output: ScalarBuffer;
@group(0) @binding(2) var<storage, read> State: SolverState;
var<workgroup> Scratch: array<{scalar}, {workgroup_size}>;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {{
    let lane = local_id.x;
    var sum: {scalar} = 0.0;
    if (State.termination == 0u) {{
        var index = lane;
        let count = arrayLength(&Partials.data);
        loop {{
            if (index >= count) {{ break; }}
            sum = sum + Partials.data[index];
            index = index + {workgroup_size}u;
        }}
    }}
    Scratch[lane] = sum;
    workgroupBarrier();
    var stride = {workgroup_size}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lane < stride) {{ Scratch[lane] = Scratch[lane] + Scratch[lane + stride]; }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if (lane == 0u) {{ Output.data[0] = Scratch[0]; }}
}}
"#
        ),
        initialize_state: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> ResidualSquared: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Rho: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> State: SolverState;
fn finite(value: {scalar}) -> bool {{ return value == value && abs(value) <= {finite_limit}; }}
@compute @workgroup_size(1)
fn main() {{
    if (State.termination != 0u) {{ return; }}
    State.initial_rr = ResidualSquared.data[0];
    State.final_rr = ResidualSquared.data[0];
    State.rho = Rho.data[0];
    State.tolerance_sq = max(State.absolute_tolerance_sq,
        State.relative_tolerance_sq * State.initial_rr);
    // A solved system has zero residual and rho; positivity is needed only to continue CG.
    if (!finite(State.initial_rr) || State.initial_rr < 0.0 || !finite(State.rho) || State.rho < 0.0) {{
        State.termination = 3u;
    }} else if (State.initial_rr <= State.tolerance_sq) {{
        State.termination = 1u;
    }} else if (State.rho == 0.0) {{
        State.termination = 3u;
    }}
}}
"#
        ),
        alpha: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> Denominator: ScalarBuffer;
@group(0) @binding(1) var<storage, read_write> State: SolverState;
fn finite(value: {scalar}) -> bool {{ return value == value && abs(value) <= {finite_limit}; }}
@compute @workgroup_size(1)
fn main() {{
    if (State.termination != 0u) {{ return; }}
    let denominator = Denominator.data[0];
    if (!finite(denominator) || denominator <= 0.0) {{
        State.termination = 3u;
        return;
    }}
    State.alpha = State.rho / denominator;
    if (!finite(State.alpha)) {{ State.termination = 3u; }}
}}
"#
        ),
        update_solution_residual: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read_write> Solution: ScalarBuffer;
@group(0) @binding(1) var<storage, read_write> Residual: ScalarBuffer;
@group(0) @binding(2) var<storage, read> Direction: ScalarBuffer;
@group(0) @binding(3) var<storage, read> Product: ScalarBuffer;
@group(0) @binding(4) var<storage, read> State: SolverState;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index >= arrayLength(&Solution.data) || State.termination != 0u) {{ return; }}
    Solution.data[index] = Solution.data[index] + State.alpha * Direction.data[index];
    Residual.data[index] = Residual.data[index] - State.alpha * Product.data[index];
}}
"#
        ),
        apply_preconditioner: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> InverseDiagonal: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Residual: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> Preconditioned: ScalarBuffer;
@group(0) @binding(3) var<storage, read> State: SolverState;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < arrayLength(&Residual.data) && State.termination == 0u) {{
        Preconditioned.data[index] = InverseDiagonal.data[index] * Residual.data[index];
    }}
}}
"#
        ),
        update_state: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> ResidualSquared: ScalarBuffer;
@group(0) @binding(1) var<storage, read> NewRho: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> State: SolverState;
fn finite(value: {scalar}) -> bool {{ return value == value && abs(value) <= {finite_limit}; }}
@compute @workgroup_size(1)
fn main() {{
    if (State.termination != 0u) {{ return; }}
    State.iterations = State.iterations + 1u;
    State.final_rr = ResidualSquared.data[0];
    let new_rho = NewRho.data[0];
    // Exact convergence sets new_rho to zero before another direction/division is needed.
    if (!finite(State.final_rr) || State.final_rr < 0.0 || !finite(new_rho) || new_rho < 0.0) {{
        State.termination = 3u;
    }} else if (State.final_rr <= State.tolerance_sq) {{
        State.termination = 1u;
    }} else if (new_rho == 0.0) {{
        State.termination = 3u;
    }} else if (State.iterations >= State.max_iterations) {{
        State.termination = 2u;
    }} else {{
        State.beta = new_rho / State.rho;
        State.rho = new_rho;
        if (!finite(State.beta)) {{ State.termination = 3u; }}
    }}
}}
"#
        ),
        update_direction: format!(
            r#"{state}
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> Preconditioned: ScalarBuffer;
@group(0) @binding(1) var<storage, read_write> Direction: ScalarBuffer;
@group(0) @binding(2) var<storage, read> State: SolverState;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < arrayLength(&Direction.data) && State.termination == 0u) {{
        Direction.data[index] = Preconditioned.data[index] + State.beta * Direction.data[index];
    }}
}}
"#
        ),
    }
}

fn state_declaration(scalar: &str) -> String {
    format!(
        r#"
struct SolverState {{
    initial_rr: {scalar},
    rho: {scalar},
    alpha: {scalar},
    beta: {scalar},
    tolerance_sq: {scalar},
    final_rr: {scalar},
    relative_tolerance_sq: {scalar},
    absolute_tolerance_sq: {scalar},
    iterations: u32,
    termination: u32,
    max_iterations: u32,
}};
"#
    )
}

pub(super) struct CgShaderSources {
    pub(super) spmv: String,
    pub(super) initialize_vectors: String,
    pub(super) dot_partials: String,
    pub(super) dot_finalize: String,
    pub(super) initialize_state: String,
    pub(super) alpha: String,
    pub(super) update_solution_residual: String,
    pub(super) apply_preconditioner: String,
    pub(super) update_state: String,
    pub(super) update_direction: String,
}
