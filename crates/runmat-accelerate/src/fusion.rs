use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Weak};

use once_cell::sync::Lazy;
use runmat_builtins::Value;
use serde::{Deserialize, Serialize};

use crate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, InstrSpan, NodeId, PrimitiveOp, ShapeInfo, ValueId,
    ValueOrigin,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionKind {
    ElementwiseChain,
    Reduction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    pub id: usize,
    pub kind: FusionKind,
    pub nodes: Vec<NodeId>,
    pub shape: ShapeInfo,
    pub span: InstrSpan,
}

pub fn detect_fusion_groups(graph: &AccelGraph) -> Vec<FusionGroup> {
    if graph.nodes.is_empty() {
        return Vec::new();
    }

    let consumer_map = build_consumer_map(graph);
    let mut assigned: HashSet<NodeId> = HashSet::new();
    let mut groups = Vec::new();
    let mut group_id = 0usize;

    for node in &graph.nodes {
        // Elementwise chains
        if !node.is_elementwise() || assigned.contains(&node.id) {
            continue;
        }
        if node.outputs.is_empty() {
            continue;
        }
        let mut current_shape = node_output_shape(graph, node);
        if matches!(current_shape, ShapeInfo::Unknown) {
            continue;
        }
        let mut chain: Vec<NodeId> = Vec::new();
        let mut frontier = node.id;
        let mut local_seen: HashSet<NodeId> = HashSet::new();

        loop {
            if !local_seen.insert(frontier) {
                break;
            }
            chain.push(frontier);
            let next = find_next_elementwise(
                graph,
                frontier,
                &assigned,
                &local_seen,
                &consumer_map,
                &current_shape,
            );
            match next {
                Some((next_id, next_shape)) => {
                    frontier = next_id;
                    current_shape = next_shape;
                }
                None => break,
            }
        }

        if chain.len() > 1 {
            for id in &chain {
                assigned.insert(*id);
            }
            let span = group_span(graph, &chain);
            groups.push(FusionGroup {
                id: group_id,
                kind: FusionKind::ElementwiseChain,
                nodes: chain,
                shape: current_shape.clone(),
                span,
            });
            group_id += 1;
        }
    }

    // Reduction singletons (basic grouping; future: include eligible producers)
    for node in &graph.nodes {
        if !node.is_reduction() || assigned.contains(&node.id) {
            continue;
        }
        let span = InstrSpan {
            start: node.span.start,
            end: node.span.end,
        };
        groups.push(FusionGroup {
            id: group_id,
            kind: FusionKind::Reduction,
            nodes: vec![node.id],
            shape: node_output_shape(graph, node),
            span,
        });
        group_id += 1;
    }

    groups
}

fn build_consumer_map(graph: &AccelGraph) -> HashMap<ValueId, HashSet<NodeId>> {
    let mut map: HashMap<ValueId, HashSet<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        for &input in &node.inputs {
            if let Some(value) = graph.value(input) {
                if matches!(value.origin, crate::graph::ValueOrigin::NodeOutput { .. }) {
                    map.entry(input).or_default().insert(node.id);
                }
            }
        }
    }
    map
}

fn node_output_shape(graph: &AccelGraph, node: &AccelNode) -> ShapeInfo {
    let mut shape = ShapeInfo::Scalar;
    for &output in &node.outputs {
        if let Some(info) = graph.value(output) {
            shape = shape.unify(&info.shape);
        }
    }
    shape
}

fn find_next_elementwise(
    graph: &AccelGraph,
    node_id: NodeId,
    assigned: &HashSet<NodeId>,
    local_seen: &HashSet<NodeId>,
    consumer_map: &HashMap<ValueId, HashSet<NodeId>>,
    current_shape: &ShapeInfo,
) -> Option<(NodeId, ShapeInfo)> {
    let node = graph.node(node_id)?;
    let mut candidate: Option<(NodeId, ShapeInfo)> = None;

    for &output in &node.outputs {
        let consumers = consumer_map.get(&output)?;
        if consumers.len() != 1 {
            return None;
        }
        let next_id = *consumers.iter().next()?;
        if next_id <= node_id || assigned.contains(&next_id) || local_seen.contains(&next_id) {
            return None;
        }
        let next_node = graph.node(next_id)?;
        if !next_node.is_elementwise() {
            return None;
        }
        // Ensure the edge we follow is actually used by next node
        if !next_node.inputs.contains(&output) {
            continue;
        }
        let next_shape = node_output_shape(graph, next_node);
        if matches!(next_shape, ShapeInfo::Unknown) {
            return None;
        }
        let unified = current_shape.unify(&next_shape);
        if matches!(unified, ShapeInfo::Unknown) {
            return None;
        }
        candidate = Some((next_id, unified));
        break;
    }

    candidate
}

fn group_span(graph: &AccelGraph, nodes: &[NodeId]) -> InstrSpan {
    let mut start = usize::MAX;
    let mut end = 0usize;
    for &id in nodes {
        if let Some(node) = graph.node(id) {
            start = start.min(node.span.start);
            end = end.max(node.span.end);
        }
    }
    if start == usize::MAX {
        start = 0;
    }
    InstrSpan { start, end }
}

#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub groups: Vec<FusionGroupPlan>,
}

#[derive(Debug, Clone)]
pub struct FusionGroupPlan {
    pub index: usize,
    pub group: FusionGroup,
    pub operations: Vec<FusionOp>,
    pub inputs: Vec<ValueId>,
    pub stack_pattern: Vec<usize>,
    pub constants: HashMap<usize, Value>,
    pub output: Option<ValueId>,
    pub kernel: FusionKernelSpec,
    // For reductions: track the ValueId of the data tensor being reduced, if identifiable
    pub reduction_data: Option<ValueId>,
}

#[derive(Debug, Clone)]
pub enum FusionOp {
    Primitive {
        op: PrimitiveOp,
        inputs: Vec<ValueId>,
        output: Option<ValueId>,
    },
    Builtin {
        name: String,
        inputs: Vec<ValueId>,
        output: Option<ValueId>,
    },
}

#[derive(Debug, Clone)]
pub struct FusionKernelSpec {
    pub kind: FusionKind,
    pub supported: bool,
}

impl FusionKernelSpec {
    fn new(kind: FusionKind, supported: bool) -> Self {
        Self { kind, supported }
    }
}

#[derive(Clone, Debug)]
pub struct ActiveFusion {
    pub kind: FusionKind,
    pub span: InstrSpan,
    pub element_count: Option<usize>,
    pub supported: bool,
}

struct ActiveContext {
    plan: Arc<FusionPlan>,
    active_group: Option<usize>,
}

static PLAN_CACHE: Lazy<RwLock<HashMap<usize, Weak<FusionPlan>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static ACTIVE_PLAN: RefCell<Option<ActiveContext>> = const { RefCell::new(None) };
}

pub fn prepare_fusion_plan(
    graph: Option<&AccelGraph>,
    groups: &[FusionGroup],
) -> Option<Arc<FusionPlan>> {
    let graph = graph?;
    if groups.is_empty() {
        return None;
    }
    let key = graph as *const AccelGraph as usize;
    if let Some(plan) = PLAN_CACHE
        .read()
        .ok()
        .and_then(|guard| guard.get(&key).and_then(|weak| weak.upgrade()))
    {
        return Some(plan);
    }

    let plan = FusionPlan::from_graph(graph, groups);
    let plan = Arc::new(plan);
    if let Ok(mut guard) = PLAN_CACHE.write() {
        guard.insert(key, Arc::downgrade(&plan));
    }
    Some(plan)
}

pub fn activate_fusion_plan(plan: Option<Arc<FusionPlan>>) {
    ACTIVE_PLAN.with(|ctx| {
        let mut slot = ctx.borrow_mut();
        *slot = plan.map(|plan| ActiveContext {
            plan,
            active_group: None,
        });
    });
}

pub fn deactivate_fusion_plan() {
    ACTIVE_PLAN.with(|ctx| {
        ctx.borrow_mut().take();
    });
}

pub fn set_current_pc(pc: usize) {
    ACTIVE_PLAN.with(|ctx| {
        if let Some(context) = ctx.borrow_mut().as_mut() {
            context.active_group = context.plan.group_for_pc(pc);
        }
    });
}

pub fn active_fusion() -> Option<ActiveFusion> {
    ACTIVE_PLAN.with(|ctx| {
        ctx.borrow()
            .as_ref()
            .and_then(|context| {
                context
                    .active_group
                    .and_then(|idx| context.plan.groups.get(idx))
            })
            .map(|plan| ActiveFusion {
                kind: plan.group.kind.clone(),
                span: plan.group.span.clone(),
                element_count: plan.element_count(),
                supported: plan.kernel.supported,
            })
    })
}

pub fn active_group_plan_clone() -> Option<FusionGroupPlan> {
    ACTIVE_PLAN.with(|ctx| {
        ctx.borrow().as_ref().and_then(|context| {
            context
                .active_group
                .and_then(|idx| context.plan.groups.get(idx).cloned())
        })
    })
}

impl FusionPlan {
    pub fn from_graph(graph: &AccelGraph, groups: &[FusionGroup]) -> Self {
        let plans = groups
            .iter()
            .enumerate()
            .map(|(idx, group)| FusionGroupPlan::new(idx, group.clone(), graph))
            .collect();
        Self { groups: plans }
    }

    fn group_for_pc(&self, pc: usize) -> Option<usize> {
        self.groups
            .iter()
            .find(|plan| pc >= plan.group.span.start && pc <= plan.group.span.end)
            .map(|plan| plan.index)
    }
}

impl From<Vec<FusionGroupPlan>> for FusionPlan {
    fn from(groups: Vec<FusionGroupPlan>) -> Self {
        Self { groups }
    }
}

impl FusionGroupPlan {
    fn new(index: usize, group: FusionGroup, graph: &AccelGraph) -> Self {
        let node_set: HashSet<NodeId> = group.nodes.iter().copied().collect();
        let mut seen_inputs: HashMap<ValueId, usize> = HashMap::new();
        let mut inputs: Vec<ValueId> = Vec::new();
        let mut stack_pattern: Vec<usize> = Vec::new();
        let mut constants: HashMap<usize, Value> = HashMap::new();
        let mut operations = Vec::new();
        let mut reduction_data: Option<ValueId> = None;
        let mut output: Option<ValueId> = None;

        let is_reduction_group = group.kind.is_reduction();
        for node_id in &group.nodes {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            for input in &node.inputs {
                let (external, is_variable, maybe_constant) = match graph.value(*input) {
                    Some(info) => match &info.origin {
                        ValueOrigin::NodeOutput { node: origin, .. }
                            if node_set.contains(origin) =>
                        {
                            (false, false, None)
                        }
                        ValueOrigin::Variable { .. } => (true, true, None),
                        ValueOrigin::Constant => (true, false, info.constant.clone()),
                        _ => (true, false, None),
                    },
                    None => (true, false, None),
                };
                if external {
                    // Special handling for reductions: do NOT include constants in inputs;
                    // only the data tensor should be an input. Constants are recorded separately.
                    if is_reduction_group {
                        if let Some(constant) = maybe_constant.clone() {
                            // Assign a synthetic key for constants; keys are not positional for reductions
                            let key = constants.len() + 1000;
                            constants.insert(key, constant);
                            continue;
                        }
                        // Only include the reduction data operand as an input
                        if let Some(data_id) = reduction_data {
                            if *input != data_id {
                                // Skip non-data external inputs for reduction groups
                                continue;
                            }
                        }
                    }

                    let input_idx = if let Some(idx) = seen_inputs.get(input) {
                        *idx
                    } else {
                        let idx = inputs.len();
                        inputs.push(*input);
                        seen_inputs.insert(*input, idx);
                        idx
                    };

                    if let Some(constant) = maybe_constant.clone() {
                        constants.insert(input_idx, constant);
                    } else if !is_variable {
                        stack_pattern.push(input_idx);
                    }
                }
            }

            let op = match &node.label {
                AccelNodeLabel::Primitive(p) => FusionOp::Primitive {
                    op: *p,
                    inputs: node.inputs.clone(),
                    output: node.outputs.get(0).copied(),
                },
                AccelNodeLabel::Builtin { name } => FusionOp::Builtin {
                    name: name.clone(),
                    inputs: node.inputs.clone(),
                    output: node.outputs.get(0).copied(),
                },
                AccelNodeLabel::Unknown => FusionOp::Primitive {
                    op: PrimitiveOp::UPlus,
                    inputs: node.inputs.clone(),
                    output: node.outputs.get(0).copied(),
                },
            };
            operations.push(op);

            if let Some(out) = node.outputs.get(0).copied() {
                output = Some(out);
            }
            // If this is a reduction builtin we recognize, record its first input as the data operand
            if let AccelNodeLabel::Builtin { name } = &node.label {
                let lname = name.to_ascii_lowercase();
                if (lname == "sum" || lname == "mean") && !node.inputs.is_empty() {
                    reduction_data = Some(node.inputs[0]);
                }
            }
        }

        let kind = group.kind.clone();
        let mut plan = Self {
            index,
            group,
            operations,
            stack_pattern,
            constants,
            inputs,
            output,
            kernel: FusionKernelSpec::new(kind, true),
            reduction_data,
        };

        // For reduction groups, externalize only the data operand as input; keep constants separate
        if plan.group.kind.is_reduction() {
            if let Some(data) = plan.reduction_data {
                plan.inputs.retain(|vid| *vid == data);
                plan.stack_pattern.clear();
            }
        }

        let supported = if plan.kernel.kind.is_elementwise() {
            plan.generate_wgsl("f32").is_some()
        } else if plan.kernel.kind.is_reduction() {
            plan.generate_reduction_wgsl("f32").is_some()
        } else {
            false
        };
        plan.kernel.supported = supported;
        plan
    }

    pub fn reduction_data_shape(&self, graph: &AccelGraph) -> Option<Vec<usize>> {
        let vid = self.reduction_data?;
        let info = graph.value(vid)?;
        match &info.shape {
            ShapeInfo::Tensor(dims) if !dims.is_empty() && dims.iter().all(|d| d.is_some()) => {
                Some(dims.iter().map(|d| d.unwrap()).collect())
            }
            _ => None,
        }
    }

    pub fn element_count(&self) -> Option<usize> {
        self.group.element_count()
    }

    pub fn constant_shape(&self, len: usize) -> Vec<usize> {
        match &self.group.shape {
            ShapeInfo::Tensor(dims) if !dims.is_empty() && dims.iter().all(|dim| dim.is_some()) => {
                dims.iter().map(|dim| dim.unwrap()).collect()
            }
            _ => vec![len],
        }
    }

    pub fn generate_wgsl(&self, scalar_ty: &str) -> Option<String> {
        if !self.kernel.kind.is_elementwise() {
            return None;
        }
        if !self.kernel.supported {
            return None;
        }
        let output_id = self.output?;
        let mut exprs: HashMap<ValueId, String> = HashMap::new();
        for (idx, input_id) in self.inputs.iter().enumerate() {
            exprs.insert(*input_id, format!("input{idx}.data[idx]"));
        }

        let mut body = String::new();
        for (node_idx, op) in self.operations.iter().enumerate() {
            let tmp_name = format!("tmp{node_idx}");
            match op {
                FusionOp::Primitive { op, inputs, output } => {
                    let expr = primitive_expr(*op, inputs, &exprs)?;
                    body.push_str(&format!("    let {tmp_name}: {scalar_ty} = {expr};\n"));
                    if let Some(out) = output {
                        exprs.insert(*out, tmp_name.clone());
                    }
                }
                FusionOp::Builtin {
                    name,
                    inputs,
                    output,
                } => {
                    let expr = builtin_expr(name, inputs, &exprs, scalar_ty)?;
                    body.push_str(&format!("    let {tmp_name}: {scalar_ty} = {expr};\n"));
                    if let Some(out) = output {
                        exprs.insert(*out, tmp_name.clone());
                    }
                }
            }
        }

        let final_expr = exprs.get(&output_id)?.clone();

        let mut shader = String::new();
        shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}> }};\n"));
        shader.push_str("struct Params {\n    len: u32;\n    _pad0: u32;\n    _pad1: u32;\n    _pad2: u32;\n}\n\n");
        // Provide portable stubs; avoid relying on backend builtins that may be missing
        if scalar_ty == "f32" {
            shader.push_str("fn isNan(x: f32) -> bool { return x != x; }\n");
            shader.push_str("fn isFinite(x: f32) -> bool { return (x == x) && (abs(x) < 3.4028234663852886e38); }\n");
            shader.push_str("fn isInf(x: f32) -> bool { return (x == x) && !(abs(x) < 3.4028234663852886e38); }\n\n");
        } else {
            shader.push_str("fn isNan(x: f64) -> bool { return x != x; }\n");
            shader.push_str("fn isFinite(x: f64) -> bool { return (x == x) && (abs(x) < f64(1.7976931348623157e308)); }\n");
            shader.push_str("fn isInf(x: f64) -> bool { return (x == x) && !(abs(x) < f64(1.7976931348623157e308)); }\n\n");
        }
        for (idx, _) in self.inputs.iter().enumerate() {
            shader.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> input{}: Tensor;\n",
                idx, idx
            ));
        }
        shader.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> output: Tensor;\n",
            self.inputs.len()
        ));
        shader.push_str(&format!(
            "@group(0) @binding({}) var<uniform> params: Params;\n\n",
            self.inputs.len() + 1
        ));
        shader.push_str("@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
        shader.push_str(
            "    let idx = gid.x;\n    if (idx >= params.len) {\n        return;\n    }\n",
        );
        shader.push_str(&body);
        shader.push_str(&format!("    output.data[idx] = {final_expr};\n}}\n"));
        Some(shader)
    }

    pub fn generate_reduction_wgsl(&self, scalar_ty: &str) -> Option<String> {
        if !self.kernel.kind.is_reduction() {
            return None;
        }
        // Minimal column-major reduction kernel template (single workgroup per slice).
        // Assumes first input is the tensor to reduce; ignores additional inputs for now.
        if self.inputs.is_empty() {
            return None;
        }
        // Determine axis from any numeric constant; 1-based MATLAB dim => 0 for columns, 1 for rows.
        let mut axis = 0usize;
        for v in self.constants.values() {
            match v {
                Value::Num(n) if *n >= 1.0 => {
                    axis = (*n as usize).saturating_sub(1);
                    break;
                }
                Value::Int(i) => {
                    let val = i.to_f64();
                    if val >= 1.0 {
                        axis = (val as usize).saturating_sub(1);
                        break;
                    }
                }
                _ => {}
            }
        }

        // Detect omitnan constant (compile-time selection)
        let omitnan = self.constants.values().any(|v| match v {
            Value::String(s) => s.eq_ignore_ascii_case("omitnan"),
            _ => false,
        });

        let mut shader = String::new();
        shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>, }};\n"));
        shader.push_str("struct MParams { nrows: u32, ncols: u32, ld: u32, flags: u32 }\n\n");
        shader.push_str(&format!(
            "@group(0) @binding(0) var<storage, read> input0: Tensor;\n"
        ));
        shader.push_str(&format!(
            "@group(0) @binding(1) var<storage, read_write> output: Tensor;\n"
        ));
        shader.push_str(&format!(
            "@group(0) @binding(2) var<uniform> params: MParams;\n\n"
        ));
        // Use a small fixed workgroup tile size to avoid driver stalls on some backends
        shader.push_str("var<workgroup> tile: array<f32, 64u>;\n\n");
        shader.push_str(&format!(
            "const OMITNAN: bool = {}bool({});\n\n",
            if scalar_ty == "f64" { "" } else { "" },
            if omitnan { "true" } else { "false" }
        ));
        shader.push_str("@compute @workgroup_size(64)\n");
        if axis == 0 {
            // Column-wise: reduce over rows; one output per column (ncols)
            shader.push_str(
                "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
            );
            shader.push_str("  let col = wid.x;\n  if (col >= params.ncols) { return; }\n");
            shader.push_str(&format!(
                "  var acc: {scalar_ty} = {}0.0;\n",
                if scalar_ty == "f64" { "f64(" } else { "" }
            ));
            if scalar_ty == "f64" {
                shader.push_str("  // close cast for f64 literal\n");
            }
            shader.push_str("  fn isNanF(x: f32) -> bool { return x != x; }\n");
            shader.push_str("  var saw_nan: bool = false;\n  var r = lid.x;\n");
            shader.push_str("  while (r < params.nrows) {\n    let v = input0.data[ (col * params.ld) + r ];\n    if (OMITNAN) { if (!isNanF(v)) { acc = acc + v; } } else { if (isNanF(v)) { saw_nan = true; } else { acc = acc + v; } }\n    r += 64u;\n  }\n");
        } else {
            // Row-wise: reduce over cols; one output per row (nrows)
            shader.push_str(
                "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
            );
            shader.push_str("  let row = wid.x;\n  if (row >= params.nrows) { return; }\n");
            shader.push_str(&format!(
                "  var acc: {scalar_ty} = {}0.0;\n",
                if scalar_ty == "f64" { "f64(" } else { "" }
            ));
            if scalar_ty == "f64" {
                shader.push_str("  // close cast for f64 literal\n");
            }
            shader.push_str("  fn isNanF(x: f32) -> bool { return x != x; }\n");
            shader.push_str("  var saw_nan: bool = false;\n  var c = lid.x;\n");
            shader.push_str("  while (c < params.ncols) {\n    let v = input0.data[ row + (c * params.ld) ];\n    if (OMITNAN) { if (!isNanF(v)) { acc = acc + v; } } else { if (isNanF(v)) { saw_nan = true; } else { acc = acc + v; } }\n    c += 64u;\n  }\n");
        }
        shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
        shader.push_str(
            "  var off = 32u;\n  loop { if (off == 0u) { break; } if (lid.x < off) {\n    let a = tile[lid.x]; let b = tile[lid.x + off];\n    if (!is_omitnan(params.flags) && (isNanF(a) || isNanF(b))) { tile[lid.x] = f32(NaN); } else { tile[lid.x] = a + b; }\n  } workgroupBarrier(); off = off / 2u; }\n",
        );
        if axis == 0 {
            shader.push_str("  if (lid.x == 0u) { output.data[col] = tile[0u]; }\n}\n");
        } else {
            shader.push_str("  if (lid.x == 0u) { output.data[row] = tile[0u]; }\n}\n");
        }
        Some(shader)
    }
}

impl FusionGroup {
    pub fn element_count(&self) -> Option<usize> {
        match &self.shape {
            ShapeInfo::Scalar => Some(1),
            ShapeInfo::Tensor(dims) => dims.iter().try_fold(1usize, |acc, dim| {
                if let Some(size) = dim.and_then(|d| acc.checked_mul(d)) {
                    Some(size)
                } else {
                    None
                }
            }),
            ShapeInfo::Unknown => None,
        }
    }
}

impl FusionKind {
    pub fn is_elementwise(&self) -> bool {
        matches!(self, FusionKind::ElementwiseChain)
    }

    pub fn is_reduction(&self) -> bool {
        matches!(self, FusionKind::Reduction)
    }
}

fn primitive_expr(
    op: PrimitiveOp,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let binary = |exprs: &HashMap<ValueId, String>| -> Option<(String, String)> {
        let lhs = exprs.get(inputs.get(0)?).cloned()?;
        let rhs = exprs.get(inputs.get(1)?).cloned()?;
        Some((lhs, rhs))
    };
    match op {
        PrimitiveOp::Add => {
            let (lhs, rhs) = binary(exprs)?;
            Some(format!("({lhs} + {rhs})"))
        }
        PrimitiveOp::Sub => {
            let (lhs, rhs) = binary(exprs)?;
            Some(format!("({lhs} - {rhs})"))
        }
        PrimitiveOp::Mul | PrimitiveOp::ElemMul => {
            let (lhs, rhs) = binary(exprs)?;
            Some(format!("({lhs} * {rhs})"))
        }
        PrimitiveOp::Div | PrimitiveOp::ElemDiv | PrimitiveOp::ElemLeftDiv => {
            let (lhs, rhs) = binary(exprs)?;
            Some(format!("({lhs} / {rhs})"))
        }
        PrimitiveOp::Pow | PrimitiveOp::ElemPow => {
            let (lhs, rhs) = binary(exprs)?;
            Some(format!("pow({lhs}, {rhs})"))
        }
        PrimitiveOp::Neg => {
            let arg = exprs.get(inputs.get(0)?).cloned()?;
            Some(format!("(-{arg})"))
        }
        PrimitiveOp::UPlus => {
            let arg = exprs.get(inputs.get(0)?).cloned()?;
            Some(format!("(+{arg})"))
        }
        _ => None,
    }
}

fn builtin_expr(
    name: &str,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
    scalar_ty: &str,
) -> Option<String> {
    let func = match name.to_ascii_lowercase().as_str() {
        "isfinite" => return builtin_unary_call("isFinite", inputs, exprs),
        "isinf" => return builtin_unary_call("isInf", inputs, exprs),
        "isnan" => return builtin_unary_call("isNan", inputs, exprs),
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "asin" => "asin",
        "acos" => "acos",
        "atan" => "atan",
        "atan2" => return builtin_binary("atan2", inputs, exprs),
        "sinh" => "sinh",
        "cosh" => "cosh",
        "tanh" => "tanh",
        "exp" => "exp",
        "log" => "log",
        "log2" => "log2",
        "sqrt" => "sqrt",
        "abs" => "abs",
        "exp2" => "exp2",
        "floor" => "floor",
        "ceil" => "ceil",
        "round" => "round",
        "trunc" => "trunc",
        _ => {
            return match name.to_ascii_lowercase().as_str() {
                "log10" => {
                    let arg = exprs.get(inputs.get(0)?).cloned()?;
                    let constant = cast_literal(scalar_ty, "0.4342944819032518");
                    Some(format!("(log({arg}) * {constant})"))
                }
                "log1p" => {
                    let arg = exprs.get(inputs.get(0)?).cloned()?;
                    let one = cast_literal(scalar_ty, "1.0");
                    Some(format!("log({arg} + {one})"))
                }
                "expm1" => {
                    let arg = exprs.get(inputs.get(0)?).cloned()?;
                    let one = cast_literal(scalar_ty, "1.0");
                    Some(format!("(exp({arg}) - {one})"))
                }
                _ => None,
            }
        }
    };
    let arg = exprs.get(inputs.get(0)?).cloned()?;
    Some(format!("{func}({arg})"))
}

fn builtin_binary(
    func: &str,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let lhs = exprs.get(inputs.get(0)?).cloned()?;
    let rhs = exprs.get(inputs.get(1)?).cloned()?;
    Some(format!("{func}({lhs}, {rhs})"))
}

fn builtin_unary_call(
    func: &str,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let arg = exprs.get(inputs.get(0)?).cloned()?;
    Some(format!("{func}({arg})"))
}

fn cast_literal(scalar_ty: &str, literal: &str) -> String {
    if scalar_ty == "f64" {
        format!("{scalar_ty}({literal})")
    } else {
        literal.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan,
        PrimitiveOp, ValueId, ValueInfo, ValueOrigin, VarKind,
    };
    use runmat_builtins::Type;
    use std::collections::HashMap as StdHashMap;

    fn simple_elementwise_graph() -> AccelGraph {
        let mut values = Vec::new();
        // Value 0: input tensor
        values.push(ValueInfo {
            id: 0,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            constant: None,
        });
        // Node 0 output value (value id 1)
        values.push(ValueInfo {
            id: 1,
            origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            constant: None,
        });
        // Node 1 output value (value id 2)
        values.push(ValueInfo {
            id: 2,
            origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            constant: None,
        });

        let node0 = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![0, 0],
            outputs: vec![1],
            span: InstrSpan { start: 10, end: 10 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let node1 = AccelNode {
            id: 1,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![1, 0],
            outputs: vec![2],
            span: InstrSpan { start: 11, end: 11 },
            tags: vec![AccelGraphTag::Elementwise],
        };

        AccelGraph {
            nodes: vec![node0, node1],
            values,
        }
    }

    #[test]
    fn detects_chain() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.nodes, vec![0, 1]);
        assert_eq!(group.kind, FusionKind::ElementwiseChain);
    }

    #[test]
    fn builds_plan_and_template() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        let plan = FusionPlan::from_graph(&graph, &groups);
        assert_eq!(plan.groups.len(), 1);
        let group_plan = &plan.groups[0];
        assert!(group_plan.kernel.supported);
        let wgsl = group_plan.generate_wgsl("f32").expect("wgsl");
        assert!(wgsl.contains("@compute"));
        assert!(group_plan.group.element_count().is_some());
    }

    #[test]
    fn stack_pattern_tracks_repeated_constants() {
        let mut values = Vec::new();
        values.push(ValueInfo {
            id: 0,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: 1,
            origin: ValueOrigin::Constant,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: 2,
            origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: 3,
            origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4)]),
            constant: None,
        });

        let node0 = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Add),
            category: AccelOpCategory::Elementwise,
            inputs: vec![0, 1],
            outputs: vec![2],
            span: InstrSpan { start: 5, end: 5 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let node1 = AccelNode {
            id: 1,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Add),
            category: AccelOpCategory::Elementwise,
            inputs: vec![2, 1],
            outputs: vec![3],
            span: InstrSpan { start: 6, end: 6 },
            tags: vec![AccelGraphTag::Elementwise],
        };

        let graph = AccelGraph {
            nodes: vec![node0, node1],
            values,
        };

        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let group_plan = &plan.groups[0];
        assert_eq!(group_plan.inputs.len(), 2);
        assert_eq!(group_plan.stack_pattern.len(), 2);
        assert!(group_plan.stack_pattern.iter().all(|idx| *idx == 1));
    }

    #[test]
    fn builtin_expr_supports_extended_set() {
        let mut exprs: StdHashMap<ValueId, String> = StdHashMap::new();
        exprs.insert(0, "v0".to_string());
        exprs.insert(1, "v1".to_string());

        let log1p = super::builtin_expr("log1p", &[0], &exprs, "f32");
        assert!(log1p.is_some());

        let log10 = super::builtin_expr("log10", &[0], &exprs, "f64");
        assert!(log10.unwrap().contains("log"));

        let expm1 = super::builtin_expr("expm1", &[0], &exprs, "f32");
        assert!(expm1.unwrap().contains("exp"));

        let floor = super::builtin_expr("floor", &[0], &exprs, "f32");
        assert_eq!(floor.unwrap(), "floor(v0)");

        let atan2 = super::builtin_expr("atan2", &[0, 1], &exprs, "f32");
        assert_eq!(atan2.unwrap(), "atan2(v0, v1)");
    }
}
