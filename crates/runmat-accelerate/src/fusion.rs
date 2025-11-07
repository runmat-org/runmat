use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Weak};

use once_cell::sync::Lazy;
use runmat_builtins::Value;
use serde::{Deserialize, Serialize};

use crate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, NodeId, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin,
};
use runmat_accelerate_api::CovNormalization;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionKind {
    ElementwiseChain,
    Reduction,
    MatmulEpilogue,
    CenteredGram,
    ImageNormalize,
    PowerStepNormalize,
    ExplainedVariance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    pub id: usize,
    pub kind: FusionKind,
    pub nodes: Vec<NodeId>,
    pub shape: ShapeInfo,
    pub span: InstrSpan,
    pub pattern: Option<FusionPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionPattern {
    CenteredGram {
        matrix: ValueId,
        normalization: CovNormalization,
    },
    ImageNormalize(ImageNormalizePattern),
    PowerStepNormalize {
        lhs: ValueId,
        rhs: ValueId,
        epsilon: f64,
    },
    ExplainedVariance {
        q: ValueId,
        g: ValueId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageNormalizePattern {
    pub input: ValueId,
    pub epsilon: ImageScalar,
    pub gain: Option<ImageScalar>,
    pub bias: Option<ImageScalar>,
    pub gamma: ImageScalar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageScalar {
    Constant(f64),
    Value(ValueId),
}

pub fn detect_fusion_groups(graph: &AccelGraph) -> Vec<FusionGroup> {
    if graph.nodes.is_empty() {
        return Vec::new();
    }

    let consumer_map = build_consumer_map(graph);
    let mut assigned: HashSet<NodeId> = HashSet::new();
    let mut groups = Vec::new();
    let mut group_id = 0usize;

    detect_image_normalize(graph, &mut assigned, &mut groups, &mut group_id);
    detect_explained_variance(graph, &mut assigned, &mut groups, &mut group_id);
    detect_power_step_normalize(graph, &mut assigned, &mut groups, &mut group_id);
    detect_centered_gram(graph, &mut assigned, &mut groups, &mut group_id);

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
                pattern: None,
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
            pattern: None,
        });
        group_id += 1;
    }

    // Matmul + simple epilogue (alpha/beta/row/col scale) chains
    for node in &graph.nodes {
        if node.category != AccelOpCategory::MatMul || assigned.contains(&node.id) {
            continue;
        }
        if node.outputs.is_empty() {
            continue;
        }
        // Require exactly one consumer chain and only elementwise ops we can fold
        let mut chain: Vec<NodeId> = vec![node.id];
        let mut frontier = node.id;
        let mut ok = false;
        loop {
            // Find single consumer of the current frontier's output
            let mut next_id_opt: Option<NodeId> = None;
            for &out in &graph.node(frontier).unwrap().outputs {
                if let Some(cons) = consumer_map.get(&out) {
                    if cons.len() == 1 {
                        next_id_opt = cons.iter().copied().next();
                    } else {
                        next_id_opt = None;
                    }
                }
            }
            let Some(next_id) = next_id_opt else { break };
            let next = graph.node(next_id).unwrap();
            if !next.is_elementwise() {
                break;
            }
            // Allow only primitive elementwise ops we can fold: add/sub/mul/div/elem variants
            let allowed = matches!(
                next.label,
                AccelNodeLabel::Primitive(PrimitiveOp::Add)
                    | AccelNodeLabel::Primitive(PrimitiveOp::Sub)
                    | AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                    | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul)
                    | AccelNodeLabel::Primitive(PrimitiveOp::Div)
                    | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
            );
            if !allowed {
                break;
            }
            chain.push(next_id);
            frontier = next_id;
            ok = true;
        }
        if ok {
            for id in &chain {
                assigned.insert(*id);
            }
            let span = group_span(graph, &chain);
            groups.push(FusionGroup {
                id: group_id,
                kind: FusionKind::MatmulEpilogue,
                nodes: chain,
                shape: node_output_shape(graph, node),
                span,
                pattern: None,
            });
            group_id += 1;
        }
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
    pub const_values: HashMap<ValueId, Value>,
    pub output: Option<ValueId>,
    pub kernel: FusionKernelSpec,
    // For reductions: track the ValueId of the data tensor being reduced, if identifiable
    pub reduction_data: Option<ValueId>,
    // For reductions: the semantic mode (sum or mean)
    pub reduction_mode: Option<ReductionMode>,
    pub pattern: Option<FusionPattern>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMode {
    Sum,
    Mean,
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
        let const_values: HashMap<ValueId, Value> = HashMap::new();
        let mut operations = Vec::new();
        let mut reduction_mode: Option<ReductionMode> = None;
        let mut reduction_data: Option<ValueId> = None;
        let mut output: Option<ValueId> = None;

        let is_reduction_group = group.kind.is_reduction();
        for node_id in &group.nodes {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            for input in &node.inputs {
                let binding = graph.var_binding(*input);
                let (external, is_variable, maybe_constant) = match graph.value(*input) {
                    Some(info) => match &info.origin {
                        ValueOrigin::NodeOutput { node: origin, .. }
                            if node_set.contains(origin) =>
                        {
                            (false, false, None)
                        }
                        ValueOrigin::Variable { .. } => (true, true, None),
                        ValueOrigin::NodeOutput { .. } if binding.is_some() => (true, true, None),
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

                    let mut newly_added = false;
                    let input_idx = if let Some(idx) = seen_inputs.get(input) {
                        *idx
                    } else {
                        let idx = inputs.len();
                        inputs.push(*input);
                        seen_inputs.insert(*input, idx);
                        newly_added = true;
                        idx
                    };

                    if let Some(constant) = maybe_constant.clone() {
                        constants.insert(input_idx, constant);
                    } else if !is_variable && newly_added {
                        stack_pattern.push(input_idx);
                    } else if !is_variable
                        && !newly_added
                        && matches!(
                            graph.value(*input).map(|v| &v.origin),
                            Some(ValueOrigin::Constant)
                        )
                    {
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
                    reduction_mode = Some(if lname == "mean" {
                        ReductionMode::Mean
                    } else {
                        ReductionMode::Sum
                    });
                }
            }
        }

        let kind = group.kind.clone();
        let pattern = group.pattern.clone();
        let mut plan = Self {
            index,
            group,
            operations,
            stack_pattern,
            constants,
            const_values,
            inputs,
            output,
            kernel: FusionKernelSpec::new(kind, true),
            reduction_data,
            reduction_mode,
            pattern,
        };

        // Record constant ValueIds for all groups for easier downstream analysis
        for node_id in &plan.group.nodes {
            if let Some(node) = graph.node(*node_id) {
                for &inp in &node.inputs {
                    if let Some(info) = graph.value(inp) {
                        if let Some(cv) = info.constant.clone() {
                            plan.const_values.insert(inp, cv);
                        }
                    }
                }
            }
        }

        // For reduction groups, externalize only real tensor dependencies; keep constants separate
        if plan.group.kind.is_reduction() {
            if let Some(data_vid) = plan.reduction_data {
                // Record constant ValueIds for codegen
                // Build dependency map from op outputs to inputs
                let mut prod: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
                for op in &plan.operations {
                    match op {
                        FusionOp::Primitive {
                            inputs,
                            output,
                            op: _,
                        } => {
                            if let Some(out) = output {
                                prod.insert(*out, inputs.clone());
                            }
                        }
                        FusionOp::Builtin {
                            name: _,
                            inputs,
                            output,
                        } => {
                            if let Some(out) = output {
                                prod.insert(*out, inputs.clone());
                            }
                        }
                    }
                }
                let original_inputs = plan.inputs.clone();
                let mut deps: Vec<ValueId> = Vec::new();
                let mut visited: HashSet<ValueId> = HashSet::new();
                let mut stack: Vec<ValueId> = vec![data_vid];
                while let Some(cur) = stack.pop() {
                    if !visited.insert(cur) {
                        continue;
                    }
                    if original_inputs.contains(&cur) {
                        if !deps.contains(&cur) {
                            deps.push(cur);
                        }
                        continue;
                    }
                    if let Some(parents) = prod.get(&cur) {
                        for p in parents {
                            stack.push(*p);
                        }
                    }
                }
                plan.inputs = deps;
                plan.stack_pattern.clear();
            }
        }

        let supported = if plan.kernel.kind.is_elementwise() {
            plan.generate_wgsl("f32").is_some()
        } else if plan.kernel.kind.is_reduction() {
            plan.generate_reduction_wgsl("f32").is_some()
        } else {
            // Non-WGSL kinds are executed via provider paths
            true
        };
        plan.kernel.supported = plan.kernel.supported && supported;
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
            // Placeholder; will be replaced by broadcasted index variable i{idx}
            exprs.insert(*input_id, format!("input{idx}.data[i{idx}]"));
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
        shader.push_str("const MAX_RANK: u32 = 128u;\n");
        shader.push_str("struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };\n");
        shader.push_str("alias PackedArray = array<PackedValue, MAX_RANK>;\n\n");
        shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>, }};\n"));
        // Broadcast-aware Params: len, offset, rank, pad, out_shape and per-input shape/stride
        shader.push_str("struct Params {\n    len: u32,\n    offset: u32,\n    rank: u32,\n    _pad: u32,\n    out_shape: PackedArray,\n");
        for idx in 0..self.inputs.len() {
            shader.push_str(&format!("    in{}_shape: PackedArray,\n", idx));
            shader.push_str(&format!("    in{}_stride: PackedArray,\n", idx));
        }
        shader.push_str("}\n\n");
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
        shader.push_str("@compute @workgroup_size(@WG@)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
        shader.push_str("    let idx = gid.x;\n    if (idx >= params.len) { return; }\n");
        shader.push_str("    let g = idx + params.offset;\n");
        shader.push_str("    // Compute N-D coordinates from global index (with chunk offset)\n    var coord: array<u32, MAX_RANK>;\n    var tmp: u32 = g;\n    var d: u32 = 0u;\n    loop { if d >= params.rank { break; } let dim = params.out_shape[d].value; if dim == 0u { coord[d] = 0u; } else { coord[d] = tmp % dim; tmp = tmp / dim; } d = d + 1u; }\n");
        // Compute broadcasted indices per input
        for (idx, _) in self.inputs.iter().enumerate() {
            shader.push_str(&format!(
                "    var i{}: u32 = 0u; d = 0u; loop {{ if d >= params.rank {{ break; }} let sd = params.in{}_shape[d].value; let st = params.in{}_stride[d].value; let c = select(coord[d], 0u, sd == 1u); i{} = i{} + c * st; d = d + 1u; }}\n",
                idx, idx, idx, idx, idx
            ));
        }
        shader.push_str(&body);
        shader.push_str(&format!("    output.data[g] = {final_expr};\n}}\n"));
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

        // Build reduction operand expression by folding the producer chain
        let data_vid = self.reduction_data?;
        let ext_input = self.inputs[0];
        let mut exprs: HashMap<ValueId, String> = HashMap::new();
        exprs.insert(ext_input, "v".to_string());
        for (vid, val) in &self.const_values {
            let lit = match val {
                Value::Num(n) => {
                    if scalar_ty == "f64" {
                        format!("f64({})", n)
                    } else {
                        format!("{}", *n as f32)
                    }
                }
                Value::Int(i) => {
                    let f = i.to_f64();
                    if scalar_ty == "f64" {
                        format!("f64({})", f)
                    } else {
                        format!("{}", f as f32)
                    }
                }
                _ => {
                    if scalar_ty == "f64" {
                        "f64(0.0)".to_string()
                    } else {
                        "0.0".to_string()
                    }
                }
            };
            exprs.insert(*vid, lit);
        }
        for op in &self.operations {
            match op {
                FusionOp::Primitive { op, inputs, output } => {
                    if let Some(out) = output {
                        if let Some(code) = primitive_expr(*op, inputs, &exprs) {
                            exprs.insert(*out, code);
                        }
                    }
                }
                FusionOp::Builtin {
                    name,
                    inputs,
                    output,
                } => {
                    if let Some(out) = output {
                        if let Some(code) = builtin_expr(name, inputs, &exprs, scalar_ty) {
                            exprs.insert(*out, code);
                        }
                    }
                }
            }
        }
        let val_expr = match exprs.get(&data_vid) {
            Some(s) => s.clone(),
            None => return None,
        };

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
        shader.push_str(&format!(
            "var<workgroup> tile: array<{scalar_ty}, @WG@u>;\n\n"
        ));
        shader.push_str(&format!(
            "const OMITNAN: bool = {};\n\n",
            if omitnan { "true" } else { "false" }
        ));
        let is_mean = matches!(self.reduction_mode, Some(ReductionMode::Mean));
        let post_scale = if is_mean {
            if axis == 0 {
                if scalar_ty == "f64" {
                    "(1.0 / f64(f32(params.nrows)))".to_string()
                } else {
                    "(1.0 / f32(params.nrows))".to_string()
                }
            } else {
                if scalar_ty == "f64" {
                    "(1.0 / f64(f32(params.ncols)))".to_string()
                } else {
                    "(1.0 / f32(params.ncols))".to_string()
                }
            }
        } else {
            if scalar_ty == "f64" {
                "f64(1.0)".to_string()
            } else {
                "1.0".to_string()
            }
        };
        // Helper(s) at module scope
        shader.push_str(&format!(
            "fn isNanF(x: {scalar}) -> bool {{ return x != x; }}\n\n",
            scalar = scalar_ty
        ));
        shader.push_str("@compute @workgroup_size(@WG@)\n");
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
            // helpers are declared at module scope
            shader.push_str("  var saw_nan: bool = false;\n  var r = lid.x;\n");
            shader.push_str(&format!(
                "  while (r < params.nrows) {{\n    let v = input0.data[ (col * params.ld) + r ];\n    let val: {scalar} = {val};\n    if (OMITNAN) {{ if (!isNanF(val)) {{ acc = acc + val; }} }} else {{ if (isNanF(val)) {{ saw_nan = true; }} else {{ acc = acc + val; }} }}\n    r += @WG@u;\n  }}\n",
                scalar = scalar_ty,
                val = val_expr
            ));
            if scalar_ty == "f64" {
                shader.push_str(
                    "  if (!OMITNAN && saw_nan) { acc = bitcast<f64>(0x7ff8000000000000u); }\n",
                );
            } else {
                shader
                    .push_str("  if (!OMITNAN && saw_nan) { acc = bitcast<f32>(0x7fc00000u); }\n");
            }
            shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
            shader.push_str(
                "  var off = (@WG@u) / 2u;\n  loop { if (off == 0u) { break; } if (lid.x < off) {\n    let a = tile[lid.x]; let b = tile[lid.x + off];\n    tile[lid.x] = a + b;\n  } workgroupBarrier(); off = off / 2u; }\n",
            );
            // Final write: apply post-scale (sum=1, mean=1/rows)
            shader.push_str(&format!(
                "  if (lid.x == 0u) {{ output.data[col] = tile[0u] * {}; }}\n}}\n",
                post_scale
            ));
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
            // helpers are declared at module scope
            shader.push_str("  var saw_nan: bool = false;\n  var c = lid.x;\n");
            shader.push_str(&format!(
                "  while (c < params.ncols) {{\n    let v = input0.data[ row + (c * params.ld) ];\n    let val: {scalar} = {val};\n    if (OMITNAN) {{ if (!isNanF(val)) {{ acc = acc + val; }} }} else {{ if (isNanF(val)) {{ saw_nan = true; }} else {{ acc = acc + val; }} }}\n    c += @WG@u;\n  }}\n",
                scalar = scalar_ty,
                val = val_expr
            ));
            if scalar_ty == "f64" {
                shader.push_str(
                    "  if (!OMITNAN && saw_nan) { acc = bitcast<f64>(0x7ff8000000000000u); }\n",
                );
            } else {
                shader
                    .push_str("  if (!OMITNAN && saw_nan) { acc = bitcast<f32>(0x7fc00000u); }\n");
            }
            shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
            shader.push_str(
                "  var off = (@WG@u) / 2u;\n  loop { if (off == 0u) { break; } if (lid.x < off) {\n    let a = tile[lid.x]; let b = tile[lid.x + off];\n    tile[lid.x] = a + b;\n  } workgroupBarrier(); off = off / 2u; }\n",
            );
            shader.push_str(&format!(
                "  if (lid.x == 0u) {{ output.data[row] = tile[0u] * {}; }}\n}}\n",
                post_scale
            ));
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

fn detect_centered_gram(
    graph: &AccelGraph,
    assigned: &mut HashSet<NodeId>,
    groups: &mut Vec<FusionGroup>,
    next_group_id: &mut usize,
) {
    for div_node in &graph.nodes {
        if assigned.contains(&div_node.id) {
            continue;
        }
        let div_op = match div_node.label {
            AccelNodeLabel::Primitive(op) => op,
            _ => continue,
        };
        if div_op != PrimitiveOp::Div && div_op != PrimitiveOp::ElemDiv {
            continue;
        }
        if div_node.inputs.len() != 2 {
            continue;
        }
        let (numerator_id, denom_id) = (div_node.inputs[0], div_node.inputs[1]);
        let denom_info = match graph.value(denom_id) {
            Some(info) => info,
            None => continue,
        };
        let denom_const = match &denom_info.constant {
            Some(Value::Num(v)) => Some(*v),
            Some(Value::Int(i)) => Some(i.to_f64()),
            _ => None,
        };
        if denom_const.is_some_and(|v| v == 0.0) {
            continue;
        }

        let mul_node_id = match graph
            .value(numerator_id)
            .and_then(|info| match &info.origin {
                ValueOrigin::NodeOutput { node, .. } => Some(*node),
                _ => None,
            }) {
            Some(id) => id,
            None => continue,
        };
        if assigned.contains(&mul_node_id) {
            continue;
        }
        let mul_node = match graph.node(mul_node_id) {
            Some(node) => node,
            None => continue,
        };
        let mul_op = match mul_node.label {
            AccelNodeLabel::Primitive(op) => op,
            _ => continue,
        };
        if mul_op != PrimitiveOp::Mul && mul_op != PrimitiveOp::ElemMul {
            continue;
        }
        if mul_node.inputs.len() != 2 {
            continue;
        }

        let mut transpose_node_id: Option<NodeId> = None;
        let mut centered_val_id: Option<ValueId> = None;
        for input_vid in &mul_node.inputs {
            let candidate_node_id =
                match graph.value(*input_vid).and_then(|info| match &info.origin {
                    ValueOrigin::NodeOutput { node, .. } => Some(*node),
                    _ => None,
                }) {
                    Some(id) => id,
                    None => continue,
                };
            if let Some(trans_node) = graph.node(candidate_node_id) {
                if matches!(
                    trans_node.label,
                    AccelNodeLabel::Primitive(PrimitiveOp::Transpose)
                ) {
                    if let Some(centered) = trans_node.inputs.get(0).copied() {
                        transpose_node_id = Some(candidate_node_id);
                        centered_val_id = Some(centered);
                        break;
                    }
                }
            }
        }

        let transpose_node_id = match transpose_node_id {
            Some(id) if !assigned.contains(&id) => id,
            _ => continue,
        };
        let centered_val_id = match centered_val_id {
            Some(id) => id,
            None => continue,
        };

        if assigned.contains(&transpose_node_id) {
            continue;
        }
        if graph.node(transpose_node_id).is_none() {
            continue;
        }

        let centered_node_id =
            match graph
                .value(centered_val_id)
                .and_then(|info| match &info.origin {
                    ValueOrigin::NodeOutput { node, .. } => Some(*node),
                    _ => None,
                }) {
                Some(id) => id,
                None => continue,
            };
        if assigned.contains(&centered_node_id) {
            continue;
        }
        let centered_node = match graph.node(centered_node_id) {
            Some(node) => node,
            None => continue,
        };
        if !matches!(
            centered_node.label,
            AccelNodeLabel::Primitive(PrimitiveOp::Sub)
        ) {
            continue;
        }
        if centered_node.inputs.len() != 2 {
            continue;
        }
        let matrix_val_id = centered_node.inputs[0];
        let mean_val_id = centered_node.inputs[1];

        let mean_node_id = match graph
            .value(mean_val_id)
            .and_then(|info| match &info.origin {
                ValueOrigin::NodeOutput { node, .. } => Some(*node),
                _ => None,
            }) {
            Some(id) => id,
            None => continue,
        };
        if assigned.contains(&mean_node_id) {
            continue;
        }
        let mean_node = match graph.node(mean_node_id) {
            Some(node) => node,
            None => continue,
        };
        match &mean_node.label {
            AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mean") => {}
            _ => continue,
        }
        if mean_node.inputs.is_empty() || mean_node.inputs[0] != matrix_val_id {
            continue;
        }

        let matrix_info = match graph.value(matrix_val_id) {
            Some(info) => info,
            None => continue,
        };
        let matrix_rows = match &matrix_info.shape {
            ShapeInfo::Tensor(dims) if !dims.is_empty() => dims[0].unwrap_or(0),
            _ => 0,
        };
        let normalization = if matrix_rows > 1 {
            if let Some(value) = denom_const {
                let unbiased = (matrix_rows as f64 - 1.0).max(1.0);
                let biased = matrix_rows as f64;
                if approx_eq(value, unbiased) {
                    CovNormalization::Unbiased
                } else if approx_eq(value, biased) {
                    CovNormalization::Biased
                } else {
                    CovNormalization::Unbiased
                }
            } else {
                CovNormalization::Unbiased
            }
        } else {
            CovNormalization::Unbiased
        };

        let mut nodes = vec![
            mean_node_id,
            centered_node_id,
            transpose_node_id,
            mul_node_id,
            div_node.id,
        ];
        nodes.sort_by_key(|node_id| {
            graph
                .node(*node_id)
                .map(|node| node.span.start)
                .unwrap_or(usize::MAX)
        });
        let span = group_span(graph, &nodes);
        let shape = node_output_shape(graph, div_node);

        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::CenteredGram,
            nodes: nodes.clone(),
            shape,
            span,
            pattern: Some(FusionPattern::CenteredGram {
                matrix: matrix_val_id,
                normalization,
            }),
        });
        *next_group_id += 1;
        for id in nodes {
            assigned.insert(id);
        }
    }
}

fn sub_uses_input_mu(node: &AccelNode, input_vid: ValueId, mu_vid: ValueId) -> bool {
    if node.inputs.len() != 2 {
        return false;
    }
    let lhs = node.inputs[0];
    let rhs = node.inputs[1];
    (lhs == input_vid && rhs == mu_vid) || (lhs == mu_vid && rhs == input_vid)
}

fn analyze_mu_chain(
    graph: &AccelGraph,
    mu_vid: ValueId,
    input_vid: ValueId,
) -> Option<(NodeId, NodeId)> {
    let (mean2_id, mean2_node) = node_from_value(graph, mu_vid)?;
    if !is_mean_with_dim(mean2_node, graph, 3.0) {
        return None;
    }
    let inner_vid = *mean2_node.inputs.get(0)?;
    let (mean1_id, mean1_node) = node_from_value(graph, inner_vid)?;
    if !is_mean_with_dim(mean1_node, graph, 2.0) {
        return None;
    }
    let source_vid = *mean1_node.inputs.get(0)?;
    if source_vid != input_vid {
        return None;
    }
    Some((mean1_id, mean2_id))
}

struct VarianceChainInfo {
    mean1: NodeId,
    mean2: NodeId,
    pow: NodeId,
    sub: NodeId,
}

fn analyze_variance_chain(
    graph: &AccelGraph,
    variance_vid: ValueId,
    input_vid: ValueId,
    mu_vid: ValueId,
) -> Option<VarianceChainInfo> {
    let (mean2_id, mean2_node) = node_from_value(graph, variance_vid)?;
    if !is_mean_with_dim(mean2_node, graph, 3.0) {
        return None;
    }
    let mean1_vid = *mean2_node.inputs.get(0)?;
    let (mean1_id, mean1_node) = node_from_value(graph, mean1_vid)?;
    if !is_mean_with_dim(mean1_node, graph, 2.0) {
        return None;
    }
    let pow_vid = *mean1_node.inputs.get(0)?;
    let (pow_id, pow_node) = node_from_value(graph, pow_vid)?;
    let pow_op = match pow_node.label {
        AccelNodeLabel::Primitive(op) => op,
        _ => return None,
    };
    if pow_op != PrimitiveOp::ElemPow && pow_op != PrimitiveOp::Pow {
        return None;
    }
    if pow_node.inputs.len() != 2 {
        return None;
    }
    let pow_base_vid = pow_node.inputs[0];
    let pow_exp_vid = pow_node.inputs[1];
    let exponent = resolve_scalar_constant(graph, pow_exp_vid)?;
    if !approx_eq(exponent, 2.0) {
        return None;
    }
    let (sub_id, sub_node) = node_from_value(graph, pow_base_vid)?;
    match sub_node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Sub) => {}
        _ => return None,
    }
    if !sub_uses_input_mu(sub_node, input_vid, mu_vid) {
        return None;
    }
    Some(VarianceChainInfo {
        mean1: mean1_id,
        mean2: mean2_id,
        pow: pow_id,
        sub: sub_id,
    })
}

fn detect_image_normalize(
    graph: &AccelGraph,
    assigned: &mut HashSet<NodeId>,
    groups: &mut Vec<FusionGroup>,
    next_group_id: &mut usize,
) {
    'outer: for pow_node in &graph.nodes {
        if assigned.contains(&pow_node.id) {
            continue;
        }
        let pow_op = match pow_node.label {
            AccelNodeLabel::Primitive(op) => op,
            _ => continue,
        };
        if pow_op != PrimitiveOp::ElemPow && pow_op != PrimitiveOp::Pow {
            continue;
        }
        if pow_node.inputs.len() != 2 {
            continue;
        }
        let base_vid = pow_node.inputs[0];
        let exponent_vid = pow_node.inputs[1];
        let (max_id, max_node) = match node_from_value(graph, base_vid) {
            Some(pair) => pair,
            None => continue,
        };
        if assigned.contains(&max_id) {
            continue;
        }
        match &max_node.label {
            AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("max") => {}
            _ => continue,
        }
        if max_node.inputs.len() != 2 {
            continue;
        }
        let max_left = max_node.inputs[0];
        let max_right = max_node.inputs[1];
        let left_zero = resolve_scalar_constant(graph, max_left)
            .map(|v| approx_eq(v, 0.0))
            .unwrap_or(false);
        let right_zero = resolve_scalar_constant(graph, max_right)
            .map(|v| approx_eq(v, 0.0))
            .unwrap_or(false);
        let (clamp_vid, _zero_vid) = if left_zero && !right_zero {
            (max_right, max_left)
        } else if right_zero && !left_zero {
            (max_left, max_right)
        } else {
            continue;
        };

        let (add_id, add_node) = match node_from_value(graph, clamp_vid) {
            Some(pair) => pair,
            None => continue,
        };
        if assigned.contains(&add_id) {
            continue;
        }
        match add_node.label {
            AccelNodeLabel::Primitive(PrimitiveOp::Add) => {}
            _ => continue,
        }
        if add_node.inputs.len() != 2 {
            continue;
        }

        // Determine normalization branch and optional gain/bias inputs.
        let left_info = node_from_value(graph, add_node.inputs[0]);
        let right_info = node_from_value(graph, add_node.inputs[1]);

        let mut nodes: Vec<NodeId> = vec![pow_node.id, max_id, add_id];

        let mut div_id: Option<NodeId> = None;
        let mut mul_id_opt: Option<NodeId> = None;
        let mut gain_scalar: Option<ImageScalar> = None;
        let mut bias_vid: Option<ValueId> = None;

        let mut assign_mul_branch = |mul_id: NodeId, mul_node: &AccelNode, other_vid: ValueId| {
            if assigned.contains(&mul_id) {
                return false;
            }
            if mul_node.inputs.len() != 2 {
                return false;
            }
            let mut local_div: Option<(NodeId, ValueId)> = None;
            let mut gain_vid: Option<ValueId> = None;
            for &inp in &mul_node.inputs {
                if let Some((child_id, child_node)) = node_from_value(graph, inp) {
                    match child_node.label {
                        AccelNodeLabel::Primitive(PrimitiveOp::Div)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemLeftDiv) => {
                            local_div = Some((child_id, inp));
                        }
                        _ => {
                            gain_vid = Some(inp);
                        }
                    }
                } else {
                    gain_vid = Some(inp);
                }
            }
            let (div_node_id, _div_vid_local) = match local_div {
                Some(pair) => pair,
                None => return false,
            };
            div_id = Some(div_node_id);
            mul_id_opt = Some(mul_id);
            gain_scalar = gain_vid.map(|vid| image_scalar_from_value(graph, vid));
            bias_vid = Some(other_vid);
            nodes.push(mul_id);
            true
        };

        let mut handled_branch = false;
        if let Some((mul_id, mul_node)) = left_info.as_ref().filter(|(_, node)| {
            matches!(
                node.label,
                AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                    | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul)
            )
        }) {
            let other = add_node.inputs[1];
            handled_branch = assign_mul_branch(*mul_id, mul_node, other);
        }

        if !handled_branch {
            if let Some((mul_id, mul_node)) = right_info.as_ref().filter(|(_, node)| {
                matches!(
                    node.label,
                    AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul)
                )
            }) {
                let other = add_node.inputs[0];
                handled_branch = assign_mul_branch(*mul_id, mul_node, other);
            }
        }

        if !handled_branch {
            let mut direct_candidate: Option<(NodeId, ValueId, ValueId)> = None;
            if let Some((node_id, node)) = left_info {
                if matches!(
                    node.label,
                    AccelNodeLabel::Primitive(PrimitiveOp::Div)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemLeftDiv)
                ) && !assigned.contains(&node_id)
                {
                    direct_candidate = Some((node_id, add_node.inputs[0], add_node.inputs[1]));
                }
            }
            if direct_candidate.is_none() {
                if let Some((node_id, node)) = right_info {
                    if matches!(
                        node.label,
                        AccelNodeLabel::Primitive(PrimitiveOp::Div)
                            | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
                            | AccelNodeLabel::Primitive(PrimitiveOp::ElemLeftDiv)
                    ) && !assigned.contains(&node_id)
                    {
                        direct_candidate = Some((node_id, add_node.inputs[1], add_node.inputs[0]));
                    }
                }
            }

            let (div_node_id_local, _div_vid_local, bias_vid_local) = match direct_candidate {
                Some(tuple) => tuple,
                None => continue 'outer,
            };
            div_id = Some(div_node_id_local);
            bias_vid = Some(bias_vid_local);
            nodes.push(div_node_id_local);
        }

        let div_node_id = match div_id {
            Some(id) => id,
            None => continue 'outer,
        };
        if assigned.contains(&div_node_id) {
            continue;
        }
        let div_node = match graph.node(div_node_id) {
            Some(node) => node,
            None => continue,
        };
        if div_node.inputs.len() != 2 {
            continue;
        }
        let (sqrt_id, sqrt_input_vid, numerator_vid) =
            if let Some(pair) = is_sqrt_node(graph, div_node.inputs[1]) {
                (pair.0, pair.1, div_node.inputs[0])
            } else if let Some(pair) = is_sqrt_node(graph, div_node.inputs[0]) {
                (pair.0, pair.1, div_node.inputs[1])
            } else {
                continue;
            };
        if assigned.contains(&sqrt_id) {
            continue;
        }

        // Numerator should be subtraction of input and mean.
        let (sub_node_id, sub_node) = match node_from_value(graph, numerator_vid) {
            Some(pair) => pair,
            None => continue,
        };
        if assigned.contains(&sub_node_id) {
            continue;
        }
        match sub_node.label {
            AccelNodeLabel::Primitive(PrimitiveOp::Sub)
            | AccelNodeLabel::Primitive(PrimitiveOp::Add) => {}
            _ => continue,
        }

        // Extract input tensor and mu value ids.
        let mut input_vid_opt: Option<ValueId> = None;
        let mut mu_vid_opt: Option<ValueId> = None;
        for &inp in &sub_node.inputs {
            if let Some((_child_id, child_node)) = node_from_value(graph, inp) {
                if matches!(child_node.label, AccelNodeLabel::Builtin { ref name } if name.eq_ignore_ascii_case("mean"))
                {
                    mu_vid_opt = Some(inp);
                } else if input_vid_opt.is_none() {
                    input_vid_opt = Some(inp);
                }
            } else if input_vid_opt.is_none() {
                input_vid_opt = Some(inp);
            }
        }
        let input_vid = match input_vid_opt {
            Some(vid) => vid,
            None => continue,
        };
        let mu_vid = match mu_vid_opt {
            Some(vid) => vid,
            None => continue,
        };

        if !sub_uses_input_mu(sub_node, input_vid, mu_vid) {
            continue;
        }

        let (mu_mean1_id, mu_mean2_id) = match analyze_mu_chain(graph, mu_vid, input_vid) {
            Some(pair) => pair,
            None => continue,
        };

        let (epsilon_add_id, epsilon_add_node) = match node_from_value(graph, sqrt_input_vid) {
            Some(pair) => pair,
            None => continue,
        };
        if assigned.contains(&epsilon_add_id) {
            continue;
        }
        if epsilon_add_node.inputs.len() != 2 {
            continue;
        }
        let (variance_vid, epsilon_vid) = {
            let left = epsilon_add_node.inputs[0];
            let right = epsilon_add_node.inputs[1];
            let left_mean = node_from_value(graph, left)
                .map(|(_, node)| matches!(node.label, AccelNodeLabel::Builtin { ref name } if name.eq_ignore_ascii_case("mean")))
                .unwrap_or(false);
            let right_mean = node_from_value(graph, right)
                .map(|(_, node)| matches!(node.label, AccelNodeLabel::Builtin { ref name } if name.eq_ignore_ascii_case("mean")))
                .unwrap_or(false);
            if left_mean && !right_mean {
                (left, right)
            } else if right_mean && !left_mean {
                (right, left)
            } else {
                continue 'outer;
            }
        };

        let variance_info = match analyze_variance_chain(graph, variance_vid, input_vid, mu_vid) {
            Some(info) => info,
            None => continue,
        };

        // Ensure numerator subtraction matches the variance subtraction as well.
        let variance_sub_node = graph.node(variance_info.sub).expect("variance sub node");
        if !sub_uses_input_mu(variance_sub_node, input_vid, mu_vid) {
            continue;
        }

        // Collect nodes and ensure no conflicts.
        nodes.extend([
            div_node_id,
            sub_node_id,
            sqrt_id,
            epsilon_add_id,
            variance_info.mean1,
            variance_info.mean2,
            variance_info.pow,
            variance_info.sub,
            mu_mean1_id,
            mu_mean2_id,
        ]);

        if let Some(mul_id) = mul_id_opt {
            if assigned.contains(&mul_id) {
                continue;
            }
        }

        if nodes.iter().any(|id| assigned.contains(id)) {
            continue;
        }

        nodes.sort();
        nodes.dedup();

        let span = group_span(graph, &nodes);
        let shape = node_output_shape(graph, pow_node);

        let pattern = ImageNormalizePattern {
            input: input_vid,
            epsilon: image_scalar_from_value(graph, epsilon_vid),
            gain: gain_scalar,
            bias: bias_vid.map(|vid| image_scalar_from_value(graph, vid)),
            gamma: image_scalar_from_value(graph, exponent_vid),
        };

        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::ImageNormalize,
            nodes: nodes.clone(),
            shape,
            span,
            pattern: Some(FusionPattern::ImageNormalize(pattern)),
        });
        *next_group_id += 1;
        for id in nodes {
            assigned.insert(id);
        }
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= scale * 1e-6
}

fn detect_power_step_normalize(
    graph: &AccelGraph,
    assigned: &mut HashSet<NodeId>,
    groups: &mut Vec<FusionGroup>,
    next_group_id: &mut usize,
) {
    'outer: for div_node in &graph.nodes {
        if assigned.contains(&div_node.id) {
            continue;
        }
        let div_op = match div_node.label {
            AccelNodeLabel::Primitive(op) => op,
            _ => continue,
        };
        if div_op != PrimitiveOp::Div && div_op != PrimitiveOp::ElemDiv {
            continue;
        }
        if div_node.inputs.len() != 2 {
            continue;
        }
        let numerator_vid = div_node.inputs[0];
        let denom_vid = div_node.inputs[1];

        let (matmul_id, matmul_node) = match node_from_value(graph, numerator_vid) {
            Some((id, node)) => (id, node),
            None => continue,
        };
        if assigned.contains(&matmul_id) {
            continue;
        }
        match &matmul_node.label {
            AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mtimes") => {}
            _ => continue,
        }
        if matmul_node.inputs.len() != 2 {
            continue;
        }

        let Some(denom_info) = analyze_power_step_denominator(graph, denom_vid, numerator_vid)
        else {
            continue;
        };
        if assigned.contains(&denom_info.sqrt_node) {
            continue;
        }
        if assigned.contains(&denom_info.sum_node) {
            continue;
        }
        if assigned.contains(&denom_info.pow_node) {
            continue;
        }
        if let Some(add_id) = denom_info.add_node {
            if assigned.contains(&add_id) {
                continue;
            }
        }
        if denom_info.pow_input != numerator_vid {
            continue;
        }

        let mut nodes = vec![matmul_id, denom_info.pow_node, denom_info.sum_node];
        if let Some(add_id) = denom_info.add_node {
            nodes.push(add_id);
        }
        nodes.push(denom_info.sqrt_node);
        nodes.push(div_node.id);

        for node_id in &nodes {
            if assigned.contains(node_id) {
                continue 'outer;
            }
        }

        nodes.sort_by_key(|node_id| {
            graph
                .node(*node_id)
                .map(|node| node.span.start)
                .unwrap_or(usize::MAX)
        });

        let span = group_span(graph, &nodes);
        let shape = node_output_shape(graph, div_node);

        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::PowerStepNormalize,
            nodes: nodes.clone(),
            shape,
            span,
            pattern: Some(FusionPattern::PowerStepNormalize {
                lhs: matmul_node.inputs[0],
                rhs: matmul_node.inputs[1],
                epsilon: denom_info.epsilon,
            }),
        });
        *next_group_id += 1;
        for id in nodes {
            assigned.insert(id);
        }
    }
}

fn detect_explained_variance(
    graph: &AccelGraph,
    assigned: &mut HashSet<NodeId>,
    groups: &mut Vec<FusionGroup>,
    next_group_id: &mut usize,
) {
    for diag_node in &graph.nodes {
        if assigned.contains(&diag_node.id) {
            continue;
        }
        match &diag_node.label {
            AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("diag") => {}
            _ => continue,
        }
        if diag_node.inputs.len() != 1 {
            continue;
        }
        let matmul2_vid = diag_node.inputs[0];
        let (matmul2_id, matmul2_node) = match node_from_value(graph, matmul2_vid) {
            Some(pair) => pair,
            None => continue,
        };
        if assigned.contains(&matmul2_id) {
            continue;
        }
        match &matmul2_node.label {
            AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mtimes") => {}
            _ => continue,
        }
        if matmul2_node.inputs.len() != 2 {
            continue;
        }

        let (matmul1_id, matmul1_node, q_vid) = if let Some((mm_id, mm_node)) =
            node_from_value(graph, matmul2_node.inputs[0])
        {
            if matches!(mm_node.label, AccelNodeLabel::Builtin { ref name } if name.eq_ignore_ascii_case("mtimes"))
            {
                (mm_id, mm_node, matmul2_node.inputs[1])
            } else {
                continue;
            }
        } else if let Some((mm_id, mm_node)) = node_from_value(graph, matmul2_node.inputs[1]) {
            if matches!(mm_node.label, AccelNodeLabel::Builtin { ref name } if name.eq_ignore_ascii_case("mtimes"))
            {
                (mm_id, mm_node, matmul2_node.inputs[0])
            } else {
                continue;
            }
        } else {
            continue;
        };

        if assigned.contains(&matmul1_id) {
            continue;
        }

        if matmul1_node.inputs.len() != 2 {
            continue;
        }

        let (transpose_id, transpose_input_vid, g_vid) =
            if let Some((t_id, src_vid)) = is_transpose_node(graph, matmul1_node.inputs[0]) {
                (t_id, src_vid, matmul1_node.inputs[1])
            } else if let Some((t_id, src_vid)) = is_transpose_node(graph, matmul1_node.inputs[1]) {
                (t_id, src_vid, matmul1_node.inputs[0])
            } else {
                continue;
            };

        if assigned.contains(&transpose_id) {
            continue;
        }

        if transpose_input_vid != q_vid {
            continue;
        }

        let mut nodes = vec![diag_node.id, matmul2_id, matmul1_id, transpose_id];
        nodes.sort_by_key(|node_id| {
            graph
                .node(*node_id)
                .map(|node| node.span.start)
                .unwrap_or(usize::MAX)
        });
        let span = group_span(graph, &nodes);
        let shape = node_output_shape(graph, diag_node);
        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::ExplainedVariance,
            nodes: nodes.clone(),
            shape,
            span,
            pattern: Some(FusionPattern::ExplainedVariance { q: q_vid, g: g_vid }),
        });
        *next_group_id += 1;
        for id in nodes {
            assigned.insert(id);
        }
    }
}

struct PowerStepDenominatorInfo {
    sqrt_node: NodeId,
    add_node: Option<NodeId>,
    sum_node: NodeId,
    pow_node: NodeId,
    pow_input: ValueId,
    epsilon: f64,
}

fn analyze_power_step_denominator(
    graph: &AccelGraph,
    denom_vid: ValueId,
    expected_source_vid: ValueId,
) -> Option<PowerStepDenominatorInfo> {
    let (sqrt_node_id, sqrt_input_vid, add_node_opt, epsilon_from_outer) =
        if let Some((sqrt_id, sqrt_in)) = is_sqrt_node(graph, denom_vid) {
            if let Some((add_node, sum_vid, epsilon_inner)) =
                extract_add_with_constant(graph, sqrt_in)
            {
                (sqrt_id, sum_vid, Some(add_node), epsilon_inner)
            } else {
                (sqrt_id, sqrt_in, None, 0.0)
            }
        } else if let Some((add_node, other_vid, epsilon_inner)) =
            extract_add_with_constant(graph, denom_vid)
        {
            let (sqrt_id, sqrt_in) = is_sqrt_node(graph, other_vid)?;
            (sqrt_id, sqrt_in, Some(add_node), epsilon_inner)
        } else {
            return None;
        };

    let (sum_node_id, sum_node) = node_from_value(graph, sqrt_input_vid)?;
    match &sum_node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("sum") => {}
        _ => return None,
    }
    if sum_node.inputs.is_empty() {
        return None;
    }
    let pow_vid = sum_node.inputs[0];
    let (pow_node_id, pow_node) = node_from_value(graph, pow_vid)?;
    let pow_input = match pow_node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::ElemPow) => {
            if pow_node.inputs.len() != 2 {
                return None;
            }
            let base = pow_node.inputs[0];
            let exponent_vid = pow_node.inputs[1];
            let exponent = value_constant_f64(graph, exponent_vid)?;
            if !approx_eq(exponent, 2.0) {
                return None;
            }
            base
        }
        _ => return None,
    };

    if pow_input != expected_source_vid {
        return None;
    }

    let epsilon = epsilon_from_outer;
    Some(PowerStepDenominatorInfo {
        sqrt_node: sqrt_node_id,
        add_node: add_node_opt,
        sum_node: sum_node_id,
        pow_node: pow_node_id,
        pow_input,
        epsilon,
    })
}

fn node_from_value<'a>(graph: &'a AccelGraph, vid: ValueId) -> Option<(NodeId, &'a AccelNode)> {
    let info = graph.value(vid)?;
    match info.origin {
        ValueOrigin::NodeOutput { node, .. } => graph.node(node).map(|n| (node, n)),
        _ => None,
    }
}

fn is_sqrt_node(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("sqrt") => {
            let input = node.inputs.get(0).copied()?;
            Some((node_id, input))
        }
        _ => None,
    }
}

fn is_transpose_node(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Transpose) => {
            let input = node.inputs.get(0).copied()?;
            Some((node_id, input))
        }
        _ => None,
    }
}

fn extract_add_with_constant(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId, f64)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Add) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(eps) = value_constant_f64(graph, rhs) {
                return Some((node_id, lhs, eps));
            }
            if let Some(eps) = value_constant_f64(graph, lhs) {
                return Some((node_id, rhs, eps));
            }
            None
        }
        AccelNodeLabel::Primitive(PrimitiveOp::Sub) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(eps) = value_constant_f64(graph, rhs) {
                return Some((node_id, lhs, -eps));
            }
            if let Some(eps) = value_constant_f64(graph, lhs) {
                return Some((node_id, rhs, eps));
            }
            None
        }
        _ => None,
    }
}

struct ConstantTrace {
    value: f64,
    nodes: Vec<NodeId>,
}

fn collect_scalar_constant(graph: &AccelGraph, vid: ValueId) -> Option<ConstantTrace> {
    let mut current = vid;
    let mut nodes: Vec<NodeId> = Vec::new();
    let mut sign = 1.0f64;
    let mut visited: HashSet<NodeId> = HashSet::new();

    loop {
        let info = graph.value(current)?;
        match &info.origin {
            ValueOrigin::Constant => {
                let base = value_info_scalar(info)?;
                return Some(ConstantTrace {
                    value: sign * base,
                    nodes,
                });
            }
            ValueOrigin::NodeOutput { node, .. } => {
                if !visited.insert(*node) {
                    return None;
                }
                let node_ref = graph.node(*node)?;
                match &node_ref.label {
                    AccelNodeLabel::Builtin { name }
                        if name.eq_ignore_ascii_case("single")
                            || name.eq_ignore_ascii_case("double") =>
                    {
                        if node_ref.inputs.len() != 1 {
                            return None;
                        }
                        nodes.push(*node);
                        current = node_ref.inputs[0];
                    }
                    AccelNodeLabel::Primitive(PrimitiveOp::Neg) => {
                        if node_ref.inputs.len() != 1 {
                            return None;
                        }
                        nodes.push(*node);
                        sign = -sign;
                        current = node_ref.inputs[0];
                    }
                    AccelNodeLabel::Primitive(PrimitiveOp::UPlus) => {
                        if node_ref.inputs.len() != 1 {
                            return None;
                        }
                        nodes.push(*node);
                        current = node_ref.inputs[0];
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

fn resolve_scalar_constant(graph: &AccelGraph, vid: ValueId) -> Option<f64> {
    collect_scalar_constant(graph, vid).map(|trace| trace.value)
}

fn image_scalar_from_value(graph: &AccelGraph, vid: ValueId) -> ImageScalar {
    if let Some(value) = resolve_scalar_constant(graph, vid) {
        ImageScalar::Constant(value)
    } else {
        ImageScalar::Value(vid)
    }
}

fn value_info_scalar(info: &ValueInfo) -> Option<f64> {
    match &info.constant {
        Some(Value::Num(v)) => Some(*v),
        Some(Value::Int(i)) => Some(i.to_f64()),
        Some(Value::Tensor(t)) if t.data.len() == 1 => Some(t.data[0]),
        Some(Value::LogicalArray(arr)) if arr.data.len() == 1 => Some(arr.data[0] as f64),
        Some(Value::Bool(flag)) => Some(if *flag { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn is_mean_with_dim(node: &AccelNode, graph: &AccelGraph, expected_dim: f64) -> bool {
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mean") => {}
        _ => return false,
    }
    if node.inputs.len() < 2 {
        return false;
    }
    let dim_vid = node.inputs[1];
    resolve_scalar_constant(graph, dim_vid)
        .map(|dim| approx_eq(dim, expected_dim))
        .unwrap_or(false)
}

fn value_constant_f64(graph: &AccelGraph, vid: ValueId) -> Option<f64> {
    resolve_scalar_constant(graph, vid)
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
        "max" => return builtin_binary("max", inputs, exprs),
        "min" => return builtin_binary("min", inputs, exprs),
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
    use runmat_builtins::{Type, Value};
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
            var_bindings: StdHashMap::new(),
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
            constant: Some(Value::Num(1.0)),
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
            var_bindings: StdHashMap::new(),
        };

        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let group_plan = &plan.groups[0];
        assert_eq!(group_plan.inputs.len(), 2);
        assert!(group_plan.stack_pattern.is_empty());
        assert!(group_plan.constants.get(&1).is_some());
        assert!(group_plan.const_values.contains_key(&1));
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


fn split_add_with_constant(
    graph: &AccelGraph,
    vid: ValueId,
) -> Option<(NodeId, ValueId, ConstantTrace)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Add) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(trace) = collect_scalar_constant(graph, rhs) {
                return Some((node_id, lhs, trace));
            }
            if let Some(trace) = collect_scalar_constant(graph, lhs) {
                return Some((node_id, rhs, trace));
            }
            None
        }
        AccelNodeLabel::Primitive(PrimitiveOp::Sub) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(mut trace) = collect_scalar_constant(graph, rhs) {
                trace.value = -trace.value;
                return Some((node_id, lhs, trace));
            }
            None
        }
        _ => None,
    }
}

fn split_mul_with_constant(
    graph: &AccelGraph,
    vid: ValueId,
) -> Option<(NodeId, ValueId, ConstantTrace)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Mul)
        | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(trace) = collect_scalar_constant(graph, rhs) {
                return Some((node_id, lhs, trace));
            }
            if let Some(trace) = collect_scalar_constant(graph, lhs) {
                return Some((node_id, rhs, trace));
            }
            None
        }
        _ => None,
    }
}

fn split_max_with_constant_zero(
    graph: &AccelGraph,
    vid: ValueId,
) -> Option<(NodeId, ValueId, ConstantTrace)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("max") => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(trace) = collect_scalar_constant(graph, rhs) {
                if approx_eq(trace.value, 0.0) {
                    return Some((node_id, lhs, trace));
                }
            }
            if let Some(trace) = collect_scalar_constant(graph, lhs) {
                if approx_eq(trace.value, 0.0) {
                    return Some((node_id, rhs, trace));
                }
            }
            None
        }
        _ => None,
    }
}

fn match_mean_with_dim(
    graph: &AccelGraph,
    vid: ValueId,
    expected_dim: f64,
) -> Option<(NodeId, ValueId)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mean") => {}
        _ => return None,
    }
    if node.inputs.len() != 2 {
        return None;
    }
    let data_vid = node.inputs[0];
    let dim_vid = node.inputs[1];
    let dim = resolve_scalar_constant(graph, dim_vid)?;
    if !approx_eq(dim, expected_dim) {
        return None;
    }
    Some((node_id, data_vid))
}

struct ImageNormalizeMatch {
    nodes: Vec<NodeId>,
    input: ValueId,
    epsilon: f64,
    gain: Option<f64>,
    bias: Option<f64>,
    gamma: Option<f64>,
}

fn analyze_image_normalize(
    graph: &AccelGraph,
    pow_node_id: NodeId,
    assigned: &HashSet<NodeId>,
) -> Option<ImageNormalizeMatch> {
    let pow_node = graph.node(pow_node_id)?;
    if !matches!(pow_node.label, AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)) {
        return None;
    }
    if pow_node.inputs.len() != 2 || pow_node.outputs.len() != 1 {
        return None;
    }

    let mut nodes: Vec<NodeId> = vec![pow_node_id];

    let gamma_trace = collect_scalar_constant(graph, pow_node.inputs[1])?;
    if gamma_trace.nodes.iter().any(|id| assigned.contains(id)) {
        return None;
    }
    nodes.extend(gamma_trace.nodes.iter().copied());
    let gamma_opt = if approx_eq(gamma_trace.value, 1.0) {
        None
    } else {
        Some(gamma_trace.value)
    };

    let (clamp_node_id, pre_bias_vid, zero_trace) = split_max_with_constant_zero(graph, pow_node.inputs[0])?;
    if assigned.contains(&clamp_node_id) {
        return None;
    }
    if zero_trace.nodes.iter().any(|id| assigned.contains(id)) {
        return None;
    }
    nodes.push(clamp_node_id);
    nodes.extend(zero_trace.nodes.iter().copied());

    let (pre_gain_vid, bias_opt) = if let Some((add_node_id, base_vid, bias_trace)) =
        split_add_with_constant(graph, pre_bias_vid)
    {
        if assigned.contains(&add_node_id) {
            return None;
        }
        if bias_trace.nodes.iter().any(|id| assigned.contains(id)) {
            return None;
        }
        nodes.push(add_node_id);
        nodes.extend(bias_trace.nodes.iter().copied());
        let bias_value = bias_trace.value;
        let bias = if approx_eq(bias_value, 0.0) {
            None
        } else {
            Some(bias_value)
        };
        (base_vid, bias)
    } else {
        (pre_bias_vid, None)
    };

    let (norm_vid, gain_opt) = if let Some((mul_node_id, base_vid, gain_trace)) =
        split_mul_with_constant(graph, pre_gain_vid)
    {
        if assigned.contains(&mul_node_id) {
            return None;
        }
        if gain_trace.nodes.iter().any(|id| assigned.contains(id)) {
            return None;
        }
        nodes.push(mul_node_id);
        nodes.extend(gain_trace.nodes.iter().copied());
        let gain_value = gain_trace.value;
        let gain = if approx_eq(gain_value, 1.0) {
            None
        } else {
            Some(gain_value)
        };
        (base_vid, gain)
    } else {
        (pre_gain_vid, None)
    };

    let (div_node_id, div_node) = node_from_value(graph, norm_vid)?;
    if assigned.contains(&div_node_id) {
        return None;
    }
    match div_node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
        | AccelNodeLabel::Primitive(PrimitiveOp::Div) => {}
        _ => return None,
    }
    if div_node.inputs.len() != 2 {
        return None;
    }

    let diff_vid = div_node.inputs[0];
    let sigma_vid = div_node.inputs[1];
    let (sigma_node_id, sigma_input_vid) = match is_sqrt_node(graph, sigma_vid) {
        Some(pair) => pair,
        None => return None,
    };
    if assigned.contains(&sigma_node_id) {
        return None;
    }
    nodes.push(div_node_id);
    nodes.push(sigma_node_id);

    let (add_node_id, mean_sq_vid, epsilon_trace) =
        split_add_with_constant(graph, sigma_input_vid)?;
    if assigned.contains(&add_node_id) {
        return None;
    }
    if epsilon_trace.nodes.iter().any(|id| assigned.contains(id)) {
        return None;
    }
    nodes.push(add_node_id);
    nodes.extend(epsilon_trace.nodes.iter().copied());
    let epsilon = epsilon_trace.value;

    let (mean_sq_dim3_id, mean_sq_dim2_vid) = match_mean_with_dim(graph, mean_sq_vid, 3.0)?;
    if assigned.contains(&mean_sq_dim3_id) {
        return None;
    }
    let (mean_sq_dim2_id, squared_diff_vid) = match_mean_with_dim(graph, mean_sq_dim2_vid, 2.0)?;
    if assigned.contains(&mean_sq_dim2_id) {
        return None;
    }
    nodes.push(mean_sq_dim3_id);
    nodes.push(mean_sq_dim2_id);

    let (square_pow_node_id, square_pow_node) = node_from_value(graph, squared_diff_vid)?;
    if assigned.contains(&square_pow_node_id) {
        return None;
    }
    if !matches!(square_pow_node.label, AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)) {
        return None;
    }
    if square_pow_node.inputs.len() != 2 {
        return None;
    }
    let exponent_trace = collect_scalar_constant(graph, square_pow_node.inputs[1])?;
    if !approx_eq(exponent_trace.value, 2.0) {
        return None;
    }
    if exponent_trace.nodes.iter().any(|id| assigned.contains(id)) {
        return None;
    }
    nodes.push(square_pow_node_id);
    nodes.extend(exponent_trace.nodes.iter().copied());

    let diff_var_vid = square_pow_node.inputs[0];
    let (diff_var_node_id, diff_var_node) = node_from_value(graph, diff_var_vid)?;
    if assigned.contains(&diff_var_node_id) {
        return None;
    }
    if !matches!(diff_var_node.label, AccelNodeLabel::Primitive(PrimitiveOp::Sub)) {
        return None;
    }
    if diff_var_node.inputs.len() != 2 {
        return None;
    }
    let imgs_vid = diff_var_node.inputs[0];
    let mu_vid = diff_var_node.inputs[1];
    nodes.push(diff_var_node_id);

    let (diff_node_id, diff_node) = node_from_value(graph, diff_vid)?;
    if assigned.contains(&diff_node_id) {
        return None;
    }
    if !matches!(diff_node.label, AccelNodeLabel::Primitive(PrimitiveOp::Sub)) {
        return None;
    }
    if diff_node.inputs.len() != 2 {
        return None;
    }
    if diff_node.inputs[0] != imgs_vid || diff_node.inputs[1] != mu_vid {
        return None;
    }
    nodes.push(diff_node_id);

    let (mean_mu_dim3_id, mean_mu_dim2_vid) = match_mean_with_dim(graph, mu_vid, 3.0)?;
    if assigned.contains(&mean_mu_dim3_id) {
        return None;
    }
    let (mean_mu_dim2_id, mean_mu_input_vid) = match_mean_with_dim(graph, mean_mu_dim2_vid, 2.0)?;
    if assigned.contains(&mean_mu_dim2_id) {
        return None;
    }
    if mean_mu_input_vid != imgs_vid {
        return None;
    }
    nodes.push(mean_mu_dim3_id);
    nodes.push(mean_mu_dim2_id);

    let input_info = graph.value(imgs_vid)?;
    let dims = match &input_info.shape {
        ShapeInfo::Tensor(dims) if dims.len() == 3 => dims,
        _ => return None,
    };
    if dims.iter().any(|dim| dim.is_none()) {
        return None;
    }

    nodes.sort_unstable();
    nodes.dedup();

    Some(ImageNormalizeMatch {
        nodes,
        input: imgs_vid,
        epsilon,
        gain: gain_opt,
        bias: bias_opt,
        gamma: gamma_opt,
    })
}