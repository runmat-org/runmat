use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock, Weak};

use once_cell::sync::Lazy;
use runmat_accelerate_api::ReductionFlavor;
use runmat_builtins::Value;
use serde::{Deserialize, Serialize};

use crate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, NodeId, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin,
};
use crate::reduction_meta::{detect_reduction_signature, ReductionAxes, ReductionBehavior};
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
    pub gamma: Option<ImageScalar>,
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
        if assigned.contains(&node.id) {
            continue;
        }
        let elementwise_like = node.is_elementwise() || is_elementwise_max_min(graph, node);
        if !elementwise_like {
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
            expand_group_with_fanout(graph, &mut chain, &assigned, &consumer_map);
            chain.sort_unstable_by_key(|id| {
                graph
                    .node(*id)
                    .map(|node| node.span.start)
                    .unwrap_or_default()
            });
            chain.dedup();
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
        if assigned.contains(&node.id) {
            continue;
        }
        if !node.is_reduction() || is_elementwise_max_min(graph, node) {
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

    merge_downstream_fanout(graph, &mut groups, &consumer_map);
    groups
}

fn expand_group_with_fanout(
    graph: &AccelGraph,
    chain: &mut Vec<NodeId>,
    assigned: &HashSet<NodeId>,
    consumer_map: &HashMap<ValueId, HashSet<NodeId>>,
) {
    let base_start = chain
        .iter()
        .filter_map(|id| graph.node(*id).map(|node| node.span.start))
        .min()
        .unwrap_or(0);
    let mut node_set: HashSet<NodeId> = chain.iter().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.nodes {
            if node_set.contains(&node.id) {
                continue;
            }
            if node.span.start < base_start {
                continue;
            }
            if assigned.contains(&node.id) {
                continue;
            }
            if !(node.is_elementwise() || is_elementwise_max_min(graph, node)) {
                continue;
            }
            if node.outputs.is_empty() {
                continue;
            }
            let mut feeds_group = false;
            let mut all_consumers_ok = true;
            for &out in &node.outputs {
                if let Some(consumers) = consumer_map.get(&out) {
                    let mut consumer_in_group = false;
                    for consumer in consumers {
                        if node_set.contains(consumer) {
                            consumer_in_group = true;
                        } else {
                            all_consumers_ok = false;
                            break;
                        }
                    }
                    if !all_consumers_ok {
                        break;
                    }
                    if consumer_in_group {
                        feeds_group = true;
                    }
                } else {
                    all_consumers_ok = false;
                    break;
                }
            }
            if !feeds_group || !all_consumers_ok {
                continue;
            }
            let mut inputs_ok = true;
            for &input in &node.inputs {
                if let Some(info) = graph.value(input) {
                    if let ValueOrigin::NodeOutput { node: producer, .. } = info.origin {
                        if !node_set.contains(&producer) {
                            if let Some(prod_node) = graph.node(producer) {
                                if prod_node.span.start >= base_start {
                                    inputs_ok = false;
                                    break;
                                }
                            } else {
                                inputs_ok = false;
                                break;
                            }
                        }
                    }
                }
            }
            if inputs_ok {
                node_set.insert(node.id);
                chain.push(node.id);
                changed = true;
            }
        }
    }
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

fn merge_downstream_fanout(
    graph: &AccelGraph,
    groups: &mut Vec<FusionGroup>,
    consumer_map: &HashMap<ValueId, HashSet<NodeId>>,
) {
    let mut changed = true;
    while changed {
        changed = false;
        let mut node_group: HashMap<NodeId, usize> = HashMap::new();
        for (idx, group) in groups.iter().enumerate() {
            if group.kind.is_elementwise() {
                for &node in &group.nodes {
                    node_group.insert(node, idx);
                }
            }
        }
        'outer: for target_idx in 0..groups.len() {
            if !groups[target_idx].kind.is_elementwise() {
                continue;
            }
            let base_start = groups[target_idx].span.start;
            let mut merge_indices: Vec<usize> = Vec::new();
            for &node_id in &groups[target_idx].nodes {
                let Some(node) = graph.node(node_id) else {
                    continue;
                };
                for &input in &node.inputs {
                    if let Some(info) = graph.value(input) {
                        if let ValueOrigin::NodeOutput { node: producer, .. } = info.origin {
                            if let Some(&source_idx) = node_group.get(&producer) {
                                if source_idx == target_idx {
                                    continue;
                                }
                                let source_group = &groups[source_idx];
                                if !source_group.kind.is_elementwise() {
                                    continue;
                                }
                                if source_group.span.start < base_start {
                                    continue;
                                }
                                if !group_consumers_subset(
                                    source_group,
                                    target_idx,
                                    groups,
                                    consumer_map,
                                    graph,
                                ) {
                                    continue;
                                }
                                merge_indices.push(source_idx);
                            }
                        }
                    }
                }
            }
            if merge_indices.is_empty() {
                continue;
            }
            merge_indices.sort_unstable();
            merge_indices.dedup();
            for idx in &merge_indices {
                let nodes = groups[*idx].nodes.clone();
                groups[target_idx].nodes.extend(nodes);
                groups[*idx].nodes.clear();
            }
            groups[target_idx]
                .nodes
                .sort_unstable_by_key(|id| graph.node(*id).map(|n| n.span.start).unwrap_or(0));
            groups[target_idx].nodes.dedup();
            groups[target_idx].span = group_span(graph, &groups[target_idx].nodes);
            changed = true;
            break 'outer;
        }
        if changed {
            groups.retain(|group| !group.nodes.is_empty());
        }
    }
}

fn group_consumers_subset(
    source_group: &FusionGroup,
    target_idx: usize,
    groups: &[FusionGroup],
    consumer_map: &HashMap<ValueId, HashSet<NodeId>>,
    graph: &AccelGraph,
) -> bool {
    let target_nodes: HashSet<NodeId> = groups[target_idx].nodes.iter().copied().collect();
    let source_nodes: HashSet<NodeId> = source_group.nodes.iter().copied().collect();
    for &node_id in &source_group.nodes {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        for &out in &node.outputs {
            if let Some(consumers) = consumer_map.get(&out) {
                for consumer in consumers {
                    if !source_nodes.contains(consumer) && !target_nodes.contains(consumer) {
                        return false;
                    }
                }
            }
        }
    }
    true
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
        if !(next_node.is_elementwise() || is_elementwise_max_min(graph, next_node)) {
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

fn is_elementwise_max_min(graph: &AccelGraph, node: &AccelNode) -> bool {
    match &node.label {
        AccelNodeLabel::Builtin { name }
            if name.eq_ignore_ascii_case("max") || name.eq_ignore_ascii_case("min") =>
        {
            if node.inputs.len() < 2 {
                return false;
            }
            !value_is_placeholder(graph, node.inputs[1])
        }
        _ => false,
    }
}

fn value_is_placeholder(graph: &AccelGraph, vid: ValueId) -> bool {
    let Some(info) = graph.value(vid) else {
        return false;
    };
    let Some(constant) = &info.constant else {
        return false;
    };
    match constant {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data.is_empty(),
        Value::CharArray(ca) => ca.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    }
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
    // For reductions: track the ValueId of the dim argument when identifiable
    pub reduction_dim: Option<ValueId>,
    // For reductions: flavor metadata (e.g., sum vs mean scaling)
    pub reduction_flavor: Option<ReductionFlavor>,
    // For reductions: axis selection metadata (e.g., explicit dims vs 'all')
    pub reduction_axes: Option<ReductionAxes>,
    pub pattern: Option<FusionPattern>,
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

fn fusion_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("RUNMAT_DEBUG_FUSION") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"),
        Err(_) => false,
    })
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

fn log_plan_stack_pattern(stage: &str, plan: &FusionGroupPlan, graph: &AccelGraph) {
    if !fusion_debug_enabled() || plan.stack_pattern.is_empty() {
        return;
    }
    let mut pattern_meta: Vec<String> = Vec::with_capacity(plan.stack_pattern.len());
    for (pos, input_idx) in plan.stack_pattern.iter().enumerate() {
        let value_id = plan.inputs.get(*input_idx).copied();
        if let Some(vid) = value_id {
            if let Some(info) = graph.value(vid) {
                let node_label = match info.origin {
                    ValueOrigin::NodeOutput { node, .. } => graph
                        .node(node)
                        .map(|n| format!("{:?}", n.label))
                        .unwrap_or_else(|| "<missing-node>".to_string()),
                    _ => String::new(),
                };
                pattern_meta.push(format!(
                    "#{}:input_idx={} vid={} origin={:?} label={}",
                    pos, input_idx, vid, info.origin, node_label
                ));
            } else {
                pattern_meta.push(format!(
                    "#{}:input_idx={} vid={} origin=<missing>",
                    pos, input_idx, vid
                ));
            }
        } else {
            pattern_meta.push(format!("#{}:input_idx={} vid=<missing>", pos, input_idx));
        }
    }
    log::debug!(
        "fusion plan {} {} stack_pattern={:?} meta={:?}",
        plan.index,
        stage,
        plan.stack_pattern,
        pattern_meta
    );
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
        let mut reduction_flavor: Option<ReductionFlavor> = None;
        let mut reduction_axes: Option<ReductionAxes> = None;
        let mut reduction_data: Option<ValueId> = None;
        let mut reduction_dim: Option<ValueId> = None;
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

                    if fusion_debug_enabled() {
                        let origin = graph.value(*input).map(|v| v.origin.clone());
                        log::debug!(
                            "fusion plan #{:?} consider input vid={} origin={:?} binding={:?} newly_added={} is_variable={} stack_candidate={}",
                            index,
                            input,
                            origin,
                            binding,
                            newly_added,
                            is_variable,
                            !is_variable && newly_added
                        );
                    }
                    if let Some(constant) = maybe_constant.clone() {
                        constants.insert(input_idx, constant);
                    } else if !is_variable && newly_added {
                        let allow_stack = match graph.value(*input) {
                            Some(info) => match info.origin {
                                ValueOrigin::NodeOutput { node, .. } => graph
                                    .node(node)
                                    .map(|n| n.span.start <= group.span.start)
                                    .unwrap_or(false),
                                _ => true,
                            },
                            None => true,
                        };
                        if allow_stack {
                            stack_pattern.push(input_idx);
                        } else if fusion_debug_enabled() {
                            log::debug!(
                                "fusion plan {} skipping stack candidate vid={} origin_after_span",
                                index,
                                input
                            );
                        }
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
                    output: node.outputs.first().copied(),
                },
                AccelNodeLabel::Builtin { name } => FusionOp::Builtin {
                    name: name.clone(),
                    inputs: node.inputs.clone(),
                    output: node.outputs.first().copied(),
                },
                AccelNodeLabel::Unknown => FusionOp::Primitive {
                    op: PrimitiveOp::UPlus,
                    inputs: node.inputs.clone(),
                    output: node.outputs.first().copied(),
                },
            };
            operations.push(op);

            if let Some(out) = node.outputs.first().copied() {
                output = Some(out);
            }
            // Generic reduction signature (no name checks)
            if node.is_reduction() {
                if let Some(sig) = detect_reduction_signature(graph, node) {
                    reduction_data = Some(sig.data_input);
                    reduction_dim = sig.dim_arg;
                    reduction_flavor = Some(match sig.behavior {
                        ReductionBehavior::MeanLike => ReductionFlavor::Mean,
                        _ => ReductionFlavor::Sum,
                    });
                    reduction_axes = Some(sig.axes.clone());
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
            reduction_dim,
            reduction_flavor,
            reduction_axes,
            pattern,
        };

        log_plan_stack_pattern("initial", &plan, graph);

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
                let original_inputs = plan.inputs.clone();
                let original_stack_pattern = plan.stack_pattern.clone();
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
                let mut deps: Vec<ValueId> = Vec::new();
                let mut visited: HashSet<ValueId> = HashSet::new();
                let mut stack: Vec<ValueId> = vec![data_vid];
                // Track extra ops we discover outside the original group that are safe to inline
                let mut extra_ops: Vec<FusionOp> = Vec::new();
                let mut added_nodes: HashSet<ValueId> = HashSet::new();
                while let Some(cur) = stack.pop() {
                    if !visited.insert(cur) {
                        continue;
                    }
                    if graph.var_binding(cur).is_some() {
                        if !deps.contains(&cur) {
                            deps.push(cur);
                        }
                        continue;
                    }
                    if let Some(info) = graph.value(cur) {
                        if matches!(info.origin, ValueOrigin::Variable { .. }) {
                            if !deps.contains(&cur) {
                                deps.push(cur);
                            }
                            continue;
                        }
                    }
                    // Do not short-circuit on the reduction_data itself; expand through its producers first.
                    if original_inputs.contains(&cur) && cur != data_vid {
                        if !deps.contains(&cur) {
                            deps.push(cur);
                        }
                        continue;
                    }
                    if let Some(parents) = prod.get(&cur) {
                        for p in parents {
                            stack.push(*p);
                        }
                        continue;
                    }
                    // If not produced by an op in this group, try to expand through safe producer nodes
                    if let Some((_, node)) = node_from_value(graph, cur) {
                        // Only consider simple arithmetic producers we know how to fold
                        match &node.label {
                            AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                            | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul)
                            | AccelNodeLabel::Primitive(PrimitiveOp::Div)
                            | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
                            | AccelNodeLabel::Primitive(PrimitiveOp::ElemLeftDiv)
                            | AccelNodeLabel::Primitive(PrimitiveOp::Add)
                            | AccelNodeLabel::Primitive(PrimitiveOp::Sub) => {
                                // Record op for codegen and traverse inputs
                                if added_nodes.insert(cur) {
                                    extra_ops.push(FusionOp::Primitive {
                                        op: match node.label {
                                            AccelNodeLabel::Primitive(op) => op,
                                            _ => PrimitiveOp::UPlus,
                                        },
                                        inputs: node.inputs.clone(),
                                        output: node.outputs.first().copied(),
                                    });
                                }
                                for &p in &node.inputs {
                                    stack.push(p);
                                }
                                continue;
                            }
                            AccelNodeLabel::Primitive(PrimitiveOp::ElemPow) => {
                                // Only accept power with constant exponent (typically 2 for squares)
                                if node.inputs.len() == 2 {
                                    if let Some(exp) = value_constant_f64(graph, node.inputs[1]) {
                                        if exp.is_finite() {
                                            if added_nodes.insert(cur) {
                                                extra_ops.push(FusionOp::Primitive {
                                                    op: PrimitiveOp::ElemPow,
                                                    inputs: node.inputs.clone(),
                                                    output: node.outputs.first().copied(),
                                                });
                                            }
                                            stack.push(node.inputs[0]);
                                            // Treat exponent as constant dependency for codegen
                                            stack.push(node.inputs[1]);
                                            continue;
                                        }
                                    }
                                }
                                // Fallback: treat as leaf dependency
                            }
                            AccelNodeLabel::Builtin { name } => {
                                // Allow simple casts to flow through (single/double)
                                if (name.eq_ignore_ascii_case("single")
                                    || name.eq_ignore_ascii_case("double"))
                                    && node.inputs.len() == 1
                                {
                                    stack.push(node.inputs[0]);
                                    continue;
                                }
                                // Unknown builtin: treat as leaf
                            }
                            _ => {
                                // Unknown producer: treat as leaf
                            }
                        }
                    }
                }
                // Ensure direct parents of the reduction data are materialized as inputs
                if let Some(parents) = prod.get(&data_vid) {
                    for &p in parents {
                        if !deps.contains(&p) {
                            // Skip trivial constants embedded in const_values; those are handled separately
                            let is_const = plan.const_values.contains_key(&p)
                                || graph.value(p).and_then(|vi| vi.constant.as_ref()).is_some();
                            if !is_const {
                                deps.push(p);
                            }
                        }
                    }
                }
                // Prepend the newly discovered ops so they are available to codegen
                // Keep original operations as well (the reduction op itself)
                if !extra_ops.is_empty() {
                    // Ensure a stable order: extra ops first
                    let mut new_ops = Vec::with_capacity(extra_ops.len() + plan.operations.len());
                    new_ops.extend(extra_ops);
                    new_ops.append(&mut plan.operations);
                    plan.operations = new_ops;
                }
                plan.inputs = deps;
                // Ensure constants referenced by any newly added operations are recorded.
                for op in &plan.operations {
                    let inputs = match op {
                        FusionOp::Primitive { inputs, .. } => inputs,
                        FusionOp::Builtin { inputs, .. } => inputs,
                    };
                    for vid in inputs {
                        if plan.const_values.contains_key(vid) {
                            continue;
                        }
                        if let Some(info) = graph.value(*vid) {
                            if let Some(cv) = info.constant.clone() {
                                plan.const_values.insert(*vid, cv);
                            }
                        }
                    }
                }

                // Rebuild stack pattern based on the dependencies that were previously sourced
                // from the execution stack.
                let mut new_stack_pattern: Vec<usize> = Vec::new();
                for (new_idx, vid) in plan.inputs.iter().enumerate() {
                    if let Some(old_idx) = original_inputs.iter().position(|v| v == vid) {
                        if original_stack_pattern.contains(&old_idx) {
                            new_stack_pattern.push(new_idx);
                        }
                    }
                }

                // Rebuild constants map using the new input ordering.
                let mut new_constants: HashMap<usize, Value> = HashMap::new();
                for (idx, vid) in plan.inputs.iter().enumerate() {
                    if let Some(value) = plan.const_values.get(vid) {
                        new_constants.insert(idx, value.clone());
                    } else if let Some(info) = graph.value(*vid) {
                        if let Some(cv) = info.constant.clone() {
                            new_constants.insert(idx, cv);
                        }
                    }
                }
                plan.constants = new_constants;

                if new_stack_pattern.is_empty() {
                    for (idx, vid) in plan.inputs.iter().enumerate() {
                        if plan.constants.contains_key(&idx) {
                            continue;
                        }
                        if let Some(info) = graph.value(*vid) {
                            if matches!(
                                info.origin,
                                ValueOrigin::Variable { .. } | ValueOrigin::Constant
                            ) {
                                continue;
                            }
                        }
                        new_stack_pattern.push(idx);
                    }
                }
                plan.stack_pattern = new_stack_pattern;
            }
        }

        // Final sanitize: for reduction groups, ensure inputs contain no constants
        if plan.group.kind.is_reduction() {
            let original_inputs = plan.inputs.clone();
            plan.inputs.retain(|vid| {
                if let Some(info) = graph.value(*vid) {
                    !matches!(info.origin, ValueOrigin::Constant)
                        && !plan.const_values.contains_key(vid)
                } else {
                    true
                }
            });
            if plan.inputs.len() != original_inputs.len() {
                let mut new_stack: Vec<usize> = Vec::new();
                for old_idx in &plan.stack_pattern {
                    if *old_idx < original_inputs.len() {
                        let vid = original_inputs[*old_idx];
                        if let Some(new_idx) = plan.inputs.iter().position(|v| *v == vid) {
                            new_stack.push(new_idx);
                        }
                    }
                }
                plan.stack_pattern = new_stack;
            }
        }

        // Determine kernel support:
        // - Elementwise: require WGSL generation at plan time.
        // - Reduction: require WGSL generation at plan time as well.
        // - Other kinds: executed via provider paths.
        let supported = if plan.kernel.kind.is_elementwise() {
            plan.generate_wgsl("f32").is_some()
        } else if plan.kernel.kind.is_reduction() {
            plan.generate_reduction_wgsl("f32").is_some()
        } else {
            true
        };
        plan.kernel.supported = plan.kernel.supported && supported;
        if !plan.kernel.supported && fusion_debug_enabled() {
            let const_ids: Vec<ValueId> = plan.const_values.keys().copied().collect();
            log::debug!(
                "fusion plan {} unsupported: kind={:?} group_kind={:?} inputs={:?} reduction_data={:?} reduction_dim={:?} const_ids={:?}",
                plan.index,
                plan.kernel.kind,
                plan.group.kind,
                plan.inputs,
                plan.reduction_data,
                plan.reduction_dim,
                const_ids
            );
            if plan.kernel.kind.is_reduction() {
                let mut seen: HashSet<ValueId> = HashSet::new();
                let mut value_info: Vec<String> = Vec::new();
                for op in &plan.operations {
                    let inputs = match op {
                        FusionOp::Primitive { inputs, .. } => inputs,
                        FusionOp::Builtin { inputs, .. } => inputs,
                    };
                    for vid in inputs {
                        if seen.insert(*vid) {
                            if let Some(info) = graph.value(*vid) {
                                value_info.push(format!(
                                    "vid={} origin={:?} constant={}",
                                    vid,
                                    info.origin,
                                    info.constant.is_some()
                                ));
                            } else {
                                value_info.push(format!("vid={} origin=<missing>", vid));
                            }
                        }
                    }
                }
                log::debug!(
                    "fusion reduction plan {} value summary: [{}]",
                    plan.index,
                    value_info.join(", ")
                );
            }
        }

        if matches!(plan.group.kind, FusionKind::CenteredGram) && plan.stack_pattern.is_empty() {
            let mut centered_stack_idxs: Vec<usize> = Vec::new();
            for (idx, vid) in plan.inputs.iter().enumerate() {
                if plan.constants.contains_key(&idx) {
                    continue;
                }
                if let Some(info) = graph.value(*vid) {
                    if matches!(info.origin, ValueOrigin::NodeOutput { .. }) {
                        centered_stack_idxs.push(idx);
                        continue;
                    }
                    if matches!(info.origin, ValueOrigin::Variable { .. }) {
                        continue;
                    }
                }
                centered_stack_idxs.push(idx);
            }
            if centered_stack_idxs.is_empty() && !plan.inputs.is_empty() {
                centered_stack_idxs.push(0);
            }
            plan.stack_pattern = centered_stack_idxs;
        }

        log_plan_stack_pattern("final", &plan, graph);

        // If the plan requires any unsupported operations, mark kernel as unsupported

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
        // Supports folding simple producer expressions over multiple inputs (e.g., sum(A.*B, dim)).
        if self.inputs.is_empty() {
            return None;
        }
        // Determine axis from the reduction builtin's explicit dim argument when available.
        // MATLAB dim is 1-based: dim=1 reduces rows (axis=0), dim=2 reduces cols (axis=1).
        let mut axis = 0usize;
        // Support 'all' via either index-keyed constants or value-id keyed const_values
        let reduce_all = self
            .constants
            .values()
            .any(|v| matches!(v, Value::String(s) if s.eq_ignore_ascii_case("all")))
            || self
                .const_values
                .values()
                .any(|v| matches!(v, Value::String(s) if s.eq_ignore_ascii_case("all")));
        if reduce_all {
            // We'll flatten in VM by setting nrows = total and ncols = 1; axis=0 works with that.
            axis = 0;
        } else if let Some(dim_vid) = self.reduction_dim {
            if let Some(v) = self.const_values.get(&dim_vid) {
                match v {
                    Value::Num(n) if *n >= 1.0 => {
                        axis = (*n as usize).saturating_sub(1);
                    }
                    Value::Int(i) => {
                        let val = i.to_f64();
                        if val >= 1.0 {
                            axis = (val as usize).saturating_sub(1);
                        }
                    }
                    _ => {}
                }
            }
        } else {
            // Fallback: scan constant table for a plausible dim
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
        // Map additional external inputs to v1, v2, ...
        for (idx, &vid) in self.inputs.iter().enumerate().skip(1) {
            exprs.insert(vid, format!("v{idx}"));
        }
        for (vid, val) in &self.const_values {
            let lit = match val {
                Value::Num(n) => {
                    if scalar_ty == "f64" {
                        format!("f64({})", n)
                    } else {
                        format!("{:?}", *n as f32)
                    }
                }
                Value::Int(i) => {
                    let f = i.to_f64();
                    if scalar_ty == "f64" {
                        format!("f64({})", f)
                    } else {
                        format!("{:?}", f as f32)
                    }
                }
                Value::Tensor(t) if t.data.len() == 1 => {
                    let scalar = t.data[0];
                    if scalar_ty == "f64" {
                        format!("f64({})", scalar)
                    } else {
                        format!("{:?}", scalar as f32)
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
        let mut progressed = true;
        while progressed {
            progressed = false;
            for op in &self.operations {
                match op {
                    FusionOp::Primitive { op, inputs, output } => {
                        if let Some(out) = output {
                            if exprs.contains_key(out) {
                                continue;
                            }
                            if let Some(code) = primitive_expr(*op, inputs, &exprs) {
                                exprs.insert(*out, code);
                                progressed = true;
                            }
                        }
                    }
                    FusionOp::Builtin {
                        name,
                        inputs,
                        output,
                    } => {
                        if let Some(out) = output {
                            if exprs.contains_key(out) {
                                continue;
                            }
                            if let Some(code) = builtin_expr(name, inputs, &exprs, scalar_ty) {
                                exprs.insert(*out, code);
                                progressed = true;
                            }
                        }
                    }
                }
            }
            if exprs.contains_key(&data_vid) {
                break;
            }
        }
        // Require a folded expression for the reduction operand; if missing, defer (no WGSL).
        let val_expr = match exprs.get(&data_vid) {
            Some(s) => s.clone(),
            None => {
                if fusion_debug_enabled() {
                    let expr_keys: Vec<ValueId> = exprs.keys().copied().collect();
                    log::debug!(
                        "fusion reduction WGSL: missing expression for data {:?}; inputs={:?} expr_keys={:?} ops={:?}",
                        data_vid,
                        self.inputs,
                        expr_keys,
                        self.operations
                    );
                }
                return None;
            }
        };

        let mut shader = String::new();
        shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>, }};\n"));
        shader.push_str("struct MParams { nrows: u32, ncols: u32, ld: u32, flags: u32 }\n\n");
        // Bind all input tensors dynamically, followed by output and params
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
            "@group(0) @binding({}) var<uniform> params: MParams;\n\n",
            self.inputs.len() + 1
        ));
        // Use a small fixed workgroup tile size to avoid driver stalls on some backends
        shader.push_str(&format!(
            "var<workgroup> tile: array<{scalar_ty}, @WG@u>;\n\n"
        ));
        shader.push_str(&format!(
            "const OMITNAN: bool = {};\n\n",
            if omitnan { "true" } else { "false" }
        ));
        // Determine mean semantics from planner-populated reduction flavor
        let is_mean = matches!(self.reduction_flavor, Some(ReductionFlavor::Mean));
        let post_scale = if is_mean {
            let dim = if axis == 0 {
                "params.nrows"
            } else {
                "params.ncols"
            };
            if scalar_ty == "f64" {
                format!("(1.0 / f64(f32({dim})))")
            } else {
                format!("(1.0 / f32({dim}))")
            }
        } else if scalar_ty == "f64" {
            "f64(1.0)".to_string()
        } else {
            "1.0".to_string()
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
            // Load row-wise values from each input and fold into expression
            {
                // Build the per-iteration loads
                let mut loop_body = String::new();
                // input0 as 'v'
                loop_body.push_str("    let v = input0.data[ (col * params.nrows) + r ];\n");
                // additional inputs as v1, v2, ...
                for (idx, _) in self.inputs.iter().enumerate().skip(1) {
                    loop_body.push_str(&format!(
                        "    let v{idx} = input{idx}.data[ (col * params.nrows) + r ];\n"
                    ));
                }
                // compute val and accumulate
                loop_body.push_str(&format!(
                    "    let val: {scalar} = {val};\n    if (OMITNAN) {{ if (!isNanF(val)) {{ acc = acc + val; }} }} else {{ if (isNanF(val)) {{ saw_nan = true; }} else {{ acc = acc + val; }} }}\n",
                scalar = scalar_ty,
                val = val_expr
            ));
                shader.push_str("  while (r < params.nrows) {\n");
                shader.push_str(&loop_body);
                shader.push_str("    r += @WG@u;\n  }\n");
            }
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
            shader.push_str("  let row = wid.x;\n  // For axis=1, number of output slices equals rows (params.ncols)\n  if (row >= params.ncols) { return; }\n");
            shader.push_str(&format!(
                "  var acc: {scalar_ty} = {}0.0;\n",
                if scalar_ty == "f64" { "f64(" } else { "" }
            ));
            if scalar_ty == "f64" {
                shader.push_str("  // close cast for f64 literal\n");
            }
            // helpers are declared at module scope
            shader.push_str("  var saw_nan: bool = false;\n  var c = lid.x;\n");
            {
                let mut loop_body = String::new();
                // input0 as 'v' — provider encodes rows in params.ncols for axis=1
                loop_body.push_str("    let v = input0.data[ row + (c * params.ncols) ];\n");
                // additional inputs as v1, v2, ...
                for (idx, _) in self.inputs.iter().enumerate().skip(1) {
                    loop_body.push_str(&format!(
                        "    let v{idx} = input{idx}.data[ row + (c * params.ncols) ];\n"
                    ));
                }
                loop_body.push_str(&format!(
                    "    let val: {scalar} = {val};\n    if (OMITNAN) {{ if (!isNanF(val)) {{ acc = acc + val; }} }} else {{ if (isNanF(val)) {{ saw_nan = true; }} else {{ acc = acc + val; }} }}\n",
                scalar = scalar_ty,
                val = val_expr
            ));
                // Iterate over reduce_len, which arrives as params.nrows when axis=1
                shader.push_str("  while (c < params.nrows) {\n");
                shader.push_str(&loop_body);
                shader.push_str("    c += @WG@u;\n  }\n");
            }
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
            ShapeInfo::Tensor(dims) => dims
                .iter()
                .try_fold(1usize, |acc, dim| dim.and_then(|d| acc.checked_mul(d))),
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
                    if let Some(centered) = trans_node.inputs.first().copied() {
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

fn detect_image_normalize(
    graph: &AccelGraph,
    assigned: &mut HashSet<NodeId>,
    groups: &mut Vec<FusionGroup>,
    next_group_id: &mut usize,
) {
    for pow_node in &graph.nodes {
        if assigned.contains(&pow_node.id) {
            continue;
        }
        let Some(match_info) = analyze_image_normalize(graph, pow_node.id, assigned) else {
            continue;
        };

        let pow_node_ref = match graph.node(pow_node.id) {
            Some(node) => node,
            None => continue,
        };

        let shape = node_output_shape(graph, pow_node_ref);
        let span = group_span(graph, &match_info.nodes);

        let pattern = ImageNormalizePattern {
            input: match_info.input,
            epsilon: match_info.epsilon.clone(),
            gain: match_info.gain.clone(),
            bias: match_info.bias.clone(),
            gamma: match_info.gamma.clone(),
        };

        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::ImageNormalize,
            nodes: match_info.nodes.clone(),
            shape,
            span: span.clone(),
            pattern: Some(FusionPattern::ImageNormalize(pattern)),
        });
        if fusion_debug_enabled() {
            log::debug!(
                "fusion: detected image normalize group id={} span={:?} nodes={:?}",
                next_group_id,
                span,
                match_info.nodes
            );
        }
        *next_group_id += 1;
        for node_id in match_info.nodes {
            assigned.insert(node_id);
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

fn node_from_value(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, &AccelNode)> {
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
            let input = node.inputs.first().copied()?;
            Some((node_id, input))
        }
        _ => None,
    }
}

fn is_transpose_node(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Transpose) => {
            let input = node.inputs.first().copied()?;
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
                            || name.eq_ignore_ascii_case("double")
                            || name.eq_ignore_ascii_case("gpuarray") =>
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

fn scalar_shape_known_one(shape: &ShapeInfo) -> bool {
    match shape {
        ShapeInfo::Scalar => true,
        ShapeInfo::Tensor(dims) => {
            if dims.is_empty() {
                return true;
            }
            dims.iter().all(|dim| matches!(dim, Some(1)))
        }
        ShapeInfo::Unknown => false,
    }
}

fn capture_image_scalar(
    graph: &AccelGraph,
    vid: ValueId,
    assigned: &HashSet<NodeId>,
    _nodes: &mut Vec<NodeId>,
) -> Option<ImageScalar> {
    if let Some(trace) = collect_scalar_constant(graph, vid) {
        if trace.nodes.iter().any(|id| assigned.contains(id)) {
            return None;
        }
        return Some(ImageScalar::Constant(trace.value));
    }
    let info = graph.value(vid)?;
    if scalar_shape_known_one(&info.shape) {
        return Some(ImageScalar::Value(vid));
    }
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "capture_image_scalar: reject vid={vid:?} shape={:?} origin={:?}",
            info.shape,
            info.origin
        );
    }
    None
}

fn peel_numeric_casts(
    graph: &AccelGraph,
    mut vid: ValueId,
    assigned: &HashSet<NodeId>,
    _nodes: &mut Vec<NodeId>,
) -> Option<ValueId> {
    loop {
        let info = graph.value(vid)?;
        match &info.origin {
            ValueOrigin::NodeOutput { node, .. } => {
                if assigned.contains(node) {
                    return None;
                }
                let node_ref = graph.node(*node)?;
                if let AccelNodeLabel::Builtin { name } = &node_ref.label {
                    if name.eq_ignore_ascii_case("single")
                        || name.eq_ignore_ascii_case("double")
                        || name.eq_ignore_ascii_case("gpuarray")
                    {
                        if node_ref.inputs.len() != 1 {
                            return None;
                        }
                        vid = node_ref.inputs[0];
                        continue;
                    }
                }
                return Some(vid);
            }
            _ => return Some(vid),
        }
    }
}

fn resolve_scalar_constant(graph: &AccelGraph, vid: ValueId) -> Option<f64> {
    collect_scalar_constant(graph, vid).map(|trace| trace.value)
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

fn value_constant_f64(graph: &AccelGraph, vid: ValueId) -> Option<f64> {
    resolve_scalar_constant(graph, vid)
}

fn primitive_expr(
    op: PrimitiveOp,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let binary = |exprs: &HashMap<ValueId, String>| -> Option<(String, String)> {
        let lhs = exprs.get(inputs.first()?).cloned()?;
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
            let arg = exprs.get(inputs.first()?).cloned()?;
            Some(format!("(-{arg})"))
        }
        PrimitiveOp::UPlus => {
            let arg = exprs.get(inputs.first()?).cloned()?;
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
        "single" | "double" | "gpuarray" => return builtin_identity(inputs, exprs),
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
                    let arg = exprs.get(inputs.first()?).cloned()?;
                    let constant = cast_literal(scalar_ty, "0.4342944819032518");
                    Some(format!("(log({arg}) * {constant})"))
                }
                "log1p" => {
                    let arg = exprs.get(inputs.first()?).cloned()?;
                    let one = cast_literal(scalar_ty, "1.0");
                    Some(format!("log({arg} + {one})"))
                }
                "expm1" => {
                    let arg = exprs.get(inputs.first()?).cloned()?;
                    let one = cast_literal(scalar_ty, "1.0");
                    Some(format!("(exp({arg}) - {one})"))
                }
                _ => None,
            }
        }
    };
    let arg = exprs.get(inputs.first()?).cloned()?;
    Some(format!("{func}({arg})"))
}

fn builtin_binary(
    func: &str,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let lhs = exprs.get(inputs.first()?).cloned()?;
    let rhs = exprs.get(inputs.get(1)?).cloned()?;
    Some(format!("{func}({lhs}, {rhs})"))
}

fn builtin_unary_call(
    func: &str,
    inputs: &[ValueId],
    exprs: &HashMap<ValueId, String>,
) -> Option<String> {
    let arg = exprs.get(inputs.first()?).cloned()?;
    Some(format!("{func}({arg})"))
}

fn builtin_identity(inputs: &[ValueId], exprs: &HashMap<ValueId, String>) -> Option<String> {
    exprs.get(inputs.first()?).cloned()
}

fn cast_literal(scalar_ty: &str, literal: &str) -> String {
    if scalar_ty == "f64" {
        format!("{scalar_ty}({literal})")
    } else {
        literal.to_string()
    }
}

fn split_add_with_scalar(
    graph: &AccelGraph,
    vid: ValueId,
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<(NodeId, ValueId, ImageScalar)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Add) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(scalar) = capture_image_scalar(graph, rhs, assigned, nodes) {
                return Some((node_id, lhs, scalar));
            }
            if let Some(scalar) = capture_image_scalar(graph, lhs, assigned, nodes) {
                return Some((node_id, rhs, scalar));
            }
            None
        }
        AccelNodeLabel::Primitive(PrimitiveOp::Sub) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(ImageScalar::Constant(value)) =
                capture_image_scalar(graph, rhs, assigned, nodes)
            {
                return Some((node_id, lhs, ImageScalar::Constant(-value)));
            }
            None
        }
        _ => None,
    }
}

fn split_mul_with_scalar(
    graph: &AccelGraph,
    vid: ValueId,
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<(NodeId, ValueId, ImageScalar)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Mul)
        | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul) => {
            if node.inputs.len() != 2 {
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(scalar) = capture_image_scalar(graph, rhs, assigned, nodes) {
                return Some((node_id, lhs, scalar));
            }
            if let Some(scalar) = capture_image_scalar(graph, lhs, assigned, nodes) {
                return Some((node_id, rhs, scalar));
            }
            None
        }
        _ => None,
    }
}

fn split_max_with_zero_scalar(
    graph: &AccelGraph,
    vid: ValueId,
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<(NodeId, ValueId)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("max") => {
            if node.inputs.len() != 2 {
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "split_max_with_zero_scalar: node {node_id:?} has {} inputs",
                        node.inputs.len()
                    );
                }
                return None;
            }
            let lhs = node.inputs[0];
            let rhs = node.inputs[1];
            if let Some(ImageScalar::Constant(value)) =
                capture_image_scalar(graph, rhs, assigned, nodes)
            {
                if approx_eq(value, 0.0) {
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "split_max_with_zero_scalar: rhs zero constant for node {node_id:?}"
                        );
                    }
                    return Some((node_id, lhs));
                }
            }
            if let Some(ImageScalar::Constant(value)) =
                capture_image_scalar(graph, lhs, assigned, nodes)
            {
                if approx_eq(value, 0.0) {
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "split_max_with_zero_scalar: lhs zero constant for node {node_id:?}"
                        );
                    }
                    return Some((node_id, rhs));
                }
            }
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "split_max_with_zero_scalar: node {node_id:?} inputs not zero constants"
                );
            }
            None
        }
        _ => None,
    }
}

fn resolve_numeric_vector_constant(graph: &AccelGraph, vid: ValueId) -> Option<Vec<f64>> {
    if let Some(scalar) = resolve_scalar_constant(graph, vid) {
        return Some(vec![scalar]);
    }
    let info = graph.value(vid)?;
    match &info.constant {
        Some(Value::Tensor(tensor)) if !tensor.data.is_empty() => Some(tensor.data.clone()),
        Some(Value::LogicalArray(arr)) if !arr.data.is_empty() => Some(
            arr.data
                .iter()
                .map(|v| if *v == 0 { 0.0 } else { 1.0 })
                .collect(),
        ),
        Some(Value::Bool(flag)) => Some(vec![if *flag { 1.0 } else { 0.0 }]),
        Some(Value::Int(iv)) => Some(vec![iv.to_f64()]),
        Some(Value::Num(num)) => Some(vec![*num]),
        _ => None,
    }
}

fn match_mean_axes(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId, Vec<f64>)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { name } if name.eq_ignore_ascii_case("mean") => {}
        _ => return None,
    }
    if node.inputs.len() < 2 {
        return None;
    }
    let data_vid = node.inputs[0];
    let dims_vid = node.inputs[1];
    let dims = resolve_numeric_vector_constant(graph, dims_vid)?;
    Some((node_id, data_vid, dims))
}

fn dims_match_unordered(found: &[f64], expected: &[f64]) -> bool {
    if found.len() != expected.len() {
        return false;
    }
    let mut a: Vec<i64> = found.iter().map(|d| d.round() as i64).collect();
    let mut b: Vec<i64> = expected.iter().map(|d| d.round() as i64).collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

fn peel_mean_dims(
    graph: &AccelGraph,
    vid: ValueId,
    expected_dims: &[f64],
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<ValueId> {
    if expected_dims.is_empty() {
        return Some(vid);
    }
    let (node_id, data_vid, dims) = match_mean_axes(graph, vid)?;
    if assigned.contains(&node_id) {
        return None;
    }
    if dims.len() == expected_dims.len() && dims_match_unordered(&dims, expected_dims) {
        nodes.push(node_id);
        return Some(data_vid);
    }
    if dims.len() == 1 && approx_eq(dims[0], expected_dims[0]) {
        nodes.push(node_id);
        return peel_mean_dims(graph, data_vid, &expected_dims[1..], assigned, nodes);
    }
    None
}

struct ImageNormalizeMatch {
    nodes: Vec<NodeId>,
    input: ValueId,
    epsilon: ImageScalar,
    gain: Option<ImageScalar>,
    bias: Option<ImageScalar>,
    gamma: Option<ImageScalar>,
}

fn analyze_image_normalize(
    graph: &AccelGraph,
    pow_node_id: NodeId,
    assigned: &HashSet<NodeId>,
) -> Option<ImageNormalizeMatch> {
    let pow_node = graph.node(pow_node_id)?;
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "image_normalize: inspect pow candidate node={pow_node_id:?} label={:?}",
            pow_node.label
        );
    }
    macro_rules! img_norm_fail {
        ($reason:expr) => {{
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "image_normalize: reject node {pow_node_id:?} reason={}",
                    $reason
                );
            }
            return None;
        }};
    }
    if !matches!(
        pow_node.label,
        AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)
    ) {
        img_norm_fail!("not elem pow");
    }
    if pow_node.inputs.len() != 2 || pow_node.outputs.len() != 1 {
        img_norm_fail!("unexpected pow arity");
    }

    let mut nodes: Vec<NodeId> = vec![pow_node_id];

    let gamma_scalar = capture_image_scalar(graph, pow_node.inputs[1], assigned, &mut nodes)?;
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("image_normalize: node {pow_node_id:?} gamma scalar={gamma_scalar:?}");
    }
    let gamma_opt = match &gamma_scalar {
        ImageScalar::Constant(value) if approx_eq(*value, 1.0) => None,
        _ => Some(gamma_scalar),
    };

    let (clamp_node_id, clamp_input_vid) =
        split_max_with_zero_scalar(graph, pow_node.inputs[0], assigned, &mut nodes)?;
    if assigned.contains(&clamp_node_id) {
        img_norm_fail!("clamp node already assigned");
    }
    nodes.push(clamp_node_id);

    let pre_bias_vid = peel_numeric_casts(graph, clamp_input_vid, assigned, &mut nodes)?;
    let (pre_gain_vid, bias_opt) = if let Some((add_node_id, base_vid, bias_scalar)) =
        split_add_with_scalar(graph, pre_bias_vid, assigned, &mut nodes)
    {
        if assigned.contains(&add_node_id) {
            img_norm_fail!("bias add already assigned");
        }
        nodes.push(add_node_id);
        let bias = match &bias_scalar {
            ImageScalar::Constant(value) if approx_eq(*value, 0.0) => None,
            _ => Some(bias_scalar),
        };
        let base_vid = peel_numeric_casts(graph, base_vid, assigned, &mut nodes)?;
        (base_vid, bias)
    } else {
        (pre_bias_vid, None)
    };

    let (mut norm_vid, gain_opt) = if let Some((mul_node_id, base_vid, gain_scalar)) =
        split_mul_with_scalar(graph, pre_gain_vid, assigned, &mut nodes)
    {
        if assigned.contains(&mul_node_id) {
            img_norm_fail!("gain mul already assigned");
        }
        nodes.push(mul_node_id);
        let gain = match &gain_scalar {
            ImageScalar::Constant(value) if approx_eq(*value, 1.0) => None,
            _ => Some(gain_scalar),
        };
        let base_vid = peel_numeric_casts(graph, base_vid, assigned, &mut nodes)?;
        (base_vid, gain)
    } else {
        (pre_gain_vid, None)
    };

    norm_vid = peel_numeric_casts(graph, norm_vid, assigned, &mut nodes)?;

    let (div_node_id, div_node) = node_from_value(graph, norm_vid)?;
    if assigned.contains(&div_node_id) {
        img_norm_fail!("div node already assigned");
    }
    match div_node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
        | AccelNodeLabel::Primitive(PrimitiveOp::Div) => {}
        _ => img_norm_fail!("not div primitive"),
    }
    if div_node.inputs.len() != 2 {
        img_norm_fail!("div arity");
    }

    let diff_vid = div_node.inputs[0];
    let sigma_vid = peel_numeric_casts(graph, div_node.inputs[1], assigned, &mut nodes)?;
    let (sigma_node_id, sigma_input_vid) = match is_sqrt_node(graph, sigma_vid) {
        Some(pair) => pair,
        None => img_norm_fail!("sigma not sqrt"),
    };
    if assigned.contains(&sigma_node_id) {
        img_norm_fail!("sqrt node already assigned");
    }
    nodes.push(div_node_id);
    nodes.push(sigma_node_id);

    let (add_node_id, mean_sq_vid, epsilon_scalar) =
        split_add_with_scalar(graph, sigma_input_vid, assigned, &mut nodes)?;
    if assigned.contains(&add_node_id) {
        img_norm_fail!("epsilon add already assigned");
    }
    nodes.push(add_node_id);
    let epsilon = epsilon_scalar;
    let mean_sq_vid = peel_numeric_casts(graph, mean_sq_vid, assigned, &mut nodes)?;

    let squared_diff_vid = peel_mean_dims(graph, mean_sq_vid, &[3.0, 2.0], assigned, &mut nodes)?;

    let (square_pow_node_id, square_pow_node) = node_from_value(graph, squared_diff_vid)?;
    if assigned.contains(&square_pow_node_id) {
        img_norm_fail!("square pow already assigned");
    }
    if !matches!(
        square_pow_node.label,
        AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)
    ) {
        img_norm_fail!("variance pow not elem pow");
    }
    if square_pow_node.inputs.len() != 2 {
        img_norm_fail!("variance pow arity");
    }
    let exponent_trace = collect_scalar_constant(graph, square_pow_node.inputs[1])?;
    if !approx_eq(exponent_trace.value, 2.0) {
        img_norm_fail!("variance exponent != 2");
    }
    if exponent_trace.nodes.iter().any(|id| assigned.contains(id)) {
        img_norm_fail!("variance exponent nodes already assigned");
    }
    nodes.push(square_pow_node_id);
    nodes.extend(exponent_trace.nodes.iter().copied());

    let diff_var_vid = square_pow_node.inputs[0];
    let (diff_var_node_id, diff_var_node) = node_from_value(graph, diff_var_vid)?;
    if assigned.contains(&diff_var_node_id) {
        img_norm_fail!("diff variance node already assigned");
    }
    if !matches!(
        diff_var_node.label,
        AccelNodeLabel::Primitive(PrimitiveOp::Sub)
    ) {
        img_norm_fail!("diff variance node not sub");
    }
    if diff_var_node.inputs.len() != 2 {
        img_norm_fail!("diff variance arity");
    }
    let imgs_vid = diff_var_node.inputs[0];
    let mu_vid = peel_numeric_casts(graph, diff_var_node.inputs[1], assigned, &mut nodes)?;
    nodes.push(diff_var_node_id);

    let (diff_node_id, diff_node) = node_from_value(graph, diff_vid)?;
    if assigned.contains(&diff_node_id) {
        img_norm_fail!("diff node already assigned");
    }
    if !matches!(diff_node.label, AccelNodeLabel::Primitive(PrimitiveOp::Sub)) {
        img_norm_fail!("diff node not sub");
    }
    if diff_node.inputs.len() != 2 {
        img_norm_fail!("diff node arity");
    }
    let diff_mu_vid = peel_numeric_casts(graph, diff_node.inputs[1], assigned, &mut nodes)?;
    if diff_node.inputs[0] != imgs_vid || diff_mu_vid != mu_vid {
        img_norm_fail!("diff inputs mismatch with variance pair");
    }
    nodes.push(diff_node_id);

    let mean_mu_input_vid = peel_mean_dims(graph, mu_vid, &[3.0, 2.0], assigned, &mut nodes)?;
    if mean_mu_input_vid != imgs_vid {
        img_norm_fail!("mean mu input mismatch");
    }

    let input_info = graph.value(imgs_vid)?;
    match &input_info.shape {
        ShapeInfo::Tensor(dims) if dims.len() >= 2 => {}
        ShapeInfo::Unknown => {}
        other => {
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "image_normalize: node {pow_node_id:?} input shape {:?}",
                    other
                );
            }
            img_norm_fail!("input not 3-d tensor");
        }
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
        let values = vec![
            // Value 0: input tensor
            ValueInfo {
                id: 0,
                origin: ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
                constant: None,
            },
            // Node 0 output value (value id 1)
            ValueInfo {
                id: 1,
                origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
                constant: None,
            },
            // Node 1 output value (value id 2)
            ValueInfo {
                id: 2,
                origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
                constant: None,
            },
        ];

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
            node_bindings: StdHashMap::new(),
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
        let values = vec![
            ValueInfo {
                id: 0,
                origin: ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4)]),
                constant: None,
            },
            ValueInfo {
                id: 1,
                origin: ValueOrigin::Constant,
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4)]),
                constant: Some(Value::Num(1.0)),
            },
            ValueInfo {
                id: 2,
                origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4)]),
                constant: None,
            },
            ValueInfo {
                id: 3,
                origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4)]),
                constant: None,
            },
        ];

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
            node_bindings: StdHashMap::new(),
        };

        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let group_plan = &plan.groups[0];
        assert_eq!(group_plan.inputs.len(), 2);
        assert!(group_plan.stack_pattern.is_empty());
        assert!(group_plan.constants.contains_key(&1));
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

        let single = super::builtin_expr("single", &[0], &exprs, "f32");
        assert_eq!(single.unwrap(), "v0");

        let double = super::builtin_expr("double", &[0], &exprs, "f64");
        assert_eq!(double.unwrap(), "v0");
    }

    #[test]
    fn fanout_chain_with_casts_supported() {
        let values = vec![
            // Base input tensor
            ValueInfo {
                id: 0,
                origin: ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(8)]),
                constant: None,
            },
            // tanh(x) output
            ValueInfo {
                id: 1,
                origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(8)]),
                constant: None,
            },
            // constant scale before casting
            ValueInfo {
                id: 2,
                origin: ValueOrigin::Constant,
                ty: Type::Num,
                shape: ShapeInfo::Scalar,
                constant: Some(Value::Num(0.1)),
            },
            // single(0.1) output
            ValueInfo {
                id: 3,
                origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
                ty: Type::Num,
                shape: ShapeInfo::Scalar,
                constant: None,
            },
            // scaled branch output
            ValueInfo {
                id: 4,
                origin: ValueOrigin::NodeOutput { node: 2, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(8)]),
                constant: None,
            },
            // final add output
            ValueInfo {
                id: 5,
                origin: ValueOrigin::NodeOutput { node: 3, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(8)]),
                constant: None,
            },
        ];

        let tanh_node = AccelNode {
            id: 0,
            label: AccelNodeLabel::Builtin {
                name: "tanh".to_string(),
            },
            category: AccelOpCategory::Elementwise,
            inputs: vec![0],
            outputs: vec![1],
            span: InstrSpan { start: 10, end: 10 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let single_node = AccelNode {
            id: 1,
            label: AccelNodeLabel::Builtin {
                name: "single".to_string(),
            },
            category: AccelOpCategory::Elementwise,
            inputs: vec![2],
            outputs: vec![3],
            span: InstrSpan { start: 11, end: 11 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let mul_node = AccelNode {
            id: 2,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![3, 0],
            outputs: vec![4],
            span: InstrSpan { start: 12, end: 12 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let add_node = AccelNode {
            id: 3,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Add),
            category: AccelOpCategory::Elementwise,
            inputs: vec![1, 4],
            outputs: vec![5],
            span: InstrSpan { start: 13, end: 13 },
            tags: vec![AccelGraphTag::Elementwise],
        };

        let graph = AccelGraph {
            nodes: vec![tanh_node, single_node, mul_node, add_node],
            values,
            var_bindings: StdHashMap::new(),
            node_bindings: StdHashMap::new(),
        };

        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);

        let plan = FusionPlan::from_graph(&graph, &groups);
        let group_plan = &plan.groups[0];
        assert!(group_plan.kernel.supported);
        let shader = group_plan.generate_wgsl("f32");
        assert!(shader
            .as_ref()
            .map(|wgsl| wgsl.contains("tanh") && wgsl.contains("output.data"))
            .unwrap_or(false));
    }
}
