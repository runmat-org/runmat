use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::OnceLock;

use runmat_accelerate_api::ReductionFlavor;
use runmat_builtins::{
    builtin_catalog_entry_by_name, MeanControl, MeanOutputClass, MEAN_CATALOG_ENTRY,
    SINGLE_CATALOG_ENTRY,
};
use runmat_runtime::builtins::common::tensor::tensor_element_len;
use runmat_types::OperatorKind;
use runmat_value::Value;
use serde::{Deserialize, Serialize};

use crate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, FusionCallOrigin, InstrSpan, NodeId,
    PrimitiveOp, ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding,
};
use crate::reduction_meta::{detect_reduction_signature, ReductionAxes, ReductionBehavior};
use runmat_accelerate_api::CovNormalization;

mod expression;
mod wgsl_source;
use wgsl_source::WgslShaderFragments;
mod scope;
pub use scope::{scope_fusion_plan, FusionPlanFuture};
mod reduction_group_nodes;
mod reduction_shader;
pub(crate) use reduction_shader::GeneratedReductionShader;
mod admission;
pub mod image_reduction;
mod real_domain;
pub(crate) mod reduction_emission;
pub mod reduction_recipe;
pub(crate) mod semantic_bindings;
pub(crate) mod semantics;
pub use semantics::exact_source_plan_identity;
pub(crate) mod typed_producer;
pub(crate) mod typed_shader;
mod work;

/// Fusion plans can contain `Value` constants, including GC handles. Keep them
/// thread-confined so the type system does not imply cross-thread GC safety.
pub type FusionPlanRef = Rc<FusionPlan>;

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
    #[serde(default)]
    pub stack_layout: Option<FusionStackLayout>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FusionStackLayout {
    pub required_stack_operands: usize,
    pub bindings: Vec<FusionStackValueBinding>,
    /// Only instruction-analysis Variable occurrences may authorize a
    /// resident host readback. Ordinal stack-pattern merges never add here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compiler_variable_bindings: Vec<FusionStackValueBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FusionStackValueBinding {
    pub value_id: ValueId,
    pub stack_offset: usize,
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
    #[serde(default)]
    pub clamp_zero: bool,
    /// A compiler-folded zero held in a live binding, rather than an
    /// unbound source literal. Execution must check the actual bound value.
    #[serde(default)]
    pub clamp_zero_value: Option<ValueId>,
    /// Digests of folded mean-axis constructors that were published into
    /// live bindings. Their actual carriers remain request inputs.
    #[serde(default)]
    pub bound_mean_axes: Vec<(ValueId, runmat_execution::Digest)>,
    /// Tensor `single` conversions in the matched chain are redundant only
    /// when its runtime input retains native single storage throughout.
    #[serde(default)]
    pub requires_single_input: bool,
    /// Exact source reduction form for a future staged execution certificate.
    /// Older plans and structurally matched but unproven routes retain `None`.
    #[serde(default)]
    pub reduction_routes: Option<Box<image_reduction::ImageReductionRoutes>>,
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
        if matches!(current_shape, ShapeInfo::Unknown | ShapeInfo::Scalar) {
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
                stack_layout: None,
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
            stack_layout: None,
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
            if !next.is_elementwise() && !matmul_boundary::is_declared_epilogue(next) {
                break;
            }
            // Allow only primitive elementwise ops we can fold: add/sub/mul and elementwise divide
            let allowed = matmul_boundary::is_declared_epilogue(next)
                || matches!(
                    next.label,
                    AccelNodeLabel::Primitive(PrimitiveOp::Add)
                        | AccelNodeLabel::Primitive(PrimitiveOp::Sub)
                        | AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemMul)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)
                        | AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)
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
                stack_layout: None,
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
        AccelNodeLabel::Builtin { name, .. }
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
    let Some(constant) = source_literal_constant(graph, vid) else {
        return false;
    };
    match constant {
        // Native integer tensors may intentionally have no f64 mirror.  Their
        // authoritative storage (and thus their emptiness) is the typed buffer.
        Value::Tensor(t) => tensor_element_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data().is_empty(),
        Value::CharArray(ca) => ca.data().is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(_) => false,
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

fn merge_stack_layout_with_stack_pattern(
    existing: Option<&FusionStackLayout>,
    inputs: &[ValueId],
    stack_pattern: &[usize],
) -> Option<FusionStackLayout> {
    if existing.is_none() && stack_pattern.is_empty() {
        return None;
    }

    let mut bindings = existing
        .map(|layout| layout.bindings.clone())
        .unwrap_or_default();
    for (stack_offset, &input_idx) in stack_pattern.iter().enumerate() {
        let &value_id = inputs.get(input_idx)?;
        if bindings.iter().any(|binding| binding.value_id == value_id) {
            continue;
        }
        bindings.push(FusionStackValueBinding {
            value_id,
            stack_offset,
        });
    }

    let required_stack_operands = existing
        .map(|layout| layout.required_stack_operands)
        .unwrap_or(0)
        .max(stack_pattern.len());

    Some(FusionStackLayout {
        required_stack_operands,
        bindings,
        compiler_variable_bindings: existing
            .map(|layout| layout.compiler_variable_bindings.clone())
            .unwrap_or_default(),
    })
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
    pub materialized_stores: Vec<FusionStoreMaterialization>,
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
pub struct FusionStoreMaterialization {
    pub value_id: ValueId,
    pub binding: VarBinding,
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
        origin: crate::graph::FusionCallOrigin,
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
    plan: FusionPlanRef,
    active_group: Option<usize>,
}

thread_local! {
    static PLAN_CACHE: RefCell<HashMap<u64, Weak<FusionPlan>>> = RefCell::new(HashMap::new());
    static ACTIVE_PLAN: RefCell<Option<ActiveContext>> = const { RefCell::new(None) };
}

fn with_plan_cache<R>(f: impl FnOnce(&mut HashMap<u64, Weak<FusionPlan>>) -> R) -> R {
    PLAN_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

fn with_active_context<R>(f: impl FnOnce(&mut Option<ActiveContext>) -> R) -> R {
    ACTIVE_PLAN.with(|ctx| {
        let mut slot = ctx.borrow_mut();
        f(&mut slot)
    })
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
    candidate_group_count: usize,
) -> Option<FusionPlanRef> {
    let graph = graph?;
    if candidate_group_count == 0 {
        if !groups.is_empty() && fusion_debug_enabled() {
            log::debug!(
                "fusion plan preparation: executable bytecode fusion groups present ({}) but semantic candidate groups are absent",
                groups.len()
            );
        }
        return None;
    }
    if groups.is_empty() {
        if candidate_group_count > 0 && fusion_debug_enabled() {
            log::debug!(
                "fusion plan preparation: semantic candidate groups present ({}) but executable bytecode fusion groups are empty",
                candidate_group_count
            );
        }
        return None;
    }
    let groups = sanitize_runtime_groups(graph, groups);
    if groups.is_empty() {
        if fusion_debug_enabled() {
            log::debug!(
                "fusion plan preparation: semantic-gated bytecode groups could not be reconciled against runtime accel graph nodes"
            );
        }
        return None;
    }
    let key = fusion_plan_cache_key(graph, &groups, candidate_group_count);
    if let Some(plan) = with_plan_cache(|cache| cache.get(&key).and_then(|weak| weak.upgrade())) {
        return Some(plan);
    }

    let plan = FusionPlan::from_graph(graph, &groups);
    let plan = Rc::new(plan);
    with_plan_cache(|cache| {
        cache.insert(key, Rc::downgrade(&plan));
    });
    Some(plan)
}

fn fusion_plan_cache_key(
    graph: &AccelGraph,
    groups: &[FusionGroup],
    candidate_group_count: usize,
) -> u64 {
    let mut text = String::new();
    let _ = write!(
        &mut text,
        "candidates={candidate_group_count};graph={graph:?};groups={groups:?}"
    );
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

mod matmul_boundary;

fn sanitize_runtime_groups(graph: &AccelGraph, groups: &[FusionGroup]) -> Vec<FusionGroup> {
    groups
        .iter()
        .filter_map(|group| {
            if group.kind == FusionKind::Reduction {
                let mut checked = group.clone();
                checked.nodes = reduction_group_nodes::checked(graph, group)?;
                return Some(checked);
            }
            let had_explicit_mapped_nodes = !group.nodes.is_empty();
            let mut sanitized = group.clone();
            sanitized.nodes.retain(|id| {
                graph
                    .node(*id)
                    .map(|node| {
                        node_matches_runtime_group_kind(graph, node, &sanitized.kind)
                            && node_within_group_span(node, &sanitized.span)
                    })
                    .unwrap_or(false)
            });
            if sanitized.nodes.is_empty() && !had_explicit_mapped_nodes {
                sanitized.nodes = graph
                    .nodes
                    .iter()
                    .filter(|node| {
                        node_matches_runtime_group_kind(graph, node, &sanitized.kind)
                            && node_within_group_span(node, &sanitized.span)
                    })
                    .map(|node| node.id)
                    .collect();
            }
            sanitized.nodes.sort_unstable_by_key(|node_id| {
                graph
                    .node(*node_id)
                    .map(|node| (node.span.start, node.span.end, node.id))
                    .unwrap_or((usize::MAX, usize::MAX, *node_id))
            });
            sanitized.nodes.dedup();
            if sanitized.nodes.is_empty()
                || (sanitized.kind == FusionKind::MatmulEpilogue
                    && !matmul_boundary::preserves_observable_values(graph, &sanitized))
            {
                None
            } else {
                Some(sanitized)
            }
        })
        .collect()
}

fn node_matches_runtime_group_kind(
    graph: &AccelGraph,
    node: &AccelNode,
    kind: &FusionKind,
) -> bool {
    match kind {
        FusionKind::ElementwiseChain => {
            node.is_elementwise()
                || node.category == AccelOpCategory::Transpose
                || is_elementwise_max_min(graph, node)
        }
        FusionKind::Reduction => node.is_reduction(),
        FusionKind::MatmulEpilogue => {
            node.category == AccelOpCategory::MatMul
                || node.is_elementwise()
                || node.category == AccelOpCategory::Transpose
                || matmul_boundary::is_declared_epilogue(node)
        }
        FusionKind::CenteredGram
        | FusionKind::ImageNormalize
        | FusionKind::PowerStepNormalize
        | FusionKind::ExplainedVariance => true,
    }
}

fn node_within_group_span(node: &AccelNode, span: &InstrSpan) -> bool {
    node.span.start >= span.start && node.span.end <= span.end
}

pub fn activate_fusion_plan(plan: Option<FusionPlanRef>) {
    with_active_context(|slot| {
        *slot = plan.map(|plan| ActiveContext {
            plan,
            active_group: None,
        });
    });
}

pub fn deactivate_fusion_plan() {
    with_active_context(|slot| {
        slot.take();
    });
}

pub fn set_current_pc(pc: usize) {
    with_active_context(|slot| {
        if let Some(context) = slot.as_mut() {
            context.active_group = context.plan.group_for_pc(pc);
        }
    });
}

pub fn active_fusion() -> Option<ActiveFusion> {
    with_active_context(|slot| {
        slot.as_ref()
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
    with_active_context(|slot| {
        slot.as_ref().and_then(|context| {
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

    /// Select from this retained plan, never ambient active-plan TLS.
    pub fn group_for_pc_clone(&self, pc: usize) -> Option<FusionGroupPlan> {
        let index = self.group_for_pc(pc)?;
        self.groups.get(index).cloned()
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
    log::trace!(
        "fusion plan {} {} stack_pattern={:?} meta={:?}",
        plan.index,
        stage,
        plan.stack_pattern,
        pattern_meta
    );
}

impl FusionGroupPlan {
    /// Float WGSL fusion kernels cannot consume the provider's native integer
    /// buffers.  Keep this check on the plan as well as at execution time: a
    /// constant otherwise becomes an f32/f64 WGSL literal before the executor
    /// gets a chance to reject the fusion.
    pub fn has_native_integer_constants(&self) -> bool {
        self.const_values
            .values()
            .chain(self.constants.values())
            .any(|value| {
                matches!(value, Value::Int(_))
                    || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
            })
    }

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
                        ValueOrigin::Constant if binding.is_none() => {
                            (true, false, info.constant.clone())
                        }
                        ValueOrigin::Constant => (true, true, None),
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
                        log::trace!(
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
                            log::trace!(
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
                AccelNodeLabel::Builtin { name, origin } => FusionOp::Builtin {
                    name: name.clone(),
                    origin: *origin,
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
            materialized_stores: Vec::new(),
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
                    if let Some(cv) = source_literal_constant(graph, inp) {
                        plan.const_values.insert(inp, cv.clone());
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
                            ..
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
                    // An external reduction-data value was computed before
                    // this group's entry. Its source operands may already
                    // have been consumed by the VM; only the materialized
                    // value has an entry-stack certificate. Traverse a data
                    // producer only when it is retained inside this group.
                    if original_inputs.contains(&cur)
                        && (cur != data_vid || !prod.contains_key(&cur))
                    {
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
                            // Builtin producers, including conversions, are
                            // materialized dependencies unless their operation
                            // is retained explicitly. A cast is not an identity:
                            // single -> double still contains Single rounding.
                            AccelNodeLabel::Builtin { .. } => {}
                            _ => {
                                // Unknown producer: treat as leaf
                            }
                        }
                    }
                }
                // Only genuinely external direct parents are materialized
                // inputs. An in-group producer output is a computed DAG value;
                // adding it as a leaf truncates typed producer traversal and
                // makes the VM gather an intermediate that was not published.
                if let Some(parents) = prod.get(&data_vid) {
                    for &p in parents {
                        if !prod.contains_key(&p) && !added_nodes.contains(&p) && !deps.contains(&p)
                        {
                            // Skip trivial constants embedded in const_values; those are handled separately
                            let is_const = plan.const_values.contains_key(&p)
                                || source_literal_constant(graph, p).is_some();
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
                        if let Some(cv) = source_literal_constant(graph, *vid) {
                            plan.const_values.insert(*vid, cv.clone());
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
                    if let Some(value) = source_literal_constant(graph, *vid) {
                        new_constants.insert(idx, value.clone());
                    }
                }
                plan.constants = new_constants;

                if new_stack_pattern.is_empty() {
                    for (idx, vid) in plan.inputs.iter().enumerate() {
                        if plan.constants.contains_key(&idx) {
                            continue;
                        }
                        if graph.var_binding(*vid).is_some()
                            || graph.value(*vid).is_some_and(|info| {
                                matches!(&info.origin, ValueOrigin::NodeOutput { node, .. }
                                    if graph.node_binding(*node).is_some())
                            })
                        {
                            // Published values are recovered from their exact
                            // binding, not from an imagined stack operand.
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
                source_literal_constant(graph, *vid).is_none()
                    && !plan.const_values.contains_key(vid)
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
        // - Elementwise: retain declared candidates; runtime facts and domain
        //   admission determine the actual typed shader before any upload.
        // - Reduction: retain structurally eligible typed producers; captured
        //   constant words and numerical admission are invocation-owned.
        // - Other kinds: executed via provider paths.
        let supported = if plan.has_native_integer_constants() {
            // Fusion WGSL has only floating scalar literals today.  Do not let
            // integer constants reach generation, where they would be rounded
            // while formatting an f32/f64 literal.
            false
        } else if plan.kernel.kind == FusionKind::CenteredGram {
            // Typed recognition is not an implementation proof. The public
            // covariance algorithm is not the authored centered product.
            false
        } else if plan.kernel.kind.is_elementwise() {
            // Keep scalar ops on the VM/runtime scalar path. Fusing scalar elementwise
            // spans can materialize scalar GPU handles that later leak into scalar-only
            // VM coercion boundaries.
            if scalar_shape_known_one(&plan.group.shape) {
                false
            } else {
                expression::structural_support(&plan)
            }
        } else if plan.kernel.kind.is_reduction() {
            reduction_emission::structural_support(&plan)
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

        if !plan.stack_pattern.is_empty() || plan.group.stack_layout.is_some() {
            plan.group.stack_layout = merge_stack_layout_with_stack_pattern(
                plan.group.stack_layout.as_ref(),
                &plan.inputs,
                &plan.stack_pattern,
            );
        }

        if plan.group.kind.is_elementwise() {
            let mut stores = Vec::new();
            for op in &plan.operations {
                let output = match op {
                    FusionOp::Primitive { output, .. } => *output,
                    FusionOp::Builtin { output, .. } => *output,
                };
                let Some(value_id) = output else {
                    continue;
                };
                let Some(binding) = graph.var_binding(value_id).cloned() else {
                    continue;
                };
                stores.push(FusionStoreMaterialization { value_id, binding });
            }
            plan.materialized_stores = stores;
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
        if self.has_native_integer_constants() {
            return None;
        }
        self.generate_wgsl_for_output(self.output?, scalar_ty)
    }

    /// Build the complete WGSL elementwise shader.
    ///
    /// The caller is responsible for the output-specific parts:
    /// - `output_bindings`: one `@group(0) @binding(…) var<storage, read_write> …` line per output.
    /// - `params_binding_idx`: the binding index for the uniform `Params` block
    ///   (`inputs.len() + num_outputs`).
    /// - `body`: the sequence of `let tmpN` assignment statements.
    /// - `final_writes`: the `output….data[g] = …;` store statements.
    fn build_wgsl_shader(
        &self,
        scalar_ty: &str,
        output_bindings: &str,
        params_binding_idx: usize,
        body: &str,
        final_writes: &str,
        include_ieee_helpers: bool,
    ) -> String {
        self.build_wgsl_shader_with_inputs(
            (self.inputs.len(), scalar_ty, scalar_ty),
            output_bindings,
            params_binding_idx,
            body,
            final_writes,
            include_ieee_helpers,
        )
    }

    fn build_wgsl_shader_with_inputs(
        &self,
        input_abi: (usize, &str, &str),
        output_bindings: &str,
        params_binding_idx: usize,
        body: &str,
        final_writes: &str,
        include_ieee_helpers: bool,
    ) -> String {
        let mut shader = String::new();
        self.append_wgsl_shader_with_inputs(
            &mut shader,
            input_abi,
            WgslShaderFragments {
                bindings: output_bindings,
                body,
                final_writes,
            },
            params_binding_idx,
            include_ieee_helpers,
        );
        shader
    }

    fn append_wgsl_shader_with_inputs(
        &self,
        shader: &mut String,
        input_abi: (usize, &str, &str),
        fragments: WgslShaderFragments<&str, &str, &str>,
        params_binding_idx: usize,
        include_ieee_helpers: bool,
    ) {
        let generated_from = shader.len();
        self.write_wgsl_shader_with_inputs(
            shader,
            input_abi,
            fragments,
            params_binding_idx,
            include_ieee_helpers,
        )
        .expect("String WGSL emission is infallible");
        if input_abi.1 == "f64" && include_ieee_helpers {
            crate::wgsl::ieee_words::append_if_missing(shader, generated_from);
        }
    }

    fn measured_wgsl_shader_bytes_with_inputs(
        &self,
        preamble_bytes: usize,
        input_abi: (usize, &str, &str),
        fragments: WgslShaderFragments<&str, &str, &str>,
        params_binding_idx: usize,
        include_ieee_helpers: bool,
    ) -> Option<usize> {
        let mut counter = wgsl_source::CountedWgslSource::with_preamble_bytes(preamble_bytes);
        self.write_wgsl_shader_with_inputs(
            &mut counter,
            input_abi,
            fragments,
            params_binding_idx,
            include_ieee_helpers,
        )
        .ok()?;
        if input_abi.1 == "f64" && include_ieee_helpers && !counter.has_word_helper() {
            counter
                .bytes()
                .checked_add(crate::wgsl::ieee_words::appended_bytes())
        } else {
            Some(counter.bytes())
        }
    }

    /// The same shell writer counts the exact fixed WGSL and indexed input
    /// geometry before producer fragments exist. Each fragment has its own
    /// checked byte authority; reserving the optional IEEE helper is
    /// conservative when a generated fragment already includes its marker.
    fn measured_wgsl_shell_bytes_with_inputs(
        &self,
        input_abi: (usize, &str, &str),
        output_binding_bytes: usize,
        params_binding_idx: usize,
        body_bytes: usize,
        final_write_bytes: usize,
        include_ieee_helpers: bool,
    ) -> Option<usize> {
        use wgsl_source::MeasuredFragment;
        let mut counter = wgsl_source::CountedWgslSource::with_preamble_bytes(0);
        self.write_wgsl_shader_with_inputs(
            &mut counter,
            input_abi,
            WgslShaderFragments {
                bindings: MeasuredFragment(output_binding_bytes),
                body: MeasuredFragment(body_bytes),
                final_writes: MeasuredFragment(final_write_bytes),
            },
            params_binding_idx,
            include_ieee_helpers,
        )
        .ok()?;
        if input_abi.1 == "f64" && include_ieee_helpers {
            counter
                .bytes()
                .checked_add(crate::wgsl::ieee_words::appended_bytes())
        } else {
            Some(counter.bytes())
        }
    }

    fn write_wgsl_shader_with_inputs<W: std::fmt::Write>(
        &self,
        shader: &mut W,
        input_abi: (usize, &str, &str),
        fragments: WgslShaderFragments<
            impl std::fmt::Display,
            impl std::fmt::Display,
            impl std::fmt::Display,
        >,
        params_binding_idx: usize,
        include_ieee_helpers: bool,
    ) -> std::fmt::Result {
        let (input_count, scalar_ty, storage_ty) = input_abi;
        let WgslShaderFragments {
            bindings: output_bindings,
            body,
            final_writes,
        } = fragments;

        // ── type definitions ──────────────────────────────────────────────────
        shader.write_str("const MAX_RANK: u32 = 128u;\n")?;
        shader.write_str(
            "struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };\n",
        )?;
        shader.write_str("alias PackedArray = array<PackedValue, MAX_RANK>;\n\n")?;
        writeln!(shader, "struct Tensor {{ data: array<{storage_ty}>, }};")?;

        // Broadcast-aware Params: len, offset, rank, pad, out_shape and per-input shape/stride
        shader.write_str("struct Params {\n    len: u32,\n    offset: u32,\n    rank: u32,\n    _pad: u32,\n    out_shape: PackedArray,\n",)?;
        for idx in 0..input_count {
            writeln!(shader, "    in{}_shape: PackedArray,", idx)?;
            writeln!(shader, "    in{}_stride: PackedArray,", idx)?;
        }
        shader.write_str("}\n\n")?;

        // ── portable helper stubs ─────────────────────────────────────────────
        // Avoid relying on backend builtins that may be missing.
        // hypot is not a WGSL builtin; define it explicitly.
        // Use the scaling form max*sqrt(1+(min/max)²) to avoid overflow when
        // a² or b² exceeds the representable range.
        // Match MATLAB's NaN precedence before guarding against Inf/Inf in the
        // scaled formula.
        if scalar_ty == "f32" {
            shader.write_str("fn isNan(x: f32) -> bool { let bits = bitcast<u32>(x); return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u; }\n")?;
            shader.write_str("fn isFinite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }\n")?;
            shader.write_str("fn isInf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }\n")?;
            shader.write_str(concat!(
                "fn hypot(a: f32, b: f32) -> f32 {\n",
                "    if isNan(a) || isNan(b) { return a + b; }\n",
                "    let lo = min(abs(a), abs(b));\n",
                "    let hi = max(abs(a), abs(b));\n",
                "    if hi == 0.0 { return 0.0; }\n",
                "    if isInf(hi) { return hi; }\n",
                "    let r = lo / hi;\n",
                "    return hi * sqrt(1.0 + r * r);\n",
                "}\n\n",
            ))?;
        } else if include_ieee_helpers {
            shader.write_str("fn isNan(x: f64) -> bool { let low = runmat_f64_low(x); let high = runmat_f64_high(x); return (high & 0x7ff00000u) == 0x7ff00000u && ((high & 0x000fffffu) != 0u || low != 0u); }\n")?;
            shader.write_str("fn isFinite(x: f64) -> bool { return (runmat_f64_high(x) & 0x7ff00000u) != 0x7ff00000u; }\n")?;
            shader.write_str("fn isInf(x: f64) -> bool { return (runmat_f64_high(x) & 0x7fffffffu) == 0x7ff00000u && runmat_f64_low(x) == 0u; }\n")?;
            shader.write_str(concat!(
                "fn hypot(a: f64, b: f64) -> f64 {\n",
                "    if isNan(a) || isNan(b) { return a + b; }\n",
                "    let lo = min(abs(a), abs(b));\n",
                "    let hi = max(abs(a), abs(b));\n",
                "    if hi == f64(0.0) { return f64(0.0); }\n",
                "    if isInf(hi) { return hi; }\n",
                "    let r = lo / hi;\n",
                "    return hi * sqrt(f64(1.0) + r * r);\n",
                "}\n\n",
            ))?;
        }

        // ── resource bindings ─────────────────────────────────────────────────
        for idx in 0..input_count {
            writeln!(
                shader,
                "@group(0) @binding({idx}) var<storage, read> input{idx}: Tensor;",
            )?;
        }
        write!(shader, "{output_bindings}")?;
        write!(
            shader,
            "@group(0) @binding({params_binding_idx}) var<uniform> params: Params;\n\n",
        )?;

        // ── compute entry point ───────────────────────────────────────────────
        shader.write_str("@compute @workgroup_size(@WG@)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n",)?;
        shader.write_str("    let idx = gid.x;\n    if (idx >= params.len) { return; }\n")?;
        shader.write_str("    let g = idx + params.offset;\n")?;

        // Compute N-D coordinates from global index (with chunk offset)
        shader.write_str("    var coord: array<u32, MAX_RANK>;\n    var tmp: u32 = g;\n    var d: u32 = 0u;\n    loop { if d >= params.rank { break; } let dim = params.out_shape[d].value; if dim == 0u { coord[d] = 0u; } else { coord[d] = tmp % dim; tmp = tmp / dim; } d = d + 1u; }\n",)?;

        // Compute broadcasted flat indices per input
        for idx in 0..input_count {
            writeln!(shader,
                "    var i{idx}: u32 = 0u; d = 0u; loop {{ if d >= params.rank {{ break; }} let sd = params.in{idx}_shape[d].value; let st = params.in{idx}_stride[d].value; let c = select(coord[d], 0u, sd == 1u); i{idx} = i{idx} + c * st; d = d + 1u; }}",
            )?;
        }

        write!(shader, "{body}")?;
        write!(shader, "{final_writes}")?;
        shader.write_str("}\n")?;
        Ok(())
    }

    /// Generate a single WGSL shader that writes all `output_ids` in one compute pass,
    /// eliminating the O(N²) redundant dispatches that arise from calling
    /// `generate_wgsl_for_output` N times.
    ///
    /// Binding layout:
    ///   0 .. inputs.len()-1          → read-only input tensors
    ///   inputs.len() + k             → read_write output tensor k
    ///   inputs.len() + output_ids.len() → uniform Params
    pub fn generate_wgsl_for_outputs(
        &self,
        output_ids: &[ValueId],
        scalar_ty: &str,
    ) -> Option<String> {
        if self.has_native_integer_constants() {
            return None;
        }
        if output_ids.is_empty() {
            return None;
        }
        if output_ids.len() == 1 {
            return self.generate_wgsl_for_output(output_ids[0], scalar_ty);
        }
        if !self.kernel.kind.is_elementwise() {
            return None;
        }
        if !self.kernel.supported {
            return None;
        }

        let mut exprs: HashMap<ValueId, String> = HashMap::new();
        for (idx, input_id) in self.inputs.iter().enumerate() {
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
                    ..
                } => {
                    let expr = builtin_expr(name, inputs, &exprs, scalar_ty)?;
                    body.push_str(&format!("    let {tmp_name}: {scalar_ty} = {expr};\n"));
                    if let Some(out) = output {
                        exprs.insert(*out, tmp_name.clone());
                    }
                }
            }
        }

        let mut final_exprs = Vec::with_capacity(output_ids.len());
        for output_id in output_ids {
            final_exprs.push(exprs.get(output_id)?.clone());
        }

        let num_outputs = output_ids.len();
        let n_inputs = self.inputs.len();

        let mut output_bindings = String::new();
        for k in 0..num_outputs {
            output_bindings.push_str(&format!(
                "@group(0) @binding({}) var<storage, read_write> output{k}: Tensor;\n",
                n_inputs + k,
            ));
        }

        let mut final_writes = String::new();
        for (k, expr) in final_exprs.iter().enumerate() {
            final_writes.push_str(&format!("    output{k}.data[g] = {expr};\n"));
        }

        Some(self.build_wgsl_shader(
            scalar_ty,
            &output_bindings,
            n_inputs + num_outputs,
            &body,
            &final_writes,
            true,
        ))
    }

    pub fn generate_wgsl_for_output(&self, output_id: ValueId, scalar_ty: &str) -> Option<String> {
        if self.has_native_integer_constants() {
            return None;
        }
        if !self.kernel.kind.is_elementwise() {
            return None;
        }
        if !self.kernel.supported {
            return None;
        }

        let mut exprs: HashMap<ValueId, String> = HashMap::new();
        for (idx, input_id) in self.inputs.iter().enumerate() {
            // Placeholder; will be resolved to the broadcasted index variable i{idx}
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
                    ..
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
        let n_inputs = self.inputs.len();

        let output_bindings =
            format!("@group(0) @binding({n_inputs}) var<storage, read_write> output: Tensor;\n",);
        let final_writes = format!("    output.data[g] = {final_expr};\n");

        Some(self.build_wgsl_shader(
            scalar_ty,
            &output_bindings,
            n_inputs + 1,
            &body,
            &final_writes,
            true,
        ))
    }

    pub(crate) fn legacy_reduction_axis(&self) -> Option<usize> {
        let reduce_all = self
            .constants
            .values()
            .chain(self.const_values.values())
            .any(|value| matches!(value, Value::String(text) if text.as_str().is_some_and(|text| text.eq_ignore_ascii_case("all"))));
        if reduce_all {
            return Some(0);
        }
        if let Some(dim_vid) = self.reduction_dim {
            return match self.const_values.get(&dim_vid) {
                Some(Value::Num(dim)) if *dim >= 1.0 => Some((*dim as usize).saturating_sub(1)),
                Some(Value::Int(_)) => None,
                _ => Some(0),
            };
        }
        for value in self.constants.values() {
            match value {
                Value::Num(dim) if *dim >= 1.0 => return Some((*dim as usize).saturating_sub(1)),
                Value::Int(_) => return None,
                _ => {}
            }
        }
        Some(0)
    }

    /// Compatibility source generator; the planner only probes whether it can
    /// generate source. Executable legacy admission supplies the selected
    /// operation's canonical flavor separately.
    pub fn generate_reduction_wgsl(&self, scalar_ty: &str) -> Option<String> {
        self.generate_reduction_wgsl_inner(
            scalar_ty,
            self.reduction_flavor.unwrap_or(ReductionFlavor::Sum),
            None,
        )
    }

    pub(crate) fn prepare_reduction_shader(
        &self,
        precision: runmat_accelerate_api::ProviderPrecision,
        flavor: ReductionFlavor,
        selected: Option<&reduction_emission::ReductionProducerEmission>,
    ) -> Option<GeneratedReductionShader> {
        let scalar_ty = match precision {
            runmat_accelerate_api::ProviderPrecision::F32 => "f32",
            runmat_accelerate_api::ProviderPrecision::F64 => "f64",
        };
        let source = self.generate_reduction_wgsl_inner(scalar_ty, flavor, selected)?;
        Some(GeneratedReductionShader::new(source, precision))
    }

    fn generate_reduction_wgsl_inner(
        &self,
        scalar_ty: &str,
        flavor: ReductionFlavor,
        selected: Option<&reduction_emission::ReductionProducerEmission>,
    ) -> Option<String> {
        if self.has_native_integer_constants() {
            return None;
        }
        if !self.kernel.kind.is_reduction() {
            return None;
        }
        // Minimal column-major reduction kernel template (single workgroup per slice).
        // Supports folding simple producer expressions over multiple inputs (e.g., sum(A.*B, dim)).
        if self.inputs.is_empty() {
            return None;
        }
        let axis = self.legacy_reduction_axis()?;
        // Detect omitnan constant (compile-time selection)
        let omitnan = self.constants.values().any(|v| match v {
            Value::String(s) => s
                .as_str()
                .is_some_and(|text| text.eq_ignore_ascii_case("omitnan")),
            _ => false,
        });

        let data_vid = self.reduction_data?;
        let (val_expr, producer_body, producer_preamble) = if let Some(emission) = selected {
            let precision = match scalar_ty {
                "f32" => runmat_accelerate_api::ProviderPrecision::F32,
                "f64" => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return None,
            };
            if emission.root() != data_vid
                || emission.precision() != precision
                || emission.input_count() != self.inputs.len()
            {
                return None;
            }
            (
                emission.expression().to_owned(),
                emission.body().to_owned(),
                emission.preamble().to_owned(),
            )
        } else {
            // Non-required producers retain the established legacy expression
            // route. A required arithmetic declaration cannot enter here.
            let ext_input = self.inputs[0];
            let mut exprs: HashMap<ValueId, String> = HashMap::new();
            exprs.insert(ext_input, "v".to_string());
            for (idx, &vid) in self.inputs.iter().enumerate().skip(1) {
                exprs.insert(vid, format!("v{idx}"));
            }
            // Captured producer scalars require the sealed typed emitter's
            // represented-word policy. This compatibility path has no literal
            // renderer: unused constants add nothing, invocation leaves win
            // shadowed IDs, and a missing captured dependency cannot execute.
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
                            ..
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
            let val_expr = match exprs.get(&data_vid) {
                Some(expression) => expression.clone(),
                None => {
                    if fusion_debug_enabled() {
                        let expr_keys: Vec<ValueId> = exprs.keys().copied().collect();
                        log::debug!(
                            "fusion reduction WGSL: missing expression for data {:?}; inputs={:?} expr_keys={:?} ops={:?}",
                            data_vid, self.inputs, expr_keys, self.operations
                        );
                    }
                    return None;
                }
            };
            (val_expr, String::new(), String::new())
        };

        let mut shader = String::new();
        shader.push_str(&producer_preamble);
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
        // The executable flavor comes from the selected operation declaration,
        // not the planner's old name-based shape heuristic.
        let is_mean = matches!(flavor, ReductionFlavor::Mean);
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
        // Render a complete typed zero for both axis templates.
        let zero = cast_literal(scalar_ty, "0.0");
        // Helper(s) at module scope
        shader.push_str(&format!(
            "fn isNanF(value: {scalar}) -> bool {{ @IEEE_NAN_PREDICATE@ }}\n",
            scalar = scalar_ty
        ));
        if scalar_ty == "f64" {
            shader.push_str(
                "fn canonicalNan() -> f64 { return runmat_f64_from_words(0u, 0x7ff80000u); }\n\n",
            );
        } else {
            shader.push_str("fn canonicalNan() -> f32 {\n  var bits: u32 = 0x7fc00000u;\n  return bitcast<f32>(bits);\n}\n\n");
        }
        shader.push_str("@compute @workgroup_size(@WG@)\n");
        if axis == 0 {
            // Column-wise: reduce over rows; one output per column (ncols)
            shader.push_str(
                "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
            );
            shader.push_str("  let col = wid.x;\n  if (col >= params.ncols) { return; }\n");
            shader.push_str(&format!("  var acc: {scalar_ty} = {zero};\n"));
            // helpers are declared at module scope
            shader.push_str("  var saw_nan: bool = false;\n  var r = lid.x;\n");
            // Load row-wise values from each input and fold into expression
            {
                // Build the per-iteration loads
                let mut loop_body = String::new();
                // input0 as 'v'
                let first_index = if let Some(emission) = selected {
                    emission.index_expression(0, axis)?
                } else {
                    "(col * params.nrows) + r".to_owned()
                };
                loop_body.push_str(&format!("    let v = input0.data[{first_index}];\n"));
                // additional inputs as v1, v2, ...
                for (idx, _) in self.inputs.iter().enumerate().skip(1) {
                    let index = if let Some(emission) = selected {
                        emission.index_expression(idx, axis)?
                    } else {
                        "(col * params.nrows) + r".to_owned()
                    };
                    loop_body.push_str(&format!("    let v{idx} = input{idx}.data[{index}];\n"));
                }
                loop_body.push_str(&producer_body);
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
            shader.push_str("  if (!OMITNAN && saw_nan) { acc = canonicalNan(); }\n");
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
            shader.push_str(&format!("  var acc: {scalar_ty} = {zero};\n"));
            // helpers are declared at module scope
            shader.push_str("  var saw_nan: bool = false;\n  var c = lid.x;\n");
            {
                let mut loop_body = String::new();
                // input0 as 'v' — provider encodes rows in params.ncols for axis=1
                let first_index = if let Some(emission) = selected {
                    emission.index_expression(0, axis)?
                } else {
                    "row + (c * params.ncols)".to_owned()
                };
                loop_body.push_str(&format!("    let v = input0.data[{first_index}];\n"));
                // additional inputs as v1, v2, ...
                for (idx, _) in self.inputs.iter().enumerate().skip(1) {
                    let index = if let Some(emission) = selected {
                        emission.index_expression(idx, axis)?
                    } else {
                        "row + (c * params.ncols)".to_owned()
                    };
                    loop_body.push_str(&format!("    let v{idx} = input{idx}.data[{index}];\n"));
                }
                loop_body.push_str(&producer_body);
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
            shader.push_str("  if (!OMITNAN && saw_nan) { acc = canonicalNan(); }\n");
            shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
            shader.push_str(
                "  var off = (@WG@u) / 2u;\n  loop { if (off == 0u) { break; } if (lid.x < off) {\n    let a = tile[lid.x]; let b = tile[lid.x + off];\n    tile[lid.x] = a + b;\n  } workgroupBarrier(); off = off / 2u; }\n",
            );
            shader.push_str(&format!(
                "  if (lid.x == 0u) {{ output.data[row] = tile[0u] * {}; }}\n}}\n",
                post_scale
            ));
        }
        Some(crate::wgsl::nan_predicate::inject(
            &shader,
            if scalar_ty == "f64" {
                crate::wgsl::NumericPrecision::F64
            } else {
                crate::wgsl::NumericPrecision::F32
            },
        ))
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
        if div_op != PrimitiveOp::ElemDiv {
            continue;
        }
        if div_node.inputs.len() != 2 {
            continue;
        }
        let (numerator_id, denom_id) = (div_node.inputs[0], div_node.inputs[1]);
        let denom_const = match source_literal_constant(graph, denom_id) {
            Some(Value::Num(v)) => Some(*v),
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
        if !mul_node.is_matrix_product() {
            continue;
        }
        // A centered Gram is EXACTLY centered.' * centered. Merely finding
        // a transpose in either operand cannot justify reordering products or
        // substituting a different live right-hand value.
        let Some((transpose_node_id, centered_val_id)) =
            is_transpose_node(graph, mul_node.inputs[0])
        else {
            continue;
        };
        if mul_node.inputs[1] != centered_val_id || assigned.contains(&transpose_node_id) {
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
        let Some((_, mean_input, axes)) = match_mean_axes(graph, mean_val_id) else {
            continue;
        };
        if mean_input != matrix_val_id || axes.as_slice() != [1.0] {
            continue;
        }

        let matrix_info = match graph.value(matrix_val_id) {
            Some(info) => info,
            None => continue,
        };
        let matrix_rows = match &matrix_info.shape {
            ShapeInfo::Tensor(dims) if dims.len() == 2 => dims[0].unwrap_or(0),
            _ => 0,
        };
        let Some(denominator) = denom_const.filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let normalization = if matrix_rows > 1 && denominator == matrix_rows as f64 - 1.0 {
            CovNormalization::Unbiased
        } else if matrix_rows > 0 && denominator == matrix_rows as f64 {
            CovNormalization::Biased
        } else {
            // Unknown/live or different divisors retain ordinary execution.
            // They are not permission to invent an unbiased normalization.
            continue;
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
            stack_layout: None,
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
            clamp_zero: match_info.clamp_zero,
            clamp_zero_value: match_info.clamp_zero_value,
            bound_mean_axes: match_info.bound_mean_axes.clone(),
            requires_single_input: match_info.requires_single_input,
            reduction_routes: match_info.reduction_routes.clone().map(Box::new),
        };

        groups.push(FusionGroup {
            id: *next_group_id,
            kind: FusionKind::ImageNormalize,
            nodes: match_info.nodes.clone(),
            shape,
            span: span.clone(),
            pattern: Some(FusionPattern::ImageNormalize(pattern)),
            stack_layout: None,
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
        if div_op != PrimitiveOp::ElemDiv {
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
            AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("mtimes") => {}
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
            stack_layout: None,
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
            AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("diag") => {}
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
            AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("mtimes") => {}
            _ => continue,
        }
        if matmul2_node.inputs.len() != 2 {
            continue;
        }

        // Matrix multiplication is ordered. This pattern represents precisely
        // diag((transpose(Q) * G) * Q), not permutations of those operands.
        let Some((matmul1_id, matmul1_node)) = node_from_value(graph, matmul2_node.inputs[0])
        else {
            continue;
        };
        if !matches!(matmul1_node.label, AccelNodeLabel::Builtin { ref name, .. } if name.eq_ignore_ascii_case("mtimes"))
        {
            continue;
        }
        let q_vid = matmul2_node.inputs[1];

        if assigned.contains(&matmul1_id) {
            continue;
        }

        if matmul1_node.inputs.len() != 2 {
            continue;
        }

        let Some((transpose_id, transpose_input_vid)) =
            is_transpose_node(graph, matmul1_node.inputs[0])
        else {
            continue;
        };
        let g_vid = matmul1_node.inputs[1];

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
            stack_layout: None,
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
        AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("sum") => {}
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
        AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("sqrt") => {
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
    // A folded unary output carries its represented value, which may differ
    // from its source literal. Keep the outermost conversion's result while
    // tracing back to an actual constant and accounting for every node.
    let mut represented = None;
    let mut visited: HashSet<NodeId> = HashSet::new();

    loop {
        let info = graph.value(current)?;
        // A constant-shaped fact on a published binding is not a frozen value.
        if graph.var_binding(current).is_some() {
            return None;
        }
        match &info.origin {
            ValueOrigin::Constant => {
                let base = value_info_scalar(info)?;
                return Some(ConstantTrace {
                    value: represented.unwrap_or(sign * base),
                    nodes,
                });
            }
            ValueOrigin::NodeOutput { node, output } => {
                if graph.node_binding(*node).is_some() {
                    return None;
                }
                if !visited.insert(*node) {
                    return None;
                }
                let node_ref = graph.node(*node)?;
                if node_ref.outputs.get(*output) != Some(&current) {
                    return None;
                }
                if node_ref
                    .tags
                    .contains(&crate::graph::AccelGraphTag::UnaryScalarFold)
                {
                    let [input] = node_ref.inputs.as_slice() else {
                        return None;
                    };
                    // The graph builder owns both the conversion and its
                    // represented result. A shape/type hint alone is not a
                    // constant, and native integer mirrors are not evidence.
                    if !matches!(
                        &info.constant,
                        Some(Value::Tensor(tensor))
                            if tensor.shape() == [1, 1]
                                && tensor.as_f32_slice().is_some_and(|values| values.len() == 1)
                    ) {
                        return None;
                    }
                    // Capture only sign changes outside this proved result.
                    // We still trace its source nodes for ownership, but any
                    // inner negation is already included in the folded value.
                    represented.get_or_insert(sign * value_info_scalar(info)?);
                    nodes.push(*node);
                    current = *input;
                    continue;
                }
                match &node_ref.label {
                    AccelNodeLabel::Builtin { name, .. }
                        if name.eq_ignore_ascii_case("gpuarray") =>
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

fn is_canonical_single_cast(node: &AccelNode) -> bool {
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &node.label
    else {
        return false;
    };
    node.inputs.len() == 1
        && builtin_catalog_entry_by_name(name).is_some_and(|entry| {
            entry.identity == SINGLE_CATALOG_ENTRY.identity
                && entry
                    .bindings
                    .iter()
                    .any(|variant| entry.binding_identity(variant) == *binding)
        })
}

fn peel_numeric_casts(
    graph: &AccelGraph,
    mut vid: ValueId,
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<ValueId> {
    loop {
        let info = graph.value(vid)?;
        match &info.origin {
            ValueOrigin::NodeOutput { node, .. } => {
                if assigned.contains(node) {
                    return None;
                }
                let node_ref = graph.node(*node)?;
                if let AccelNodeLabel::Builtin { name, .. } = &node_ref.label {
                    let single_cast = is_canonical_single_cast(node_ref);
                    if name.eq_ignore_ascii_case("gpuarray") || single_cast {
                        if node_ref.inputs.len() != 1 {
                            return None;
                        }
                        if single_cast {
                            let input_shape = &graph.value(node_ref.inputs[0])?.shape;
                            // Mean/elementwise outputs in a compiled vecdim
                            // chain may remain Unknown even though the source
                            // image is a known tensor. A known scalar cannot
                            // be a tensor cast in this path. The complete
                            // image matcher and execution's native-single
                            // input guard own the remaining dynamic proof.
                            if !matches!(&info.shape, ShapeInfo::Tensor(_) | ShapeInfo::Unknown)
                                || !matches!(input_shape, ShapeInfo::Tensor(_) | ShapeInfo::Unknown)
                            {
                                return None;
                            }
                            // The executor checks authoritative input storage before
                            // admitting a chain that passes through this conversion.
                            nodes.push(*node);
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
        // Float fusion pattern matching must not establish eligibility by
        // rounding a native integer scalar.
        Some(Value::Int(_)) => None,
        // Do not establish eligibility for a floating fusion by reading the
        // lossy f64 mirror of a native integer tensor.
        Some(Value::Tensor(t)) if t.integer_storage().is_none() && t.len() == 1 => t
            .numeric_value_at(0)
            .map(runmat_value::NumericScalar::materialize_f64),
        Some(Value::LogicalArray(arr)) if arr.data.len() == 1 => Some(arr.data[0] as f64),
        Some(Value::Bool(flag)) => Some(if *flag { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn value_constant_f64(graph: &AccelGraph, vid: ValueId) -> Option<f64> {
    resolve_scalar_constant(graph, vid)
}

/// A numeric fact attached to a variable or published node is not a frozen
/// input value: the binding may have changed before this plan executes.
fn source_literal_constant(graph: &AccelGraph, vid: ValueId) -> Option<&Value> {
    graph.source_literal_constant(vid)
}

pub(crate) trait ExpressionLookup {
    fn get(&self, id: &ValueId) -> Option<&String>;
}

impl ExpressionLookup for HashMap<ValueId, String> {
    fn get(&self, id: &ValueId) -> Option<&String> {
        HashMap::get(self, id)
    }
}

impl ExpressionLookup for Vec<(ValueId, String)> {
    fn get(&self, id: &ValueId) -> Option<&String> {
        self.binary_search_by_key(id, |(key, _)| *key)
            .ok()
            .map(|index| &self[index].1)
    }
}

fn primitive_expr(
    op: PrimitiveOp,
    inputs: &[ValueId],
    exprs: &impl ExpressionLookup,
) -> Option<String> {
    primitive_expr_checked(op, inputs, exprs).ok().flatten()
}

fn primitive_expr_checked(
    op: PrimitiveOp,
    inputs: &[ValueId],
    exprs: &impl ExpressionLookup,
) -> anyhow::Result<Option<String>> {
    // Every admitted scalar primitive is unary or binary. Borrow the source
    // expressions while rendering; cloning both operands would create a
    // second request-local payload outside the counted producer geometry.
    if inputs.len() > 2 {
        return Ok(None);
    }
    let mut arguments = [""; 2];
    for (slot, input) in arguments.iter_mut().zip(inputs) {
        let Some(expression) = exprs.get(input) else {
            return Ok(None);
        };
        *slot = expression.as_str();
    }
    let arguments = &arguments[..inputs.len()];
    match op {
        PrimitiveOp::Add => scalar_operator_expr(OperatorKind::Add, arguments),
        PrimitiveOp::Sub => scalar_operator_expr(OperatorKind::Subtract, arguments),
        PrimitiveOp::Mul | PrimitiveOp::ElemMul => {
            scalar_operator_expr(OperatorKind::ElementwiseMultiply, arguments)
        }
        PrimitiveOp::ElemDiv => scalar_operator_expr(OperatorKind::ElementwiseDivide, arguments),
        PrimitiveOp::ElemLeftDiv => {
            scalar_operator_expr(OperatorKind::ElementwiseLeftDivide, arguments)
        }
        PrimitiveOp::Pow | PrimitiveOp::ElemPow => {
            let [lhs, rhs] = arguments else {
                return Ok(None);
            };
            if expr_is_literal(rhs, 2.0) {
                return Ok(Some(format!("({lhs} * {lhs})")));
            }
            Ok(Some(format!("pow({lhs}, {rhs})")))
        }
        PrimitiveOp::Neg => scalar_operator_expr(OperatorKind::UnaryMinus, arguments),
        PrimitiveOp::UPlus => scalar_operator_expr(OperatorKind::UnaryPlus, arguments),
        _ => Ok(None),
    }
}

/// One WGSL spelling for both the legacy primitive route and checked scalar
/// programs. Domain admission decides which operations may reach this lowerer.
fn scalar_operator_expr(
    operation: OperatorKind,
    arguments: &[impl std::fmt::Display],
) -> anyhow::Result<Option<String>> {
    let arity = match operation {
        OperatorKind::Add
        | OperatorKind::Subtract
        | OperatorKind::ElementwiseMultiply
        | OperatorKind::ElementwiseDivide
        | OperatorKind::ElementwiseLeftDivide => 2,
        OperatorKind::UnaryMinus | OperatorKind::UnaryPlus => 1,
        _ => return Ok(None),
    };
    if arguments.len() != arity {
        return Ok(None);
    }
    let mut measured = wgsl_source::CountedWgslSource::with_preamble_bytes(0);
    write_scalar_operator_expr(&mut measured, operation, arguments)?;
    let mut expression = String::new();
    expression.try_reserve_exact(measured.bytes())?;
    write_scalar_operator_expr(&mut expression, operation, arguments)?;
    if expression.len() != measured.bytes() {
        return Err(std::fmt::Error.into());
    }
    Ok(Some(expression))
}

/// The operation spelling is shared by legacy emission, checked scalar
/// lowering, and allocation-free geometry measurement.
pub(super) fn write_scalar_operator_expr(
    writer: &mut impl std::fmt::Write,
    operation: OperatorKind,
    arguments: &[impl std::fmt::Display],
) -> std::fmt::Result {
    match (operation, arguments) {
        (OperatorKind::Add, [a, b]) => write!(writer, "({a} + {b})"),
        (OperatorKind::Subtract, [a, b]) => write!(writer, "({a} - {b})"),
        (OperatorKind::ElementwiseMultiply, [a, b]) => write!(writer, "({a} * {b})"),
        (OperatorKind::ElementwiseDivide, [a, b]) => write!(writer, "({a} / {b})"),
        (OperatorKind::ElementwiseLeftDivide, [a, b]) => write!(writer, "({b} / {a})"),
        (OperatorKind::UnaryMinus, [a]) => write!(writer, "(-{a})"),
        // WGSL has no unary `+`; admission already establishes the class.
        (OperatorKind::UnaryPlus, [a]) => write!(writer, "{a}"),
        _ => Err(std::fmt::Error),
    }
}

fn expr_is_literal(expr: &str, expected: f64) -> bool {
    let trimmed = expr.trim();
    let literal = trimmed
        .strip_prefix("f64(")
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(trimmed);
    literal
        .parse::<f64>()
        .map(|value| (value - expected).abs() <= f64::EPSILON)
        .unwrap_or(false)
}

fn builtin_expr(
    name: &str,
    inputs: &[ValueId],
    exprs: &impl ExpressionLookup,
    scalar_ty: &str,
) -> Option<String> {
    if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(name) {
        let scalar_ty = match scalar_ty {
            "f32" => runmat_runtime::builtins::common::spec::ScalarType::F32,
            "f64" => runmat_runtime::builtins::common::spec::ScalarType::F64,
            _ => return None,
        };
        return expression::canonical_expression(entry, inputs, exprs, scalar_ty);
    }
    let func = match name.to_ascii_lowercase().as_str() {
        "isfinite" => return builtin_unary_call("isFinite", inputs, exprs),
        "isinf" => return builtin_unary_call("isInf", inputs, exprs),
        "isnan" => return builtin_unary_call("isNan", inputs, exprs),
        "gpuarray" => return builtin_identity(inputs, exprs),
        "fix" => return builtin_unary_call("trunc", inputs, exprs),
        "sign" => return builtin_unary_call("sign", inputs, exprs),
        "heaviside" => {
            let arg = exprs.get(inputs.first()?).cloned()?;
            let zero = cast_literal(scalar_ty, "0.0");
            let half = cast_literal(scalar_ty, "0.5");
            let one = cast_literal(scalar_ty, "1.0");
            return Some(format!(
                "select(select(select({zero}, {one}, ({arg} > {zero})), {half}, ({arg} == {zero})), {arg}, isNan({arg}))"
            ));
        }
        "mod" => {
            let lhs = exprs.get(inputs.first()?).cloned()?;
            let rhs = exprs.get(inputs.get(1)?).cloned()?;
            // When rhs is infinite and lhs is finite, MATLAB sign-corrects: returns lhs when
            // signs match, rhs (±Inf) when they differ. The general formula produces NaN here
            // (inf * 0 = NaN), so we must short-circuit. A zero divisor is a separate
            // documented convention: mod(lhs, 0) returns lhs.
            return Some(format!(
                "select(select(({lhs} - {rhs} * floor({lhs} / {rhs})), select({rhs}, {lhs}, ({lhs} == 0.0 || sign({lhs}) == sign({rhs}))), (isInf({rhs}) && isFinite({lhs}))), {lhs}, ({rhs} == 0.0))"
            ));
        }
        "rem" => {
            let lhs = exprs.get(inputs.first()?).cloned()?;
            let rhs = exprs.get(inputs.get(1)?).cloned()?;
            return Some(format!(
                "select(({lhs} - {rhs} * trunc({lhs} / {rhs})), {lhs}, (isInf({rhs}) && isFinite({lhs})))"
            ));
        }
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "asin" => "asin",
        "acos" => "acos",
        "atan" => "atan",
        "atan2" => return builtin_binary("atan2", inputs, exprs),
        "hypot" => return builtin_binary("hypot", inputs, exprs),
        "pow2" => {
            if inputs.len() == 1 {
                return builtin_unary_call("exp2", inputs, exprs);
            }
            return None;
        }
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
        "asinh" => return builtin_unary_call("asinh", inputs, exprs),
        "acosh" => return builtin_unary_call("acosh", inputs, exprs),
        "atanh" => return builtin_unary_call("atanh", inputs, exprs),
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

fn builtin_binary(func: &str, inputs: &[ValueId], exprs: &impl ExpressionLookup) -> Option<String> {
    let lhs = exprs.get(inputs.first()?).cloned()?;
    let rhs = exprs.get(inputs.get(1)?).cloned()?;
    Some(format!("{func}({lhs}, {rhs})"))
}

fn builtin_unary_call(
    func: &str,
    inputs: &[ValueId],
    exprs: &impl ExpressionLookup,
) -> Option<String> {
    let arg = exprs.get(inputs.first()?).cloned()?;
    Some(format!("{func}({arg})"))
}

fn builtin_identity(inputs: &[ValueId], exprs: &impl ExpressionLookup) -> Option<String> {
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
) -> Option<(NodeId, ValueId, Option<ValueId>)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    match &node.label {
        AccelNodeLabel::Builtin { .. } if image_reduction::is_canonical_max_node(node) => {
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
                if value.to_bits() == 0.0f64.to_bits() {
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "split_max_with_zero_scalar: rhs zero constant for node {node_id:?}"
                        );
                    }
                    return Some((node_id, lhs, None));
                }
            }
            if let Some(ImageScalar::Constant(value)) =
                capture_image_scalar(graph, lhs, assigned, nodes)
            {
                if value.to_bits() == 0.0f64.to_bits() {
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "split_max_with_zero_scalar: lhs zero constant for node {node_id:?}"
                        );
                    }
                    return Some((node_id, rhs, None));
                }
            }
            // A published compiler fold is not a source literal. It may be
            // recognized structurally, but remains an external live input
            // whose exact bound value is checked before provider effects.
            for (control, base) in [(rhs, lhs), (lhs, rhs)] {
                if image_reduction::bound_folded_zero(graph, control, node.span.start) {
                    return Some((node_id, base, Some(control)));
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
    let constant =
        source_literal_constant(graph, vid).or_else(|| graph.folded_literal_numeric_vector(vid))?;
    match constant {
        // Axis and dimension vectors feed floating shader literals below, so
        // native integer tensors must take the normal unfused path rather than
        // being derived from their f64 mirror.
        Value::Tensor(tensor) if tensor.integer_storage().is_none() && !tensor.is_empty() => {
            Some(tensor.materialize_f64())
        }
        Value::LogicalArray(arr) if !arr.data.is_empty() => Some(
            arr.data
                .iter()
                .map(|v| if *v == 0 { 0.0 } else { 1.0 })
                .collect(),
        ),
        Value::Bool(flag) => Some(vec![if *flag { 1.0 } else { 0.0 }]),
        Value::Int(_) => None,
        Value::Num(num) => Some(vec![*num]),
        _ => None,
    }
}

fn match_mean_axes(graph: &AccelGraph, vid: ValueId) -> Option<(NodeId, ValueId, Vec<f64>)> {
    let (node_id, node) = node_from_value(graph, vid)?;
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &node.label
    else {
        return None;
    };
    let [variant] = MEAN_CATALOG_ENTRY.bindings else {
        return None;
    };
    if !name.eq_ignore_ascii_case(MEAN_CATALOG_ENTRY.identity.name)
        || *binding != MEAN_CATALOG_ENTRY.binding_identity(variant)
    {
        return None;
    }
    if !(2..=3).contains(&node.inputs.len()) {
        return None;
    }
    if let Some(control_vid) = node.inputs.get(2) {
        let control_value = source_literal_constant(graph, *control_vid)?;
        let control = match control_value {
            Value::String(text) => text.as_str().and_then(MeanControl::from_public_option),
            Value::CharArray(chars) if chars.rows == 1 => {
                MeanControl::from_public_code_units(chars.data().iter().map(|unit| unit.raw()))
            }
            _ => None,
        };
        if control != Some(MeanControl::Output(MeanOutputClass::Native)) {
            return None;
        }
    }
    let data_vid = node.inputs[0];
    let dims_vid = node.inputs[1];
    let dims = image_reduction::resolve_selected_mean_axes(graph, dims_vid, node.span.start)?;
    Some((node_id, data_vid, dims))
}

fn dims_match_unordered(found: &[f64], expected: &[f64]) -> bool {
    if found.len() != expected.len() {
        return false;
    }
    let mut a = found.to_vec();
    let mut b = expected.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    a == b
}

struct ImageNormalizeMatch {
    nodes: Vec<NodeId>,
    input: ValueId,
    epsilon: ImageScalar,
    gain: Option<ImageScalar>,
    bias: Option<ImageScalar>,
    gamma: Option<ImageScalar>,
    clamp_zero: bool,
    clamp_zero_value: Option<ValueId>,
    bound_mean_axes: Vec<(ValueId, runmat_execution::Digest)>,
    requires_single_input: bool,
    reduction_routes: Option<image_reduction::ImageReductionRoutes>,
}

fn analyze_image_normalize(
    graph: &AccelGraph,
    pow_node_id: NodeId,
    assigned: &HashSet<NodeId>,
) -> Option<ImageNormalizeMatch> {
    let pow_node = graph.node(pow_node_id)?;
    if log::log_enabled!(log::Level::Trace) {
        log::trace!(
            "image_normalize: inspect pow candidate node={pow_node_id:?} label={:?}",
            pow_node.label
        );
    }
    macro_rules! img_norm_fail {
        ($reason:expr) => {{
            if log::log_enabled!(log::Level::Trace) {
                log::trace!(
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
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("image_normalize: node {pow_node_id:?} gamma scalar={gamma_scalar:?}");
    }
    let gamma_opt = match &gamma_scalar {
        ImageScalar::Constant(value) if approx_eq(*value, 1.0) => None,
        _ => Some(gamma_scalar),
    };

    let mut clamp_nodes = Vec::new();
    let pow_base_vid = peel_numeric_casts(graph, pow_node.inputs[0], assigned, &mut clamp_nodes)?;
    let (clamp_input_vid, clamp_zero, clamp_zero_value) =
        if let Some((clamp_node_id, clamp_input_vid, zero_value)) =
            split_max_with_zero_scalar(graph, pow_base_vid, assigned, &mut clamp_nodes)
        {
            if assigned.contains(&clamp_node_id) {
                img_norm_fail!("clamp node already assigned");
            }
            nodes.extend(clamp_nodes);
            nodes.push(clamp_node_id);
            (clamp_input_vid, true, zero_value)
        } else {
            nodes.extend(clamp_nodes);
            (pow_base_vid, false, None)
        };

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
        AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv) => {}
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
    let variance_observed = mean_sq_vid;
    let mean_sq_vid = peel_numeric_casts(graph, mean_sq_vid, assigned, &mut nodes)?;

    let variance =
        image_reduction::peel_mean_dims(graph, mean_sq_vid, &[3.0, 2.0], assigned, &mut nodes)?;
    let squared_diff_vid = variance.source;

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

    let variance_route = variance
        .route
        .and_then(|route| route.with_cast_path(graph, variance_observed, mean_sq_vid));

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
    let mu_variance_observed = diff_var_node.inputs[1];
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
    let mu_output_observed = diff_node.inputs[1];
    let diff_mu_vid = peel_numeric_casts(graph, diff_node.inputs[1], assigned, &mut nodes)?;
    if diff_node.inputs[0] != imgs_vid || diff_mu_vid != mu_vid {
        img_norm_fail!("diff inputs mismatch with variance pair");
    }
    nodes.push(diff_node_id);

    let mu = image_reduction::peel_mean_dims(graph, mu_vid, &[3.0, 2.0], assigned, &mut nodes)?;
    let mean_mu_input_vid = mu.source;
    if mean_mu_input_vid != imgs_vid {
        img_norm_fail!("mean mu input mismatch");
    }
    let mu_route = mu
        .route
        .and_then(|route| route.with_cast_path(graph, mu_variance_observed, mu_vid))
        .and_then(|route| route.with_cast_path(graph, mu_output_observed, diff_mu_vid));

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

    let requires_single_input = nodes
        .iter()
        .any(|id| graph.node(*id).is_some_and(is_canonical_single_cast));
    nodes.sort_unstable();
    nodes.dedup();

    let reduction_routes =
        mu_route
            .zip(variance_route)
            .map(|(mu, variance)| image_reduction::ImageReductionRoutes {
                mu,
                variance,
                stage_producers: Some(image_reduction::ImageStageProducers {
                    mu: mu_vid,
                    variance: mean_sq_vid,
                    sigma: sigma_vid,
                    pre_clamp_output: clamp_zero.then_some(clamp_input_vid),
                    clamp_output: clamp_zero.then_some(pow_base_vid),
                    output: pow_node.outputs[0],
                }),
            });
    let bound_mean_axes =
        image_reduction::collect_bound_mean_axes(graph, &nodes, reduction_routes.as_ref())?;

    Some(ImageNormalizeMatch {
        nodes,
        input: imgs_vid,
        epsilon,
        gain: gain_opt,
        bias: bias_opt,
        gamma: gamma_opt,
        clamp_zero,
        clamp_zero_value,
        bound_mean_axes,
        requires_single_input,
        reduction_routes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan,
        PrimitiveOp, ValueId, ValueInfo, ValueOrigin, VarKind,
    };
    use runmat_builtins::Type;
    use runmat_value::{IntValue, Value};
    use std::collections::HashMap as StdHashMap;

    #[path = "reduction_entry_leaves.rs"]
    mod reduction_entry_leaves;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn image_single_cast_requires_the_selected_catalog_binding() {
        let [variant] = SINGLE_CATALOG_ENTRY.bindings else {
            panic!("single catalog binding must be unambiguous");
        };
        let mut node = AccelNode {
            id: 0,
            label: AccelNodeLabel::Builtin {
                name: "SINGLE".into(),
                origin: FusionCallOrigin::Named {
                    binding: SINGLE_CATALOG_ENTRY.binding_identity(variant),
                },
            },
            category: AccelOpCategory::Elementwise,
            inputs: vec![0],
            outputs: vec![1],
            span: InstrSpan { start: 0, end: 0 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        assert!(is_canonical_single_cast(&node));
        let AccelNodeLabel::Builtin { name, .. } = &mut node.label else {
            unreachable!()
        };
        *name = "double".into();
        assert!(!is_canonical_single_cast(&node));
        node.label = AccelNodeLabel::Builtin {
            name: "single".into(),
            origin: FusionCallOrigin::NamedUnresolved,
        };
        assert!(!is_canonical_single_cast(&node));
    }

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
    fn live_variable_fact_never_becomes_a_frozen_fusion_input() {
        let mut graph = simple_elementwise_graph();
        graph.values[0].constant = Some(Value::Num(41.0));
        let group = detect_fusion_groups(&graph).remove(0);
        let plan = FusionGroupPlan::new(0, group, &graph);
        let slot = plan.inputs.iter().position(|value| *value == 0).unwrap();
        assert!(!plan.constants.contains_key(&slot));
        assert!(!plan.const_values.contains_key(&0));

        graph.values[0].origin = ValueOrigin::Constant;
        graph.var_bindings.insert(
            0,
            crate::graph::VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        );
        let group = detect_fusion_groups(&graph).remove(0);
        let plan = FusionGroupPlan::new(0, group, &graph);
        let slot = plan.inputs.iter().position(|value| *value == 0).unwrap();
        assert!(!plan.constants.contains_key(&slot));
        assert!(!plan.const_values.contains_key(&0));
    }

    #[test]
    fn prepare_fusion_plan_requires_semantic_candidate_groups() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);

        let plan = prepare_fusion_plan(Some(&graph), &groups, 0);
        assert!(
            plan.is_none(),
            "bytecode groups alone should not produce an executable fusion plan without semantic candidate evidence"
        );
    }

    #[test]
    fn prepare_fusion_plan_allows_semantic_gated_groups() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);

        let plan = prepare_fusion_plan(Some(&graph), &groups, 1);
        assert!(
            plan.is_some(),
            "semantic candidate evidence should allow executable fusion plan preparation"
        );
    }

    #[test]
    fn prepare_fusion_plan_cache_is_thread_local_rc() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);

        let first = prepare_fusion_plan(Some(&graph), &groups, 1).expect("first plan");
        let second = prepare_fusion_plan(Some(&graph), &groups, 1).expect("cached plan");

        assert!(
            std::rc::Rc::ptr_eq(&first, &second),
            "fusion plans should be reused within the current thread without implying Send/Sync ownership"
        );
    }

    #[test]
    fn prepare_fusion_plan_cache_distinguishes_graph_content() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        let first = prepare_fusion_plan(Some(&graph), &groups, 1).expect("first plan");

        let mut changed_graph = simple_elementwise_graph();
        changed_graph.nodes[0].label = AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv);
        let changed_groups = detect_fusion_groups(&changed_graph);
        let second =
            prepare_fusion_plan(Some(&changed_graph), &changed_groups, 1).expect("second plan");

        assert!(
            !std::rc::Rc::ptr_eq(&first, &second),
            "fusion plan cache must not reuse a plan for a different graph that happens to occupy a reused address"
        );
    }

    #[test]
    fn prepare_fusion_plan_recovers_empty_group_nodes_from_contained_runtime_span() {
        let graph = simple_elementwise_graph();
        let groups = vec![FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            span: InstrSpan { start: 10, end: 10 },
            pattern: None,
            stack_layout: None,
        }];

        let plan = prepare_fusion_plan(Some(&graph), &groups, 1)
            .expect("runtime group sanitization should recover contained elementwise nodes");
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(
            plan.groups[0].group.nodes,
            vec![0],
            "runtime sanitization should recover a compatible contained node for empty group mapping"
        );
    }

    #[test]
    fn prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_graph_is_too_far() {
        let graph = simple_elementwise_graph();
        let groups = vec![FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            span: InstrSpan { start: 20, end: 20 },
            pattern: None,
            stack_layout: None,
        }];

        let plan = prepare_fusion_plan(Some(&graph), &groups, 1);
        assert!(
            plan.is_none(),
            "runtime sanitization should reject empty group mapping when no compatible nearby nodes exist"
        );
    }

    #[test]
    fn prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_node_covers_group_span() {
        let values = vec![
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
            ValueInfo {
                id: 1,
                origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                ty: Type::tensor(),
                shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
                constant: None,
            },
        ];
        let graph = AccelGraph {
            nodes: vec![AccelNode {
                id: 0,
                label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
                category: AccelOpCategory::Elementwise,
                inputs: vec![0, 0],
                outputs: vec![1],
                span: InstrSpan { start: 10, end: 12 },
                tags: vec![AccelGraphTag::Elementwise],
            }],
            values,
            var_bindings: StdHashMap::new(),
            node_bindings: StdHashMap::new(),
        };
        let groups = vec![FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            span: InstrSpan { start: 11, end: 11 },
            pattern: None,
            stack_layout: None,
        }];

        let plan = prepare_fusion_plan(Some(&graph), &groups, 1);
        assert!(
            plan.is_none(),
            "runtime sanitization should reject covering runtime-node spans when semantic group spans are narrower"
        );
    }

    #[test]
    fn prepare_fusion_plan_rejects_stale_mapped_nodes_without_runtime_remap() {
        let graph = simple_elementwise_graph();
        let groups = vec![FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
            span: InstrSpan { start: 10, end: 10 },
            pattern: None,
            stack_layout: None,
        }];

        let plan = prepare_fusion_plan(Some(&graph), &groups, 1);
        assert!(
            plan.is_none(),
            "runtime sanitization should reject stale mapped nodes instead of remapping from runtime graph scan"
        );
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
    fn counted_wgsl_source_matches_emission_with_f64_helper() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let group = &plan.groups[0];
        let bindings = "@group(0) @binding(1) var<storage, read_write> output0: Tensor;\n";
        let body = "    let typed0: f64 = input0.data[i0];\n";
        let writes = "    output0.data[g] = typed0;\n";
        let mut source = "// retained preamble\n".to_owned();
        let fragments = WgslShaderFragments {
            bindings,
            body,
            final_writes: writes,
        };
        let measured = group
            .measured_wgsl_shader_bytes_with_inputs(
                source.len(),
                (1, "f64", "f64"),
                fragments,
                2,
                true,
            )
            .unwrap();
        group.append_wgsl_shader_with_inputs(&mut source, (1, "f64", "f64"), fragments, 2, true);
        assert_eq!(measured, source.len());
        assert!(source.contains("fn runmat_f64_from_words("));
    }

    #[test]
    fn counted_math_helpers_match_emitted_source_bytes() {
        for precision in [
            runmat_accelerate_api::ProviderPrecision::F32,
            runmat_accelerate_api::ProviderPrecision::F64,
        ] {
            let mut power = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
            crate::kernel_math::power::write_functions(precision, &mut power).unwrap();
            assert_eq!(
                power.bytes(),
                crate::kernel_math::power::functions(precision).len()
            );

            let mut round = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
            crate::kernel_math::rounding::write_functions(precision, &mut round).unwrap();
            assert_eq!(
                round.bytes(),
                crate::kernel_math::rounding::functions(precision).len()
            );

            let mut remainder = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
            crate::kernel_math::floating_remainder::write_functions(precision, &mut remainder)
                .unwrap();
            assert_eq!(
                remainder.bytes(),
                crate::kernel_math::floating_remainder::functions(precision).len()
            );
        }
    }

    #[test]
    fn classification_helpers_use_ieee_encoding_not_finite_thresholds() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let single = plan.groups[0].generate_wgsl("f32").unwrap();
        assert!(single.contains("(bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u"));
        assert!(single.contains("(bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u"));
        let double = plan.groups[0].generate_wgsl("f64").unwrap();
        assert!(double.contains("(runmat_f64_high(x) & 0x7ff00000u) != 0x7ff00000u"));
        assert!(double.contains("runmat_f64_low(x) == 0u"));
        assert!(!double.contains("0x7ff0000000000000u"));
    }

    #[test]
    fn wide_integer_constant_declines_float_fusion_before_wgsl_generation() {
        let mut graph = simple_elementwise_graph();
        let wide = u64::MAX;
        let poisoned_f64_mirror = wide as f64;
        assert_eq!(poisoned_f64_mirror, (wide - 1) as f64);
        graph.values.push(ValueInfo {
            id: 3,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::Int(IntValue::U64(wide))),
        });
        graph.nodes[0].inputs = vec![0, 3];

        let groups = detect_fusion_groups(&graph);
        let plan = FusionPlan::from_graph(&graph, &groups);
        let group = &plan.groups[0];
        assert!(group.has_native_integer_constants());
        assert!(!group.kernel.supported);
        assert!(group.generate_wgsl("f32").is_none());
    }

    #[test]
    fn all_native_integer_constant_classes_decline_float_fusion() {
        for constant in [
            IntValue::I8(7),
            IntValue::I16(7),
            IntValue::I32(7),
            IntValue::I64(i64::MAX),
            IntValue::U8(7),
            IntValue::U16(7),
            IntValue::U32(7),
            IntValue::U64(u64::MAX),
        ] {
            let mut graph = simple_elementwise_graph();
            graph.values.push(ValueInfo {
                id: 3,
                origin: ValueOrigin::Constant,
                ty: Type::Num,
                shape: ShapeInfo::Scalar,
                constant: Some(Value::Int(constant)),
            });
            graph.nodes[0].inputs = vec![0, 3];

            let groups = detect_fusion_groups(&graph);
            let plan = FusionPlan::from_graph(&graph, &groups);
            let group = &plan.groups[0];
            assert!(group.has_native_integer_constants());
            assert!(!group.kernel.supported);
            assert!(group.generate_wgsl("f32").is_none());
        }
    }

    #[test]
    fn native_integer_tensor_constants_decline_float_fusion() {
        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![1_u64 << 63]),
            vec![1, 1],
        )
        .expect("integer tensor");
        let value = Value::Tensor(tensor);
        let info = ValueInfo {
            id: 99,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Tensor(vec![Some(1), Some(1)]),
            constant: Some(value.clone()),
        };

        assert_eq!(value_info_scalar(&info), None);
        let mut graph = simple_elementwise_graph();
        graph.values.push(info);
        assert!(!value_is_placeholder(&graph, 99));
        assert_eq!(resolve_numeric_vector_constant(&graph, 99), None);
    }

    #[test]
    fn bound_dimension_vector_fact_is_not_an_inline_literal() {
        let mut graph = simple_elementwise_graph();
        graph.values.push(ValueInfo {
            id: 3,
            origin: ValueOrigin::Constant,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(1), Some(2)]),
            constant: Some(Value::Tensor(
                runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )),
        });
        assert_eq!(
            resolve_numeric_vector_constant(&graph, 3),
            Some(vec![1.0, 2.0])
        );
        graph.var_bindings.insert(
            3,
            crate::graph::VarBinding {
                kind: VarKind::Global,
                index: 3,
            },
        );
        assert_eq!(resolve_numeric_vector_constant(&graph, 3), None);
    }

    #[test]
    fn image_mean_axes_require_exact_source_dimensions() {
        assert!(dims_match_unordered(&[3.0, 2.0], &[2.0, 3.0]));
        assert!(!dims_match_unordered(&[2.4, 3.0], &[2.0, 3.0]));
        assert!(!dims_match_unordered(&[2.0 + 1e-7], &[2.0]));
        assert!(!dims_match_unordered(&[f64::NAN, 3.0], &[2.0, 3.0]));
        assert!(!dims_match_unordered(&[f64::INFINITY, 3.0], &[2.0, 3.0]));
    }

    fn folded_single_clamp_graph(input: Value, folded: Option<Value>) -> AccelGraph {
        AccelGraph {
            values: vec![
                ValueInfo {
                    id: 0,
                    origin: ValueOrigin::Variable {
                        kind: VarKind::Global,
                        index: 0,
                    },
                    ty: Type::tensor(),
                    shape: ShapeInfo::Tensor(vec![Some(2)]),
                    constant: None,
                },
                ValueInfo {
                    id: 1,
                    origin: ValueOrigin::Constant,
                    ty: Type::Num,
                    shape: ShapeInfo::Scalar,
                    constant: Some(input),
                },
                ValueInfo {
                    id: 2,
                    origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                    ty: Type::Tensor {
                        shape: Some(vec![Some(1), Some(1)]),
                    },
                    shape: ShapeInfo::Tensor(vec![Some(1), Some(1)]),
                    constant: folded,
                },
                ValueInfo {
                    id: 3,
                    origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
                    ty: Type::tensor(),
                    shape: ShapeInfo::Tensor(vec![Some(2)]),
                    constant: None,
                },
            ],
            nodes: vec![
                AccelNode {
                    id: 0,
                    label: AccelNodeLabel::Builtin {
                        origin: Default::default(),
                        name: "single".into(),
                    },
                    category: AccelOpCategory::Elementwise,
                    inputs: vec![1],
                    outputs: vec![2],
                    span: InstrSpan { start: 0, end: 0 },
                    tags: vec![AccelGraphTag::Elementwise, AccelGraphTag::UnaryScalarFold],
                },
                AccelNode {
                    id: 1,
                    label: AccelNodeLabel::Builtin {
                        origin: FusionCallOrigin::Named {
                            binding: runmat_builtins::MAX_CATALOG_ENTRY
                                .binding_identity(&runmat_builtins::MAX_CATALOG_ENTRY.bindings[0]),
                        },
                        name: "max".into(),
                    },
                    category: AccelOpCategory::Elementwise,
                    inputs: vec![0, 2],
                    outputs: vec![3],
                    span: InstrSpan { start: 1, end: 1 },
                    tags: vec![AccelGraphTag::Elementwise],
                },
            ],
            var_bindings: StdHashMap::new(),
            node_bindings: StdHashMap::new(),
        }
    }

    fn folded_single(value: f32) -> Value {
        Value::Tensor(runmat_value::Tensor::from_f32(vec![value], vec![1, 1]).unwrap())
    }

    #[test]
    fn image_clamp_accepts_only_a_certified_folded_single_zero() {
        let graph = folded_single_clamp_graph(Value::Num(0.0), Some(folded_single(0.0)));
        let trace = collect_scalar_constant(&graph, 2).expect("certified single zero");
        assert_eq!(trace.value.to_bits(), 0.0f64.to_bits());
        assert_eq!(trace.nodes, vec![0]);
        assert_eq!(
            split_max_with_zero_scalar(&graph, 3, &HashSet::new(), &mut Vec::new()),
            Some((1, 0, None))
        );

        let assigned = HashSet::from([0]);
        assert_eq!(
            split_max_with_zero_scalar(&graph, 3, &assigned, &mut Vec::new()),
            None,
            "a conversion already owned by another group cannot certify the clamp"
        );

        let serialized = serde_json::to_string(&graph).expect("serialize graph");
        let restored: AccelGraph = serde_json::from_str(&serialized).expect("deserialize graph");
        assert!(collect_scalar_constant(&restored, 2).is_none());
    }

    #[test]
    fn constant_traces_reject_bound_sources_and_published_fold_outputs() {
        let original = folded_single_clamp_graph(Value::Num(0.0), Some(folded_single(0.0)));
        assert!(collect_scalar_constant(&original, 2).is_some());

        let mut bound_source = original.clone();
        bound_source.var_bindings.insert(
            1,
            crate::graph::VarBinding {
                kind: VarKind::Global,
                index: 1,
            },
        );
        assert!(collect_scalar_constant(&bound_source, 2).is_none());

        let mut published_fold = original;
        published_fold.node_bindings.insert(
            0,
            crate::graph::VarBinding {
                kind: VarKind::Global,
                index: 2,
            },
        );
        assert!(collect_scalar_constant(&published_fold, 2).is_none());
    }

    #[test]
    fn image_clamp_rejects_nonzero_unknown_and_unsafe_integer_controls() {
        let mut untagged = folded_single_clamp_graph(Value::Num(0.0), Some(folded_single(0.0)));
        untagged.nodes[0]
            .tags
            .retain(|tag| *tag != AccelGraphTag::UnaryScalarFold);
        let mut wrong_output = folded_single_clamp_graph(Value::Num(0.0), Some(folded_single(0.0)));
        wrong_output.nodes[0].outputs.clear();
        let mut wrong_arity = folded_single_clamp_graph(Value::Num(0.0), Some(folded_single(0.0)));
        wrong_arity.nodes[0].inputs.push(1);
        for graph in [
            folded_single_clamp_graph(Value::Num(0.25), Some(folded_single(0.25))),
            folded_single_clamp_graph(Value::Num(0.0), None),
            folded_single_clamp_graph(Value::Int(IntValue::U64(u64::MAX)), None),
            folded_single_clamp_graph(
                Value::Int(IntValue::U64(u64::MAX)),
                Some(folded_single(0.0)),
            ),
            folded_single_clamp_graph(Value::Num(0.0), Some(Value::Num(0.0))),
            untagged,
            wrong_output,
            wrong_arity,
        ] {
            assert_eq!(
                split_max_with_zero_scalar(&graph, 3, &HashSet::new(), &mut Vec::new()),
                None
            );
        }
    }

    #[test]
    fn image_scalar_trace_preserves_single_rounding_and_unclamped_source() {
        let graph = folded_single_clamp_graph(Value::Num(0.1), Some(folded_single(0.1)));
        let trace = collect_scalar_constant(&graph, 2).expect("rounded single constant");
        assert_eq!(trace.value.to_bits(), ((0.1f64 as f32) as f64).to_bits());
        assert_eq!(trace.nodes, vec![0]);
        assert_eq!(collect_scalar_constant(&graph, 1).unwrap().value, 0.1);
        assert_eq!(
            split_max_with_zero_scalar(&graph, 3, &HashSet::new(), &mut Vec::new()),
            None,
            "nonzero max is not an image-normalize clamp"
        );
    }

    #[test]
    fn certified_fold_sign_counts_outer_but_not_inner_negation() {
        // The current graph owner folds literal single calls. This exercises
        // the generic proof tag if a later owner folds a unary expression
        // whose source itself contains a negation.
        let mut graph = folded_single_clamp_graph(Value::Num(2.0), Some(folded_single(-2.0)));
        graph.values.push(ValueInfo {
            id: 4,
            origin: ValueOrigin::NodeOutput { node: 2, output: 0 },
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: None,
        });
        graph.values.push(ValueInfo {
            id: 5,
            origin: ValueOrigin::NodeOutput { node: 3, output: 0 },
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: None,
        });
        graph.nodes[0].inputs = vec![4];
        graph.nodes.push(AccelNode {
            id: 2,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Neg),
            category: AccelOpCategory::Elementwise,
            inputs: vec![1],
            outputs: vec![4],
            span: InstrSpan { start: 2, end: 2 },
            tags: vec![AccelGraphTag::Unary, AccelGraphTag::Elementwise],
        });
        graph.nodes.push(AccelNode {
            id: 3,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Neg),
            category: AccelOpCategory::Elementwise,
            inputs: vec![2],
            outputs: vec![5],
            span: InstrSpan { start: 3, end: 3 },
            tags: vec![AccelGraphTag::Unary, AccelGraphTag::Elementwise],
        });

        let inner = collect_scalar_constant(&graph, 2).expect("certified folded inner negation");
        assert_eq!(inner.value, -2.0);
        assert_eq!(inner.nodes, vec![0, 2]);

        let outer = collect_scalar_constant(&graph, 5).expect("outer negation of folded value");
        assert_eq!(outer.value, 2.0);
        assert_eq!(outer.nodes, vec![3, 0, 2]);
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
    fn builtin_expressions_respect_canonical_specs() {
        let mut exprs: StdHashMap<ValueId, String> = StdHashMap::new();
        exprs.insert(0, "v0".to_string());
        exprs.insert(1, "v1".to_string());

        let log1p = super::builtin_expr("log1p", &[0], &exprs, "f32");
        assert!(
            log1p.is_none(),
            "canonical domain boundary must not borrow a legacy expression"
        );

        let log10 = super::builtin_expr("log10", &[0], &exprs, "f64");
        assert!(log10.is_none());

        let expm1 = super::builtin_expr("expm1", &[0], &exprs, "f32");
        assert!(expm1.is_none());

        let floor = super::builtin_expr("floor", &[0], &exprs, "f32");
        assert_eq!(floor.unwrap(), "floor(v0)");

        let atan2 = super::builtin_expr("atan2", &[0, 1], &exprs, "f32");
        assert_eq!(atan2.unwrap(), "select(atan2(v0, v1), select(v0, 0.0, bitcast<u32>(v1) == 0x80000000u), (v0 == 0.0) && (v1 == 0.0))");

        let asinh = super::builtin_expr("asinh", &[0], &exprs, "f32");
        assert_eq!(asinh.unwrap(), "asinh(v0)");

        let acosh = super::builtin_expr("acosh", &[0], &exprs, "f32");
        assert!(acosh.is_none());

        let atanh = super::builtin_expr("atanh", &[0], &exprs, "f32");
        assert!(atanh.is_none());

        let hypot = super::builtin_expr("hypot", &[0, 1], &exprs, "f32");
        assert_eq!(hypot.unwrap(), "hypot(v0, v1)");

        let sign = super::builtin_expr("sign", &[0], &exprs, "f32");
        assert_eq!(sign.unwrap(), "sign(v0)");

        let heaviside = super::builtin_expr("heaviside", &[0], &exprs, "f32").unwrap();
        assert!(heaviside.contains("0.5"));
        assert!(heaviside.contains("isNan(v0)"));

        let fix = super::builtin_expr("fix", &[0], &exprs, "f32");
        assert_eq!(fix.unwrap(), "select(trunc(v0), 0.0, trunc(v0) == 0.0)");

        let modulo = super::builtin_expr("mod", &[0, 1], &exprs, "f32");
        // Remainder declarations require shared numerical helpers. This
        // legacy emitter cannot publish them; typed shader emission owns that.
        assert!(modulo.is_none());

        let rem = super::builtin_expr("rem", &[0, 1], &exprs, "f32");
        assert!(rem.is_none());

        let pow2 = super::builtin_expr("pow2", &[0], &exprs, "f32");
        assert_eq!(pow2.unwrap(), "exp(v0 * 0.69314718055994529)");

        let single = super::builtin_expr("single", &[0], &exprs, "f32");
        assert_eq!(single.unwrap(), "v0");

        let double = super::builtin_expr("double", &[0], &exprs, "f64");
        assert_eq!(double.unwrap(), "(v0)");
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
                origin: Default::default(),
                name: "tanh".into(),
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
                origin: Default::default(),
                name: "single".into(),
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
