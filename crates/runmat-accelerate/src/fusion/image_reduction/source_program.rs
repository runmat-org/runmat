//! Exact selected graph transcript for the whole image source region.
//!
//! This records source operations and every external source leaf. It is
//! not a represented-arithmetic proof or an executable provider program.

use std::collections::{HashMap, HashSet};

use crate::fusion::{FusionGroup, FusionKind, FusionPattern};
use crate::graph::{
    AccelGraph, AccelNodeLabel, FusionCallOrigin, InstrSpan, NodeId, ValueId, ValueOrigin,
};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::Digest;

mod folded_leaf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSourceStep {
    pub node: NodeId,
    pub pc: InstrSpan,
    pub label: AccelNodeLabel,
    pub inputs: Vec<ValueId>,
    pub output: ValueId,
    pub selected_binding: Option<BuiltinBindingIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSourceLeaf {
    /// Canonical typed digest retains class, rank/shape, exact floating bits,
    /// integer/logical storage, complex components, axes, and string flags.
    /// Unsupported graph constants reject the transcript instead of vanishing.
    Constant { id: ValueId, digest: Digest },
    /// The source graph cannot substitute a stale compile-time constant for
    /// this input. A later request owner must bind its actual invocation value.
    Live { id: ValueId },
}

/// A graph-only source prerequisite. Source order, casts, canonical builtin
/// bindings, and stage producers are retained, but numerical and provider
/// admission remain deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSourceTranscript {
    pub steps: Vec<ImageSourceStep>,
    pub leaves: Vec<ImageSourceLeaf>,
    pub checkpoints: Vec<ValueId>,
}

impl ImageSourceTranscript {
    pub fn from_graph(graph: &AccelGraph, group: &FusionGroup) -> Option<Self> {
        if group.kind != FusionKind::ImageNormalize {
            return None;
        }
        let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref()? else {
            return None;
        };
        let routes = pattern.reduction_routes.as_ref()?;
        if !routes.matches_source_graph(graph) {
            return None;
        }
        let producers = routes.stage_producers()?;
        if pattern.clamp_zero != producers.clamp_output.is_some() {
            return None;
        }
        let mut checkpoints = vec![producers.mu, producers.sigma];
        checkpoints.extend(producers.pre_clamp_output);
        checkpoints.extend(producers.clamp_output);
        checkpoints.push(producers.output);

        let mut steps = Vec::with_capacity(group.nodes.len());
        let mut producer_by_value = HashMap::new();
        let mut seen_nodes = HashSet::new();
        for &id in &group.nodes {
            let node = graph.node(id)?;
            let [output] = node.outputs.as_slice() else {
                return None;
            };
            if node.id != id
                || !seen_nodes.insert(id)
                || node.span.start > node.span.end
                || node.span.start < group.span.start
                || node.span.end > group.span.end
                || !matches!(&graph.value(*output)?.origin,
                    ValueOrigin::NodeOutput { node: owner, output: 0 } if *owner == id)
                || producer_by_value.insert(*output, id).is_some()
            {
                return None;
            }
            let selected_binding = match &node.label {
                AccelNodeLabel::Primitive(_) => None,
                AccelNodeLabel::Builtin {
                    name,
                    origin: FusionCallOrigin::Named { binding },
                } => {
                    let entry = runmat_builtins::builtin_catalog_entry_by_name(name)?;
                    if entry.identity != binding.builtin
                        || entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
                        || entry.contract.workspace_effect.is_some()
                        || entry.contract.environment_effect.is_some()
                        || !entry
                            .bindings
                            .iter()
                            .any(|variant| entry.binding_identity(variant) == *binding)
                    {
                        return None;
                    }
                    Some(*binding)
                }
                _ => return None,
            };
            steps.push(ImageSourceStep {
                node: id,
                pc: node.span.clone(),
                label: node.label.clone(),
                inputs: node.inputs.clone(),
                output: *output,
                selected_binding,
            });
        }
        if steps.is_empty() {
            return None;
        }
        steps.sort_by_key(|step| (step.pc.start, step.pc.end, step.node));
        if steps
            .windows(2)
            .any(|pair| pair[0].pc.end >= pair[1].pc.start)
        {
            return None;
        }
        let source_order: HashMap<_, _> = steps
            .iter()
            .enumerate()
            .map(|(position, step)| (step.output, position))
            .collect();
        let mut external = HashSet::new();
        let mut folded_digests = HashMap::new();
        for (position, step) in steps.iter().enumerate() {
            for &input in &step.inputs {
                if let Some(&producer) = source_order.get(&input) {
                    if producer >= position {
                        return None;
                    }
                } else {
                    let info = graph.value(input)?;
                    match &info.origin {
                        ValueOrigin::Constant | ValueOrigin::Variable { .. } => {}
                        ValueOrigin::NodeOutput { node, output }
                            if graph.node(*node).is_some_and(|external| {
                                external.outputs.get(*output) == Some(&input)
                                    && external.span.end < group.span.start
                            }) => {}
                        ValueOrigin::NodeOutput { .. } => {
                            folded_digests
                                .insert(input, folded_leaf::digest(graph, input, step.pc.start)?);
                        }
                        _ => return None,
                    }
                    external.insert(input);
                }
            }
        }
        if checkpoints.iter().any(|id| !source_order.contains_key(id))
            || checkpoints
                .windows(2)
                .any(|pair| source_order[&pair[0]] >= source_order[&pair[1]])
        {
            return None;
        }
        let mut external = external.into_iter().collect::<Vec<_>>();
        external.sort_unstable();
        let mut leaves = Vec::new();
        for id in external {
            let info = graph.value(id)?;
            if let Some(digest) = folded_digests.get(&id) {
                leaves.push(ImageSourceLeaf::Constant {
                    id,
                    digest: *digest,
                });
                continue;
            }
            match &info.origin {
                ValueOrigin::Constant => leaves.push(ImageSourceLeaf::Constant {
                    id,
                    digest: super::super::semantics::exact_constant_digest(info.constant.as_ref()?)
                        .ok()?,
                }),
                ValueOrigin::Variable { .. } | ValueOrigin::NodeOutput { .. } => {
                    leaves.push(ImageSourceLeaf::Live { id });
                }
                _ => return None,
            }
        }
        Some(Self {
            steps,
            leaves,
            checkpoints,
        })
    }
}
