//! Source-form evidence for image-normalization means. This is metadata, not
//! permission to execute a staged provider session.

use super::{dims_match_unordered, match_mean_axes, node_from_value};
use crate::graph::{AccelGraph, AccelNodeLabel, FusionCallOrigin, NodeId, ValueId, ValueOrigin};
use runmat_builtins::{MeanControl, MeanOutputClass, MAX_CATALOG_ENTRY, MEAN_CATALOG_ENTRY};
use runmat_value::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMeanAxis {
    Height,
    Width,
}

impl ImageMeanAxis {
    fn from_exact_dim(dim: f64) -> Option<Self> {
        if dim == 2.0 {
            Some(Self::Height)
        } else if dim == 3.0 {
            Some(Self::Width)
        } else {
            None
        }
    }

    fn zero_based(self) -> usize {
        match self {
            Self::Height => 1,
            Self::Width => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMeanOutput {
    Default,
    Native,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMeanCall {
    pub node: NodeId,
    pub input: ValueId,
    pub axes_value: ValueId,
    pub output: ValueId,
    /// Graph provenance is an expected binding, never runtime selection.
    pub origin: FusionCallOrigin,
    pub output_class: ImageMeanOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMeanForm {
    /// One public vecdim call; a staged provider route must prove its own
    /// numerical equivalence to the selected ordinary implementation.
    VecDims {
        call: ImageMeanCall,
        axes: [ImageMeanAxis; 2],
    },
    /// Two public calls, each with its own output conversion boundary.
    Nested {
        inner: ImageMeanCall,
        inner_axis: ImageMeanAxis,
        outer: ImageMeanCall,
        outer_axis: ImageMeanAxis,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMeanCastPath {
    pub consumer_value: ValueId,
    pub peeled_value: ValueId,
    pub nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMeanRoute {
    pub form: ImageMeanForm,
    pub cast_paths: Vec<ImageMeanCastPath>,
}

mod provider_plan;
mod source_invocation;
mod source_controls;
mod source_extension;
mod source_moment_publications;
mod source_output_hooks;
pub(crate) use source_output_hooks::{SelectedBinaryStep, SelectedRealStep};
mod source_program;
pub use provider_plan::{
    ImageMeanAndPowerSource, ImageMeanProviderCall, ImageMeanProviderHook, ImageMeanProviderPlan,
    ImageSourcePower,
};
pub(crate) use source_controls::bound_image_controls_match;
pub(super) use source_controls::{
    bound_folded_zero, collect_bound_mean_axes, is_canonical_max_node, resolve_selected_mean_axes,
};
pub use source_extension::selected_group_with_publications;
pub(crate) use source_invocation::{
    ImageSelectedBinaryCall, ImageSelectedBinaryCalls, ImageSelectedMomentBindings,
    ImageSelectedMomentConversion, ImageSelectedRealCall, ImageSelectedRealCalls,
    ImageSelectedSigmaOperations, ImageSelectedSigmaRaw,
};
pub use source_invocation::{
    ImageSelectedPower, ImageSourceAdmissionFailure, ImageSourceInvocation, ImageSourcePublication,
};
pub use source_program::{ImageSourceLeaf, ImageSourceStep, ImageSourceTranscript};

impl ImageMeanRoute {
    pub(super) fn for_each_axes_value(&self, mut visit: impl FnMut(ValueId)) {
        match &self.form {
            ImageMeanForm::VecDims { call, .. } => visit(call.axes_value),
            ImageMeanForm::Nested { inner, outer, .. } => {
                visit(inner.axes_value);
                visit(outer.axes_value);
            }
        }
    }

    pub(super) fn with_cast_path(
        mut self,
        graph: &AccelGraph,
        consumer_value: ValueId,
        peeled_value: ValueId,
    ) -> Option<Self> {
        let mut nodes = Vec::new();
        let mut current = consumer_value;
        let mut visited = HashSet::new();
        while current != peeled_value {
            let ValueOrigin::NodeOutput { node, output } = &graph.value(current)?.origin else {
                return None;
            };
            if !visited.insert(*node) {
                return None;
            }
            let producer = graph.node(*node)?;
            if producer.outputs.get(*output) != Some(&current)
                || producer.inputs.len() != 1
                || !super::is_canonical_single_cast(producer)
            {
                return None;
            }
            nodes.push(*node);
            current = producer.inputs[0];
        }
        self.cast_paths.push(ImageMeanCastPath {
            consumer_value,
            peeled_value,
            nodes,
        });
        Some(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageReductionRoutes {
    pub mu: ImageMeanRoute,
    pub variance: ImageMeanRoute,
    /// Graph producers for mean, sigma, and each source-visible output
    /// checkpoint. These are not VM store IDs or permission to replace the
    /// source statements; the VM joins the actual PC-ordered stores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_producers: Option<ImageStageProducers>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageStageProducers {
    pub mu: ValueId,
    pub variance: ValueId,
    pub sigma: ValueId,
    /// Optional source-visible checkpoints before and after a selected clamp.
    /// Presence alone does not claim a store; the VM joins actual PC stores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_clamp_output: Option<ValueId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clamp_output: Option<ValueId>,
    pub output: ValueId,
}

impl ImageMeanRoute {
    pub fn output(&self) -> ValueId {
        match &self.form {
            ImageMeanForm::VecDims { call, .. } => call.output,
            ImageMeanForm::Nested { outer, .. } => outer.output,
        }
    }
}

impl ImageReductionRoutes {
    pub fn stage_producers(&self) -> Option<ImageStageProducers> {
        let producers = self.stage_producers?;
        (self.mu.output() == producers.mu
            && self.variance.output() == producers.variance
            && (producers.pre_clamp_output.is_some() == producers.clamp_output.is_some()))
        .then_some(producers)
    }

    pub fn matches_source_graph(&self, graph: &AccelGraph) -> bool {
        let Some(producers) = self.stage_producers() else {
            return false;
        };
        if producers
            .clamp_output
            .is_some_and(|value| !selected_clamp(graph, value))
        {
            return false;
        }
        [
            (&self.mu, producers.mu),
            (&self.variance, producers.variance),
        ]
        .into_iter()
        .all(|(route, value)| {
            exact_route(graph, value).is_some_and(|exact| exact.form == route.form)
                && route.cast_paths.iter().all(|path| {
                    ImageMeanRoute {
                        form: route.form.clone(),
                        cast_paths: Vec::new(),
                    }
                    .with_cast_path(graph, path.consumer_value, path.peeled_value)
                    .is_some_and(|checked| {
                        checked.cast_paths.as_slice() == std::slice::from_ref(path)
                    })
                })
        })
    }
}

fn selected_clamp(graph: &AccelGraph, value: ValueId) -> bool {
    let Some(ValueOrigin::NodeOutput { node, output }) = graph.value(value).map(|v| &v.origin)
    else {
        return false;
    };
    let Some(producer) = graph.node(*node) else {
        return false;
    };
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &producer.label
    else {
        return false;
    };
    *output == 0
        && producer.outputs.as_slice() == [value]
        && producer.inputs.len() == 2
        && runmat_builtins::builtin_catalog_entry_by_name(name).is_some_and(|entry| {
            entry.identity == MAX_CATALOG_ENTRY.identity
                && entry.identity == binding.builtin
                && entry
                    .bindings
                    .iter()
                    .any(|variant| entry.binding_identity(variant) == *binding)
        })
}

/// An exact graph-value lineage, allowing only selected canonical `single`
/// calls after a stage producer. The caller owns the MIR statement boundary.
pub fn publication_follows_stage(
    graph: &AccelGraph,
    producer: ValueId,
    published: ValueId,
    statement: std::ops::Range<usize>,
) -> bool {
    let Some(ValueOrigin::NodeOutput { node, output }) = graph.value(producer).map(|v| &v.origin)
    else {
        return false;
    };
    let Some(source) = graph.node(*node) else {
        return false;
    };
    if source.outputs.get(*output) != Some(&producer)
        || source.span.start < statement.start
        || source.span.end >= statement.end
    {
        return false;
    }
    let mut current = published;
    let mut seen = HashSet::new();
    while current != producer {
        let Some(ValueOrigin::NodeOutput { node, output }) =
            graph.value(current).map(|v| &v.origin)
        else {
            return false;
        };
        if !seen.insert(*node) {
            return false;
        }
        let Some(conversion) = graph.node(*node) else {
            return false;
        };
        if conversion.outputs.get(*output) != Some(&current)
            || conversion.span.start < statement.start
            || conversion.span.end >= statement.end
            || !super::is_canonical_single_cast(conversion)
        {
            return false;
        }
        current = conversion.inputs[0];
    }
    true
}

pub(super) struct PeeledMean {
    pub source: ValueId,
    pub route: Option<ImageMeanRoute>,
}

fn canonical_call(
    graph: &AccelGraph,
    output: ValueId,
) -> Option<(ImageMeanCall, Vec<ImageMeanAxis>)> {
    let (node, candidate) = node_from_value(graph, output)?;
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &candidate.label
    else {
        return None;
    };
    let [variant] = MEAN_CATALOG_ENTRY.bindings else {
        return None;
    };
    if !name.eq_ignore_ascii_case(MEAN_CATALOG_ENTRY.identity.name)
        || *binding != MEAN_CATALOG_ENTRY.binding_identity(variant)
        || !(2..=3).contains(&candidate.inputs.len())
        || candidate.outputs.as_slice() != [output]
    {
        return None;
    }
    let dims = resolve_selected_mean_axes(graph, candidate.inputs[1], candidate.span.start)?;
    if dims.is_empty() || dims.len() > 2 {
        return None;
    }
    let axes = dims
        .into_iter()
        .map(ImageMeanAxis::from_exact_dim)
        .collect::<Option<Vec<_>>>()?;
    let output_class = exact_output_class(graph, candidate.inputs.get(2).copied())?;
    Some((
        ImageMeanCall {
            node,
            input: candidate.inputs[0],
            axes_value: candidate.inputs[1],
            output,
            origin: FusionCallOrigin::Named { binding: *binding },
            output_class,
        },
        axes,
    ))
}

fn exact_call(graph: &AccelGraph, output: ValueId, axis: ImageMeanAxis) -> Option<ImageMeanCall> {
    let (call, axes) = canonical_call(graph, output)?;
    (axes.as_slice() == [axis]).then_some(call)
}

fn exact_vec_call(
    graph: &AccelGraph,
    output: ValueId,
) -> Option<(ImageMeanCall, [ImageMeanAxis; 2])> {
    let (call, axes) = canonical_call(graph, output)?;
    let axes: [ImageMeanAxis; 2] = axes.try_into().ok()?;
    matches!(
        axes,
        [ImageMeanAxis::Height, ImageMeanAxis::Width]
            | [ImageMeanAxis::Width, ImageMeanAxis::Height]
    )
    .then_some((call, axes))
}

fn exact_output_class(graph: &AccelGraph, control: Option<ValueId>) -> Option<ImageMeanOutput> {
    let Some(control) = control else {
        return Some(ImageMeanOutput::Default);
    };
    let option = match graph.value(control)?.constant.as_ref()? {
        Value::String(text) => text.as_str().and_then(MeanControl::from_public_option),
        Value::CharArray(chars) if chars.rows == 1 => {
            MeanControl::from_public_code_units(chars.data().iter().map(|unit| unit.raw()))
        }
        _ => None,
    }?;
    match option {
        MeanControl::Output(MeanOutputClass::Native) => Some(ImageMeanOutput::Native),
        _ => None,
    }
}

/// Retain the established structural detector while recording a stricter,
/// optional route. Approximate axes can never form staged evidence.
pub(super) fn peel_mean_dims(
    graph: &AccelGraph,
    vid: ValueId,
    expected_dims: &[f64],
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<PeeledMean> {
    let source = structural_peel(graph, vid, expected_dims, assigned, nodes)?;
    Some(PeeledMean {
        source,
        route: (expected_dims == [3.0, 2.0])
            .then(|| exact_route(graph, vid))
            .flatten(),
    })
}

fn structural_peel(
    graph: &AccelGraph,
    vid: ValueId,
    expected_dims: &[f64],
    assigned: &HashSet<NodeId>,
    nodes: &mut Vec<NodeId>,
) -> Option<ValueId> {
    if expected_dims.is_empty() {
        return Some(vid);
    }
    let vid = super::peel_numeric_casts(graph, vid, assigned, nodes)?;
    let (node, data, dims) = match_mean_axes(graph, vid)?;
    if assigned.contains(&node) {
        return None;
    }
    if dims.len() == expected_dims.len() && dims_match_unordered(&dims, expected_dims) {
        nodes.push(node);
        return Some(data);
    }
    if dims.len() == 1 && dims[0] == expected_dims[0] {
        nodes.push(node);
        return structural_peel(graph, data, &expected_dims[1..], assigned, nodes);
    }
    None
}

fn exact_route(graph: &AccelGraph, vid: ValueId) -> Option<ImageMeanRoute> {
    if let Some((call, axes)) = exact_vec_call(graph, vid) {
        return Some(ImageMeanRoute {
            form: ImageMeanForm::VecDims { call, axes },
            cast_paths: Vec::new(),
        });
    }
    let outer = exact_call(graph, vid, ImageMeanAxis::Width)?;
    let inner_value =
        super::peel_numeric_casts(graph, outer.input, &HashSet::new(), &mut Vec::new())?;
    let inner = exact_call(graph, inner_value, ImageMeanAxis::Height)?;
    let consumer = outer.input;
    ImageMeanRoute {
        form: ImageMeanForm::Nested {
            inner_axis: ImageMeanAxis::Height,
            outer_axis: ImageMeanAxis::Width,
            inner,
            outer,
        },
        cast_paths: Vec::new(),
    }
    .with_cast_path(graph, consumer, inner_value)
}

#[cfg(test)]
#[path = "image_reduction/tests.rs"]
mod tests;
