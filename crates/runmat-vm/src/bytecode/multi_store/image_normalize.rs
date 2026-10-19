//! Source-ordered ImageNormalize publications joined to selected MIR stores.
//! This is a structural proposal, not arithmetic, provider, or cost admission.

use runmat_accelerate::fusion::{
    image_reduction::{
        publication_follows_stage, selected_group_with_publications, ImageSourcePublication,
        ImageSourceTranscript,
    },
    FusionGroup, FusionGroupPlan, FusionKind, FusionPattern, FusionPlan,
};
use runmat_accelerate::graph::{AccelGraph, ValueId, ValueOrigin, VarBinding};
use runmat_types::{ProgramPointId, RegionValueId};
use std::rc::Rc;
use std::sync::Arc;

use super::{ProspectiveStore, SelectedMultiStoreGraph};
use crate::bytecode::Bytecode;
use crate::layout::VmSlotId;

#[path = "image_normalize/statement_region.rs"]
mod statement_region;
use statement_region::{selected_image_region, ImageStatementRegion};
#[path = "image_normalize/capture.rs"]
mod capture;
pub(crate) use capture::CapturedImageInputs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageNormalizePublicationJoin {
    pcs: std::ops::Range<usize>,
    stores: Vec<ImageRegionStore>,
    live_outs: Vec<ImageLiveOut>,
}

/// An installed candidate is derived with its graph and publication join in
/// one selected bytecode walk. Its plan is only a source description until
/// runtime binding, physical placement, and the ordered VM driver admit it.
#[derive(Debug)]
pub(crate) struct SelectedImageRegion {
    pub(crate) plan: FusionGroupPlan,
    pub(crate) publication: Rc<ImageNormalizePublicationJoin>,
    pub(crate) source_publications: Vec<ImageSourcePublication>,
    pub(crate) source: Arc<ImageSourceTranscript>,
    entry_inputs: Vec<ImageEntryInput>,
}

#[derive(Debug)]
enum ImageEntryInput {
    Constant,
    FoldedConstant,
    Binding(VarBinding),
}

#[derive(Debug)]
pub(crate) struct SelectedImageRegionSet {
    pub(crate) graph: AccelGraph,
    pub(crate) regions: Vec<SelectedImageRegion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageCheckpoint {
    Mu,
    Sigma,
    PreClamp,
    Clamp,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageStoreDestination {
    Checkpoint(ImageCheckpoint),
    /// A compiler-owned MIR slot still needs its exact value after a skipped
    /// region, even when it has no lexical source binding.
    Intermediate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRegionStore {
    pub store: ProspectiveStore,
    /// Exact value at the actual selected VM store. A stage's raw reduction
    /// or SQRT identity is dependency evidence, not the published value.
    pub public_output: ValueId,
    pub raw_stage_dependency: Option<ValueId>,
    pub statement: ProgramPointId,
    pub slot: VmSlotId,
    pub mir_defs: Vec<RegionValueId>,
    pub destination: ImageStoreDestination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLiveOutSource {
    LastStore(ImageStoreOrdinal),
    PreserveEntry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageStoreOrdinal(usize);

impl ImageStoreOrdinal {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageLiveOut {
    pub value: RegionValueId,
    pub slot: VmSlotId,
    pub source: ImageLiveOutSource,
}

impl ImageNormalizePublicationJoin {
    pub(crate) fn source_publications(&self) -> Option<Vec<ImageSourcePublication>> {
        self.stores
            .iter()
            .filter(|entry| matches!(entry.destination, ImageStoreDestination::Checkpoint(_)))
            .map(|entry| {
                Some(ImageSourcePublication {
                    producer: entry.raw_stage_dependency?,
                    published: entry.public_output,
                    statement: self.pcs.start..entry.store.pc.checked_add(1)?,
                })
            })
            .collect()
    }

    pub fn pcs(&self) -> std::ops::Range<usize> {
        self.pcs.clone()
    }

    /// All selected bytecode stores, including compiler intermediates, in PC
    /// order. A later executor must satisfy every live-out and publication.
    pub fn ordered_stores(&self) -> &[ImageRegionStore] {
        &self.stores
    }

    pub fn store_at(&self, ordinal: ImageStoreOrdinal) -> Option<&ImageRegionStore> {
        self.stores.get(ordinal.index())
    }

    pub fn live_outs(&self) -> &[ImageLiveOut] {
        &self.live_outs
    }

    pub fn checkpoint(&self, checkpoint: ImageCheckpoint) -> Option<&ProspectiveStore> {
        self.stores.iter().find_map(|entry| {
            (entry.destination == ImageStoreDestination::Checkpoint(checkpoint))
                .then_some(&entry.store)
        })
    }

    pub fn output_checkpoints(&self) -> impl Iterator<Item = &ProspectiveStore> {
        self.stores.iter().filter_map(|entry| {
            matches!(
                entry.destination,
                ImageStoreDestination::Checkpoint(
                    ImageCheckpoint::PreClamp | ImageCheckpoint::Clamp | ImageCheckpoint::Final
                )
            )
            .then_some(&entry.store)
        })
    }
}

impl Bytecode {
    /// Derive graph, ordered stores, and image groups from one selected
    /// bytecode walk. No caller-supplied graph or coincident ValueIds are an
    /// authority. This does not permit staged provider execution.
    pub fn prospective_image_normalize_publications(&self) -> Vec<ImageNormalizePublicationJoin> {
        self.selected_image_regions()
            .map(|selected| {
                selected
                    .regions
                    .into_iter()
                    .map(|region| region.publication.as_ref().clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn selected_image_regions(&self) -> Option<SelectedImageRegionSet> {
        let projection = self.fusion_metadata.stage_projection.as_ref()?;
        if self.validate_fusion_stage_projection().is_err() {
            return None;
        }
        let selected = SelectedMultiStoreGraph::for_image(self)?;
        let groups: Vec<_> = selected
            .graph
            .detect_fusion_groups()
            .into_iter()
            .filter(|group| group.kind == FusionKind::ImageNormalize)
            .collect();
        let mut regions = groups
            .iter()
            .filter_map(|group| {
                let region = selected_image_region(self, projection, &selected, group)?;
                if self
                    .catch_scopes
                    .get(region.pcs.start)
                    .copied()
                    .flatten()
                    .is_some()
                {
                    return None;
                }
                let publication = reconcile_image_publications(&region, &selected.graph, group)?;
                let source_publications = publication.source_publications()?;
                let extended =
                    selected_group_with_publications(&selected.graph, group, &source_publications)?;
                let plan = FusionPlan::from_graph(&selected.graph, std::slice::from_ref(&extended))
                    .groups
                    .into_iter()
                    .next()?;
                let source = Arc::new(ImageSourceTranscript::from_graph(
                    &selected.graph,
                    &plan.group,
                )?);
                let entry_inputs = capture::sealed_entry_inputs(
                    &selected.graph,
                    &selected.stores,
                    &plan,
                    &publication,
                    &source,
                )?;
                Some(SelectedImageRegion {
                    plan,
                    publication: Rc::new(publication),
                    source_publications,
                    source,
                    entry_inputs,
                })
            })
            .collect::<Vec<_>>();
        regions.sort_by_key(|region| region.publication.pcs.start);
        if regions
            .windows(2)
            .any(|pair| pair[0].publication.pcs.end > pair[1].publication.pcs.start)
        {
            return None;
        }
        Some(SelectedImageRegionSet {
            graph: selected.graph,
            regions,
        })
    }
}

/// This private join consumes only the graph held by SelectedMultiStoreGraph.
/// Runtime binding, arithmetic, cost, and ordered error publication remain
/// separate requirements before any executable certificate can exist.
fn reconcile_image_publications(
    region: &ImageStatementRegion<'_>,
    graph: &AccelGraph,
    group: &FusionGroup,
) -> Option<ImageNormalizePublicationJoin> {
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref()? else {
        return None;
    };
    if group.kind != FusionKind::ImageNormalize
        || group.span.start < region.pcs.start
        || group.span.end >= region.pcs.end
        || region.stores.len() < 3
    {
        return None;
    }
    let routes = pattern.reduction_routes.as_ref()?;
    let producers = routes.stage_producers()?;
    if pattern.clamp_zero != producers.clamp_output.is_some() {
        return None;
    }
    if !routes.matches_source_graph(graph) {
        return None;
    }
    // Preserve the entire selected arithmetic/cast/binding source transcript.
    // This is still only a proposal: no represented numerical or provider
    // arithmetic proof is minted here.
    let _source = ImageSourceTranscript::from_graph(graph, group)?;
    let mut source_outputs = Vec::with_capacity(3);
    source_outputs.extend(
        producers
            .pre_clamp_output
            .map(|value| (value, ImageCheckpoint::PreClamp)),
    );
    source_outputs.extend(
        producers
            .clamp_output
            .map(|value| (value, ImageCheckpoint::Clamp)),
    );
    source_outputs.push((producers.output, ImageCheckpoint::Final));
    let mut sources = vec![producers.mu, producers.sigma];
    sources.extend(source_outputs.iter().map(|(value, _)| *value));
    if sources
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != sources.len()
    {
        return None;
    }
    let belongs_to_group = |value: ValueId| {
        let Some(ValueOrigin::NodeOutput { node, output }) =
            graph.value(value).map(|info| &info.origin)
        else {
            return false;
        };
        group.nodes.contains(node)
            && graph
                .node(*node)
                .is_some_and(|producer| producer.outputs.get(*output) == Some(&value))
    };
    if ![
        producers.mu,
        producers.variance,
        producers.sigma,
        producers.pre_clamp_output.unwrap_or(producers.output),
        producers.clamp_output.unwrap_or(producers.output),
        producers.output,
    ]
    .into_iter()
    .all(belongs_to_group)
    {
        return None;
    }
    let named_indices = region
        .stores
        .iter()
        .enumerate()
        .filter_map(|(index, store)| store.named.then_some(index))
        .collect::<Vec<_>>();
    if named_indices.len() > sources.len() || named_indices.len() < 3 {
        return None;
    }
    let mut roles = vec![None; region.stores.len()];
    let mut raw_dependencies = vec![None; region.stores.len()];
    for (index, producer) in sources.iter().take(2).enumerate() {
        let selected = &region.stores[named_indices[index]];
        let statement = &selected.stage.pc;
        if !(statement.start..statement.end).contains(&selected.store.pc)
            || !publication_follows_stage(
                graph,
                *producer,
                selected.store.value_id,
                region.pcs.start..statement.end,
            )
        {
            return None;
        }
        roles[named_indices[index]] = Some(if index == 0 {
            ImageCheckpoint::Mu
        } else {
            ImageCheckpoint::Sigma
        });
        raw_dependencies[named_indices[index]] = Some(*producer);
    }
    let mut next_source = 0;
    for index in named_indices.iter().copied().skip(2) {
        let selected = &region.stores[index];
        let statement = &selected.stage.pc;
        if !(statement.start <= selected.store.pc && selected.store.pc < statement.end) {
            return None;
        }
        let matched = source_outputs[next_source..]
            .iter()
            .position(|(producer, _)| {
                publication_follows_stage(
                    graph,
                    *producer,
                    selected.store.value_id,
                    region.pcs.start..statement.end,
                )
            })?;
        next_source += matched + 1;
        roles[index] = Some(source_outputs[next_source - 1].1);
        raw_dependencies[index] = Some(source_outputs[next_source - 1].0);
    }
    let named = named_indices
        .iter()
        .map(|index| &region.stores[*index].store)
        .collect::<Vec<_>>();
    if region
        .stores
        .windows(2)
        .any(|pair| pair[0].store.pc >= pair[1].store.pc)
        || named[0].binding == named[1].binding
        || named[2..]
            .iter()
            .any(|store| store.binding == named[0].binding || store.binding == named[1].binding)
        || named[2..]
            .iter()
            .any(|store| store.binding != named[2].binding)
        || next_source != source_outputs.len()
    {
        return None;
    }
    let stores = region
        .stores
        .iter()
        .enumerate()
        .map(|(index, selected)| ImageRegionStore {
            store: selected.store.clone(),
            public_output: selected.store.value_id,
            raw_stage_dependency: raw_dependencies[index],
            statement: selected.stage.contract.point,
            slot: selected.slot,
            mir_defs: selected.defs.clone(),
            destination: roles[index]
                .map(ImageStoreDestination::Checkpoint)
                .unwrap_or(ImageStoreDestination::Intermediate),
        })
        .collect();
    Some(ImageNormalizePublicationJoin {
        pcs: region.pcs.clone(),
        stores,
        live_outs: region.live_outs.clone(),
    })
}

#[cfg(test)]
#[path = "image_normalize/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "image_normalize/transcript_tests.rs"]
mod transcript_tests;
