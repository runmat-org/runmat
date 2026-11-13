use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    CandidateOutputResidency, ExecutionCandidateDescriptor, PlacementResourceSnapshot,
    PlacementSignature, ProviderInputStageDemand, ProviderInputStageDemands, ResourceAccount,
};

mod source_output;
pub use source_output::SourceOutputOrdinal;
mod output_authority;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ResourceBoundUnavailable {
    NoPeakMemoryBound,
    NoScratchBound,
    NoRetainedValueBound,
}

/// A byte bound for a resource with the matching meaning. A limit on one
/// allocation is not a bound on peak memory or transient scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceByteBound {
    Known(u64),
    Unavailable(ResourceBoundUnavailable),
}

/// A route's simultaneously live provider allocation bound. Unlike scratch,
/// this may include borrowed inputs and the published output. The already-live
/// portion is subtracted exactly once from an available-memory comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatePeakLiveDemand {
    pub peak_bytes: u64,
    pub already_live_bytes: u64,
}

impl CandidatePeakLiveDemand {
    pub fn incremental_bytes(self) -> Option<u64> {
        self.peak_bytes.checked_sub(self.already_live_bytes)
    }
}

impl From<u64> for ResourceByteBound {
    fn from(bytes: u64) -> Self {
        Self::Known(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateResourceDemand {
    pub cpu_millicores: u32,
    pub retained_bytes: ResourceByteBound,
    pub scratch_bytes: ResourceByteBound,
    /// Additional host workspace that coexists with a provider operation's
    /// physical peak. `None` means this older candidate did not declare the
    /// domain, not that a zero-byte bound was proved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_host_workspace: Option<ResourceByteBound>,
    /// Sequential provider-execution input preparation, each stage including
    /// prior operation-created values still live. None is legacy unmodeled,
    /// not a proven absence of input readbacks or uploads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_input_stages: Option<super::ProviderInputStageDemands>,
    pub queue_slots: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_live: Option<CandidatePeakLiveDemand>,
}

impl Default for CandidateResourceDemand {
    fn default() -> Self {
        Self {
            cpu_millicores: 0,
            retained_bytes: ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoRetainedValueBound,
            ),
            scratch_bytes: ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            provider_host_workspace: None,
            provider_input_stages: None,
            queue_slots: 0,
            peak_live: None,
        }
    }
}

impl CandidateResourceDemand {
    /// Charge one owner-issued observation envelope to the candidate peak.
    /// Input-stage peaks are alternatives, so each receives the same added
    /// fence/query requirement; retained output is not charged again.
    pub fn with_provider_observation_peak(self, bytes: u64) -> Option<Self> {
        let add = |bound| {
            ResourceAccount::default()
                .checked_add(bound)?
                .checked_add(ResourceByteBound::Known(bytes))
                .map(ResourceAccount::bound)
        };
        let mut demand = self;
        demand.scratch_bytes = add(demand.scratch_bytes)?;
        if let Some(peak) = &mut demand.peak_live {
            peak.peak_bytes = peak.peak_bytes.checked_add(bytes)?;
        }
        demand.provider_input_stages = match demand.provider_input_stages {
            None => None,
            Some(stages) => {
                let charged = stages
                    .iter()
                    .map(|stage| {
                        Some(ProviderInputStageDemand {
                            incremental_peak_bytes: add(stage.incremental_peak_bytes)?,
                            ..stage
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(ProviderInputStageDemands::sequential(charged)?)
            }
        };
        Some(demand)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraphCandidate {
    pub descriptor: ExecutionCandidateDescriptor,
    pub resources: CandidateResourceDemand,
    /// Public outputs in source order. Absence is legacy one-output authority
    /// for consumed nodes; only a single-node, edge-free whole-work envelope
    /// may retain mixed outputs with aggregate resource facts alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_facts: Option<Vec<PlacementGraphOutputFact>>,
    /// Provider failure may fall back only when outputs remain uncommitted.
    pub transactional_results: bool,
}

/// A public output's post-publication residency and retained byte bound.
/// Provider execution peak and readback staging remain separate resource facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraphOutputFact {
    pub residency: CandidateOutputResidency,
    pub retained_bytes: ResourceByteBound,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraphNode {
    pub identity: String,
    pub candidates: Vec<PlacementGraphCandidate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraphEdge {
    pub from: u32,
    pub to: u32,
    /// The producing public output, not a provider-internal physical result.
    /// Omission is accepted only for a one-output source node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_output: Option<SourceOutputOrdinal>,
    pub bytes: u64,
    pub host_to_provider_ns: u64,
    pub provider_to_host_ns: u64,
    pub cross_provider_ns: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraph {
    /// Nodes are topologically ordered. Edges must point from a lower index to
    /// a higher index.
    pub nodes: Vec<PlacementGraphNode>,
    pub edges: Vec<PlacementGraphEdge>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementGraphLimits {
    pub max_nodes: u32,
    pub max_candidates_per_node: u32,
    pub max_edges: u32,
    pub max_providers: u32,
    pub max_frontier_states: u32,
    pub max_expansions: u32,
}

impl Default for PlacementGraphLimits {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_candidates_per_node: 8,
            max_edges: 256,
            max_providers: 16,
            max_frontier_states: 256,
            max_expansions: 16_384,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementPlanRequest {
    pub signature: PlacementSignature,
    pub graph: PlacementGraph,
    pub limits: PlacementGraphLimits,
    pub resources: PlacementResourceSnapshot,
    pub deterministic: bool,
    /// When true, provider candidates must guarantee staged results. A
    /// pre-commit failure may use an executable fallback if one was offered.
    pub require_transactional_results: bool,
}

impl PlacementPlanRequest {
    /// Bind the exact graph output contract before submitting a request. A
    /// changed role, byte bound, or source ordinal changes feedback identity.
    pub fn bind_graph_output_authority(&mut self) {
        self.signature.graph_output_authority = output_authority::digest(&self.graph);
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        self.signature.validate()?;
        if self.graph.nodes.is_empty() {
            return Err("placement graph must contain at least one node");
        }
        if self.graph.nodes.len() > self.limits.max_nodes as usize {
            return Err("placement graph exceeds node limit");
        }
        if self.graph.edges.len() > self.limits.max_edges as usize {
            return Err("placement graph exceeds edge limit");
        }
        if self.resources.providers.len() > self.limits.max_providers as usize {
            return Err("placement resources exceed provider limit");
        }
        let mut node_identities = BTreeSet::new();
        let mut output_arities = Vec::with_capacity(self.graph.nodes.len());
        let whole_work_envelope = self.graph.nodes.len() == 1 && self.graph.edges.is_empty();
        for node in &self.graph.nodes {
            if node.identity.is_empty()
                || node.identity.len() > 128
                || node.identity.chars().any(char::is_control)
            {
                return Err("placement node identity is invalid");
            }
            if !node_identities.insert(node.identity.as_str()) {
                return Err("placement node identities must be unique");
            }
            if node.candidates.is_empty()
                || node.candidates.len() > self.limits.max_candidates_per_node as usize
            {
                return Err("placement node candidate count is invalid");
            }
            if node
                .candidates
                .iter()
                .all(|candidate| candidate.descriptor.kind.is_provider())
                && (!whole_work_envelope || !self.require_transactional_results)
            {
                return Err("provider-only placement requires one transactional whole-work node");
            }
            let mut candidates = BTreeSet::new();
            let mut candidate_arities = Vec::with_capacity(node.candidates.len());
            for candidate in &node.candidates {
                candidate.descriptor.validate()?;
                if matches!(
                    candidate.descriptor.kind,
                    super::ExecutionCandidateKind::OriginalVmRegion
                        | super::ExecutionCandidateKind::OriginalNativeRegion
                ) && (!whole_work_envelope
                    || candidate.output_facts.is_some()
                    || candidate.transactional_results)
                {
                    return Err("original VM region is only a source-ordered whole-work route");
                }
                let arity = source_output::output_arity(candidate)?;
                candidate_arities.push(arity);
                if !candidates.insert(candidate.descriptor.identity.as_str()) {
                    return Err("candidate identities must be unique within a node");
                }
            }
            if !whole_work_envelope && candidate_arities.iter().any(Option::is_none) {
                return Err("placement source candidates lack matching output roles");
            }
            if let Some(first) = candidate_arities.iter().copied().flatten().next() {
                if candidate_arities
                    .iter()
                    .flatten()
                    .any(|candidate| *candidate != first)
                {
                    return Err("placement candidates disagree on public output count");
                }
            }
            output_arities.push(candidate_arities);
        }
        let mut edges = BTreeSet::new();
        for edge in &self.graph.edges {
            if edge.from >= edge.to || edge.to as usize >= self.graph.nodes.len() {
                return Err("placement edges must follow topological node order");
            }
            let arities = &output_arities[edge.from as usize];
            let Some(arity) = arities[0] else {
                return Err("placement source candidates lack matching output roles");
            };
            if arities.iter().any(|candidate| *candidate != Some(arity)) {
                return Err("placement source candidates lack matching output roles");
            }
            let ordinal = match edge.source_output {
                Some(ordinal) => ordinal,
                None if arity == 1 => SourceOutputOrdinal::new(0),
                None => return Err("mixed placement edge requires a source-output ordinal"),
            };
            if ordinal.index() as usize >= arity {
                return Err("placement edge source-output ordinal is out of range");
            }
            if !edges.insert((edge.from, ordinal, edge.to)) {
                return Err("placement graph edges must be unique");
            }
        }
        if self.limits.max_frontier_states == 0
            || self.limits.max_expansions == 0
            || self.limits.max_edges == 0
            || self.limits.max_providers == 0
        {
            return Err("placement search limits must be non-zero");
        }
        self.resources.validate()?;
        if self.signature.graph_output_authority != output_authority::digest(&self.graph) {
            return Err("placement graph output authority does not match signature");
        }
        Ok(())
    }
}

#[cfg(test)]
mod source_output_tests;

#[cfg(test)]
mod tests {
    use crate::{
        CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState, Digest,
        EstimateConfidence, EstimateSource, ExecutionCandidateKind, ExecutionCostComponents,
        ExecutionCostEstimate, PlacementRevision, ProviderResourceSnapshot,
    };

    use super::*;

    pub(super) fn candidate(identity: &str, provider: bool) -> PlacementGraphCandidate {
        let kind = if provider {
            ExecutionCandidateKind::ProviderFusion
        } else {
            ExecutionCandidateKind::SharedRuntime
        };
        PlacementGraphCandidate {
            descriptor: ExecutionCandidateDescriptor {
                identity: identity.into(),
                region: None,
                kind,
                execution_location: if provider {
                    CandidateExecutionLocation::Provider { device_id: 7 }
                } else {
                    CandidateExecutionLocation::Host
                },
                preparation: CandidatePreparationState::Warm,
                cost: ExecutionCostEstimate {
                    components: ExecutionCostComponents {
                        execution_ns: 1,
                        ..ExecutionCostComponents::default()
                    },
                    scratch_bytes: Some(0),
                    confidence: EstimateConfidence::Exact,
                    source: EstimateSource::Synthetic,
                },
                // Execution and resulting residency are intentionally
                // independent: provider work may materialize a host result.
                output_residency: CandidateOutputResidency::Host,
                presentation: crate::CandidateOutputPresentation::Direct,
                guards: Vec::new(),
            },
            resources: CandidateResourceDemand {
                retained_bytes: 0.into(),
                scratch_bytes: 0.into(),
                ..CandidateResourceDemand::default()
            },
            output_facts: None,
            transactional_results: true,
        }
    }

    pub(super) fn request() -> PlacementPlanRequest {
        PlacementPlanRequest {
            signature: PlacementSignature {
                region: None,
                operation: "test.graph".into(),
                runtime_facts: Digest::sha256(b"facts"),
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: None,
                    catalog: Digest::sha256(b"catalog"),
                    compiler: Digest::sha256(b"compiler"),
                    authority: crate::PlacementExecutionAuthorityRevision::Provider(
                        Digest::sha256(b"provider"),
                    ),
                    policy: Digest::sha256(b"policy"),
                },
            },
            graph: PlacementGraph {
                nodes: vec![PlacementGraphNode {
                    identity: "node".into(),
                    candidates: vec![candidate("cpu", false), candidate("gpu-host", true)],
                }],
                edges: Vec::new(),
            },
            limits: PlacementGraphLimits::default(),
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(1_024),
                cancellation_requested: false,
                providers: vec![ProviderResourceSnapshot {
                    device_id: 7,
                    capacity_bytes: Some(1_024),
                    live_bytes: 0,
                    reclaimable_bytes: 0,
                    scratch_available_bytes: Some(1_024),
                    queue_depth: Some(0),
                    queue_limit: Some(1),
                    lost: false,
                    epoch: 1,
                }],
                epoch: 1,
            },
            deterministic: true,
            require_transactional_results: true,
        }
    }

    #[test]
    fn graph_contract_round_trips_with_independent_execution_and_output_locations() {
        let request = request();
        request.validate().unwrap();
        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: PlacementPlanRequest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, request);
        let known = serde_json::to_value(ResourceByteBound::Known(16)).unwrap();
        assert_eq!(known, 16);
        let decoded: ResourceByteBound = serde_json::from_value(known).unwrap();
        assert_eq!(decoded, ResourceByteBound::Known(16));
        let unavailable = serde_json::to_value(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ))
        .unwrap();
        assert_eq!(unavailable, "NoScratchBound");
        assert_eq!(
            CandidateResourceDemand::default().scratch_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
        );
    }

    #[test]
    fn graph_contract_rejects_unbounded_or_duplicate_edges() {
        let mut request = request();
        request.graph.nodes.push(PlacementGraphNode {
            identity: "second".into(),
            candidates: vec![candidate("cpu.second", false)],
        });
        let edge = PlacementGraphEdge {
            from: 0,
            to: 1,
            source_output: None,
            bytes: 1,
            host_to_provider_ns: 1,
            provider_to_host_ns: 1,
            cross_provider_ns: 1,
        };
        request.graph.edges = vec![edge, edge];
        assert_eq!(
            request.validate().unwrap_err(),
            "placement graph edges must be unique"
        );

        request.graph.edges = vec![edge];
        request.limits.max_edges = 0;
        assert!(request.validate().is_err());
    }

    #[test]
    fn provider_only_requires_one_transactional_whole_work_node() {
        let mut request = request();
        request.graph.nodes[0].candidates.remove(0);
        request
            .validate()
            .expect("provider-only whole-work is executable");

        request.require_transactional_results = false;
        assert_eq!(
            request.validate().unwrap_err(),
            "provider-only placement requires one transactional whole-work node"
        );

        request.require_transactional_results = true;
        request.graph.nodes.push(PlacementGraphNode {
            identity: "second".into(),
            candidates: vec![candidate("cpu.second", false)],
        });
        assert_eq!(
            request.validate().unwrap_err(),
            "provider-only placement requires one transactional whole-work node"
        );
    }
}
