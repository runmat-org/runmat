use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    reduction::{ReductionStrategy, SelectedReductionWork},
    AccelProvider, ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderWorkload,
};
use runmat_time::{duration_ns_saturating, Instant};
use runmat_value::Value;

use crate::fusion::FusionKind;

use super::{
    provider_rejection_token, record_event, tensor_representation, PlacementAttribute,
    PlacementCorrelationId, PlacementEventKind, PlacementVariant,
};

pub const fn fusion_operation_token(kind: &FusionKind) -> &'static str {
    match kind {
        FusionKind::ElementwiseChain => "fusion.elementwise",
        FusionKind::Reduction => "fusion.reduction",
        FusionKind::MatmulEpilogue => "fusion.matmul_epilogue",
        FusionKind::CenteredGram => "fusion.centered_gram",
        FusionKind::ImageNormalize => "fusion.image_normalize",
        FusionKind::PowerStepNormalize => "fusion.power_step_normalize",
        FusionKind::ExplainedVariance => "fusion.explained_variance",
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FusionPlacementObserver {
    correlation: Option<PlacementCorrelationId>,
}

/// Host-only structural preparation performed before any selected builtin
/// domain proof or provider interaction. Counters record attempted work, even
/// when the next malformed binding rejects the plan.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FusionLeafIndexWork {
    pub input_ids_indexed: u64,
    pub source_constants_indexed: u64,
    pub producer_ids_checked: u64,
    pub unique_leaves: u64,
    pub resident_inputs_checked: u64,
}

/// Request-local exact scalar proof work, separate from selected domain probes.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FusionScalarEvidenceWork {
    pub attempted: u64,
    pub produced: u64,
}

impl FusionPlacementObserver {
    pub(crate) fn new(correlation: Option<PlacementCorrelationId>) -> Self {
        Self { correlation }
    }

    pub(crate) fn leaf_index_preparation(
        &self,
        started: Instant,
        work: FusionLeafIndexWork,
        completed: bool,
    ) {
        let attributes = [
            ("completed", u64::from(completed)),
            ("input_ids_indexed", work.input_ids_indexed),
            ("source_constants_indexed", work.source_constants_indexed),
            ("producer_ids_checked", work.producer_ids_checked),
            ("unique_leaves", work.unique_leaves),
            ("resident_inputs_checked", work.resident_inputs_checked),
        ]
        .into_iter()
        .map(|(key, value)| PlacementAttribute {
            key: key.to_string(),
            value,
        })
        .collect::<Vec<_>>();
        self.event(
            PlacementEventKind::Prepare,
            None,
            Some("fusion_leaf_index_wall"),
            Some(duration_ns_saturating(started.elapsed())),
            None,
            &attributes,
        );
    }

    pub(crate) fn scalar_evidence_preparation(
        &self,
        duration_ns: u64,
        work: FusionScalarEvidenceWork,
        completed: bool,
    ) {
        if work.attempted == 0 {
            return;
        }
        let attributes = [
            ("completed", u64::from(completed)),
            ("attempted", work.attempted),
            ("produced", work.produced),
        ]
        .into_iter()
        .map(|(key, value)| PlacementAttribute {
            key: key.to_string(),
            value,
        })
        .collect::<Vec<_>>();
        self.event(
            PlacementEventKind::Prepare,
            None,
            Some("fusion_scalar_evidence_wall"),
            Some(duration_ns),
            None,
            &attributes,
        );
    }

    pub(crate) fn select_provider(
        &self,
        provider: &dyn AccelProvider,
        kind: &FusionKind,
        inputs: &[&Value],
        outputs: &[ProviderRepresentation],
        workload: ProviderWorkload,
    ) -> Result<ProviderFeasibilityQuery> {
        let started = Instant::now();
        let inputs = inputs
            .iter()
            .filter_map(|value| tensor_representation(value, provider.precision()))
            .collect();
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(fusion_operation_token(kind)),
            family: ProviderOperationFamily::Fusion,
            inputs,
            outputs: outputs.to_vec(),
            workload,
        };
        let feasibility = provider.query_feasibility(&query);
        let duration_ns = Some(duration_ns_saturating(started.elapsed()));
        match feasibility {
            ProviderFeasibility::Supported { estimate } => {
                let attributes = [
                    ("elements", query.workload.elements),
                    ("scalar_work_units", query.workload.scalar_work_units),
                    ("batch", query.workload.batch),
                    ("transient_bytes", estimate.transient_bytes),
                    ("output_bytes", estimate.output_bytes),
                    ("dispatches", estimate.dispatches.map(u64::from)),
                ]
                .into_iter()
                .filter_map(|(key, value)| {
                    value.map(|value| PlacementAttribute {
                        key: key.to_string(),
                        value,
                    })
                })
                .collect::<Vec<_>>();
                self.event(
                    PlacementEventKind::Candidate,
                    Some(PlacementVariant::ProviderFusion),
                    Some("provider_feasible"),
                    duration_ns,
                    None,
                    &attributes,
                );
                Ok(query)
            }
            ProviderFeasibility::Rejected { rejection } => {
                let reason = provider_rejection_token(rejection.code);
                self.event(
                    PlacementEventKind::Selected,
                    Some(PlacementVariant::SharedRuntime),
                    Some(reason),
                    duration_ns,
                    None,
                    &[],
                );
                Err(anyhow!("fusion provider rejected operation: {reason}"))
            }
        }
    }

    pub(crate) fn selected(
        &self,
        variant: PlacementVariant,
        reason: &'static str,
        estimated_ns: Option<u64>,
    ) {
        self.event(
            PlacementEventKind::Selected,
            Some(variant),
            Some(reason),
            None,
            None,
            &estimated_ns
                .map(|value| PlacementAttribute {
                    key: "estimated_total_ns".to_string(),
                    value,
                })
                .into_iter()
                .collect::<Vec<_>>(),
        );
    }

    pub(crate) fn provider_candidate(&self) {
        self.event(
            PlacementEventKind::Candidate,
            Some(PlacementVariant::ProviderFusion),
            Some("provider_feasible"),
            None,
            None,
            &[],
        );
    }

    /// The immutable selected physical recipe was submitted and its output
    /// admitted. This is enqueue evidence, not fence completion or elapsed
    /// provider time. `bytes` is the selected partial scratch allocation.
    pub(crate) fn typed_reduction_submitted(
        &self,
        strategy: ReductionStrategy,
        work: SelectedReductionWork,
    ) {
        let pass_count = match strategy {
            ReductionStrategy::SinglePass => 1,
            ReductionStrategy::TwoPass => 2,
        };
        self.event(
            PlacementEventKind::Queue,
            Some(PlacementVariant::ProviderFusion),
            Some("typed_reduction_submitted"),
            None,
            Some(work.partial_requested_bytes),
            &[PlacementAttribute {
                key: "pass_count".into(),
                value: pass_count,
            }],
        );
    }

    pub(crate) fn provider_rejected(&self, code: runmat_accelerate_api::ProviderRejectionCode) {
        self.event(
            PlacementEventKind::Selected,
            Some(PlacementVariant::SharedRuntime),
            Some(provider_rejection_token(code)),
            None,
            None,
            &[],
        );
    }

    /// Generate shader source on the host. Device pipeline compilation happens
    /// inside provider execution, not while preparing a candidate's source.
    pub(crate) fn shader_source_generation(&self, started: Instant) {
        self.timed(
            PlacementEventKind::Prepare,
            started,
            None,
            Some("generate_wgsl"),
        );
    }

    pub(crate) fn admission(
        &self,
        started: Instant,
        work: &runmat_runtime::builtins::common::spec::FusionAdmissionWork,
        completed: bool,
    ) {
        use runmat_runtime::builtins::common::spec::FusionAdmissionWork;
        if let FusionAdmissionWork::DomainProof { stages, .. } = work {
            use runmat_runtime::builtins::common::spec::FusionProofStage;
            let mut host_elements = 0u64;
            let mut provider_elements = Some(0u64);
            let mut readback_elements = 0u64;
            let mut graph_examined = 0u64;
            let mut graph_indexed = 0u64;
            let mut leaves_indexed = 0u64;
            let mut range_operands = 0u64;
            let mut control_values = 0u64;
            let mut control_shape_dimensions = 0u64;
            let mut control_string_bytes = 0u64;
            let mut control_char_elements = 0u64;
            for stage in stages {
                match stage {
                    FusionProofStage::GraphIndex { operations, leaves } => {
                        graph_indexed = graph_indexed.saturating_add(*operations);
                        leaves_indexed = leaves_indexed.saturating_add(*leaves);
                    }
                    FusionProofStage::GraphWalk { examined } => {
                        graph_examined = graph_examined.saturating_add(*examined);
                    }
                    FusionProofStage::RangePropagation { operands } => {
                        range_operands = range_operands.saturating_add(*operands);
                    }
                    FusionProofStage::ControlMaterialization {
                        values,
                        shape_dimensions,
                        string_bytes,
                        char_elements,
                    } => {
                        control_values = control_values.saturating_add(*values);
                        control_shape_dimensions =
                            control_shape_dimensions.saturating_add(*shape_dimensions);
                        control_string_bytes = control_string_bytes.saturating_add(*string_bytes);
                        control_char_elements =
                            control_char_elements.saturating_add(*char_elements);
                    }
                    FusionProofStage::HostScan { elements } => {
                        host_elements = host_elements.saturating_add(*elements);
                    }
                    FusionProofStage::Provider { input, .. } => {
                        provider_elements = provider_elements
                            .and_then(|total| input.checked_element_count()?.checked_add(total));
                    }
                    FusionProofStage::ScalarReadback { elements } => {
                        readback_elements = readback_elements.saturating_add(*elements);
                    }
                }
            }
            let mut attributes = vec![
                PlacementAttribute {
                    key: "completed".into(),
                    value: u64::from(completed),
                },
                PlacementAttribute {
                    key: "proof_stages".into(),
                    value: stages.len() as u64,
                },
                PlacementAttribute {
                    key: "host_elements".into(),
                    value: host_elements,
                },
                PlacementAttribute {
                    key: "graph_examined".into(),
                    value: graph_examined,
                },
                PlacementAttribute {
                    key: "graph_indexed".into(),
                    value: graph_indexed,
                },
                PlacementAttribute {
                    key: "leaves_indexed".into(),
                    value: leaves_indexed,
                },
                PlacementAttribute {
                    key: "range_operands".into(),
                    value: range_operands,
                },
                PlacementAttribute {
                    key: "control_values".into(),
                    value: control_values,
                },
                PlacementAttribute {
                    key: "control_shape_dimensions".into(),
                    value: control_shape_dimensions,
                },
                PlacementAttribute {
                    key: "control_string_bytes".into(),
                    value: control_string_bytes,
                },
                PlacementAttribute {
                    key: "control_char_elements".into(),
                    value: control_char_elements,
                },
                PlacementAttribute {
                    key: "scalar_readback_elements".into(),
                    value: readback_elements,
                },
            ];
            if let Some(provider_elements) = provider_elements {
                attributes.push(PlacementAttribute {
                    key: "provider_elements".into(),
                    value: provider_elements,
                });
            }
            self.event(
                PlacementEventKind::Prepare,
                Some(PlacementVariant::ProviderFusion),
                Some("domain_proof_wall"),
                Some(duration_ns_saturating(started.elapsed())),
                None,
                &attributes,
            );
            return;
        }
        let (reason, elements) = match work {
            FusionAdmissionWork::None => return,
            FusionAdmissionWork::SignProof { host_elements, .. } => {
                ("sign_proof_wall", Some(*host_elements))
            }
            FusionAdmissionWork::Host { elements } => ("domain_host_preparation", Some(*elements)),
            FusionAdmissionWork::ProviderProbe { inputs, .. } => (
                "domain_provider_probe_wall",
                inputs.iter().try_fold(0u64, |sum, input| {
                    sum.checked_add(input.checked_element_count()?)
                }),
            ),
            FusionAdmissionWork::DomainProof { .. } => unreachable!("handled above"),
        };
        let mut attributes = vec![PlacementAttribute {
            key: "completed".into(),
            value: u64::from(completed),
        }];
        if let Some(elements) = elements {
            attributes.push(PlacementAttribute {
                key: "input_elements".into(),
                value: elements,
            });
        }
        self.event(
            PlacementEventKind::Prepare,
            Some(PlacementVariant::ProviderFusion),
            Some(reason),
            Some(duration_ns_saturating(started.elapsed())),
            None,
            &attributes,
        );
    }

    pub(crate) fn upload(&self, started: Instant, bytes: Option<u64>) {
        self.timed(PlacementEventKind::Upload, started, bytes, None);
    }

    pub(crate) fn queue(&self, started: Instant) {
        self.timed(
            PlacementEventKind::Queue,
            started,
            None,
            Some("host_dispatch"),
        );
    }

    pub(crate) fn queue_unmeasured(&self) {
        self.event(
            PlacementEventKind::Queue,
            Some(PlacementVariant::ProviderFusion),
            Some("provider_queue_timing_unavailable"),
            None,
            None,
            &[],
        );
    }

    pub(crate) fn kernel_unmeasured(&self) {
        self.event(
            PlacementEventKind::Kernel,
            Some(PlacementVariant::ProviderFusion),
            Some("provider_timing_unavailable"),
            None,
            None,
            &[],
        );
    }

    pub(crate) fn synchronize(&self, started: Instant, bytes: Option<u64>) {
        self.timed(PlacementEventKind::Synchronize, started, bytes, None);
    }

    fn timed(
        &self,
        kind: PlacementEventKind,
        started: Instant,
        bytes: Option<u64>,
        reason: Option<&str>,
    ) {
        self.event(
            kind,
            Some(PlacementVariant::ProviderFusion),
            reason,
            Some(duration_ns_saturating(started.elapsed())),
            bytes,
            &[],
        );
    }

    fn event(
        &self,
        kind: PlacementEventKind,
        variant: Option<PlacementVariant>,
        reason: Option<&str>,
        duration_ns: Option<u64>,
        bytes: Option<u64>,
        attributes: &[PlacementAttribute],
    ) {
        if let Some(correlation) = self.correlation {
            record_event(
                correlation,
                kind,
                variant,
                reason,
                duration_ns,
                bytes,
                attributes,
            );
        }
    }
}

#[cfg(test)]
#[path = "fusion/typed_reduction_trace_tests.rs"]
mod typed_reduction_trace_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderResidency, ProviderStorage,
    };
    use runmat_runtime::builtins::common::spec::{FusionAdmissionWork, FusionProofStage};

    #[test]
    fn observer_records_candidate_and_cost_selection() {
        let _recorder = super::super::recorder::ScopedRecorder::new();
        super::super::reset();
        let trace = super::super::begin_trace("fusion.test", None);
        let observer = FusionPlacementObserver::new(Some(trace.correlation()));
        observer.provider_candidate();
        observer.selected(PlacementVariant::ProviderFusion, "complete_cost", Some(42));
        trace.complete(Some(PlacementVariant::ProviderFusion), None);
        let report = super::super::report();
        let events = &report.traces.last().unwrap().events;
        assert!(events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Candidate));
        assert!(events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Selected));
    }

    #[test]
    fn unrepresentable_probe_size_omits_provider_count_without_losing_other_work() {
        let _recorder = super::super::recorder::ScopedRecorder::new();
        super::super::reset();
        let trace = super::super::begin_trace("fusion.unknown_probe_size", None);
        let observer = FusionPlacementObserver::new(Some(trace.correlation()));
        observer.admission(
            Instant::now(),
            &FusionAdmissionWork::DomainProof {
                binding: runmat_builtins::BuiltinBindingIdentity {
                    builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
                    variant: "default",
                },
                provider_precision: runmat_accelerate_api::ProviderPrecision::F64,
                stages: vec![
                    FusionProofStage::GraphIndex {
                        operations: 2,
                        leaves: 1,
                    },
                    FusionProofStage::Provider {
                        operation: ProviderOperationIdentity::new("reduce_min"),
                        input: ProviderRepresentation {
                            element_type: ProviderElementType::F64,
                            storage: ProviderStorage::DenseReal,
                            layout: ProviderLayout::ColumnMajorContiguous,
                            shape: vec![u64::MAX, 2],
                            residency: ProviderResidency::Device,
                        },
                    },
                ],
            },
            false,
        );
        trace.complete(None, None);
        let report = super::super::report();
        let event = report
            .traces
            .last()
            .unwrap()
            .events
            .iter()
            .find(|event| {
                event.kind == PlacementEventKind::Prepare
                    && event.reason.as_deref() == Some("domain_proof_wall")
            })
            .expect("domain proof event");
        assert!(event
            .attributes
            .iter()
            .any(|attribute| attribute.key == "graph_indexed" && attribute.value == 2));
        assert!(!event
            .attributes
            .iter()
            .any(|attribute| attribute.key == "provider_elements"));
    }
}
