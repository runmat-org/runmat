use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    AccelProvider, ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderWorkload,
};
use runmat_time::{duration_ns_saturating, Instant};
use runmat_value::Value;

use crate::fusion::FusionKind;

use super::{
    dense_output_representation, provider_rejection_token, record_event, tensor_representation,
    PlacementAttribute, PlacementCorrelationId, PlacementEventKind, PlacementVariant,
};

pub fn fusion_operation_token(kind: &FusionKind) -> &'static str {
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

impl FusionPlacementObserver {
    pub(crate) fn new(correlation: Option<PlacementCorrelationId>) -> Self {
        Self { correlation }
    }

    pub(crate) fn select_provider(
        &self,
        provider: &dyn AccelProvider,
        kind: &FusionKind,
        inputs: &[Value],
        output_shapes: &[Vec<usize>],
        elements: Option<usize>,
        batch: Option<usize>,
    ) -> Result<ProviderFeasibilityQuery> {
        let started = Instant::now();
        let inputs = inputs
            .iter()
            .filter_map(|value| tensor_representation(value, provider.precision()))
            .collect();
        let outputs = output_shapes
            .iter()
            .map(|shape| dense_output_representation(provider.precision(), shape))
            .collect();
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(fusion_operation_token(kind)),
            family: ProviderOperationFamily::Fusion,
            inputs,
            outputs,
            workload: ProviderWorkload {
                elements: elements.and_then(|value| u64::try_from(value).ok()),
                flops: None,
                batch: batch.and_then(|value| u64::try_from(value).ok()),
            },
        };
        let feasibility = provider.query_feasibility(&query);
        let duration_ns = Some(duration_ns_saturating(started.elapsed()));
        match feasibility {
            ProviderFeasibility::Supported { estimate } => {
                let attributes = [
                    ("elements", query.workload.elements),
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

    pub(crate) fn compile(&self, started: Instant) {
        self.timed(
            PlacementEventKind::Compile,
            started,
            None,
            Some("generate_wgsl"),
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
mod tests {
    use super::*;

    #[test]
    fn observer_records_candidate_and_cost_selection() {
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
}
