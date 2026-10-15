use super::*;

#[test]
fn selected_typed_reduction_queue_reports_passes_and_exact_partial_scratch() {
    let _recorder = super::super::recorder::ScopedRecorder::new();
    for (strategy, partial, passes) in [
        (ReductionStrategy::SinglePass, 0, 1_u32),
        (ReductionStrategy::TwoPass, 64, 2_u32),
    ] {
        super::super::reset();
        let trace = super::super::begin_trace("fusion.reduction", None);
        let observer = FusionPlacementObserver::new(Some(trace.correlation()));
        observer.typed_reduction_submitted(
            strategy,
            SelectedReductionWork {
                output_requested_bytes: 16,
                partial_requested_bytes: partial,
                dispatches: passes,
                workgroup_bytes: 128,
                uniform_bytes: 8,
                producer_scalar_work_units: 100,
                terminal_scalar_work_units: 30,
            },
        );
        trace.complete(Some(PlacementVariant::ProviderFusion), None);
        let report = super::super::report();
        let queued = report
            .traces
            .last()
            .unwrap()
            .events
            .iter()
            .filter(|event| {
                event.kind == PlacementEventKind::Queue
                    && event.reason.as_deref() == Some("typed_reduction_submitted")
            })
            .collect::<Vec<_>>();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].variant, Some(PlacementVariant::ProviderFusion));
        assert_eq!(queued[0].duration_ns, None);
        assert_eq!(queued[0].bytes, Some(partial));
        assert_eq!(
            queued[0].attributes,
            vec![PlacementAttribute {
                key: "pass_count".into(),
                value: u64::from(passes),
            }]
        );
    }
}
