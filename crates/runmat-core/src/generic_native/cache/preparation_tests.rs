use super::*;
use runmat_execution::{
    CandidateObservationEvidence, EstimateConfidence, ObservedCandidateDuration,
};
use runmat_runtime::context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts};
use std::cell::Cell;

struct StatisticsOnly {
    owner: runmat_accelerate::placement::PlacementSession,
    fresh: Cell<bool>,
}
impl RuntimePlacementService for StatisticsOnly {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host: Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        self.owner.whole_work_host_service_revision(program, host)
    }
    fn observed_candidate_duration(
        &self,
        _: &runmat_execution::PlacementSignature,
        _: &str,
        kind: runmat_execution::ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        if !self.fresh.get() {
            return CandidateObservationEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::StaleObservations,
            );
        }
        CandidateObservationEvidence::Observed(ObservedCandidateDuration {
            elapsed_ns: if kind == runmat_execution::ExecutionCandidateKind::SharedRuntime {
                100
            } else {
                200
            },
            confidence: EstimateConfidence::High,
        })
    }
    fn select_whole_work(
        &self,
        _: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, runmat_runtime::RuntimeError>
    {
        panic!("retired metadata must never reserve an execution or exploration permit")
    }
    fn plan(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, runmat_runtime::RuntimeError> {
        RuntimePlacementService::plan(&self.owner, request)
    }
    fn observe(
        &self,
        feedback: runmat_execution::PlacementFeedback,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        RuntimePlacementService::observe(&self.owner, feedback)
    }
    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.owner, invalidation);
    }
}

#[test]
fn retired_recipe_prices_without_retaining_code_and_stale_evidence_restores_exploration() {
    let mut session = crate::RunMatSession::with_options(false, false).unwrap();
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "preparation-quote@1",
            "quote.m",
            "function y = quote(x)\ny=x+1;\nend\n",
        ),
        None,
    ))
    .unwrap();
    let mut cache = GenericNativeCache::with_tiering_config(runmat_jit::tiering::TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 2,
        deterministic: true,
        ..Default::default()
    });
    let profile = cache
        .representation_profile(&[runmat_value::Value::Num(1.0)])
        .unwrap();
    cache
        .resolve_or_compile(
            &unit,
            Some("quote"),
            runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
        )
        .unwrap();
    for _ in 0..3 {
        cache
            .observe_invocation(&unit, Some("quote"), Some(&profile), 100)
            .unwrap();
    }
    let specialized = cache
        .resolve_or_schedule(&unit, Some("quote"), Some(&profile))
        .unwrap()
        .unwrap();
    assert!(matches!(
        specialized.tier,
        runmat_jit::entry::PublishedTier::Specialized { .. }
    ));
    let frozen = specialized.dependencies.clone();
    let weak_code = Rc::downgrade(&specialized.executor);
    let before = cache.compilation_count();
    let current = cache.dependencies.snapshot_all();
    cache.entries.enforce_limits(&current, 1, u64::MAX).unwrap();
    drop(specialized);
    assert!(
        weak_code.upgrade().is_none(),
        "quote metadata must not retain code"
    );
    let key = entry_key(&unit, Some("quote")).unwrap();
    let quote = cache
        .entries
        .preparation_quote(&key, profile.digest, &current)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&frozen, &quote.dependencies));
    assert!(cache
        .entries
        .preparation_quote(&key, Digest::sha256(b"other profile"), &current)
        .is_none());
    let statistics = Rc::new(StatisticsOnly {
        owner: Default::default(),
        fresh: Cell::new(true),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(statistics.clone()));
    let arguments = [runmat_value::Value::Num(1.0)];
    let invocation = crate::generic_native::placement::ReadyInvocation {
        unit: &unit,
        preferred_function: Some("quote"),
        arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&arguments),
        requested_outputs: 1,
        profile: Some(profile.digest),
    };
    let chosen = cache
        .resolve_or_schedule_with_placement(invocation, Some(&profile), &runtime, false)
        .unwrap()
        .published
        .unwrap();
    assert_eq!(chosen.tier, runmat_jit::entry::PublishedTier::Generic);
    assert_eq!(cache.compilation_count(), before);
    statistics.fresh.set(false);
    let chosen = cache
        .resolve_or_schedule_with_placement(invocation, Some(&profile), &runtime, false)
        .unwrap()
        .published
        .unwrap();
    assert!(matches!(
        chosen.tier,
        runmat_jit::entry::PublishedTier::Specialized { .. }
    ));
    assert_eq!(
        cache.compilation_count(),
        before + 1,
        "stale evidence must restore bounded exploration"
    );
    assert!(
        cache
            .entries
            .preparation_quote(&key, profile.digest, &Default::default())
            .is_none(),
        "missing dependency generation cannot certify an equivalent recipe"
    );
}
