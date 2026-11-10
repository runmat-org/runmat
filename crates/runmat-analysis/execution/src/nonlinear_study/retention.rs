//! Borrowed admission before retaining response metadata; no decoded prefix history inventory.
use super::*;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use serde::{ser::SerializeSeq, Serialize, Serializer};

struct Attempts<'a>(&'a [PlannedResourceUsageAttempt]);
impl Serialize for Attempts<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for attempt in self.0 {
            sequence.serialize_element(&(&attempt.plan, &attempt.accounting))?;
        }
        sequence.end()
    }
}
struct StateWireInventory<'a>(&'a [ImportedSemanticState]);
impl Serialize for StateWireInventory<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for state in self.0 {
            sequence.serialize_element(&state.state)?;
        }
        sequence.end()
    }
}
pub(super) type ProposedClosure<'a> = (
    NonlinearSolidStudyResultPhase,
    &'a AnalysisWorkloadRequest,
    &'a ProgramExecutionResponse,
    &'a AnalysisPhaseExecutionEvidence,
);

impl NonlinearSolidStudyCoordinator {
    pub(super) fn admit_retention(
        &self,
        proposed: Option<ProposedClosure<'_>>,
        plan: Option<&FeaResourcePlan>,
    ) -> AnalysisBridgeResult<u64> {
        let ceiling = self.preparation.product.workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        let profile = profile_serialized(
            &(
                &self.preparation.product.compiled,
                &self.preparation.product.workload,
                &self.preparation.mesh,
                &self.closures,
                proposed,
                Attempts(&self.attempts),
                plan,
                self.block.as_ref().map(|b| &b.block),
                self.source_host
                    .as_ref()
                    .map(|h| (&h.workload, &h.artifact_access)),
                &self.source_plan,
                self.recovery_task.as_ref().map(|r| {
                    (
                        &r.request,
                        &r.host.workload,
                        &r.host.artifact_access,
                        &r.resource_plan,
                    )
                }),
                self.fields.as_ref().map(|f| (&f.manifest, &f.output)),
            ),
            CanonicalLimits::content_bounded(
                usize::try_from(ceiling)
                    .map_err(|_| limit("nonlinear retention limit exceeds address space"))?,
                65_536,
                64,
            ),
        )
        .map_err(AnalysisBridgeError::from)?;
        // Cold replay profiles borrowed state and imports separately admitted wire
        // objects. It never encodes this entire coordinator. Root publication owns
        // its separate encoding peak; keep the conservative producer allowance.
        let state_limits = CanonicalLimits::content_bounded(
            usize::try_from(ceiling)
                .map_err(|_| limit("nonlinear state retention exceeds address space"))?,
            65_536,
            64,
        );
        let encoded = if self.import_workspace_bytes.is_some() {
            0
        } else {
            profile
                .encoded_bytes_upper_bound
                .checked_add(
                    profile_serialized(&StateWireInventory(self.states()), state_limits)?
                        .encoded_bytes_upper_bound,
                )
                .ok_or_else(|| limit("nonlinear state wire inventory overflow"))?
        };
        let state_bytes = runmat_analysis_fea::execution::semantic_state_retained_storage_bytes(
            self.states().iter().map(|s| &s.state),
            state_limits,
        )?;
        let bytes = profile
            .materialization_resident_bytes_estimate
            .checked_add(encoded)
            .and_then(|n| n.checked_add(state_bytes))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| limit("nonlinear coordinator retention overflow"))?;
        let mut transport = self
            .tail
            .as_ref()
            .map_or(Ok(0), |tail| tail.retained_storage_bytes())?;
        let mut add = |bytes: u64| -> AnalysisBridgeResult<()> {
            transport = transport
                .checked_add(bytes)
                .ok_or_else(|| limit("nonlinear transport storage overflow"))?;
            Ok(())
        };
        if let Some(block) = &self.block {
            add(block.objects.retained_storage_bytes()?)?;
            add(block.receipt_objects.retained_storage_bytes()?)?;
        }
        if let Some(task) = &self.block_task {
            for input in &task.inputs {
                add(input.retained_storage_bytes()?)?;
            }
            add(task.program.retained_storage_bytes()?)?;
        }
        if let Some(fields) = &self.fields {
            add(fields.catalog_objects.retained_storage_bytes()?)?;
        }
        if let Some(task) = &self.recovery_task {
            for input in &task.inputs {
                add(input.retained_storage_bytes()?)?;
            }
            add(task.program.retained_storage_bytes()?)?;
        }
        add(self.observation_state.retained_storage_bytes(ceiling)?)?;
        let import_workspace = self
            .import_workspace_bytes
            .unwrap_or_else(|| plan.map_or(0, |p| p.request.memory_bytes));
        let bytes = bytes
            .checked_add(transport)
            .and_then(|n| n.checked_add(self.external_retained_bytes))
            .and_then(|n| n.checked_add(import_workspace))
            .ok_or_else(|| limit("nonlinear coordinator retention overflow"))?;
        if bytes > ceiling {
            return Err(crate::error::CoordinatorRetentionLimit {
                cause: runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: match self.phase() {
                        NonlinearSolidStudyPhase::ReadyForBlock { .. }
                        | NonlinearSolidStudyPhase::AwaitingBlock { .. } => {
                            "nonlinear block coordinator resident bytes"
                        }
                        NonlinearSolidStudyPhase::ReadyForFieldRecovery
                        | NonlinearSolidStudyPhase::AwaitingFieldRecovery => {
                            "nonlinear recovery coordinator resident bytes"
                        }
                        NonlinearSolidStudyPhase::ReadyForObservationFanout
                        | NonlinearSolidStudyPhase::AwaitingObservationPartitions
                        | NonlinearSolidStudyPhase::ReadyForObservationReduction { .. }
                        | NonlinearSolidStudyPhase::AwaitingObservationReduction { .. }
                        | NonlinearSolidStudyPhase::ReadyForObservationAssembly
                        | NonlinearSolidStudyPhase::AwaitingObservationAssembly => {
                            "nonlinear observation coordinator resident bytes"
                        }
                        NonlinearSolidStudyPhase::ReadyForAccuracy
                        | NonlinearSolidStudyPhase::AwaitingAccuracy => {
                            "nonlinear accuracy coordinator resident bytes"
                        }
                        NonlinearSolidStudyPhase::ReadyForState { .. }
                        | NonlinearSolidStudyPhase::AwaitingState { .. } => {
                            "nonlinear state coordinator resident bytes"
                        }
                        _ => "nonlinear publication coordinator resident bytes",
                    },
                    required: bytes,
                    maximum: ceiling,
                },
                decoded: profile.materialization_resident_bytes_estimate
                    .checked_add(state_bytes)
                    .ok_or_else(|| limit("nonlinear state decoded inventory overflow"))?,
                encoded,
                coordinator: std::mem::size_of::<Self>() as u64,
                transport,
                caller: self.external_retained_bytes,
                import_workspace,
            }.into());
        }
        // Prior responses contain only immutable references. Their entire stored inventory
        // remains bounded by the caller's shared artifact policy, independently of heap estimates.
        let mut count = 0usize;
        let mut total = 0u64;
        for response in self
            .closures
            .iter()
            .map(|c| &c.response)
            .chain(proposed.map(|p| p.2))
        {
            let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = response
            else {
                return Err(invalid(
                    "nonlinear retained completion must be externalized",
                ));
            };
            count = count
                .checked_add(result_objects.len())
                .ok_or_else(|| limit("nonlinear retained object count overflow"))?;
            for object in result_objects {
                total = total
                    .checked_add(object.encoded_length)
                    .ok_or_else(|| limit("nonlinear retained object bytes overflow"))?;
            }
        }
        let limits = self.preparation.output_policy.inventory_limits();
        if count > limits.max_objects || total > limits.max_total_bytes {
            return Err(limit(
                "original nonlinear prefix exceeds retained artifact inventory",
            ));
        }
        Ok(bytes)
    }
}
