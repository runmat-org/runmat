//! Local original-field temporal blocks. This is not a trusted checkpoint or native task.
mod resources;
const RESULT_PREFIX: &[u8] = b"runmat-fluid-temporal-block-result/v1\0";
use super::{invalid, AnalysisExecutionResult, FluidTransientSchedule};
use crate::{
    excitation::ProjectionSample,
    physics::cfd::canonical::{
        CanonicalFluidFields, CanonicalFluidInput, CanonicalFluidProblem, CanonicalFluidResult,
    },
};
use runmat_analysis_core::{AnalysisConstraint, AnalysisModel, ProcedureSettings, ProcedureStepId};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use std::{ops::Range, sync::Arc};

/// Frozen numerical intent borrowing original authority and initial U/p-reference/p/q.
/// The selected range indexes intervals; [a,b) ends at schedule point b.
pub struct FluidTransientBlock<'a> {
    input: CanonicalFluidInput<'a>,
    initial: CanonicalFluidFields<'a>,
    schedule: FluidTransientSchedule,
    range: Range<usize>,
    retain: Vec<usize>,
    identity: Digest,
    reservation: FluidTransientBlockReservation,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockReservation {
    pub resident_bytes_upper_bound: u64,
    pub operator_work_upper_bound: u64,
    pub result_encoded_bytes_upper_bound: u64,
}
/// Accepted-interval evidence kept for every interval of a block: the solve counts, the
/// accepted residuals and the digest of the interval's original fields. Fields themselves are
/// retained only where the authored outputs sample them (see [`retained_ordinals`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidIntervalEvidence {
    pub ordinal: usize,
    pub outer_iterations: usize,
    pub linear_iterations: u64,
    pub momentum_relative: f64,
    pub continuity_relative: f64,
    pub fields_digest: Digest,
}
/// Portable numerical claims. Decoding does not independently validate equations or lineage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockResult {
    identity: Digest,
    start_ordinal: usize,
    end_ordinal: usize,
    intervals: Vec<FluidIntervalEvidence>,
    /// Retained intervals in ascending ordinal: authored samples, the block terminal, and the
    /// predecessor of each so every retained interval is recoverable.
    retained: Arc<Vec<(usize, CanonicalFluidResult)>>,
}
impl FluidTransientBlockResult {
    /// Structural transport only. Original source-bound recovery remains mandatory.
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(runmat_canonical_codec::encode_contract(
            RESULT_PREFIX,
            "fluid-temporal-block-result",
            self,
            super::LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(runmat_canonical_codec::decode_contract(
            RESULT_PREFIX,
            "fluid-temporal-block-result",
            bytes,
            super::LIMITS,
        )?)
    }
    /// Retained original intervals with their schedule interval ordinals.
    pub fn retained_results(&self) -> &[(usize, CanonicalFluidResult)] {
        &self.retained
    }
    /// Evidence of every accepted interval of the block.
    pub fn intervals(&self) -> &[FluidIntervalEvidence] {
        &self.intervals
    }
    pub fn terminal(&self) -> Option<&CanonicalFluidResult> {
        self.retained
            .last()
            .filter(|(ordinal, _)| ordinal + 1 == self.end_ordinal)
            .map(|(_, result)| result)
    }
}
/// Digest of an interval's original fields (velocity, pressure reference and fluctuation,
/// face flux): binds non-retained intervals' evidence to the fields they accepted.
pub(crate) fn fields_digest(result: &CanonicalFluidResult) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(runmat_canonical_codec::encode_contract(
        b"runmat-fluid-interval-fields/v1\0",
        "fluid-interval-fields",
        &(
            &result.velocity_m_per_s,
            &result.pressure_reference_pa,
            &result.pressure_fluctuation_pa,
            &result.face_volume_flux_m3_per_s,
        ),
        super::LIMITS,
    )?))
}
/// Interval ordinals whose fields a block retains: every interval ending at an authored sample
/// time (or all of them for every-increment sampling), the block terminal, and the predecessor
/// of each retained interval inside the block.
pub(crate) fn retained_ordinals(
    step: &runmat_analysis_core::ProcedureStep,
    schedule: &FluidTransientSchedule,
    range: &Range<usize>,
) -> Vec<usize> {
    use runmat_analysis_core::SamplingPolicy;
    let points = schedule.points_s();
    let matches_time = |ordinal: usize, t: f64| {
        let end = points[ordinal + 1];
        (end - t).abs() <= 1e-9 * end.abs().max(t.abs()).max(f64::MIN_POSITIVE)
    };
    let mut keep = std::collections::BTreeSet::new();
    keep.insert(range.end - 1);
    for spec in step.observations.iter().filter_map(|o| o.spec()) {
        for ordinal in range.clone() {
            let sampled = match &spec.sampling {
                SamplingPolicy::EveryAcceptedIncrement => true,
                SamplingPolicy::TimeSeconds(times) => {
                    times.iter().any(|t| matches_time(ordinal, *t))
                }
                SamplingPolicy::TimeIntervalSeconds {
                    start,
                    end,
                    interval,
                } => {
                    let t = points[ordinal + 1];
                    t >= *start - 1e-9 * start.abs()
                        && t <= *end + 1e-9 * end.abs()
                        && *interval > 0.0
                        && {
                            let k = ((t - start) / interval).round();
                            matches_time(ordinal, start + k * interval)
                        }
                }
                _ => false,
            };
            if sampled {
                keep.insert(ordinal);
            }
        }
    }
    let predecessors = keep
        .iter()
        .filter(|ordinal| **ordinal > range.start)
        .map(|ordinal| ordinal - 1)
        .collect::<Vec<_>>();
    keep.extend(predecessors);
    keep.into_iter().collect()
}
impl<'a> FluidTransientBlock<'a> {
    pub fn prepare(
        model: &'a AnalysisModel,
        step_id: &ProcedureStepId,
        input: CanonicalFluidInput<'a>,
        initial: CanonicalFluidFields<'a>,
        range: Range<usize>,
    ) -> AnalysisExecutionResult<Self> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid temporal block cancelled"));
        }
        // Borrowed source profiling precedes model validation, time union and field traversal.
        let metadata = runmat_canonical_codec::profile_serialized(model, super::LIMITS)?;
        if metadata.materialization_resident_bytes_estimate > input.policy.resident_byte_budget() {
            return Err(invalid(
                "fluid block source profile exceeds resident ceiling",
            ));
        }
        runmat_analysis_core::validate_model(model).map_err(invalid)?;
        let mut selected = model
            .sequence
            .steps
            .iter()
            .filter(|s| &s.step_id == step_id);
        let step = selected
            .next()
            .filter(|_| selected.next().is_none())
            .ok_or_else(|| invalid("fluid block needs exactly one selected step"))?;
        let mismatch = if step.settings != ProcedureSettings::None {
            Some("procedure settings")
        } else if input.signals != model.signals {
            Some("signals")
        } else if input.boundary_operators != model.boundaries {
            Some("boundary operators")
        } else if model
            .constraints
            .iter()
            .filter_map(|c| match c {
                AnalysisConstraint::Fluid(c) => Some(c),
                _ => None,
            })
            .ne(input.constraints.iter())
            || model.constraints.len() != input.constraints.len()
        {
            Some("constraints")
        } else if !step.activations.is_empty() {
            Some("step activations")
        } else {
            None
        };
        if let Some(what) = mismatch {
            return Err(invalid(format!(
                "fluid block {what} differ from the authored source or need unsupported activation"
            )));
        }
        let selected_actions =
            crate::excitation::active_actions(model, Some(step)).map_err(invalid)?;
        if selected_actions
            .map(|a| a.action)
            .ne(input.excitations.iter())
        {
            return Err(invalid(
                "fluid block actions differ from shared selected cases",
            ));
        }
        let schedule = FluidTransientSchedule::derive_with_ceiling(
            model,
            step,
            input
                .policy
                .resident_byte_budget()
                .min(step.accuracy.resources.resident_byte_budget()),
        )?;
        if range.start >= range.end || range.end >= schedule.points_s().len() {
            return Err(invalid(
                "fluid block interval ordinals exceed authored schedule",
            ));
        }
        let retain = retained_ordinals(step, &schedule, &range);
        let reservation = resources::reserve(
            &input,
            initial,
            &schedule,
            &range,
            retain.len(),
            metadata.materialization_resident_bytes_estimate,
            step.accuracy.resources.resident_byte_budget(),
        )?;
        let bytes = runmat_canonical_codec::encode_contract(
            b"runmat-fluid-temporal-block/v1\0",
            "fluid-temporal-block",
            &(
                model.solver_identity_contract(),
                step_id,
                &range,
                &retain,
                schedule.points_s(),
                initial.velocity_m_per_s.canonical_values(),
                initial.pressure_reference_pa.canonical_values(),
                initial.pressure_fluctuation_pa.canonical_values(),
                initial.face_volume_flux_m3_per_s.canonical_values(),
                input.policy,
                // The fluid domain comes from the compiled material properties of the source,
                // which already reconciles any model-level domain.
                input.material,
                input.topology.mesh_digest,
                input.topology.projection_digest,
            ),
            super::LIMITS,
        )?;
        Ok(Self {
            input,
            initial,
            schedule,
            range,
            retain,
            identity: Digest::sha256(bytes),
            reservation,
        })
    }
    pub fn schedule(&self) -> &FluidTransientSchedule {
        &self.schedule
    }
    pub fn reservation(&self) -> FluidTransientBlockReservation {
        self.reservation
    }
    fn problem(&self, end: usize) -> AnalysisExecutionResult<CanonicalFluidProblem<'_>> {
        CanonicalFluidProblem::prepare_at(
            CanonicalFluidInput {
                source: self.input.source,
                topology: self.input.topology,
                material: self.input.material,
                constraints: self.input.constraints,
                boundary_operators: self.input.boundary_operators,
                signals: self.input.signals,
                excitations: self.input.excitations,
                policy: self.input.policy,
            },
            ProjectionSample::TimeSeconds {
                value: self.schedule.points_s()[end],
                approach: self.schedule.arrival(end).unwrap(),
            },
        )
        .map_err(invalid)
    }
    pub fn execute(&self) -> AnalysisExecutionResult<FluidTransientBlockResult> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid temporal block cancelled"));
        }
        let mut intervals = Vec::new();
        intervals
            .try_reserve_exact(self.range.len())
            .map_err(invalid)?;
        let mut retained: Vec<(usize, CanonicalFluidResult)> = Vec::new();
        retained
            .try_reserve_exact(self.retain.len())
            .map_err(invalid)?;
        // Only the latest interval is held beyond the retained set: it is the next predecessor.
        let mut latest: Option<CanonicalFluidResult> = None;
        for ordinal in self.range.clone() {
            let previous = latest
                .as_ref()
                .map_or(self.initial, |r| r.original_fields());
            let p = self.problem(ordinal + 1)?;
            let interval = p
                .prepare_backward_euler(
                    self.schedule.points_s()[ordinal],
                    self.schedule.points_s()[ordinal + 1],
                    previous,
                )
                .map_err(invalid)?;
            let result = interval.solve().map_err(super::fluid_error)?;
            intervals.push(FluidIntervalEvidence {
                ordinal,
                outer_iterations: result.outer_iterations,
                linear_iterations: result.linear_iterations,
                momentum_relative: result.residual.momentum_relative,
                continuity_relative: result.residual.continuity_relative,
                fields_digest: fields_digest(&result)?,
            });
            if self.retain.binary_search(&ordinal).is_ok() {
                retained.push((ordinal, result.clone()));
            }
            latest = Some(result);
        }
        Ok(FluidTransientBlockResult {
            identity: self.identity,
            start_ordinal: self.range.start,
            end_ordinal: self.range.end,
            intervals,
            retained: Arc::new(retained),
        })
    }
    /// Bind every retained interval to its evidence, and re-evaluate each retained interval
    /// whose predecessor is retained (or is the block initial state) against it. This never solves,
    /// interpolates, repairs or reconstructs fields from absolute Pa.
    pub fn recover(&self, result: &FluidTransientBlockResult) -> AnalysisExecutionResult<()> {
        if result.identity != self.identity
            || result.start_ordinal != self.range.start
            || result.end_ordinal != self.range.end
            || result.intervals.len() != self.range.len()
            || result
                .intervals
                .iter()
                .zip(self.range.clone())
                .any(|(evidence, ordinal)| evidence.ordinal != ordinal)
            || result.retained.len() != self.retain.len()
            || result
                .retained
                .iter()
                .zip(&self.retain)
                .any(|((ordinal, _), expected)| ordinal != expected)
        {
            return Err(invalid("fluid block scope/ordinal/field inventory differs"));
        }
        for (index, (ordinal, original)) in result.retained.iter().enumerate() {
            let evidence = &result.intervals[ordinal - self.range.start];
            if evidence.fields_digest != fields_digest(original)?
                || evidence.outer_iterations != original.outer_iterations
                || evidence.linear_iterations != original.linear_iterations
            {
                return Err(invalid("retained fluid interval differs from its evidence"));
            }
            // A retained interval whose predecessor is not retained is itself only the
            // predecessor of the next retained one: its fields are bound by the digest above.
            let previous = if *ordinal == self.range.start {
                self.initial
            } else {
                match index.checked_sub(1).map(|i| &result.retained[i]) {
                    Some((before, fields)) if before + 1 == *ordinal => fields.original_fields(),
                    _ => continue,
                }
            };
            let p = self.problem(ordinal + 1)?;
            p.prepare_backward_euler(
                self.schedule.points_s()[*ordinal],
                self.schedule.points_s()[ordinal + 1],
                previous,
            )
            .map_err(invalid)?
            .recover(original)
            .map_err(invalid)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use runmat_analysis_core::*;

    fn sampled(sampling: SamplingPolicy) -> ObservationRequest {
        ObservationRequest::Field(ObservationSpec {
            observation_id: ObservationId("velocity".into()),
            quantity: ObservationQuantity::Velocity,
            value_shape: ObservationValueShape::Vector3,
            target: ObservationTarget::WholeModel,
            source_location: FieldLocation::Element,
            output_location: FieldLocation::Element,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling,
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::Automatic,
        })
    }

    /// A block keeps the fields its outputs sample (plus each sample's predecessor and the
    /// block terminal), and evidence for every interval; a march of n steps does not hold n
    /// field sets.
    #[test]
    fn blocks_retain_sampled_intervals_their_predecessors_and_the_terminal() {
        let (workload, _) = crate::execution::fluid_test_workload();
        let mut model = workload.model;
        let mut step = ProcedureStep::fluid_transient("march");
        step.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.,
            start: 0.,
            end: 1.,
            interval_count: 10,
        };
        model.sequence = AnalysisSequence::from_steps("retention", vec![step.clone()]);
        let schedule = FluidTransientSchedule::derive(&model, &step).unwrap();
        assert_eq!(retained_ordinals(&step, &schedule, &(0..10)), vec![8, 9]);
        assert_eq!(retained_ordinals(&step, &schedule, &(0..1)), vec![0]);
        step.observations = vec![sampled(SamplingPolicy::TimeSeconds(vec![0.3, 0.6]))];
        // t = 0.3 ends interval 2 and t = 0.6 ends interval 5.
        assert_eq!(
            retained_ordinals(&step, &schedule, &(0..10)),
            vec![1, 2, 4, 5, 8, 9]
        );
        assert_eq!(retained_ordinals(&step, &schedule, &(2..6)), vec![2, 4, 5]);
        step.observations = vec![sampled(SamplingPolicy::TimeIntervalSeconds {
            start: 0.5,
            end: 1.,
            interval: 0.25,
        })];
        // 0.5 and 1.0 are schedule points; 0.75 is not, so no interval ends there.
        assert_eq!(
            retained_ordinals(&step, &schedule, &(0..10)),
            vec![3, 4, 8, 9]
        );
        step.observations = vec![sampled(SamplingPolicy::EveryAcceptedIncrement)];
        assert_eq!(
            retained_ordinals(&step, &schedule, &(0..10)),
            (0..10).collect::<Vec<_>>()
        );
    }
}
