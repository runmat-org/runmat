//! Exact executable native alternatives, shared by standalone native planning
//! and the provider composition boundary. This only constructs candidates;
//! the existing placement service still owns selection and feedback permits.
use super::super::NumericRegionWorkload;
use runmat_execution::*;
use runmat_types::RegionId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNumericRegionCpuKind {
    Generic,
    Vectorized,
}

pub struct NativeNumericRegionCpuCandidates {
    pub generic: PlacementGraphCandidate,
    pub vectorized: PlacementGraphCandidate,
}

impl NativeNumericRegionCpuCandidates {
    pub fn identity_storage_bytes(region: RegionId) -> u64 {
        let digits = |value: u32| {
            if value == 0 {
                1
            } else {
                value.ilog10() as usize + 1
            }
        };
        (2 * (4 + 2 + digits(region.function.0) + 2 + digits(region.ordinal))
            + "specialized".len()
            + "vectorized".len()) as u64
    }
    pub fn for_verified_program(
        region: RegionId,
        operations: u32,
        workload: NumericRegionWorkload,
        outputs: usize,
    ) -> Option<Self> {
        let elements = u64::try_from(workload.elements).ok()?;
        let retained = super::super::output_retained_bytes(workload, outputs).ok()?;
        let candidate = |kind: NativeNumericRegionCpuKind, rate, overhead| {
            let vectorized = kind == NativeNumericRegionCpuKind::Vectorized;
            let label = if vectorized {
                "vectorized"
            } else {
                "specialized"
            };
            let digits = |value: u32| {
                if value == 0 {
                    1
                } else {
                    value.ilog10() as usize + 1
                }
            };
            let mut identity = String::with_capacity(
                4 + label.len() + 2 + digits(region.function.0) + 2 + digits(region.ordinal),
            );
            use std::fmt::Write;
            write!(
                &mut identity,
                "cpu.{label}.f{}.r{}",
                region.function.0, region.ordinal
            )
            .expect("writing a String is infallible");
            PlacementGraphCandidate {
                descriptor: ExecutionCandidateDescriptor {
                    identity,
                    region: Some(region),
                    kind: if vectorized {
                        ExecutionCandidateKind::VectorizedNativeCpu
                    } else {
                        ExecutionCandidateKind::SpecializedNativeCpu
                    },
                    execution_location: CandidateExecutionLocation::Host,
                    preparation: CandidatePreparationState::Ready,
                    cost: ExecutionCostEstimate {
                        components: ExecutionCostComponents {
                            execution_ns: elements
                                .saturating_mul(u64::from(operations))
                                .saturating_mul(rate)
                                .saturating_add(overhead),
                            ..Default::default()
                        },
                        scratch_bytes: Some(0),
                        confidence: EstimateConfidence::Prior,
                        source: EstimateSource::Compiler,
                    },
                    output_residency: CandidateOutputResidency::Host,
                    presentation: CandidateOutputPresentation::Direct,
                    guards: Vec::new(),
                },
                resources: if vectorized {
                    CandidateResourceDemand {
                        cpu_millicores: 1_000,
                        retained_bytes: retained.into(),
                        scratch_bytes: 0.into(),
                        provider_host_workspace: None,
                        provider_input_stages: None,
                        queue_slots: 0,
                        peak_live: None,
                    }
                } else {
                    CandidateResourceDemand {
                        cpu_millicores: 1_000,
                        retained_bytes: retained.into(),
                        ..Default::default()
                    }
                },
                output_facts: None,
                transactional_results: true,
            }
        };
        Some(Self {
            generic: candidate(NativeNumericRegionCpuKind::Generic, 20, 10_000),
            vectorized: candidate(NativeNumericRegionCpuKind::Vectorized, 3, 1_000),
        })
    }
}
