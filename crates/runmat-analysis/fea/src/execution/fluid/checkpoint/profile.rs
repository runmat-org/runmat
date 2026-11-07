//! One actual checkpoint schema, streamed at the largest possible history inventory.
use super::*;
use serde::ser::SerializeSeq;
pub(super) struct Repeated<T> {
    sample: T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut sequence = s.serialize_seq(Some(self.count))?;
        for _ in 0..self.count {
            sequence.serialize_element(&self.sample)?;
        }
        sequence.end()
    }
}
pub(super) fn complete_profile(
    p: &PreparedFluidTransient<'_>,
    scope: &Scope,
) -> AnalysisExecutionResult<runmat_canonical_codec::SerializedCanonicalProfile> {
    let schedule = super::super::FluidTransientSchedule::derive(
        &p.workload.model,
        p.workload.selected_step()?,
    )?;
    let count = schedule.points_s().len() - 1;
    let source = &p.source;
    let authorities = crate::physics::cfd::canonical::potential_boundary_authorities(
        &source.constraints,
        &p.workload.model.boundaries,
        &p.workload.model.excitations,
    );
    // One single-interval block per interval is the maximum ledger/container inventory.
    // Original arrays and boundary metadata use the shared actual block profile.
    let original = super::super::profile::original_block_profile(
        source.topology.cells.len(),
        source.topology.faces.len(),
        authorities,
        1,
        1,
    );
    Ok(runmat_canonical_codec::profile_serialized(
        &Data {
            schema_version: 1_u16,
            scope,
            next_interval: count,
            next_output_ordinal: count,
            charged_operator_work: u64::MAX,
            blocks: Repeated {
                count,
                sample: Entry {
                    start_interval: 0,
                    end_interval: 1,
                    charged_operator_work: u64::MAX,
                    original,
                },
            },
        },
        LIMITS,
    )?)
}
pub(super) fn require_complete_budget(
    p: &PreparedFluidTransient<'_>,
    scope: &Scope,
) -> AnalysisExecutionResult<u64> {
    let profile = complete_profile(p, scope)?;
    // Imported predecessor and newly constructed full checkpoint can coexist. Keeping
    // this envelope fixed across blocks makes numerical identity independent of split.
    let overhead = profile
        .materialization_resident_bytes_estimate
        .checked_mul(2)
        .and_then(|v| v.checked_add(profile.encoded_bytes_upper_bound))
        .ok_or_else(|| invalid("fluid checkpoint complete history overflow"))
        .and_then(|later| p.source.resident_with(later))?;
    let remaining = p
        .workload
        .resident_ceiling()?
        .checked_sub(overhead)
        .filter(|v| *v > 0)
        .ok_or_else(|| invalid("fluid checkpoint history exhausts numerical ceiling"))?;
    Ok(remaining)
}
