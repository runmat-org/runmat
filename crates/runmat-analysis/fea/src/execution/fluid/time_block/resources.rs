//! Whole numerical block bounds before original predecessor traversal or solve.
use super::*;
pub(super) fn reserve(
    input: &CanonicalFluidInput<'_>,
    initial: CanonicalFluidFields<'_>,
    schedule: &FluidTransientSchedule,
    range: &Range<usize>,
    retained: usize,
    metadata: u64,
    ceiling: u64,
) -> AnalysisExecutionResult<FluidTransientBlockReservation> {
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or_else(|| invalid("fluid block byte/work overflow"))
    };
    let mul = |a: u64, b: u64| {
        a.checked_mul(b)
            .ok_or_else(|| invalid("fluid block byte/work overflow"))
    };
    let authorities = crate::physics::cfd::canonical::potential_boundary_authorities(
        input.constraints,
        input.boundary_operators,
        input.excitations,
    );
    let output = super::super::profile::original_block_encoding_profile(
        input.topology.cells.len(),
        input.topology.faces.len(),
        authorities,
        range.len(),
        retained,
    )?;
    let history = output.materialization_resident_bytes_estimate;
    let profile_work = output.encoded_bytes_upper_bound;
    // Retained history and interval evidence coexist with one numerical interval, its
    // current output and the previous interval (the next predecessor).
    let working = super::super::profile::original_block_encoding_profile(
        input.topology.cells.len(),
        input.topology.faces.len(),
        crate::physics::cfd::canonical::potential_boundary_authorities(
            input.constraints,
            input.boundary_operators,
            input.excitations,
        ),
        0,
        2,
    )?
    .materialization_resident_bytes_estimate;
    let extra = add(
        add(add(mul(metadata, 2)?, history)?, working)?,
        mul(schedule.points_s().len() as u64, 64)?,
    )?;
    let maximum = input.policy.resident_byte_budget().min(ceiling);
    if extra >= maximum {
        return Err(invalid(
            "fluid block history/source exceeds resident ceiling",
        ));
    }
    let mut resident = extra;
    let mut work = add(profile_work, mul(metadata, 64)?)?;
    for ordinal in range.clone() {
        let mut policy = input.policy;
        policy.set_resident_byte_budget(maximum - extra);
        let p = CanonicalFluidProblem::prepare_at(
            CanonicalFluidInput {
                source: input.source,
                topology: input.topology,
                material: input.material,
                constraints: input.constraints,
                boundary_operators: input.boundary_operators,
                signals: input.signals,
                excitations: input.excitations,
                policy,
            },
            ProjectionSample::TimeSeconds {
                value: schedule.points_s()[ordinal + 1],
                approach: schedule.arrival(ordinal + 1).unwrap(),
            },
        )
        .map_err(invalid)?;
        let r = p
            .prepare_backward_euler(
                schedule.points_s()[ordinal],
                schedule.points_s()[ordinal + 1],
                initial,
            )
            .map_err(invalid)?
            .reservation();
        resident = resident.max(add(extra, r.resident_bytes_upper_bound)?);
        work = add(work, r.operator_work_upper_bound)?;
    }
    if resident > maximum || work > input.policy.operator_work_budget() {
        return Err(invalid("fluid block cumulative work/resident ceiling"));
    }
    Ok(FluidTransientBlockReservation {
        resident_bytes_upper_bound: resident,
        operator_work_upper_bound: work,
        result_encoded_bytes_upper_bound: profile_work,
    })
}
