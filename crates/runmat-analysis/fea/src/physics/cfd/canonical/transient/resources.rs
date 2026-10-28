//! Additional interval storage/work beyond the original prepared solver peak.
use super::*;
use serde::{ser::SerializeSeq, Serialize, Serializer};
struct Inventory<T>(usize, T);
impl<T: Serialize> Serialize for Inventory<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0))?;
        for _ in 0..self.0 {
            sequence.serialize_element(&self.1)?;
        }
        sequence.end()
    }
}
pub(super) fn reserve(p: &CanonicalFluidProblem<'_>) -> Result<CanonicalFluidReservation, Error> {
    let mut result = p.reservation;
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or(Error::Resource("transient arithmetic overflow"))
    };
    let mul = |a: u64, b: u64| {
        a.checked_mul(b)
            .ok_or(Error::Resource("transient arithmetic overflow"))
    };
    // Count-only streaming profile has the exact tuple/container shape of the
    // identity contract, without visiting predecessor values or allocating fields.
    let profile = runmat_canonical_codec::profile_serialized(
        &(
            p.numerical_input_identity,
            0.0_f64,
            1.0_f64,
            Inventory(p.topology.cells.len(), [0.0_f64; 3]),
            Inventory(p.topology.cells.len(), 0.0_f64),
            Inventory(p.topology.cells.len(), 0.0_f64),
            Inventory(p.topology.faces.len(), 0.0_f64),
        ),
        super::super::INPUT_IDENTITY_LIMITS,
    )
    .map_err(|_| Error::Resource("transient identity profile"))?;
    // Original predecessor, warm fields, component validation and the shared
    // codec's actual structural profile. No entire time history retained.
    result.resident_bytes_upper_bound = add(
        result.resident_bytes_upper_bound,
        add(
            profile.materialization_resident_bytes_estimate,
            add(mul(result.cells, 512)?, mul(result.faces, 64)?)?,
        )?,
    )?;
    result.operator_work_upper_bound = add(
        result.operator_work_upper_bound,
        add(
            mul(profile.encoded_bytes_upper_bound, 4)?,
            mul(
                result.cells,
                add(1024, mul(p.policy.maximum_outer_iterations as u64, 1024)?)?,
            )?,
        )?,
    )?;
    if result.resident_bytes_upper_bound > p.policy.resident_byte_budget()
        || result.resident_bytes_upper_bound > isize::MAX as u64
    {
        return Err(Error::Resource("transient resident bytes"));
    }
    if result.operator_work_upper_bound > p.policy.operator_work_budget() {
        return Err(Error::Resource("transient operator work"));
    }
    Ok(result)
}
