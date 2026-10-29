//! Integer primitives whose semantics must not depend on signed SPIR-V
//! remainder for negative operands.

/// Euclidean residue for an i32 coordinate and a positive u32 period.
/// Unsigned magnitude also handles i32::MIN without signed negation overflow.
pub(crate) const POSITIVE_PERIOD_RESIDUE: &str = r#"
fn runmat_i32_positive_period_residue(value: i32, period: u32) -> u32 {
    let bits = bitcast<u32>(value);
    if (value >= 0) {
        return bits % period;
    }
    let magnitude = (~bits) + 1u;
    let residue = magnitude % period;
    return select(period - residue, 0u, residue == 0u);
}
"#;
