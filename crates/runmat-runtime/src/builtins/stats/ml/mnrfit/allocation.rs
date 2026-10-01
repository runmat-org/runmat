//! Checked host workspace allocation for multinomial fitting.
use super::{invalid, BuiltinResult};

pub(super) fn checked_elements(rows: usize, columns: usize, role: &str) -> BuiltinResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| invalid(format!("mnrfit: {role} dimensions overflow")))
}

pub(super) fn filled(len: usize, value: f64, role: &str) -> BuiltinResult<Vec<f64>> {
    let mut out = reserved(len, role)?;
    out.resize(len, value);
    Ok(out)
}

pub(super) fn reserved(len: usize, role: &str) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
    Ok(out)
}

pub(super) fn copy(values: &[f64], role: &str) -> BuiltinResult<Vec<f64>> {
    let mut out = reserved(values.len(), role)?;
    out.extend_from_slice(values);
    Ok(out)
}

pub(super) fn columns(count: usize, rows: usize, role: &str) -> BuiltinResult<Vec<Vec<f64>>> {
    checked_elements(count, rows, role)?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
    for _ in 0..count {
        out.push(filled(rows, 0.0, role)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_shape_overflow_rejects_without_touching_memory() {
        let err = checked_elements(usize::MAX, 2, "information matrix")
            .expect_err("element product must be checked");
        assert!(err.message().contains("dimensions overflow"));
        let err = columns(usize::MAX, 2, "outcomes")
            .expect_err("nested columns must check total cells first");
        assert!(err.message().contains("dimensions overflow"));
    }
}
