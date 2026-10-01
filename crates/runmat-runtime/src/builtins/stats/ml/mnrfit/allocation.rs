//! Checked host workspace allocation for multinomial fitting.
use super::{invalid, BuiltinResult, FitCooperation};

pub(super) fn checked_elements(rows: usize, columns: usize, role: &str) -> BuiltinResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| invalid(format!("mnrfit: {role} dimensions overflow")))
}

pub(super) async fn filled_cooperatively(
    len: usize,
    value: f64,
    role: &str,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = reserved(len, role)?;
    while out.len() < len {
        check()?;
        let next = out.len().saturating_add(16_384).min(len);
        let completed = next - out.len();
        out.resize(next, value);
        cooperation.completed(completed as u64).await?;
    }
    Ok(out)
}

pub(super) fn reserved(len: usize, role: &str) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
    Ok(out)
}

pub(super) async fn copy_cooperatively(
    values: &[f64],
    role: &str,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    check()?;
    let mut out = reserved(values.len(), role)?;
    for chunk in values.chunks(16_384) {
        check()?;
        out.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(out)
}

pub(super) async fn columns(
    count: usize,
    rows: usize,
    role: &str,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    checked_elements(count, rows, role)?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
    for _ in 0..count {
        out.push(filled_cooperatively(rows, 0.0, role, check, cooperation).await?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn host_shape_overflow_rejects_without_touching_memory() {
        let err = checked_elements(usize::MAX, 2, "information matrix")
            .expect_err("element product must be checked");
        assert!(err.message().contains("dimensions overflow"));
        let err = block_on(columns(
            usize::MAX,
            2,
            "outcomes",
            &|| Ok(()),
            &mut FitCooperation::new(None),
        ))
        .expect_err("nested columns must check total cells first");
        assert!(err.message().contains("dimensions overflow"));
    }
}
