//! Full output over borrowed floating/logical columns; native scalar algebra.
use super::*;
pub(super) async fn multiply<T: Element>(
    left: Operand<'_>,
    right: Operand<'_>,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<Value> {
    let (m, n) = (left.rows(), right.cols());
    let count = m.checked_mul(n).ok_or_else(|| {
        failure(
            crate::context::host_allocation::HostAllocationError::Overflow,
            runtime,
        )
    })?;
    if count == 0 {
        return T::full(Vec::new(), m, n)
            .map_err(|e| crate::build_runtime_error(e).with_builtin(NAME).build());
    }
    let mut work = runtime
        .map(OperatorWork::new)
        .transpose()
        .map_err(|e| operator_error(e, runtime))?;
    let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| failure(e, runtime))?;
    for _ in 0..count {
        values.push(T::zero());
        tick(&mut work, runtime).await?;
    }
    for column in 0..n {
        for (inner, coefficient) in right.column(column) {
            let coefficient = T::convert(coefficient).expect("selected compatible class");
            for (row, value) in left.column(inner) {
                T::add_product(
                    &mut values[column * m + row],
                    T::convert(value).expect("selected compatible class"),
                    coefficient,
                );
                tick(&mut work, runtime).await?;
            }
            tick(&mut work, runtime).await?;
        }
        tick(&mut work, runtime).await?;
    }
    if let Some(work) = &mut work {
        work.flush().await.map_err(|e| operator_error(e, runtime))?;
    }
    T::full(values, m, n).map_err(|e| crate::build_runtime_error(e).with_builtin(NAME).build())
}
