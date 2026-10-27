//! Admission certificates are independent of the regularized optimizer kernel.
use super::*;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use std::collections::BTreeMap;

/// Normalize the public symmetric part once; keep only its upper triangle.
pub(super) async fn symmetric(
    h: Matrix,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Matrix> {
    let n = h.n;
    let mut pairs = vec![BTreeMap::<usize, (f64, f64)>::new(); n];
    for col in 0..n {
        for entry in h.colptr[col]..h.colptr[col + 1] {
            let row = h.rowval[entry];
            let value = h.nzval[entry];
            let pair = pairs[row.max(col)].entry(row.min(col)).or_default();
            if row <= col {
                pair.0 = value;
            } else {
                pair.1 = value;
            }
            if row == col {
                pair.1 = value;
            }
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut changed = false;
    let mut ptr = Vec::with_capacity(n + 1);
    let mut rows = Vec::new();
    let mut values = Vec::new();
    ptr.push(0);
    for column in pairs {
        for (row, (a, b)) in column {
            changed |= a != b;
            // Rational conversion avoids overflow and double-rounding in the
            // symmetric average, including the smallest subnormal inputs.
            let value = if a == b {
                a
            } else {
                ((BigRational::from_float(a).expect("finite matrix")
                    + BigRational::from_float(b).expect("finite matrix"))
                    / BigRational::from_integer(2.into()))
                .to_f64()
                .ok_or_else(|| invalid("symmetric H is outside binary64 range"))?
            };
            if value != 0.0 {
                rows.push(row);
                values.push(value);
            }
            cooperation.completed(1).await?;
        }
        ptr.push(values.len());
    }
    if changed {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: NAME,
            identifier: "RunMat:quadprog:NonSymmetricH",
            message: "quadprog: H is not symmetric; using (H+H')/2",
            show_identifier: true,
        })?;
    }
    Ok(Matrix::new(n, n, ptr, rows, values))
}
pub(super) async fn certify(
    p: &Matrix,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    conic::convexity::certify(
        p,
        runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
        cooperation,
    )
    .await
}
