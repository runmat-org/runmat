//! Original-context borrowed rectangular least squares through the same RRQR/COD.
use super::*;
pub(crate) struct BorrowedProblem<'a> {
    pub rows: usize,
    pub columns: usize,
    pub pointers: &'a [usize],
    pub indices: &'a [usize],
    pub values: &'a [f64],
    pub rhs: &'a [f64],
}
pub(crate) async fn solve_borrowed(
    problem: BorrowedProblem<'_>,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<Option<PassiveSolution>> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        owner.name,
        "finite host-memory execution is unsupported for borrowed rectangular QR/COD fill",
    )?;
    let invalid = |error| qr_error(error, runtime, owner);
    if problem.rhs.len() != problem.rows {
        return Err(invalid(SparseCompleteQrError::DimensionMismatch));
    }
    let factor = match SparseCompleteQr::<f64>::factor_borrowed(
        problem.rows,
        problem.columns,
        problem.pointers,
        problem.indices,
        problem.values,
        runtime,
    )
    .await
    {
        Ok(factor) => factor,
        Err(SparseCompleteQrError::NumericalBreakdown) => return Ok(None),
        Err(error) => return Err(invalid(error)),
    };
    let mut output = runmat_value::exact_capacity_vec(problem.columns)
        .map_err(|error| allocation(error, owner))?;
    output.resize(problem.columns, 0.);
    match factor
        .solve_optional_with_output(
            problem.rhs,
            1,
            false,
            runtime,
            std::convert::identity,
            |index, value| {
                *output
                    .get_mut(index)
                    .ok_or(SparseCompleteQrError::DimensionMismatch)? = value;
                Ok(())
            },
        )
        .await
    {
        Ok(()) => {}
        Err(SparseCompleteQrError::NumericalBreakdown) => return Ok(None),
        Err(error) => return Err(invalid(error)),
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    Ok(Some(PassiveSolution {
        values: output,
        rank: factor.rank(),
    }))
}
