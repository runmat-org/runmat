//! Reuse actual unchanged active rows only after the current seed is feasible.
use super::Rows;
use crate::builtins::math::optim::common::{
    csc_rows::CscRows, model_policy::Policy, ScalarCooperation,
};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix as Matrix;
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    pub rows: Rows<'a>,
    pub roles: &'a [runmat_value::LinearRowRole],
}
pub(crate) struct StoredRows {
    pub matrix: Matrix,
    pub rhs: Vec<f64>,
    pub roles: Vec<runmat_value::LinearRowRole>,
    pub equalities: usize,
}
use runmat_value::LinearRowRole as Role;
use runmat_value::{
    warm_start::LinearWorkingSetData, ExactArrayStorage, HostNumericBuffer, NumericStorage,
    SparseCscValidation, SparseTensor,
};
fn role_key(role: Role) -> (u8, usize, u8) {
    match role {
        Role::Equality(i) => (0, i, 0),
        Role::Inequality(i) => (1, i, 0),
        Role::Lower(i) => (2, i, 0),
        Role::Upper(i) => (2, i, 1),
    }
}
fn locate(roles: &[(RoleKey, usize)], role: Role) -> Option<usize> {
    roles
        .binary_search_by_key(&role_key(role), |r| r.0)
        .ok()
        .map(|i| roles[i].1)
}
type RoleKey = (u8, usize, u8);
struct Geometry<'a> {
    index: &'a CscRows,
    values: &'a [f64],
    rhs: &'a [f64],
}
async fn same_row(
    left: &Geometry<'_>,
    a: usize,
    right: &Geometry<'_>,
    b: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if left.rhs[a].to_bits() != right.rhs[b].to_bits() {
        return Ok(false);
    }
    let x = left.index.row(a);
    let y = right.index.row(b);
    if x.len() != y.len() {
        return Ok(false);
    }
    for (x, y) in x.iter().zip(y) {
        if x.column != y.column || left.values[x.entry].to_bits() != right.values[y.entry].to_bits()
        {
            return Ok(false);
        }
        c.completed(1).await?;
    }
    c.completed(1).await?;
    Ok(true)
}
pub(in crate::builtins::math::optim) async fn initial(
    model: View<'_>,
    saved: &LinearWorkingSetData,
    activity: &[f64],
    tolerance: f64,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<bool>>> {
    let old = saved.constraints();
    if old.cols != model.rows.a.n {
        return Err(policy.invalid("warm start requires the same number of variables"));
    }
    if saved.equalities() != model.rows.equalities {
        return Ok(None);
    }
    let old_values = old
        .as_f64_slice()
        .ok_or_else(|| policy.invalid("warm state lost its double coefficients"))?;
    let old_rhs = saved
        .rhs()
        .as_f64_slice()
        .ok_or_else(|| policy.invalid("warm state lost its double rhs"))?;
    let old_index = CscRows::prepare(
        old.cols,
        &old.col_ptrs,
        &old.row_indices,
        |e| policy.storage(e),
        |e| policy.reserve(e),
        c,
    )
    .await?;
    let new_index = CscRows::prepare(
        model.rows.a.n,
        &model.rows.a.colptr,
        &model.rows.a.rowval,
        |e| policy.storage(e),
        |e| policy.reserve(e),
        c,
    )
    .await?;
    let left = Geometry {
        index: &old_index,
        values: old_values,
        rhs: old_rhs,
    };
    let right = Geometry {
        index: &new_index,
        values: &model.rows.a.nzval,
        rhs: model.rows.b,
    };
    // Any equality edit discards the old working set, not its new exact rows.
    for row in 0..model.rows.equalities {
        if !same_row(&left, row, &right, row, c).await? {
            return Ok(None);
        }
    }
    if activity.len() != model.rows.b.len() || model.roles.len() != model.rows.b.len() {
        return Err(policy.invalid("warm row activity and roles disagree"));
    }
    let role_order = role_order(model.roles, policy, c).await?;
    let mut active = crate::builtins::math::optim::common::solver_workspace::capacity(
        model.rows.b.len(),
        policy.identity,
    )?;
    for (row, (&value, &rhs)) in activity.iter().zip(model.rows.b).enumerate() {
        active.push(row < model.rows.equalities || value == rhs);
        c.completed(1).await?;
    }
    for (old_row, &selected) in saved.active().iter().enumerate() {
        if selected {
            let role = saved.roles()[old_row];
            if let Some(row) = locate(&role_order, role) {
                // Projection or phase-I may have changed the actual seed.
                // A formerly active but now slack row is not an equality.
                if (activity[row] - model.rows.b[row]).abs() <= tolerance
                    && same_row(&left, old_row, &right, row, c).await?
                {
                    active[row] = true;
                }
            }
        }
        c.completed(1).await?;
    }
    Ok(Some(active))
}
pub(in crate::builtins::math::optim) async fn retain(
    rows: StoredRows,
    active: Vec<bool>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<LinearWorkingSetData> {
    let StoredRows {
        matrix,
        rhs,
        roles,
        equalities,
    } = rows;
    let Matrix {
        m,
        n,
        colptr,
        rowval,
        nzval,
    } = matrix;
    let mut check = SparseCscValidation::new(m, n, colptr.into(), rowval.into(), nzval.len())
        .map_err(|e| policy.invalid(e))?;
    let quantum = std::num::NonZeroUsize::new(1024).expect("nonzero quantum");
    loop {
        let done = check.advance(quantum).map_err(|e| policy.invalid(e))?;
        c.completed(1024).await?;
        if done {
            break;
        }
    }
    let parts = check
        .into_validated()
        .map_err(|_| policy.invalid("incomplete warm constraint topology"))?;
    let matrix = SparseTensor::from_validated_numeric(
        parts,
        HostNumericBuffer::from_numeric_storage(NumericStorage::F64(nzval)),
    )
    .map_err(|e| policy.invalid(e))?;
    let rhs = HostNumericBuffer::from_numeric_storage(NumericStorage::F64(rhs));
    LinearWorkingSetData::new(
        matrix,
        rhs,
        ExactArrayStorage::shared_unbounded(roles),
        ExactArrayStorage::shared_unbounded(active),
        equalities,
    )
    .map_err(|e| policy.invalid(e))
}
async fn role_order(
    roles: &[Role],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<(RoleKey, usize)>> {
    let mut order = runmat_value::exact_capacity_vec(roles.len()).map_err(|e| policy.storage(e))?;
    for (row, &role) in roles.iter().enumerate() {
        order.push((role_key(role), row));
        c.completed(1).await?;
    }
    use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
    let mut sort = StableSortWork::new(&mut order, |a, b| a.0.cmp(&b.0));
    loop {
        match sort.advance().map_err(|source| policy.reserve(source.0))? {
            SortProgress::Pending(work) => c.completed(work).await?,
            SortProgress::Complete(work) => {
                c.completed(work).await?;
                break;
            }
        }
    }
    drop(sort);
    Ok(order)
}
