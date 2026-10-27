//! One retained sparse absolute-distance lift, not a per-individual model clone.
use super::super::{allocation, invalid, model, ScalarCooperation};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;

pub(in crate::builtins::math::optim) struct Input {
    pub matrix: CscMatrix<f64>,
    pub rhs: Vec<f64>,
    pub equalities: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub integers: Vec<usize>,
}

impl Input {
    pub async fn from_pairs(
        inequalities: (CscMatrix<f64>, Vec<f64>),
        equalities: (CscMatrix<f64>, Vec<f64>),
        lower: Vec<f64>,
        upper: Vec<f64>,
        integers: Vec<usize>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        super::workspace_admission(cooperation)?;
        let (a, b) = inequalities;
        let (aeq, beq) = equalities;
        let mut unbounded_lower = runmat_value::exact_capacity_vec(a.m).map_err(allocation)?;
        for _ in 0..a.m {
            unbounded_lower.push(f64::NEG_INFINITY);
            cooperation.completed(1).await?;
        }
        let (matrix, rhs, equalities) =
            super::super::normalize::constraints(a, unbounded_lower, b, aeq, beq, cooperation)
                .await?;
        Ok(Self {
            matrix,
            rhs,
            equalities,
            lower,
            upper,
            integers,
        })
    }
}

pub(super) async fn validate(
    input: &Input,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let n = input.lower.len();
    if input.matrix.n != n
        || input.upper.len() != n
        || input.matrix.m != input.rhs.len()
        || input.equalities > input.rhs.len()
        || n.checked_add(1) != Some(input.matrix.colptr.len())
        || input.matrix.rowval.len() != input.matrix.nzval.len()
        || input.matrix.colptr.first() != Some(&0)
        || input.matrix.colptr.last() != Some(&input.matrix.nzval.len())
    {
        return Err(invalid("invalid original feasibility geometry"));
    }
    let mut previous_integer = None;
    for &column in &input.integers {
        if column >= n || previous_integer.is_some_and(|previous| previous >= column) {
            return Err(invalid(
                "feasibility integer indices must be sorted and distinct",
            ));
        }
        previous_integer = Some(column);
        cooperation.completed(1).await?;
    }

    for (&lower, &upper) in input.lower.iter().zip(&input.upper) {
        if lower.is_nan() || upper.is_nan() {
            return Err(invalid("original feasibility bounds must not be NaN"));
        }
        cooperation.completed(1).await?;
    }
    for &value in &input.rhs {
        if !value.is_finite() {
            return Err(invalid(
                "original feasibility right-hand sides must be finite",
            ));
        }
        cooperation.completed(1).await?;
    }
    for column in 0..n {
        let first = input.matrix.colptr[column];
        let end = input.matrix.colptr[column + 1];
        if first > end || end > input.matrix.nzval.len() {
            return Err(invalid("invalid original feasibility CSC pointers"));
        }
        let mut previous = None;
        for slot in first..end {
            let row = input.matrix.rowval[slot];
            if row >= input.matrix.m
                || previous.is_some_and(|old| old >= row)
                || !input.matrix.nzval[slot].is_finite()
            {
                return Err(invalid(
                    "original feasibility CSC must be finite and canonical",
                ));
            }
            previous = Some(row);
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}

pub(super) async fn lift(
    input: Input,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::Workspace> {
    let Input {
        matrix,
        rhs,
        equalities,
        mut lower,
        mut upper,
        integers,
    } = input;
    let n = lower.len();
    let variables = n
        .checked_mul(2)
        .ok_or_else(|| invalid("distance variable count overflow"))?;
    let rows = matrix
        .m
        .checked_add(variables)
        .ok_or_else(|| invalid("distance row count overflow"))?;
    let entries = n
        .checked_mul(4)
        .and_then(|extra| matrix.nzval.len().checked_add(extra))
        .ok_or_else(|| invalid("distance sparse size overflow"))?;
    let mut pointers = runmat_value::exact_capacity_vec(
        variables
            .checked_add(1)
            .ok_or_else(|| invalid("distance pointer count overflow"))?,
    )
    .map_err(allocation)?;
    let mut indices = runmat_value::exact_capacity_vec(entries).map_err(allocation)?;
    let mut coefficients = runmat_value::exact_capacity_vec(entries).map_err(allocation)?;
    let mut lifted_rhs = runmat_value::exact_capacity_vec(rows).map_err(allocation)?;
    for value in rhs {
        lifted_rhs.push(value);
        cooperation.completed(1).await?;
    }
    for _ in 0..variables {
        lifted_rhs.push(0.0);
        cooperation.completed(1).await?;
    }
    pointers.push(0);
    for column in 0..n {
        for slot in matrix.colptr[column]..matrix.colptr[column + 1] {
            indices.push(matrix.rowval[slot]);
            coefficients.push(matrix.nzval[slot]);
            cooperation.completed(1).await?;
        }
        indices.push(matrix.m + 2 * column);
        coefficients.push(1.0);
        indices.push(matrix.m + 2 * column + 1);
        coefficients.push(-1.0);
        pointers.push(coefficients.len());
        cooperation.completed(1).await?;
    }
    // Original coefficients are now owned only by the lift. Release consumed
    // original CSC before constructing the admitted workspace's bound rows.
    let first_row = matrix.m;
    drop(matrix);
    for column in 0..n {
        indices.push(first_row + 2 * column);
        coefficients.push(-1.0);
        indices.push(first_row + 2 * column + 1);
        coefficients.push(-1.0);
        pointers.push(coefficients.len());
        cooperation.completed(1).await?;
    }
    // Bound carriers are extended only once, never recopied for a repair.
    lower
        .try_reserve_exact(n)
        .map_err(|error| super::super::reserve_error(error, "distance lower bounds"))?;
    upper
        .try_reserve_exact(n)
        .map_err(|error| super::super::reserve_error(error, "distance upper bounds"))?;
    let mut objective = runmat_value::exact_capacity_vec(variables).map_err(allocation)?;
    for _ in 0..n {
        objective.push(0.0);
        cooperation.completed(1).await?;
    }
    for _ in 0..n {
        lower.push(0.0);
        upper.push(f64::INFINITY);
        objective.push(1.0);
        cooperation.completed(1).await?;
    }
    model::Workspace::new(
        model::Problem {
            a: CscMatrix::new(rows, variables, pointers, indices, coefficients),
            b: lifted_rhs,
            f: objective,
            equalities,
            lower,
            upper,
            integers,
            semis: Vec::new(),
            initial: None,
        },
        cooperation,
    )
    .await
}
