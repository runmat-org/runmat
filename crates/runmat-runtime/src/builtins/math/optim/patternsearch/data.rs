//! Selected numeric roles are admitted before any provider transfer. Original
//! sparse constraints retain CSC form; bound rows have one stored coefficient.
use super::{buffer, input::Input, invalid, linear::Linear, push};
use crate::builtins::math::optim::common::{
    conic, constrained_numeric as numeric, ScalarCooperation,
};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use numeric::{Geometry, Numeric, Policy};
use runmat_value::Value;

pub(super) struct Prepared {
    pub initial: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
    pub linear: Linear,
    pub inconsistent_bounds: bool,
    pub short_bounds: bool,
    pub bound_constrained: bool,
    pub linearly_constrained: bool,
    pub search_window: Option<super::search::Window>,
    pub original_bounds: Option<super::search::Bounds>,
}

pub(super) fn preflight(
    input: &Input<'_>,
    policy: &Policy,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let mut roles = Vec::with_capacity(9);
    roles.push((input.initial, "x0", Geometry::Vector));
    for (pair, label) in [
        (input.inequality, "linear inequality"),
        (input.equality, "linear equality"),
    ] {
        if let Some((matrix, rhs)) = pair {
            roles.push((matrix, label, Geometry::MatrixAny));
            roles.push((rhs, label, Geometry::Vector));
        }
    }
    if let Some(value) = input.lower {
        roles.push((value, "lower bounds", Geometry::Vector));
    }
    if let Some(value) = input.upper {
        roles.push((value, "upper bounds", Geometry::Vector));
    }
    for (value, label, geometry) in roles {
        if !numeric::preflight(value, label, geometry, policy, runtime)? {
            return Err(invalid(format!("{label} must be real")));
        }
    }
    Ok(())
}

async fn finite(
    value: &Value,
    label: &str,
    geometry: Geometry,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Vec<usize>, Vec<f64>)> {
    let Numeric::Real { shape, values } =
        numeric::read(value.clone(), label, geometry, policy, cooperation).await?
    else {
        return Err(invalid(format!("{label} must be finite and real")));
    };
    Ok((shape, values))
}

async fn matrix(
    value: &Value,
    n: usize,
    label: &str,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<CscMatrix<f64>> {
    let value = numeric::prepare(
        value.clone(),
        label,
        Geometry::MatrixAny,
        policy,
        cooperation,
    )
    .await?
    .ok_or_else(|| invalid(format!("{label} must be real")))?;
    if let Some(matrix) = conic::matrix::parse(
        value,
        conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail| invalid(format!("{label} {detail}")),
    )
    .await?
    {
        if matrix.n != n {
            return Err(invalid(format!("{label} must have numel(x0) columns")));
        }
        return Ok(matrix);
    }
    let mut pointers = buffer(
        n.checked_add(1)
            .ok_or_else(|| invalid("column count overflow"))?,
    )?;
    for _ in 0..=n {
        pointers.push(0);
        cooperation.completed(1).await?;
    }
    Ok(CscMatrix::new(0, n, pointers, Vec::new(), Vec::new()))
}

async fn pair(
    pair: Option<(&Value, &Value)>,
    n: usize,
    label: &str,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(CscMatrix<f64>, Vec<f64>)> {
    let Some((a, b)) = pair else {
        let mut pointers = buffer(
            n.checked_add(1)
                .ok_or_else(|| invalid("column count overflow"))?,
        )?;
        for _ in 0..=n {
            pointers.push(0);
            cooperation.completed(1).await?;
        }
        return Ok((
            CscMatrix::new(0, n, pointers, Vec::new(), Vec::new()),
            Vec::new(),
        ));
    };
    let matrix = matrix(a, n, label, policy, cooperation).await?;
    let (_, rhs) = finite(b, label, Geometry::Vector, policy, cooperation).await?;
    if rhs.len() != matrix.m {
        return Err(invalid(format!("{label} row and rhs extents differ")));
    }
    Ok((matrix, rhs))
}

async fn bounds(
    value: Option<&Value>,
    n: usize,
    lower: bool,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Vec<f64>, bool)> {
    let mut values = if let Some(value) = value {
        let value = numeric::prepare(
            value.clone(),
            "bounds",
            Geometry::Vector,
            policy,
            cooperation,
        )
        .await?
        .ok_or_else(|| invalid("bounds must be real"))?;
        let Numeric::Real { values, .. } =
            numeric::materialize_raw(value, "bounds", policy, cooperation).await?
        else {
            return Err(invalid("bounds must be real"));
        };
        values
    } else {
        Vec::new()
    };
    if values.len() > n || values.iter().any(|value| value.is_nan()) {
        return Err(invalid("bounds must not contain NaN or exceed numel(x0)"));
    }
    let short = !values.is_empty() && values.len() < n;
    while values.len() < n {
        push(
            &mut values,
            if lower {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
        )?;
        cooperation.completed(1).await?;
    }
    Ok((values, short))
}

pub(super) async fn prepare(
    input: &Input<'_>,
    policy: &Policy,
    retain_search_window: bool,
    retain_original_bounds: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    preflight(input, policy, cooperation.runtime())?;
    let (shape, initial) =
        finite(input.initial, "x0", Geometry::Vector, policy, cooperation).await?;
    if initial.is_empty() {
        return Err(invalid("x0 must be nonempty"));
    }
    let n = initial.len();
    let (a, inequality_rhs) = pair(
        input.inequality,
        n,
        "linear inequality",
        policy,
        cooperation,
    )
    .await?;
    let (aeq, equality_rhs) =
        pair(input.equality, n, "linear equality", policy, cooperation).await?;
    let (lower, short_lower) = bounds(input.lower, n, true, policy, cooperation).await?;
    let (upper, short_upper) = bounds(input.upper, n, false, policy, cooperation).await?;
    let equalities = equality_rhs.len();
    let linearly_constrained = a.m > 0 || aeq.m > 0;
    let mut rhs = equality_rhs;
    for value in inequality_rhs {
        push(&mut rhs, value)?;
        cooperation.completed(1).await?;
    }
    let mut lower_rows = buffer(n)?;
    let mut upper_rows = buffer(n)?;
    let mut inconsistent_bounds = false;
    let mut bound_constrained = false;
    for (&lower, &upper) in lower.iter().zip(&upper) {
        inconsistent_bounds |=
            lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY;
        bound_constrained |= lower.is_finite() || upper.is_finite();
        lower_rows.push(if lower.is_finite() {
            let row = rhs.len();
            push(&mut rhs, -lower)?;
            Some(row)
        } else {
            None
        });
        upper_rows.push(if upper.is_finite() {
            let row = rhs.len();
            push(&mut rhs, upper)?;
            Some(row)
        } else {
            None
        });
        cooperation.completed(1).await?;
    }
    let mut pointers = buffer(
        n.checked_add(1)
            .ok_or_else(|| invalid("column count overflow"))?,
    )?;
    let mut rows = Vec::new();
    let mut values = Vec::new();
    pointers.push(0);
    for column in 0..n {
        for (matrix, offset) in [(&aeq, 0), (&a, equalities)] {
            for slot in matrix.colptr[column]..matrix.colptr[column + 1] {
                push(&mut rows, offset + matrix.rowval[slot])?;
                push(&mut values, matrix.nzval[slot])?;
                cooperation.completed(1).await?;
            }
        }
        if let Some(row) = lower_rows[column] {
            push(&mut rows, row)?;
            push(&mut values, -1.0)?;
        }
        if let Some(row) = upper_rows[column] {
            push(&mut rows, row)?;
            push(&mut values, 1.0)?;
        }
        pointers.push(rows.len());
        cooperation.completed(1).await?;
    }
    let matrix = CscMatrix::new(rhs.len(), n, pointers, rows, values);
    let linear = Linear::new(matrix, rhs, equalities, cooperation).await?;
    let (search_window, original_bounds) = if inconsistent_bounds {
        (None, None)
    } else if retain_original_bounds {
        // Only a mutable SearchFcn can need both DIFFERENT numerical domains.
        // The RBF box retains the original endpoints; LHS fills absent bounds.
        let window = if retain_search_window {
            let mut l = buffer(n)?;
            let mut u = buffer(n)?;
            for (&lower, &upper) in lower.iter().zip(&upper) {
                l.push(lower);
                u.push(upper);
                cooperation.completed(1).await?;
            }
            Some(super::search::Window::new(l, u, cooperation).await?)
        } else {
            None
        };
        (window, Some(super::search::Bounds::new(lower, upper)?))
    } else if retain_search_window {
        (
            Some(super::search::Window::new(lower, upper, cooperation).await?),
            None,
        )
    } else {
        (None, None)
    };
    Ok(Prepared {
        initial,
        shape,
        scalar: matches!(
            input.initial,
            Value::Num(_) | Value::Int(_) | Value::Bool(_)
        ),
        linear,
        inconsistent_bounds,
        short_bounds: short_lower || short_upper,
        bound_constrained,
        linearly_constrained,
        search_window,
        original_bounds,
    })
}
