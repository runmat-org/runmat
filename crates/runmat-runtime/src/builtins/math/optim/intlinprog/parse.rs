//! Public positional/problem forms; only declared numeric roles are inspected.
use super::*;
use crate::context::RuntimeContext;
use clarabel::algebra::CscMatrix;
use numeric::Role;
pub(super) async fn parse(
    args: crate::call::BuiltinArguments,
    runtime: &RuntimeContext,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(model::Problem, controls::Controls)> {
    let count = args.len();
    let values = if count == 1 {
        let Value::Struct(problem) = args.into_iter().next().unwrap() else {
            return Err(invalid("one argument requires a scalar problem struct"));
        };
        let solver = problem
            .fields
            .get("solver")
            .ok_or_else(|| invalid("problem requires solver"))?;
        if runmat_builtins::OptimSolver::parse(&common::field_name(solver)?)
            != Some(runmat_builtins::OptimSolver::Intlinprog)
        {
            return Err(invalid("problem solver must be intlinprog"));
        }
        for field in ["f", "intcon", "options"] {
            if !problem.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        [
            "f", "intcon", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
        ]
        .into_iter()
        .map(|name| problem.fields.get(name).cloned().unwrap_or_else(empty))
        .collect::<Vec<_>>()
    } else {
        if !matches!(count, 4 | 6 | 8 | 9 | 10) {
            return Err(argument("expected f,intcon,A,b and complete optional Aeq/beq,lb/ub pairs, then x0 and options"));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(10, empty);
        values
    };
    // The b cell contains exactly the two selected numeric range roles. Do not
    // recursively traverse arbitrary cells, captures or unknown problem fields.
    let ranges = range_parts(&values[3])?;
    let domain_object = domains::validate_input(&values[1], runtime)?;
    for (index, role) in [
        (0, Role::Objective),
        (1, Role::IntegerIndices),
        (2, Role::InequalityMatrix),
        (4, Role::EqualityMatrix),
        (5, Role::EqualityRhs),
        (6, Role::Lower),
        (7, Role::Upper),
        (8, Role::Initial),
    ] {
        if index != 1 || domain_object.is_none() {
            numeric::preflight(&values[index], role, Some(runtime))?;
        }
    }
    for value in &ranges {
        numeric::preflight(value, Role::InequalityRhs, Some(runtime))?;
    }
    for value in ranges.iter().chain(std::iter::once(&values[5])) {
        if crate::value_fact::value_fact(value).storage == runmat_types::StorageFact::Sparse {
            return Err(invalid("b and beq must be full vectors"));
        }
    }
    let hints = [matrix_dimensions(&values[2]), matrix_dimensions(&values[4])];
    let mut values = values.into_iter();
    let f = values.next().unwrap();
    let integers = values.next().unwrap();
    let a = values.next().unwrap();
    let _ = values.next();
    let aeq = values.next().unwrap();
    let beq = values.next().unwrap();
    let lb = values.next().unwrap();
    let ub = values.next().unwrap();
    let x0 = values.next().unwrap();
    let controls = controls::Controls::parse(values.next().unwrap(), runtime, cooperation).await?;
    let mut f = numeric::vector(f, true, false, cooperation).await?;
    let a = numeric::matrix(a, cooperation).await?;
    let aeq = numeric::matrix(aeq, cooperation).await?;
    let beq = numeric::vector(beq, false, false, cooperation).await?;
    let mut lb = numeric::vector(lb, true, true, cooperation).await?;
    let mut ub = numeric::vector(ub, true, true, cooperation).await?;
    let x0 = numeric::vector(x0, true, false, cooperation).await?;
    let n = if !f.is_empty() {
        f.len()
    } else {
        a.as_ref()
            .map(|v| v.n)
            .into_iter()
            .chain(aeq.as_ref().map(|v| v.n))
            .chain(hints.into_iter().flatten().map(|(_, cols)| cols))
            .chain([lb.len(), ub.len(), x0.len()])
            .max()
            .unwrap_or(0)
    };
    if f.is_empty() {
        f.try_reserve(n)
            .map_err(|e| reserve_error(e, "feasibility objective"))?;
        f.resize(n, 0.0);
    }
    if lb.len() > n || ub.len() > n || !x0.is_empty() && x0.len() != n {
        return Err(invalid(
            "bounds and initial-point arrays exceed the variable geometry",
        ));
    }
    lb.try_reserve(n - lb.len())
        .map_err(|e| reserve_error(e, "lower bounds"))?;
    lb.resize(n, f64::NEG_INFINITY);
    ub.try_reserve(n - ub.len())
        .map_err(|e| reserve_error(e, "upper bounds"))?;
    ub.resize(n, f64::INFINITY);
    let a = match a {
        Some(a) => a,
        None => empty_matrix(hints[0], n)?,
    };
    let aeq = match aeq {
        Some(a) => a,
        None => empty_matrix(hints[1], n)?,
    };
    if a.n != n || aeq.n != n || aeq.m != beq.len() {
        return Err(invalid(
            "constraint dimensions must agree with the variable count",
        ));
    }
    let mut ranges = ranges.into_iter();
    let (lower, upper) = if let Some(upper) = ranges.next() {
        let lower = numeric::vector(upper, false, true, cooperation).await?;
        if let Some(upper) = ranges.next() {
            (
                lower,
                numeric::vector(upper, false, true, cooperation).await?,
            )
        } else {
            let upper = lower;
            let mut lower = runmat_value::exact_capacity_vec(upper.len()).map_err(allocation)?;
            lower.resize(upper.len(), f64::NEG_INFINITY);
            (lower, upper)
        }
    } else {
        unreachable!("one or two range roles")
    };
    if lower.len() != a.m || upper.len() != a.m {
        return Err(invalid("b must have one bound per A row"));
    }
    let (integer_columns, semis) =
        domains::decode(integers, domain_object, n, &mut lb, &ub, cooperation).await?;
    let (a, b, equalities) = normalize::constraints(a, lower, upper, aeq, beq, cooperation).await?;
    Ok((
        model::Problem {
            a,
            b,
            f,
            equalities,
            lower: lb,
            upper: ub,
            integers: integer_columns,
            semis,
            initial: if x0.is_empty() { None } else { Some(x0) },
        },
        controls,
    ))
}
fn range_parts(value: &Value) -> BuiltinResult<Vec<Value>> {
    if let Value::Cell(cell) = value {
        cell.validate_layout().map_err(invalid)?;
        if cell.data().len() != 2 {
            return Err(invalid("range b must be the two-element cell {bl,bu}"));
        }
        Ok(cell.data().to_vec())
    } else {
        Ok(vec![value.clone()])
    }
}
fn matrix_dimensions(value: &Value) -> Option<(usize, usize)> {
    let fact = crate::value_fact::value_fact(value);
    let dims = fact.shape.known_dims()?;
    if dims.len() <= 2 {
        Some((
            dims.first()
                .copied()
                .flatten()
                .or(Some(1).filter(|_| dims.is_empty()))?,
            dims.get(1)
                .copied()
                .flatten()
                .or(Some(1).filter(|_| dims.len() < 2))?,
        ))
    } else {
        None
    }
}
fn empty_matrix(hint: Option<(usize, usize)>, n: usize) -> BuiltinResult<CscMatrix<f64>> {
    let (rows, cols) = match hint {
        Some((0, 0)) | None => (0, n),
        Some(shape) => shape,
    };
    let count = cols
        .checked_add(1)
        .ok_or_else(|| invalid("matrix column count overflow"))?;
    let mut pointers = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
    pointers.resize(count, 0);
    Ok(CscMatrix::new(rows, cols, pointers, Vec::new(), Vec::new()))
}
