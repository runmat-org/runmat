//! Public matrix/problem grammar; no duplicated workspace or private solver loader.
use super::*;

pub(super) async fn problem(
    args: crate::call::BuiltinArguments,
    timing: Option<&SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let count = args.len();
    let values = if count == 1 {
        let Value::Struct(value) = args.into_iter().next().unwrap() else {
            return Err(invalid(
                "single-argument form requires a scalar problem struct",
            ));
        };
        let solver = value
            .fields
            .get("solver")
            .ok_or_else(|| invalid("problem requires solver"))?;
        if runmat_builtins::OptimSolver::parse(&crate::builtins::math::optim::common::field_name(
            solver,
        )?) != Some(runmat_builtins::OptimSolver::Quadprog)
        {
            return Err(invalid("problem solver must be quadprog"));
        }
        for field in ["H", "f", "options"] {
            if !value.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        [
            "H", "f", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
        ]
        .into_iter()
        .map(|name| value.fields.get(name).cloned().unwrap_or_else(empty))
        .collect::<Vec<_>>()
    } else {
        if !matches!(count, 2 | 4 | 6 | 8 | 9 | 10) {
            return Err(argument(
                "expected H,f, complete A/b, Aeq/beq, lb/ub pairs, then optional x0 and options",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(10, empty);
        values
    };
    prepared_values(values, timing, cooperation).await
}
pub(super) async fn prepared_values(
    values: Vec<Value>,
    timing: Option<&SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    // Complete datatype/intent admission precedes all control/operand transfers.
    for (value, extension) in values.iter().take(9).zip([
        &runmat_builtins::QUADPROG_INTEGER_H_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_F_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_A_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_B_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_AEQ_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_BEQ_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_LB_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_UB_EXTENSION,
        &runmat_builtins::QUADPROG_INTEGER_X0_EXTENSION,
    ]) {
        numeric::preflight(
            value,
            extension,
            cooperation.runtime(),
            numeric::CompatibilityRole::Data,
        )?;
    }
    let mut values = values.into_iter();
    let h = values.next().unwrap();
    let f = values.next().unwrap();
    let a = values.next().unwrap();
    let b = values.next().unwrap();
    let aeq = values.next().unwrap();
    let beq = values.next().unwrap();
    let lb = values.next().unwrap();
    let ub = values.next().unwrap();
    let x0 = values.next().unwrap();
    let options = values.next().unwrap();
    // Reject unsupported control declarations before touching any provider data.
    let mut controls = options::Controls::parse(options, cooperation).await?;
    controls.bind_timing(timing, cooperation)?;
    if controls.active.is_some()
        && [&h, &f, &a, &b, &aeq, &beq, &lb, &ub, &x0]
            .iter()
            .any(|value| matches!(value, Value::SparseTensor(_)))
    {
        return Err(invalid("active-set does not accept public sparse operands"));
    }
    let h = numeric::matrix(h, cooperation)
        .await?
        .ok_or_else(|| invalid("H must be a nonempty square matrix"))?;
    if h.m != h.n {
        return Err(invalid("H must be square"));
    }
    let n = h.n;
    let mut f = numeric::vector(f, false, true, cooperation).await?;
    if f.is_empty() {
        f = vec![0.0; n];
    }
    if f.len() != n {
        return Err(invalid("f must have one element per H column"));
    }
    let (a, b) = pair(a, b, n, cooperation).await?;
    let (aeq, beq) = pair(aeq, beq, n, cooperation).await?;
    let lb = bound(lb, n, f64::NEG_INFINITY, cooperation).await?;
    let ub = bound(ub, n, f64::INFINITY, cooperation).await?;
    let x0 = numeric::vector(x0, false, false, cooperation).await?;
    if !x0.is_empty() && x0.len() != n {
        return Err(invalid("x0 must have one element per H column"));
    }
    let p = psd::symmetric(h, cooperation).await?;
    Ok(Problem {
        p,
        f,
        a,
        b,
        aeq,
        beq,
        lb,
        ub,
        x0,
        controls,
    })
}
async fn pair(
    a: Value,
    b: Value,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Matrix, Vec<f64>)> {
    let a = numeric::matrix(a, cooperation).await?;
    let b = numeric::vector(b, false, false, cooperation).await?;
    match a {
        None if b.is_empty() => Ok((Matrix::zeros((0, n)), b)),
        Some(a) if a.n == n && a.m == b.len() => Ok((a, b)),
        _ => Err(invalid(
            "constraint matrix/RHS dimensions must agree with H",
        )),
    }
}
async fn bound(
    value: Value,
    n: usize,
    default: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = numeric::vector(value, true, true, cooperation).await?;
    if values.len() > n {
        return Err(invalid("bound has more elements than H columns"));
    }
    if !values.is_empty() && values.len() < n {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: NAME,
            identifier: "RunMat:quadprog:ShortBounds",
            message: "quadprog: unspecified trailing bounds are infinite",
            show_identifier: true,
        })?;
    }
    values.resize(n, default);
    Ok(values)
}
