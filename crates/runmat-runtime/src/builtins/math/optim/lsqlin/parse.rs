//! Public matrix/problem grammar; no duplicated workspace or private solver loader.
use super::*;

pub(super) fn problem<'a>(
    args: crate::call::BuiltinArguments,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Problem>> + 'a {
    crate::execution::after_result(
        futures::future::lazy(move |_| arguments(args)),
        move |values| prepared_values(values, cooperation),
    )
}
fn arguments(args: crate::call::BuiltinArguments) -> BuiltinResult<Vec<Value>> {
    let count = args.len();
    let values = if count == 1 {
        let Value::Struct(value) = args.into_iter().next().unwrap() else {
            return Err(invalid(
                "single-argument form requires a scalar problem struct",
            ));
        };
        if value.fields.get("x0").is_some_and(super::warm::is_value) {
            return Err(invalid(
                "warm start is not available with the problem structure form",
            ));
        }
        let solver = value
            .fields
            .get("solver")
            .ok_or_else(|| invalid("problem requires solver"))?;
        if runmat_builtins::OptimSolver::parse(&crate::builtins::math::optim::common::field_name(
            solver,
        )?) != Some(runmat_builtins::OptimSolver::Lsqlin)
        {
            return Err(invalid("problem solver must be lsqlin"));
        }
        for field in ["C", "d", "options"] {
            if !value.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        [
            "C", "d", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
        ]
        .into_iter()
        .map(|name| value.fields.get(name).cloned().unwrap_or_else(empty))
        .collect::<Vec<_>>()
    } else {
        if !matches!(count, 4 | 6 | 8 | 9 | 10) {
            return Err(argument(
                "expected C,d, complete A/b, Aeq/beq, lb/ub pairs, then optional x0 and options",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(10, empty);
        values
    };
    Ok(values)
}
pub(super) async fn prepared_values(
    values: Vec<Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    if values.len() != 10 {
        return Err(invalid("normalized least-squares input count disagrees"));
    }
    options::Controls::preflight_container(&values[9])?;
    crate::builtins::math::optim::common::preflight_catalog_facts(
        &runmat_builtins::LSQLIN_CATALOG_ENTRY,
        &runmat_builtins::LSQLIN_ERROR_INVALID_INPUT,
        values.iter().enumerate().map(|(index, value)| {
            if index == 9 && matches!(value, Value::Struct(_)) {
                runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::Struct(
                    runmat_types::StructFact::scalar(Default::default(), false),
                ))
            } else {
                crate::value_fact::value_fact(value)
            }
        }),
    )?;
    // Admit all numerical roles before the first transfer, not piecemeal after
    // an earlier automatic argument has already downloaded.
    for (value, extension) in values.iter().take(9).zip([
        &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_D_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_A_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_B_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_AEQ_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_BEQ_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_LB_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_UB_EXTENSION,
        &runmat_builtins::LSQLIN_INTEGER_X0_EXTENSION,
    ]) {
        numeric::compatibility(
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
    let linear_constraints = has_rows(&a) || has_rows(&aeq);
    // Pure selection refuses unsupported active data before even a live control
    // carrier can acquire provider authority. Actual preparation has one owner.
    if options::Controls::selected(&options, linear_constraints)?
        == runmat_builtins::OptimAlgorithm::ActiveSet
    {
        if matches!(&x0,Value::Tensor(value)if value.is_empty()) {
            return Err(invalid("active-set requires a nonempty x0"));
        }
        if [&h, &f, &a, &b, &aeq, &beq, &lb, &ub, &x0]
            .into_iter()
            .any(|value| matches!(value, Value::SparseTensor(_)))
        {
            return Err(invalid("active-set requires full input data; use interior-point or trust-region-reflective for sparse inputs"));
        }
    }
    let controls = options::Controls::parse(options, linear_constraints, cooperation).await?;
    let c = numeric::matrix(h, cooperation)
        .await?
        .ok_or_else(|| invalid("C must be a nonempty real matrix"))?;
    let n = c.n;
    let d = numeric::vector(f, false, false, cooperation).await?;
    if d.len() != c.m {
        return Err(invalid("d must have one element per C row"));
    }
    let (a, b) = pair(a, b, n, cooperation).await?;
    let (aeq, beq) = pair(aeq, beq, n, cooperation).await?;
    let lb = bound(lb, n, f64::NEG_INFINITY, cooperation).await?;
    let ub = bound(ub, n, f64::INFINITY, cooperation).await?;
    let x0 = numeric::vector(x0, false, true, cooperation).await?;
    if !x0.is_empty() && x0.len() != n {
        return Err(invalid("x0 must have one element per C column"));
    }
    Ok(Problem {
        c,
        d,
        a,
        b,
        aeq,
        beq,
        lb,
        ub,
        x0,
        controls,
        weights: objective::Weights::Uniform,
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
            "constraint matrix/RHS dimensions must agree with C",
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
        return Err(invalid("bound has more elements than C columns"));
    }
    if !values.is_empty() && values.len() < n {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: NAME,
            identifier: "RunMat:lsqlin:ShortBounds",
            message: "lsqlin: unspecified trailing bounds are infinite",
            show_identifier: true,
        })?;
    }
    values.resize(n, default);
    Ok(values)
}

fn has_rows(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => !value.is_empty(),
        Value::SparseTensor(value) => value.rows != 0 && value.cols != 0,
        Value::GpuTensor(value) => !value.shape().contains(&0),
        _ => true,
    }
}
