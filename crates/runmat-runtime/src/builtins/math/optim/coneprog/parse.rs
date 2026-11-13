//! Current positional/problem/container forms, normalized once to sparse rows.
use super::*;
use model::SocConstraint;

pub(super) async fn problem(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeProgram> {
    if args.len() == 1 {
        let Value::Struct(problem) = args.into_iter().next().expect("single problem") else {
            return Err(invalid(
                "single-argument form requires a scalar problem struct",
            ));
        };
        return from_struct(problem, runtime, cooperation).await;
    }
    let mut args = args.into_iter();
    let f = args.next().expect("preflight objective");
    let soc = args.next().expect("preflight cones");
    let mut rest = args.collect::<Vec<_>>();
    let option = if rest.len() % 2 == 1 {
        options_value(rest.pop().expect("odd options"))?
    } else {
        None
    };
    // All option metadata/text is admitted before any objective/constraint
    // provider access, using the original invocation context.
    let options = options::Controls::parse(option, runtime, cooperation).await?;
    let f = numeric::vector("f", f, FiniteMode::Finite, cooperation).await?;
    if f.is_empty() {
        return Err(invalid("f must be a nonempty numeric vector"));
    }
    let n = f.len();
    let soc = cones(soc, n, cooperation).await?;
    let mut rest = rest.into_iter();
    let (a_ineq, b_ineq) = pair(
        "A",
        rest.next().unwrap_or_else(empty_value),
        "b",
        rest.next().unwrap_or_else(empty_value),
        n,
        cooperation,
    )
    .await?;
    let (a_eq, b_eq) = pair(
        "Aeq",
        rest.next().unwrap_or_else(empty_value),
        "beq",
        rest.next().unwrap_or_else(empty_value),
        n,
        cooperation,
    )
    .await?;
    let lb = bound("lb", rest.next(), n, f64::NEG_INFINITY, cooperation).await?;
    let ub = bound("ub", rest.next(), n, f64::INFINITY, cooperation).await?;
    Ok(ConeProgram {
        f,
        soc,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
        options,
    })
}

async fn from_struct(
    problem: StructValue,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeProgram> {
    let option = optional_field(&problem, "options")
        .map(options_value)
        .transpose()?
        .flatten();
    let options = options::Controls::parse(option, runtime, cooperation).await?;
    let f = numeric::vector(
        "f",
        required_field(&problem, "f")?,
        FiniteMode::Finite,
        cooperation,
    )
    .await?;
    if f.is_empty() {
        return Err(invalid("f must be a nonempty numeric vector"));
    }
    let n = f.len();
    let soc = cones(
        optional_field(&problem, "socConstraints").unwrap_or_else(empty_value),
        n,
        cooperation,
    )
    .await?;
    let (a_ineq, b_ineq) = pair(
        "A",
        optional_field(&problem, "Aineq")
            .or_else(|| optional_field(&problem, "A"))
            .unwrap_or_else(empty_value),
        "b",
        optional_field(&problem, "bineq")
            .or_else(|| optional_field(&problem, "b"))
            .unwrap_or_else(empty_value),
        n,
        cooperation,
    )
    .await?;
    let (a_eq, b_eq) = pair(
        "Aeq",
        optional_field(&problem, "Aeq").unwrap_or_else(empty_value),
        "beq",
        optional_field(&problem, "beq").unwrap_or_else(empty_value),
        n,
        cooperation,
    )
    .await?;
    let lb = bound(
        "lb",
        optional_field(&problem, "lb"),
        n,
        f64::NEG_INFINITY,
        cooperation,
    )
    .await?;
    let ub = bound(
        "ub",
        optional_field(&problem, "ub"),
        n,
        f64::INFINITY,
        cooperation,
    )
    .await?;
    Ok(ConeProgram {
        f,
        soc,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
        options,
    })
}

fn options_value(value: Value) -> BuiltinResult<Option<StructValue>> {
    match value {
        Value::Struct(value) => Ok(Some(value)),
        value if is_empty_value(&value) => Ok(None),
        _ => Err(invalid("options must be a scalar struct or []")),
    }
}
async fn pair(
    matrix_label: &str,
    a: Value,
    rhs_label: &str,
    b: Value,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(MatrixInput, Vec<f64>)> {
    let a = numeric_matrix(matrix_label, a, cooperation).await?;
    let b = numeric::vector(rhs_label, b, FiniteMode::Finite, cooperation).await?;
    match a {
        None if b.is_empty() => Ok((MatrixInput::zeros((0, n)), b)),
        None => Err(invalid(
            "a nonempty right-hand side requires a constraint matrix",
        )),
        Some(a) if a.n != n || a.m != b.len() => Err(invalid(
            "constraint columns and right-hand-side lengths must match f and matrix rows",
        )),
        Some(a) => Ok((a, b)),
    }
}
async fn bound(
    label: &str,
    value: Option<Value>,
    n: usize,
    default: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    match value {
        None => Ok(vec![default; n]),
        Some(value) if is_empty_value(&value) => Ok(vec![default; n]),
        Some(value) => {
            let value = numeric::vector(label, value, FiniteMode::Bounds, cooperation).await?;
            if value.len() != n {
                return Err(invalid("bound length must match f"));
            }
            Ok(value)
        }
    }
}

async fn cones(
    value: Value,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<SocConstraint>> {
    let value = if matches!(&value, Value::GpuTensor(_)) {
        numeric::gather(value, cooperation).await?
    } else {
        value
    };
    if is_empty_value(&value) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    match value {
        Value::Struct(value) => out.push(single(value, n, cooperation).await?),
        Value::StructArray(array) => {
            for element in array.elements() {
                out.push(single(element.to_owned(), n, cooperation).await?);
                cooperation.completed(1).await?;
            }
        }
        Value::Cell(array) => {
            for value in array.data() {
                let Value::Struct(value) = value else {
                    return Err(invalid(
                        "cone container entries must be scalar secondordercone descriptors",
                    ));
                };
                out.push(single(value.clone(), n, cooperation).await?);
                cooperation.completed(1).await?;
            }
        }
        Value::OutputList(values) => {
            for value in &values {
                let Value::Struct(value) = value else {
                    return Err(invalid(
                        "cone container entries must be scalar secondordercone descriptors",
                    ));
                };
                out.push(single(value.clone(), n, cooperation).await?);
                cooperation.completed(1).await?;
            }
        }
        _ => {
            return Err(invalid(
                "socConstraints must be descriptors, descriptor containers or []",
            ))
        }
    }
    Ok(out)
}
async fn single(
    value: StructValue,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<SocConstraint> {
    let a = numeric_matrix(
        "socConstraints.A",
        required_field(&value, "A")?,
        cooperation,
    )
    .await?
    .ok_or_else(|| invalid("cone A must be nonempty"))?;
    let b = numeric::vector(
        "socConstraints.b",
        required_field(&value, "b")?,
        FiniteMode::Finite,
        cooperation,
    )
    .await?;
    let d = numeric::vector(
        "socConstraints.d",
        required_field(&value, "d")?,
        FiniteMode::Finite,
        cooperation,
    )
    .await?;
    let gamma = numeric::scalar(
        "socConstraints.gamma",
        required_field(&value, "gamma")?,
        cooperation,
    )
    .await?;
    if a.n != n || b.len() != a.m || d.len() != n {
        return Err(invalid(
            "cone A, b and d dimensions must match each other and f",
        ));
    }
    Ok(SocConstraint { a, b, d, gamma })
}
pub(super) fn optional_field(value: &StructValue, name: &str) -> Option<Value> {
    value
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}
pub(super) fn required_field(value: &StructValue, name: &str) -> BuiltinResult<Value> {
    optional_field(value, name).ok_or_else(|| invalid(&format!("missing required field {name}")))
}
pub(super) fn is_empty_value(value: &Value) -> bool {
    matches!(value,Value::Tensor(t) if t.is_empty())
        || matches!(value,Value::SparseTensor(t) if t.rows==0 || t.cols==0)
}
pub(super) fn empty_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}
fn invalid(detail: &str) -> RuntimeError {
    coneprog_error(&ERROR_INVALID_INPUT, detail)
}
