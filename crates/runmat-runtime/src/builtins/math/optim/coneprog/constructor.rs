//! Public MATLAB signs are preserved verbatim in the descriptor fields.
use super::*;
pub(super) async fn construct(
    a: Value,
    b: Value,
    d: Value,
    gamma: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    use runmat_builtins::OptimNumericInput as Role;
    for (label, value, role) in [
        ("A", &a, Role::NonemptySparseMatrix),
        ("b", &b, Role::OptionalVector),
        ("d", &d, Role::OptionalVector),
        ("gamma", &gamma, Role::Scalar),
    ] {
        preflight::numeric(label, value, role)?;
    }
    let (a, rows, cols) = numeric::descriptor_matrix(a, cooperation).await?;
    let b = numeric::vector("b", b, FiniteMode::Finite, cooperation).await?;
    let d = numeric::vector("d", d, FiniteMode::Finite, cooperation).await?;
    let gamma = numeric::scalar("gamma", gamma, cooperation).await?;
    if b.len() != rows || d.len() != cols {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "secondordercone: b and d lengths must match rows and columns of A",
        ));
    }
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("A", a);
    fields.insert("b", output::column(b));
    fields.insert("d", output::column(d));
    fields.insert("gamma", Value::Num(gamma));
    fields.insert("Type", Value::from("secondordercone"));
    cooperation.completed(0).await?;
    Ok(Value::Struct(fields.finish()))
}
