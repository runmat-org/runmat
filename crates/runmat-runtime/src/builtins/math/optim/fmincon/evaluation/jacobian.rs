//! Public GC is nvariables-by-nconstraints. The solver owns its transpose in
//! CSC form, including for sparse callbacks; no dense Jacobian is required.
use super::super::numeric::{self, Geometry, Numeric, Policy};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::{build_runtime_error, BuiltinResult};
use clarabel::algebra::CscMatrix;
use runmat_value::Value;

pub(super) enum Jacobian {
    Real(CscMatrix<f64>),
    Undefined,
}

fn capacity<T>(length: usize, policy: &Policy) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(length).map_err(|source| {
        build_runtime_error(format!(
            "{}: Jacobian workspace allocation failed",
            policy.owner.name
        ))
        .with_builtin(policy.owner.name)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
    })
}

pub(super) async fn read(
    value: Value,
    variables: usize,
    constraints: usize,
    role: &str,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Jacobian> {
    let Some(value) = numeric::prepare(
        value,
        role,
        Geometry::Matrix(variables, constraints),
        policy,
        cooperation,
    )
    .await?
    else {
        return Ok(Jacobian::Undefined);
    };
    let columns = variables
        .checked_add(1)
        .ok_or_else(|| numeric::invalid(policy, role, "extent overflow"))?;
    match value {
        Value::SparseTensor(value) => {
            let mut validation = runmat_value::SparseCscValidation::new(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                value.row_indices.clone(),
                value.nnz(),
            )
            .map_err(|detail| numeric::invalid(policy, role, &detail))?;
            let quantum = std::num::NonZeroUsize::new(1024).expect("positive work quantum");
            loop {
                let done = validation
                    .advance(quantum)
                    .map_err(|detail| numeric::invalid(policy, role, &detail))?;
                cooperation.completed(1024).await?;
                if done {
                    break;
                }
            }
            let mut counts = capacity(columns, policy)?;
            for _ in 0..columns {
                counts.push(0usize);
                cooperation.completed(1).await?;
            }
            for &row in value.row_indices.iter() {
                counts[row + 1] += 1;
                cooperation.completed(1).await?;
            }
            for column in 0..variables {
                counts[column + 1] += counts[column];
                cooperation.completed(1).await?;
            }
            let mut next = capacity(variables, policy)?;
            for &offset in &counts[..variables] {
                next.push(offset);
                cooperation.completed(1).await?;
            }
            let mut rows = capacity(value.nnz(), policy)?;
            let mut entries = capacity(value.nnz(), policy)?;
            for _ in 0..value.nnz() {
                rows.push(0usize);
                entries.push(0.0);
                cooperation.completed(1).await?;
            }
            for constraint in 0..constraints {
                for source in value.col_ptrs[constraint]..value.col_ptrs[constraint + 1] {
                    let number = if value.is_logical() {
                        1.0
                    } else {
                        numeric::exact(
                            value
                                .numeric_host_buffer()
                                .and_then(|buffer| buffer.value_at(source))
                                .ok_or_else(|| {
                                    numeric::invalid(policy, role, "sparse backing is absent")
                                })?,
                        )
                        .ok_or_else(|| {
                            numeric::invalid(
                                policy,
                                role,
                                "integers must be exactly representable as double",
                            )
                        })?
                    };
                    if !number.is_finite() {
                        return Ok(Jacobian::Undefined);
                    }
                    let variable = value.row_indices[source];
                    let destination = next[variable];
                    rows[destination] = constraint;
                    entries[destination] = number;
                    next[variable] += 1;
                    cooperation.completed(1).await?;
                }
                cooperation.completed(1).await?;
            }
            Ok(Jacobian::Real(CscMatrix::new(
                constraints,
                variables,
                counts,
                rows,
                entries,
            )))
        }
        value => {
            let Numeric::Real { values, .. } =
                numeric::materialize(value, role, policy, cooperation).await?
            else {
                return Ok(Jacobian::Undefined);
            };
            let mut offsets = capacity(columns, policy)?;
            let mut rows = capacity(values.len(), policy)?;
            let mut entries = capacity(values.len(), policy)?;
            offsets.push(0);
            for variable in 0..variables {
                for constraint in 0..constraints {
                    let number = values[constraint * variables + variable];
                    if number != 0.0 {
                        rows.push(constraint);
                        entries.push(number);
                    }
                    cooperation.completed(1).await?;
                }
                offsets.push(rows.len());
            }
            Ok(Jacobian::Real(CscMatrix::new(
                constraints,
                variables,
                offsets,
                rows,
                entries,
            )))
        }
    }
}
