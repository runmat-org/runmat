//! Assemble the sole original CSC model directly from admitted dense/CSC
//! headers. Equality rows retain their typed prefix; no linear row becomes a
//! nonlinear callback lane or a copied dense normal-equation workspace.
use super::*;
use crate::builtins::math::optim::intlinprog::feasibility::Input;
use clarabel::algebra::CscMatrix;

pub(in super::super) struct Prepared {
    pub objective: Value,
    pub rng: Option<Value>,
    pub model: Input,
    pub inconsistent: bool,
}
impl Raw {
    pub(in super::super) async fn prepare(
        self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared> {
        let lower = numeric::prepare(self.lower, "lb", Geometry::Vector, cooperation).await?;
        let upper = numeric::prepare(self.upper, "ub", Geometry::Vector, cooperation).await?;
        if lower.len() == 0 || lower.len() != upper.len() {
            return Err(invalid(
                "lb and ub must be nonempty finite vectors of the same length",
            ));
        }
        let variables = lower.len();
        let lower = vector(&lower, "lb", cooperation).await?;
        let upper = vector(&upper, "ub", cooperation).await?;
        let intcon =
            numeric::prepare(self.integers, "intcon", Geometry::Vector, cooperation).await?;
        let mut integers =
            runmat_value::exact_capacity_vec(intcon.len().min(variables)).map_err(allocation)?;
        for index in 0..intcon.len() {
            let index = runmat_types::decode_shape_dimension(intcon.get(index))
                .map_err(|_| invalid("intcon must contain positive exact indices within the original variable extent"))?
                .checked_sub(1).filter(|&index| index < variables)
                .ok_or_else(|| invalid("intcon index is outside the original variable extent"))?;
            if let Err(position) = integers.binary_search(&index) {
                let work = integers.len() - position;
                integers.insert(position, index);
                cooperation.completed(work as u64).await?;
            }
            cooperation.completed(1).await?;
        }
        let a = numeric::prepare(self.inequality, "A", Geometry::MatrixAny, cooperation).await?;
        let b = numeric::prepare(self.rhs, "b", Geometry::Vector, cooperation).await?;
        let e = numeric::prepare(self.equality, "Aeq", Geometry::MatrixAny, cooperation).await?;
        let d = numeric::prepare(self.equality_rhs, "beq", Geometry::Vector, cooperation).await?;
        let inequalities = rows(&a, variables, "A")?;
        let equalities = rows(&e, variables, "Aeq")?;
        if b.len() != inequalities || d.len() != equalities {
            return Err(invalid(
                "linear right-hand sides must match their matrix row extents",
            ));
        }
        let rows = equalities
            .checked_add(inequalities)
            .ok_or_else(|| invalid("original linear row extent overflow"))?;
        let mut rhs = runmat_value::exact_capacity_vec(rows).map_err(allocation)?;
        for array in [&d, &b] {
            for index in 0..array.len() {
                let value = array.get(index);
                if !value.is_finite() {
                    return Err(invalid("linear right-hand sides must be finite"));
                }
                rhs.push(value);
                cooperation.completed(1).await?;
            }
        }
        let mut pointers = runmat_value::exact_capacity_vec(
            variables
                .checked_add(1)
                .ok_or_else(|| invalid("original CSC pointer extent overflow"))?,
        )
        .map_err(allocation)?;
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        pointers.push(0);
        for column in 0..variables {
            for (array, count, offset) in [(&e, equalities, 0), (&a, inequalities, equalities)] {
                if count == 0 {
                    continue;
                }
                for (row, value) in array.column(column) {
                    if !value.is_finite() {
                        return Err(invalid("linear coefficients must be finite"));
                    }
                    if value != 0.0 {
                        row_indices.try_reserve(1).map_err(reserve)?;
                        values.try_reserve(1).map_err(reserve)?;
                        row_indices.push(offset + row);
                        values.push(value);
                    }
                    cooperation.completed(1).await?;
                }
            }
            pointers.push(values.len());
            cooperation.completed(1).await?;
        }
        let mut inconsistent = false;
        for index in 0..variables {
            inconsistent |= lower[index] > upper[index]
                || integers.binary_search(&index).is_ok()
                    && lower[index].ceil() > upper[index].floor();
            cooperation.completed(1).await?;
        }
        Ok(Prepared {
            objective: self.objective,
            rng: self.rng,
            model: Input {
                matrix: CscMatrix::new(rows, variables, pointers, row_indices, values),
                rhs,
                equalities,
                lower,
                upper,
                integers,
            },
            inconsistent,
        })
    }
}
fn rows(array: &DoubleArray, variables: usize, role: &str) -> BuiltinResult<usize> {
    if array.rows() == 0 && (array.columns() == 0 || array.columns() == variables) {
        return Ok(0);
    }
    if array.columns() != variables {
        return Err(invalid(format!("{role} must have nvars columns")));
    }
    Ok(array.rows())
}
async fn vector(
    array: &DoubleArray,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = runmat_value::exact_capacity_vec(array.len()).map_err(allocation)?;
    for index in 0..array.len() {
        let value = array.get(index);
        if !value.is_finite() {
            return Err(invalid(format!("{role} must be finite")));
        }
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(values)
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: original model allocation failed")
        .with_builtin(super::super::NAME)
        .with_source(source)
        .build()
}
fn reserve(source: std::collections::TryReserveError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: original CSC allocation failed")
        .with_builtin(super::super::NAME)
        .with_source(source)
        .build()
}
