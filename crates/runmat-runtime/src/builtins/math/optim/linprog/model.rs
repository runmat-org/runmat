//! Original LP rows retain their identities through sparse conic normalization.
use super::*;
use clarabel::{
    algebra::CscMatrix,
    solver::{NonnegativeConeT, SupportedConeT, ZeroConeT},
};

pub(in crate::builtins::math::optim) struct LinearProgram {
    pub f: Vec<f64>,
    pub a_ineq: clarabel::algebra::CscMatrix<f64>,
    pub b_ineq: Vec<f64>,
    pub a_eq: clarabel::algebra::CscMatrix<f64>,
    pub b_eq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
}

pub(in crate::builtins::math::optim) use crate::builtins::math::optim::common::conic::LinearRowRole as RowRole;

pub(super) struct ConicModel {
    pub a: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub f: Vec<f64>,
    pub roles: Vec<RowRole>,
    pub objective_scale: f64,
    pub row_scale: Vec<f64>,
    pub equality_rows: usize,
    pub inequality_rows: usize,
    pub seed: Vec<f64>,
}
impl ConicModel {
    pub async fn new(
        problem: LinearProgram,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let n = problem.f.len();
        let me = problem.b_eq.len();
        let mi = problem.b_ineq.len();
        let mut roles = Vec::new();
        let mut b = Vec::new();
        for (row, &rhs) in problem.b_eq.iter().enumerate() {
            roles.push(RowRole::Equality(row));
            b.push(rhs);
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        for (row, &rhs) in problem.b_ineq.iter().enumerate() {
            roles.push(RowRole::Inequality(row));
            b.push(rhs);
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        let mut lower = vec![None; n];
        let mut upper = vec![None; n];
        let mut seed: Vec<f64> = vec![0.0; n];
        for j in 0..n {
            if problem.lb[j].is_finite() {
                lower[j] = Some(b.len());
                roles.push(RowRole::Lower(j));
                b.push(-problem.lb[j]);
                seed[j] = seed[j].max(problem.lb[j]);
            }
            cooperation.completed(1).await?;
        }
        for j in 0..n {
            if problem.ub[j].is_finite() {
                upper[j] = Some(b.len());
                roles.push(RowRole::Upper(j));
                b.push(problem.ub[j]);
                seed[j] = seed[j].min(problem.ub[j]);
            }
            cooperation.completed(1).await?;
        }
        let nz = problem
            .a_eq
            .nzval
            .len()
            .checked_add(problem.a_ineq.nzval.len())
            .and_then(|count| count.checked_add(lower.iter().filter(|row| row.is_some()).count()))
            .and_then(|count| count.checked_add(upper.iter().filter(|row| row.is_some()).count()))
            .ok_or_else(|| invalid("sparse model size overflow"))?;
        let mut pointers = runmat_value::exact_capacity_vec(
            n.checked_add(1)
                .ok_or_else(|| invalid("model column count overflow"))?,
        )
        .map_err(allocation)?;
        let mut indices = runmat_value::exact_capacity_vec(nz).map_err(allocation)?;
        let mut values = runmat_value::exact_capacity_vec(nz).map_err(allocation)?;
        let mut row_scale = vec![0.0_f64; b.len()];
        pointers.push(0);
        for col in 0..n {
            for (matrix, offset) in [(&problem.a_eq, 0), (&problem.a_ineq, me)] {
                for first in (matrix.colptr[col]..matrix.colptr[col + 1]).step_by(1024) {
                    let end = first.saturating_add(1024).min(matrix.colptr[col + 1]);
                    for entry in first..end {
                        let row = matrix.rowval[entry]
                            .checked_add(offset)
                            .ok_or_else(|| invalid("model row count overflow"))?;
                        indices.push(row);
                        row_scale[row] = row_scale[row].max(matrix.nzval[entry].abs());
                        values.push(matrix.nzval[entry]);
                    }
                    cooperation.completed((end - first) as u64).await?;
                }
            }
            if let Some(row) = lower[col] {
                indices.push(row);
                values.push(-1.0);
                row_scale[row] = 1.0;
            }
            if let Some(row) = upper[col] {
                indices.push(row);
                values.push(1.0);
                row_scale[row] = 1.0;
            }
            pointers.push(values.len());
        }
        let mut objective_scale = 0.0_f64;
        for chunk in problem.f.chunks(1024) {
            for value in chunk {
                objective_scale = objective_scale.max(value.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let objective_scale = if objective_scale == 0.0 {
            1.0
        } else {
            objective_scale
        };
        Ok(Self {
            a: CscMatrix::new(b.len(), n, pointers, indices, values),
            b,
            f: problem.f,
            roles,
            objective_scale,
            row_scale,
            equality_rows: me,
            inequality_rows: mi,
            seed,
        })
    }
    pub fn cones(&self) -> Vec<SupportedConeT<f64>> {
        let mut cones = Vec::new();
        if self.equality_rows > 0 {
            cones.push(ZeroConeT(self.equality_rows));
        }
        let inequalities = self.b.len() - self.equality_rows;
        if inequalities > 0 {
            cones.push(NonnegativeConeT(inequalities));
        }
        cones
    }
    pub fn work_units(&self) -> u64 {
        self.a
            .nzval
            .len()
            .saturating_add(self.b.len())
            .saturating_add(self.f.len()) as u64
    }
}

pub(in crate::builtins::math::optim) struct Multipliers {
    pub ineqlin: Vec<f64>,
    pub eqlin: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Multipliers {
    pub(super) async fn from_dual(
        model: &ConicModel,
        z: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut result = Self {
            ineqlin: vec![0.0; model.inequality_rows],
            eqlin: vec![0.0; model.equality_rows],
            lower: vec![0.0; model.f.len()],
            upper: vec![0.0; model.f.len()],
        };
        for (index, (role, &value)) in model.roles.iter().zip(z).enumerate() {
            match *role {
                RowRole::Equality(i) => result.eqlin[i] = value,
                RowRole::Inequality(i) => result.ineqlin[i] = value,
                RowRole::Lower(i) => result.lower[i] = value,
                RowRole::Upper(i) => result.upper[i] = value,
            }
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(0).await?;
        Ok(result)
    }
    pub(super) fn into_value(self) -> Value {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("ineqlin", vector_value(self.ineqlin));
        fields.insert("eqlin", vector_value(self.eqlin));
        fields.insert("lower", vector_value(self.lower));
        fields.insert("upper", vector_value(self.upper));
        Value::Struct(fields.finish())
    }
}

pub(in crate::builtins::math::optim) struct LinprogOutcome {
    pub x: Option<Vec<f64>>,
    pub fval: Option<f64>,
    pub lambda: Option<Multipliers>,
    pub exitflag: i32,
    pub iterations: usize,
    pub constrviolation: f64,
    pub primalfeasibility: f64,
    pub dualfeasibility: f64,
    pub dualitygap: f64,
    pub firstorderopt: f64,
    pub message: String,
}
impl LinprogOutcome {
    pub fn failure(exitflag: i32, iterations: usize, message: &str) -> Self {
        Self {
            x: None,
            fval: None,
            lambda: None,
            exitflag,
            iterations,
            constrviolation: f64::NAN,
            primalfeasibility: f64::NAN,
            dualfeasibility: f64::NAN,
            dualitygap: f64::NAN,
            firstorderopt: f64::NAN,
            message: message.into(),
        }
    }
    pub fn infeasible(message: &str) -> Self {
        Self::failure(-2, 0, message)
    }
    pub fn unbounded(iterations: usize) -> Self {
        Self::failure(-3, iterations, "Problem is feasible and unbounded below.")
    }
}
fn invalid(message: &str) -> RuntimeError {
    linprog_error_with_detail(&LINPROG_ERROR_INVALID_INPUT, message)
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("linprog: sparse model allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .build()
}
