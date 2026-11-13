//! SOC-specific sparse sign/row/dual mapping; shared mechanics only borrow it.
use super::*;
use clarabel::solver::{NonnegativeConeT, SecondOrderConeT, SupportedConeT, ZeroConeT};

pub(super) struct SocConstraint {
    pub a: MatrixInput,
    pub b: Vec<f64>,
    pub d: Vec<f64>,
    pub gamma: f64,
}
pub(super) struct ConeProgram {
    pub f: Vec<f64>,
    pub soc: Vec<SocConstraint>,
    pub a_ineq: MatrixInput,
    pub b_ineq: Vec<f64>,
    pub a_eq: MatrixInput,
    pub b_eq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub options: options::Controls,
}
#[derive(Clone, Copy)]
enum RowRole {
    Equality(usize),
    Inequality(usize),
    Lower(usize),
    Upper(usize),
    Soc(usize),
    SocTail,
}
pub(super) struct ConicModel {
    pub a: MatrixInput,
    pub b: Vec<f64>,
    pub f: Vec<f64>,
    pub cones: Vec<SupportedConeT<f64>>,
    pub row_scale: Vec<f64>,
    pub objective_scale: f64,
    pub seed: Vec<f64>,
    roles: Vec<RowRole>,
    equality_rows: usize,
    inequality_rows: usize,
    soc_count: usize,
}
impl ConicModel {
    pub fn view(&self) -> conic::Model<'_> {
        conic::Model {
            p: None,
            a: &self.a,
            b: &self.b,
            f: &self.f,
            cones: &self.cones,
            identity: runmat_builtins::CONEPROG_CATALOG_ENTRY.identity,
            objective_scale: self.objective_scale,
        }
    }
    pub async fn new(
        problem: ConeProgram,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, options::Controls)> {
        let controls = problem.options;
        let n = problem.f.len();
        let me = problem.b_eq.len();
        let mi = problem.b_ineq.len();
        let mut roles = Vec::new();
        let mut b = Vec::new();
        for (row, &value) in problem.b_eq.iter().enumerate() {
            roles.push(RowRole::Equality(row));
            b.push(value);
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        for (row, &value) in problem.b_ineq.iter().enumerate() {
            roles.push(RowRole::Inequality(row));
            b.push(value);
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        let mut lower = vec![None; n];
        let mut upper = vec![None; n];
        let mut seed = vec![0.0_f64; n];
        for col in 0..n {
            if problem.lb[col].is_finite() {
                lower[col] = Some(b.len());
                roles.push(RowRole::Lower(col));
                b.push(-problem.lb[col]);
                seed[col] = seed[col].max(problem.lb[col]);
            }
            cooperation.completed(1).await?;
        }
        for col in 0..n {
            if problem.ub[col].is_finite() {
                upper[col] = Some(b.len());
                roles.push(RowRole::Upper(col));
                b.push(problem.ub[col]);
                seed[col] = seed[col].min(problem.ub[col]);
            }
            cooperation.completed(1).await?;
        }
        let mut cones = Vec::new();
        if me > 0 {
            cones.push(ZeroConeT(me));
        }
        if b.len() > me {
            cones.push(NonnegativeConeT(b.len() - me));
        }
        let mut offsets = Vec::new();
        let mut nz = problem
            .a_eq
            .nzval
            .len()
            .checked_add(problem.a_ineq.nzval.len())
            .and_then(|count| {
                count.checked_add(lower.iter().filter(|value| value.is_some()).count())
            })
            .and_then(|count| {
                count.checked_add(upper.iter().filter(|value| value.is_some()).count())
            })
            .ok_or_else(overflow)?;
        for (index, soc) in problem.soc.iter().enumerate() {
            let dimension = soc.a.m.checked_add(1).ok_or_else(overflow)?;
            b.len().checked_add(dimension).ok_or_else(overflow)?;
            offsets.push(b.len());
            roles.push(RowRole::Soc(index));
            b.push(-soc.gamma);
            for (row, &value) in soc.b.iter().enumerate() {
                roles.push(RowRole::SocTail);
                b.push(-value);
                if row % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cones.push(SecondOrderConeT(dimension));
            nz = nz
                .checked_add(soc.a.nzval.len())
                .and_then(|count| {
                    count.checked_add(soc.d.iter().filter(|&&value| value != 0.0).count())
                })
                .ok_or_else(overflow)?;
            cooperation.completed(1).await?;
        }
        let mut pointers = runmat_value::exact_capacity_vec(n.checked_add(1).ok_or_else(overflow)?)
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
                            .ok_or_else(overflow)?;
                        indices.push(row);
                        values.push(matrix.nzval[entry]);
                        row_scale[row] = row_scale[row].max(matrix.nzval[entry].abs());
                    }
                    cooperation.completed((end - first) as u64).await?;
                }
            }
            for (row, value) in [(lower[col], -1.0), (upper[col], 1.0)] {
                if let Some(row) = row {
                    indices.push(row);
                    values.push(value);
                    row_scale[row] = 1.0;
                }
            }
            for (soc, &offset) in problem.soc.iter().zip(&offsets) {
                if soc.d[col] != 0.0 {
                    indices.push(offset);
                    values.push(-soc.d[col]);
                    row_scale[offset] = row_scale[offset].max(soc.d[col].abs());
                }
                for first in (soc.a.colptr[col]..soc.a.colptr[col + 1]).step_by(1024) {
                    let end = first.saturating_add(1024).min(soc.a.colptr[col + 1]);
                    for entry in first..end {
                        let row = offset
                            .checked_add(1)
                            .and_then(|row| row.checked_add(soc.a.rowval[entry]))
                            .ok_or_else(overflow)?;
                        indices.push(row);
                        values.push(-soc.a.nzval[entry]);
                        row_scale[row] = row_scale[row].max(soc.a.nzval[entry].abs());
                    }
                    cooperation.completed((end - first) as u64).await?;
                }
            }
            pointers.push(values.len());
            cooperation.completed(1).await?;
        }
        let mut objective_scale = 0.0_f64;
        for chunk in problem.f.chunks(1024) {
            for value in chunk {
                objective_scale = objective_scale.max(value.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        if objective_scale == 0.0 {
            objective_scale = 1.0;
        }
        Ok((
            Self {
                a: MatrixInput::new(b.len(), n, pointers, indices, values),
                b,
                f: problem.f,
                cones,
                row_scale,
                objective_scale,
                seed,
                roles,
                equality_rows: me,
                inequality_rows: mi,
                soc_count: problem.soc.len(),
            },
            controls,
        ))
    }
    pub async fn multipliers(
        &self,
        z: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Lambda> {
        if z.len() != self.b.len() {
            return Err(coneprog_error(
                &ERROR_INVALID_INPUT,
                "dual result geometry does not match original conic rows",
            ));
        }
        let mut lambda = Lambda {
            ineqlin: vec![0.0; self.inequality_rows],
            eqlin: vec![0.0; self.equality_rows],
            lower: vec![0.0; self.f.len()],
            upper: vec![0.0; self.f.len()],
            soc: vec![0.0; self.soc_count],
        };
        for (index, (role, &value)) in self.roles.iter().zip(z).enumerate() {
            match *role {
                RowRole::Equality(i) => lambda.eqlin[i] = value,
                RowRole::Inequality(i) => lambda.ineqlin[i] = value,
                RowRole::Lower(i) => lambda.lower[i] = value,
                RowRole::Upper(i) => lambda.upper[i] = value,
                RowRole::Soc(i) => lambda.soc[i] = value,
                RowRole::SocTail => {}
            }
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(0).await?;
        Ok(lambda)
    }
}
pub(super) struct Lambda {
    pub ineqlin: Vec<f64>,
    pub eqlin: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub soc: Vec<f64>,
}
pub(super) struct ConeOutcome {
    pub x: Option<Vec<f64>>,
    pub fval: Option<f64>,
    pub lambda: Option<Lambda>,
    pub exitflag: i32,
    pub iterations: usize,
    pub constrviolation: f64,
    pub primalfeasibility: f64,
    pub dualfeasibility: f64,
    pub dualitygap: f64,
    pub message: String,
}
impl ConeOutcome {
    pub fn failure(flag: i32, iterations: usize, message: &str) -> Self {
        Self {
            x: None,
            fval: None,
            lambda: None,
            exitflag: flag,
            iterations,
            constrviolation: f64::NAN,
            primalfeasibility: f64::NAN,
            dualfeasibility: f64::NAN,
            dualitygap: f64::NAN,
            message: message.into(),
        }
    }
}
fn overflow() -> RuntimeError {
    coneprog_error(&ERROR_INVALID_INPUT, "sparse conic model size overflow")
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("coneprog: sparse model allocation failed")
        .with_builtin(CONEPROG)
        .with_source(source)
        .build()
}
