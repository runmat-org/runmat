//! QP-specific original-row mapping; bounds contribute one stored entry each.
use super::*;
use clarabel::solver::{NonnegativeConeT, SupportedConeT, ZeroConeT};
pub(super) struct Problem {
    pub p: Matrix,
    pub f: Vec<f64>,
    pub a: Matrix,
    pub b: Vec<f64>,
    pub aeq: Matrix,
    pub beq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub x0: Vec<f64>,
    pub controls: options::Controls,
}
use conic::LinearRowRole as Row;
pub(super) struct Model {
    pub p: Matrix,
    pub f: Vec<f64>,
    pub a: Matrix,
    pub b: Vec<f64>,
    pub cones: Vec<SupportedConeT<f64>>,
    pub eq: usize,
    pub objective_scale: f64,
    pub seed: Vec<f64>,
    pub(super) roles: Vec<Row>,
    mi: usize,
}
pub(in crate::builtins::math::optim) struct Lambda {
    pub ineqlin: Vec<f64>,
    pub eqlin: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Model {
    pub fn view(&self) -> conic::Model<'_> {
        conic::Model {
            p: Some(&self.p),
            a: &self.a,
            b: &self.b,
            f: &self.f,
            cones: &self.cones,
            identity: runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
            objective_scale: self.objective_scale,
        }
    }
    pub async fn new(
        problem: Problem,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, options::Controls)> {
        let n = problem.f.len();
        let eq = problem.beq.len();
        let mi = problem.b.len();
        let mut b = problem.beq;
        let mut roles = (0..eq).map(Row::Equality).collect::<Vec<_>>();
        for (index, value) in problem.b.into_iter().enumerate() {
            b.push(value);
            roles.push(Row::Inequality(index));
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        let mut lower = vec![None; n];
        let mut upper = vec![None; n];
        let mut seed = if problem.controls.active.is_some() {
            problem.x0
        } else {
            vec![0.0_f64; n]
        };
        for col in 0..n {
            if problem.lb[col].is_finite() {
                lower[col] = Some(b.len());
                b.push(-problem.lb[col]);
                roles.push(Row::Lower(col));
                seed[col] = seed[col].max(problem.lb[col]);
            }
            cooperation.completed(1).await?;
        }
        for col in 0..n {
            if problem.ub[col].is_finite() {
                upper[col] = Some(b.len());
                b.push(problem.ub[col]);
                roles.push(Row::Upper(col));
                seed[col] = seed[col].min(problem.ub[col]);
            }
            cooperation.completed(1).await?;
        }
        let capacity = problem
            .a
            .nzval
            .len()
            .checked_add(problem.aeq.nzval.len())
            .and_then(|x| x.checked_add(b.len() - eq - mi))
            .ok_or_else(|| invalid("model size overflow"))?;
        let mut ptr = runmat_value::exact_capacity_vec(
            n.checked_add(1)
                .ok_or_else(|| invalid("model size overflow"))?,
        )
        .map_err(allocation)?;
        let mut rows = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
        let mut values = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
        ptr.push(0);
        for col in 0..n {
            for (a, offset) in [(&problem.aeq, 0), (&problem.a, eq)] {
                for entry in a.colptr[col]..a.colptr[col + 1] {
                    let row = a.rowval[entry]
                        .checked_add(offset)
                        .ok_or_else(|| invalid("row size overflow"))?;
                    rows.push(row);
                    values.push(a.nzval[entry]);
                    if entry % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
            }
            for (row, value) in [(lower[col], -1.0), (upper[col], 1.0)] {
                if let Some(row) = row {
                    rows.push(row);
                    values.push(value);
                }
            }
            ptr.push(values.len());
            cooperation.completed(1).await?;
        }
        let mut cones = Vec::new();
        if eq > 0 {
            cones.push(ZeroConeT(eq));
        }
        if b.len() > eq {
            cones.push(NonnegativeConeT(b.len() - eq));
        }
        let mut scale = 0.0_f64;
        for chunk in problem.f.chunks(1024).chain(problem.p.nzval.chunks(1024)) {
            for value in chunk {
                scale = scale.max(value.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let objective_scale = if scale == 0.0 { 1.0 } else { scale };
        Ok((
            Self {
                p: problem.p,
                f: problem.f,
                a: Matrix::new(b.len(), n, ptr, rows, values),
                b,
                cones,
                eq,
                objective_scale,
                seed,
                roles,
                mi,
            },
            problem.controls,
        ))
    }
    pub async fn lambda(
        &self,
        z: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Lambda> {
        let n = self.f.len();
        let mut out = Lambda {
            ineqlin: vec![0.0; self.mi],
            eqlin: vec![0.0; self.eq],
            lower: vec![0.0; n],
            upper: vec![0.0; n],
        };
        for (row, role) in self.roles.iter().enumerate() {
            match *role {
                Row::Equality(i) => out.eqlin[i] = z[row],
                Row::Inequality(i) => out.ineqlin[i] = z[row],
                Row::Lower(i) => out.lower[i] = z[row],
                Row::Upper(i) => out.upper[i] = z[row],
            }
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        Ok(out)
    }
}
pub(in crate::builtins::math::optim) struct Outcome {
    pub method: Method,
    pub x: Option<Vec<f64>>,
    pub fval: Option<f64>,
    pub lambda: Option<Lambda>,
    pub flag: i32,
    pub iterations: usize,
    pub violation: f64,
    pub optimality: f64,
    pub gap: f64,
    pub message: &'static str,
}
impl Outcome {
    pub fn failure(flag: i32, iterations: usize, message: &'static str) -> Self {
        Self {
            method: Method::InteriorPoint,
            x: None,
            fval: None,
            lambda: None,
            flag,
            iterations,
            violation: f64::NAN,
            optimality: f64::NAN,
            gap: f64::NAN,
            message,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Method {
    InteriorPoint,
    ActiveSet(ActiveTermination),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum ActiveTermination {
    Optimal,
    IterationLimit,
    TimeLimit,
    ObjectiveLimit,
    StepStagnation,
    Numerical,
    Unverified,
    Infeasible,
    Unbounded,
    Nonconvex,
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("quadprog: model allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .build()
}
