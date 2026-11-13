//! Sparse residual lifting; public constraints retain their original row roles.
use super::*;
use clarabel::solver::{NonnegativeConeT, SupportedConeT, ZeroConeT};

pub(super) struct Problem {
    pub c: Matrix,
    pub d: Vec<f64>,
    pub a: Matrix,
    pub b: Vec<f64>,
    pub aeq: Matrix,
    pub beq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub x0: Vec<f64>,
    pub controls: options::Controls,
    pub weights: objective::Weights,
}
enum Row {
    Residual,
    Equality(usize),
    Inequality(usize),
    Lower(usize),
    Upper(usize),
}
pub(super) struct Model {
    pub c: Matrix,
    pub d: Vec<f64>,
    pub p: Matrix,
    pub a: Matrix,
    pub b: Vec<f64>,
    pub f: Vec<f64>,
    pub eq: usize,
    pub cones: Vec<SupportedConeT<f64>>,
    pub seed: Vec<f64>,
    pub objective_scale: f64,
    pub weighting: objective::Weighting,
    roles: Vec<Row>,
    neq: usize,
    nineq: usize,
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
            identity: runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
            objective_scale: self.objective_scale,
        }
    }
    pub async fn new(
        problem: Problem,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, options::Controls)> {
        let Problem {
            c,
            d,
            a,
            b,
            aeq,
            beq,
            lb,
            ub,
            x0: _,
            controls,
            weights,
        } = problem;
        let n = c.n;
        let m = c.m;
        let variables = n
            .checked_add(m)
            .ok_or_else(|| invalid("lifted dimension overflow"))?;
        let eq = m
            .checked_add(beq.len())
            .ok_or_else(|| invalid("row dimension overflow"))?;
        let neq = beq.len();
        let nineq = b.len();
        let mut rhs = d.clone();
        let mut roles = (0..m).map(|_| Row::Residual).collect::<Vec<_>>();
        for (i, v) in beq.into_iter().enumerate() {
            rhs.push(v);
            roles.push(Row::Equality(i));
        }
        for (i, v) in b.into_iter().enumerate() {
            rhs.push(v);
            roles.push(Row::Inequality(i));
        }
        let mut lower = vec![None; n];
        let mut upper = vec![None; n];
        let mut seed = vec![0.0; variables];
        for i in 0..n {
            if lb[i].is_finite() {
                lower[i] = Some(rhs.len());
                rhs.push(-lb[i]);
                roles.push(Row::Lower(i));
                seed[i] = 0.0_f64.max(lb[i]);
            }
            if ub[i].is_finite() {
                upper[i] = Some(rhs.len());
                rhs.push(ub[i]);
                roles.push(Row::Upper(i));
                seed[i] = seed[i].min(ub[i]);
            }
            cooperation.completed(1).await?;
        }
        // Set the residual part of the seed from the actual original equation.
        for i in 0..m {
            seed[n + i] = -d[i];
        }
        for col in 0..n {
            for entry in c.colptr[col]..c.colptr[col + 1] {
                seed[n + c.rowval[entry]] += c.nzval[entry] * seed[col];
            }
            cooperation
                .completed((c.colptr[col + 1] - c.colptr[col]) as u64 + 1)
                .await?;
        }
        let bound_count = rhs.len() - eq - nineq;
        let capacity = c
            .nzval
            .len()
            .checked_add(a.nzval.len())
            .and_then(|v| v.checked_add(aeq.nzval.len()))
            .and_then(|v| v.checked_add(bound_count))
            .and_then(|v| v.checked_add(m))
            .ok_or_else(|| invalid("lifted storage overflow"))?;
        let mut ptr = runmat_value::exact_capacity_vec(
            variables
                .checked_add(1)
                .ok_or_else(|| invalid("lifted storage overflow"))?,
        )
        .map_err(allocation)?;
        let mut rows = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
        let mut values = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
        ptr.push(0);
        for col in 0..n {
            for (matrix, offset) in [(&c, 0), (&aeq, m), (&a, eq)] {
                for k in matrix.colptr[col]..matrix.colptr[col + 1] {
                    rows.push(matrix.rowval[k] + offset);
                    values.push(matrix.nzval[k]);
                    if k % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
            }
            // Bounds were appended in variable order, lower before upper.
            for (row, value) in [(lower[col], -1.0), (upper[col], 1.0)] {
                if let Some(row) = row {
                    rows.push(row);
                    values.push(value);
                }
            }
            ptr.push(values.len());
            cooperation.completed(1).await?;
        }
        for i in 0..m {
            rows.push(i);
            values.push(-1.0);
            ptr.push(values.len());
            cooperation.completed(1).await?;
        }
        // Only m diagonal entries; never C'C or a dense identity.
        let mut pp = runmat_value::exact_capacity_vec(variables + 1).map_err(allocation)?;
        pp.extend(std::iter::repeat_n(0, n + 1));
        pp.extend(1..=m);
        let (weights, objective_scale, weighting) = weights.into_values(m, cooperation).await?;
        let p = Matrix::new(variables, variables, pp, (n..variables).collect(), weights);
        let mut cones = vec![ZeroConeT(eq)];
        if rhs.len() > eq {
            cones.push(NonnegativeConeT(rhs.len() - eq));
        }
        Ok((
            Self {
                c,
                d,
                p,
                a: Matrix::new(rhs.len(), variables, ptr, rows, values),
                b: rhs,
                f: vec![0.0; variables],
                eq,
                cones,
                seed,
                objective_scale,
                weighting,
                roles,
                neq,
                nineq,
            },
            controls,
        ))
    }
    pub(super) fn weight(&self, row: usize) -> Option<f64> {
        self.p.nzval.get(row).copied()
    }
    pub(super) fn external_role_count(&self) -> usize {
        self.roles.len() - self.c.m
    }
    pub(super) fn external_role(&self, index: usize) -> Option<conic::LinearRowRole> {
        Some(match *self.roles.get(index.checked_add(self.c.m)?)? {
            Row::Residual => return None,
            Row::Equality(row) => conic::LinearRowRole::Equality(row),
            Row::Inequality(row) => conic::LinearRowRole::Inequality(row),
            Row::Lower(column) => conic::LinearRowRole::Lower(column),
            Row::Upper(column) => conic::LinearRowRole::Upper(column),
        })
    }
    pub async fn lambda(
        &self,
        z: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Lambda> {
        let mut out = Lambda {
            ineqlin: vec![0.0; self.nineq],
            eqlin: vec![0.0; self.neq],
            lower: vec![0.0; self.c.n],
            upper: vec![0.0; self.c.n],
        };
        for (i, role) in self.roles.iter().enumerate() {
            match *role {
                Row::Residual => {}
                Row::Equality(j) => out.eqlin[j] = z[i],
                Row::Inequality(j) => out.ineqlin[j] = z[i],
                Row::Lower(j) => out.lower[j] = z[i],
                Row::Upper(j) => out.upper[j] = z[i],
            }
            if i % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        Ok(out)
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum ActiveTermination {
    Optimal,
    IterationLimit,
    ObjectiveLimit,
    IllConditioned,
    NoDirection,
    Unverified,
    Infeasible,
}
pub(in crate::builtins::math::optim) enum Method {
    InteriorPoint,
    Reflective { cg_iterations: usize },
    ActiveSet { termination: ActiveTermination },
}
impl Method {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::InteriorPoint => ALGORITHM,
            Self::Reflective { .. } => "trust-region-reflective",
            Self::ActiveSet { .. } => "active-set",
        }
    }
}
pub(in crate::builtins::math::optim) struct Outcome {
    pub method: Method,
    pub x: Option<Vec<f64>>,
    pub residual: Option<Vec<f64>>,
    pub resnorm: Option<f64>,
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
            residual: None,
            resnorm: None,
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
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("lsqlin: model allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .build()
}
