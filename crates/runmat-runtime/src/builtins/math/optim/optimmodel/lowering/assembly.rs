//! Final sparse solver assembly. Global decision column pointers appear only
//! here; original array/row roles remain in the retained formulation.
use super::*;
use clarabel::algebra::CscMatrix;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;
use runmat_value::optimization_variable::OptimizationVariableDomain;

use runmat_value::optimization_variable::OptimizationBoundRole;
use terms::{Coefficients, Term};
pub(super) struct Rows {
    pub equality: bool,
    pub start: usize,
    pub count: usize,
}
#[derive(Clone, Copy)]
pub(super) enum ObjectiveKind {
    Affine,
    Constant,
    LeastSquares,
    Quadratic,
}
pub(super) enum Cost {
    Affine(Vec<f64>),
    Residual,
    Quadratic,
}
pub(super) struct Assembled {
    pub cost: Cost,
    pub integers: Vec<usize>,
    pub semis: Vec<super::super::super::intlinprog::compiled::Semi>,
    pub a_ineq: CscMatrix<f64>,
    pub b_ineq: Vec<f64>,
    pub a_eq: CscMatrix<f64>,
    pub b_eq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub rows: Vec<Rows>,
}
pub(super) fn assemble<'a>(
    program: &'a Program,
    formulation: &'a formulation::Formulation,
    objective_kind: ObjectiveKind,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = Lowered<Assembled>> + 'a {
    assemble_domains(
        program,
        formulation,
        objective_kind,
        Domains::Continuous,
        policy,
        cooperation,
    )
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Domains {
    Continuous,
    Integer,
}
pub(super) async fn assemble_domains(
    program: &Program,
    formulation: &formulation::Formulation,
    objective_kind: ObjectiveKind,
    domains: Domains,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Assembled> {
    let n = program.layout.columns;
    let mut cost = match objective_kind {
        ObjectiveKind::Affine | ObjectiveKind::Constant => {
            Cost::Affine(runmat_value::exact_capacity_vec(n).map_err(|e| policy.storage(e))?)
        }
        ObjectiveKind::LeastSquares => Cost::Residual,
        ObjectiveKind::Quadratic => Cost::Quadratic,
    };
    let mut integers = Vec::new();
    let mut semis = Vec::new();
    let mut lb = runmat_value::exact_capacity_vec(n).map_err(|e| policy.storage(e))?;
    let mut ub = runmat_value::exact_capacity_vec(n).map_err(|e| policy.storage(e))?;
    for &index in &program.layout.order {
        let variable = &program.layout.variables[index];
        if domains == Domains::Continuous
            && variable.domain != OptimizationVariableDomain::Continuous
        {
            return Err(
                policy.invalid("continuous model solve cannot ignore an integer variable domain")
            );
        }
        for row in 0..variable.count {
            if matches!(
                variable.domain,
                OptimizationVariableDomain::Integer | OptimizationVariableDomain::SemiInteger
            ) {
                integers.try_reserve(1).map_err(|e| policy.reserve(e))?;
                integers.push(variable.start + row);
            }
            if let Cost::Affine(f) = &mut cost {
                f.push(0.);
            }
            let lower = variable
                .view
                .definition()
                .bound_value(OptimizationBoundRole::Lower, row)
                .ok_or_else(|| policy.invalid("original lower bound exceeds its definition"))?;
            let upper = variable
                .view
                .definition()
                .bound_value(OptimizationBoundRole::Upper, row)
                .ok_or_else(|| policy.invalid("original upper bound exceeds its definition"))?;
            if variable.domain.is_semi() {
                semis.try_reserve(1).map_err(|e| policy.reserve(e))?;
                semis.push(
                    super::super::super::intlinprog::compiled::Semi::from_bounds(
                        variable.start + row,
                        lower,
                        upper,
                        variable.domain == OptimizationVariableDomain::SemiInteger,
                    )?,
                );
                lb.push(0.);
            } else {
                lb.push(lower);
            }
            ub.push(upper);
            cooperation.completed(1).await?;
        }
    }
    if matches!(objective_kind, ObjectiveKind::Affine) {
        if let (Cost::Affine(f), Some(objective)) = (&mut cost, &formulation.objective) {
            let array = program.array(objective.root, policy)?;
            let sign = if objective.sense == Sense::Minimize {
                1.
            } else {
                -1.
            };
            for term in array.coefficients.entries() {
                if term.row != 0 {
                    return Err(policy.invalid("scalar objective has a nonscalar coefficient row"));
                }
                f[term.column] = sign * term.value;
                cooperation.completed(1).await?;
            }
        }
    }
    let mut inequality = Vec::new();
    let mut equality = Vec::new();
    let mut b_ineq = Vec::new();
    let mut b_eq = Vec::new();
    let mut rows = runmat_value::exact_capacity_vec(formulation.constraints.len())
        .map_err(|e| policy.storage(e))?;
    for constraint in &formulation.constraints {
        let (left, right) = if constraint.relation == Relation::GreaterEqual {
            (constraint.right, constraint.left)
        } else {
            (constraint.left, constraint.right)
        };
        let residual = match program
            .array(left, policy)?
            .clone()
            .add(
                program.array(right, policy)?.clone(),
                runmat_types::OperatorKind::Subtract,
                constraint.shape.clone(),
                n,
                policy,
                cooperation,
            )
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        let is_equality = constraint.relation == Relation::Equal;
        let (entries, rhs) = if is_equality {
            (&mut equality, &mut b_eq)
        } else {
            (&mut inequality, &mut b_ineq)
        };
        let count = residual.count(policy)?;
        let start = rhs.len();
        rhs.try_reserve(count).map_err(|e| policy.reserve(e))?;
        for row in 0..count {
            if !residual.assigned.at(row, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            rhs.push(-residual.offset.at(row, policy)?);
            cooperation.completed(1).await?;
        }
        entries
            .try_reserve(residual.coefficients.nonzeros())
            .map_err(|e| policy.reserve(e))?;
        for term in residual.coefficients.entries() {
            entries.push(Term {
                row: term
                    .row
                    .checked_add(start)
                    .ok_or_else(|| policy.invalid("constraint row overflow"))?,
                ..term
            });
            cooperation.completed(1).await?;
        }
        rows.push(Rows {
            equality: is_equality,
            start,
            count,
        });
    }
    let a_ineq = match matrix(inequality, b_ineq.len(), n, policy, cooperation).await? {
        Ok(matrix) => matrix,
        Err(stop) => return Ok(Err(stop)),
    };
    let a_eq = match matrix(equality, b_eq.len(), n, policy, cooperation).await? {
        Ok(matrix) => matrix,
        Err(stop) => return Ok(Err(stop)),
    };
    Ok(Ok(Assembled {
        cost,
        integers,
        semis,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
        rows,
    }))
}
pub(super) async fn matrix(
    entries: Vec<Term>,
    m: usize,
    n: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<CscMatrix<f64>> {
    let compact = match Coefficients::from_entries(entries, m, n, policy, cooperation).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let mut pointers = runmat_value::exact_capacity_vec(
        n.checked_add(1)
            .ok_or_else(|| policy.invalid("solver column pointer overflow"))?,
    )
    .map_err(|e| policy.storage(e))?;
    let mut indices =
        runmat_value::exact_capacity_vec(compact.nonzeros()).map_err(|e| policy.storage(e))?;
    let mut values =
        runmat_value::exact_capacity_vec(compact.nonzeros()).map_err(|e| policy.storage(e))?;
    pointers.push(0);
    for column in 0..n {
        for (row, value) in compact.column(column) {
            indices.push(row);
            values.push(value);
            cooperation.completed(1).await?;
        }
        pointers.push(values.len());
        cooperation.completed(1).await?;
    }
    Ok(Ok(CscMatrix::new(m, n, pointers, indices, values)))
}
