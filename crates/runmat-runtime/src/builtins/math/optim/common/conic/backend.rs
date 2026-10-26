//! One owned cooperative Clarabel solve, with domain-owned iteration policy.
use super::Model;
use crate::{build_runtime_error, BuiltinResult};
use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultInfo, DefaultSettings, DefaultSolver},
};

pub(crate) async fn run<X, F, P>(
    model: &Model<'_>,
    feasibility: bool,
    settings: DefaultSettings<f64>,
    context: &mut X,
    progress: F,
) -> BuiltinResult<DefaultSolver<f64>>
where
    F: for<'a> std::ops::AsyncFnMut(&'a mut X, &'a DefaultInfo<f64>) -> BuiltinResult<P>,
    P: Into<clarabel::solver::ProgressControl>,
{
    let owned_p = if !feasibility && model.objective_scale != 1.0 {
        model.p.map(|original| {
            let mut scaled = original.clone();
            for value in &mut scaled.nzval {
                *value /= model.objective_scale;
            }
            scaled
        })
    } else {
        None
    };
    let zero = CscMatrix::zeros((model.f.len(), model.f.len()));
    let p = if feasibility {
        &zero
    } else {
        owned_p.as_ref().or(model.p).unwrap_or(&zero)
    };
    let q = if feasibility {
        vec![0.0; model.f.len()]
    } else {
        model
            .f
            .iter()
            .map(|value| value / model.objective_scale)
            .collect()
    };
    let solver =
        DefaultSolver::new(p, &q, model.a, model.b, model.cones, settings).map_err(|source| {
            build_runtime_error(format!(
                "{}: sparse solver initialization failed",
                model.identity.name
            ))
            .with_builtin(model.identity.name)
            .with_source(source)
            .build()
        })?;
    // Upstream owns its working data now. Retire input-only temporaries before
    // any host turn; the solver itself remains owned through await/error/drop.
    drop(q);
    drop(owned_p);
    drop(zero);
    solver.solve_with_progress(context, progress).await
}
