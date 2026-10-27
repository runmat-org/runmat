//! Build/run and terminal callbacks are distinct polls of the same admitted
//! future. No backend-construction frame remains while original validation runs.
use super::*;
mod terminal;

pub(super) fn solve<'a, V: super::compiled::Validation + 'a>(
    problem: LinearProgram,
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    products: super::compiled::Products,
    validation: &'a mut V,
) -> impl std::future::Future<Output = BuiltinResult<LinprogOutcome>> + 'a {
    crate::execution::after_result(
        prepare(problem, cooperation, controls, products, validation),
        terminal::finish,
    )
}

struct BackendPoint {
    status: SolverStatus,
    x: Vec<f64>,
    z: Vec<f64>,
}
struct Prepared<'a, V> {
    model: ConicModel,
    point: BackendPoint,
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    products: super::compiled::Products,
    validation: &'a mut V,
}
async fn prepare<'a, V: super::compiled::Validation>(
    problem: LinearProgram,
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    products: super::compiled::Products,
    validation: &'a mut V,
) -> BuiltinResult<Prepared<'a, V>> {
    let model = ConicModel::new(problem, cooperation).await?;
    let mut solver = run(&model, false, cooperation, controls).await?;
    let point = BackendPoint {
        status: solver.solution.status,
        x: std::mem::take(&mut solver.solution.x),
        z: std::mem::take(&mut solver.solution.z),
    };
    // Keep exactly the original candidate/dual backings. The numerical kernel
    // is no longer used and retires before any original-domain callback.
    drop(solver);
    Ok(Prepared {
        model,
        point,
        cooperation,
        controls,
        products,
        validation,
    })
}
