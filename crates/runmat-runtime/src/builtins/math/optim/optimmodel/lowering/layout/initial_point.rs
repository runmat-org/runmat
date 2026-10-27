//! Original named/shaped initial values have one conversion owner per definition.
use super::super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn initial_point(
    source: &evaluation::Plan,
    layout: &layout::Layout,
    supplied: &Value,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    crate::context::host_allocation::require_unbounded_workspace(
        c.runtime(),
        policy.identity.name,
        "model initial-point projection requires unbounded host workspace",
    )?;
    let values = evaluation::prepare_point(source, supplied, policy, c).await?;
    let mut x = runmat_value::exact_capacity_vec(layout.columns).map_err(|e| policy.storage(e))?;
    for &index in &layout.order {
        let variable = &layout.variables[index];
        let value = values
            .get(&variable.view.selection().definition())
            .ok_or_else(|| policy.invalid("initial point lost an original definition"))?;
        for row in 0..variable.count {
            let value = numerical::number(value.value(), row, policy)?;
            if !value.is_finite() {
                return Err(policy.invalid("initial-point coordinates must be finite"));
            }
            x.push(value);
            c.completed(1).await?;
        }
    }
    Ok(x)
}
