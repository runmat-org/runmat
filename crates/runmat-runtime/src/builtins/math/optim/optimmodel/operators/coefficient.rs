//! Structural constant coefficients retain their original expression DAG.
//! No evaluation, cancellation of variable terms, or floating-point folding
//! establishes constancy at construction.
use super::super::*;
use runmat_gc::GcHandle;

pub(super) enum Shape {
    Scalar,
    Array,
}

pub(super) async fn is_constant(
    root: GcHandle,
    node: &Node,
    shape: Shape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if matches!(shape, Shape::Scalar) && !node.shape.iter().all(|&extent| extent == 1) {
        return Ok(false);
    }
    if matches!(&node.kind, Kind::Constant(_) | Kind::Zero) {
        return Ok(true);
    }
    let plan = evaluation::Plan::prepare(root, policy, cooperation).await?;
    for handle in &plan.order {
        match &plan.nodes[handle].node().kind {
            Kind::Variable(_) | Kind::Unassigned | Kind::EmptyInitializer => return Ok(false),
            Kind::Builtin { function, .. } => {
                let deterministic = runmat_builtins::builtin_catalog_entry_by_name(&function.0)
                    .is_some_and(|entry| {
                        entry
                            .contract
                            .effect_set()
                            .is_deterministic_value_function()
                    });
                if !deterministic {
                    return Ok(false);
                }
            }
            Kind::Zero
            | Kind::Constant(_)
            | Kind::Labels { .. }
            | Kind::Unary { .. }
            | Kind::Binary { .. }
            | Kind::Index { .. }
            | Kind::Assign { .. }
            | Kind::Concatenate { .. } => {}
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
