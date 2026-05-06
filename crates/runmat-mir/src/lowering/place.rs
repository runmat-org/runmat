use crate::MirPlace;
use runmat_hir::{HirPlace, SemanticError};

use super::MirLoweringContext;

pub(crate) fn lower_place(
    ctx: &MirLoweringContext,
    place: &HirPlace,
) -> Result<MirPlace, SemanticError> {
    Ok(match place {
        HirPlace::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        _ => {
            return Err(SemanticError::new(
                "MIR lowering for place is not implemented yet",
            ))
        }
    })
}
