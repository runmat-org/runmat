//! Shared structural matrix-product identity. Never infer a compiled owner
//! from a builtin spelling or mistake elementwise multiplication for matmul.
use super::{AccelNode, AccelNodeLabel, FusionCallOrigin, PrimitiveOp};
use runmat_builtins::MTIMES_CATALOG_ENTRY;
use runmat_types::OperatorKind;

pub(super) fn is_matrix_product(node: &AccelNode) -> bool {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return false;
    }
    match &node.label {
        AccelNodeLabel::Primitive(PrimitiveOp::Mul) => true,
        AccelNodeLabel::Builtin { name, origin }
            if name.eq_ignore_ascii_case(MTIMES_CATALOG_ENTRY.identity.name) =>
        {
            match origin {
                FusionCallOrigin::Operator(OperatorKind::MatrixMultiply) => true,
                FusionCallOrigin::Named { binding } => MTIMES_CATALOG_ENTRY
                    .bindings
                    .iter()
                    .any(|variant| MTIMES_CATALOG_ENTRY.binding_identity(variant) == *binding),
                _ => false,
            }
        }
        _ => false,
    }
}
