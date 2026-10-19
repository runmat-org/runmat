//! Structural compiler candidates from the existing expression declaration.
//! This does not admit operand values, options, numerical domains or providers:
//! the selected fusion executor still performs its request-local admission.

use crate::bytecode::FusionInstructionKind;
#[cfg(feature = "native-accel")]
use runmat_builtins::{builtin_catalog_entry_by_name, builtin_functions, AccelTag};
use runmat_builtins::{
    BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinFusionPolicy, BuiltinPurity,
};
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, FusionExprContext, FusionExprInput, FusionKernelTemplate,
    FusionStructuralOperation,
};
use runmat_types::EffectKind;

pub(crate) fn canonical_kind(
    entry: &BuiltinCatalogEntry,
    argument_count: Option<usize>,
) -> Option<FusionInstructionKind> {
    if entry.placement.fusion != BuiltinFusionPolicy::Candidate
        || entry.contract.purity != BuiltinPurity::Pure
        || entry.contract.workspace_effect.is_some()
        || entry.contract.environment_effect.is_some()
        || entry.contract.async_behavior == BuiltinAsyncBehavior::RequiresAsyncRuntime
        || entry
            .contract
            .effects
            .iter()
            .any(|effect| !matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend))
    {
        return None;
    }
    // The bytecode call ABI carries the builtin identity, not a selected method
    // variant. Only a unique binding can be resolved without inventing one.
    let [binding] = entry.bindings else {
        return None;
    };
    let declaration = canonical_fusion_declaration(entry.binding_identity(binding))?;
    if let Some(operation) = declaration.structural {
        if argument_count.is_some_and(|count| count != operation.input_arity()) {
            return None;
        }
        return Some(match operation {
            FusionStructuralOperation::MatrixMultiply => FusionInstructionKind::Matmul,
        });
    }
    if declaration
        .spec
        .elementwise
        .as_ref()
        .is_some_and(|template| accepts_arity(template, argument_count))
    {
        // Elementwise execution awaits typed input/output adapters. MaySuspend
        // here is supported numerical adaptation, not arbitrary async work.
        Some(FusionInstructionKind::Elementwise)
    } else if declaration
        .spec
        .reduction
        .as_ref()
        .is_some_and(|template| accepts_arity(template, argument_count))
    {
        Some(FusionInstructionKind::Reduction)
    } else {
        None
    }
}

fn accepts_arity(template: &FusionKernelTemplate, argument_count: Option<usize>) -> bool {
    let Some(count) = argument_count else {
        return !template.scalar_precisions.is_empty();
    };
    // Ask the operation's existing expression builder, rather than reconstruct
    // optional-control signatures or maintain a second per-identity arity table.
    let inputs = vec![FusionExprInput::text("operand"); count];
    template.scalar_precisions.iter().any(|&scalar_ty| {
        template
            .wgsl_body
            .render(&FusionExprContext {
                scalar_ty,
                inputs: &inputs,
                constants: &[],
            })
            .is_ok()
    })
}

#[cfg(feature = "native-accel")]
pub(crate) fn call_kind(
    name: &str,
    argument_count: usize,
    output_count: usize,
) -> Option<FusionInstructionKind> {
    if output_count != 1 {
        return None;
    }
    if let Some(entry) = builtin_catalog_entry_by_name(name) {
        // A canonical rejection must not fall through to a stale legacy tag.
        return canonical_kind(entry, Some(argument_count));
    }
    let registered = builtin_functions();
    let builtin = registered.iter().find(|builtin| builtin.name == name)?;
    if builtin.accel_tags.contains(&AccelTag::MatMul) {
        Some(FusionInstructionKind::Matmul)
    } else if builtin.accel_tags.contains(&AccelTag::Reduction) {
        Some(FusionInstructionKind::Reduction)
    } else if builtin.accel_tags.iter().any(|tag| {
        matches!(
            tag,
            AccelTag::Unary | AccelTag::Elementwise | AccelTag::Transpose
        )
    }) {
        Some(FusionInstructionKind::Elementwise)
    } else {
        None
    }
}
