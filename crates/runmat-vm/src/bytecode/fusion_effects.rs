//! Shared selected-statement effect admission, not execution authority.
//! Canonical numerical adapters may throw or suspend; runtime binding, value
//! domain and kernel admission still belong to their existing operation owners.
use super::{CompiledFusionStageStatement, FusionInstructionKind, Instr};
use runmat_types::EffectKind;

pub(super) fn elementwise_statement_effects_are_compatible(
    instructions: &[Instr],
    statement: &CompiledFusionStageStatement,
) -> bool {
    if statement.contract.effects.0.is_empty() {
        return true;
    }
    let Some(instructions) = instructions.get(statement.pc.start..statement.pc.end) else {
        return false;
    };
    let mut signals = instructions
        .iter()
        .filter(|instruction| super::compile::instr_fusion_signal_kind(instruction).is_some());
    let Some(Instr::CallBuiltinMulti(name, arguments, 1)) = signals.next() else {
        // An intrinsic cannot borrow a named implementation's effect contract.
        return false;
    };
    if signals.next().is_some() {
        return false;
    }
    let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(name) else {
        return false;
    };
    crate::accel::fusion_declaration::canonical_kind(entry, Some(*arguments))
        == Some(FusionInstructionKind::Elementwise)
        && statement.contract.effects.0.iter().all(|effect| {
            matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend)
                && entry.contract.effects.contains(effect)
        })
}
