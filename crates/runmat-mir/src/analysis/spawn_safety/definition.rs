//! The same lexical-capture policy for a standalone compiled definition.
use super::{
    analyze_capture_facts, CaptureFacts, MirAssembly, MirBody, SpawnSafetyFact, SpawnSafetyReason,
};
use runmat_hir::FunctionId;

pub fn function_spawn_safety(
    assembly: &MirAssembly,
    function: FunctionId,
) -> Option<SpawnSafetyFact> {
    assembly
        .bodies
        .get(&function)
        .map(|body| capture_safety(&capture_facts(assembly, body)))
}

pub(super) fn capture_facts(assembly: &MirAssembly, body: &MirBody) -> CaptureFacts {
    let mut facts = analyze_capture_facts(body);
    // Ordinary anonymous captures are value snapshots. Do not confuse them
    // with nested functions' live ancestor cells. Transit cells remain denied;
    // the canonical lexical projection owns that distinction for both backends.
    if !facts.reads_captures.is_empty()
        && assembly
            .functions
            .get(&body.function)
            .is_some_and(|metadata| {
                metadata.source_role == runmat_types::FunctionSourceRole::Anonymous
                    && facts.reads_captures.iter().all(|binding| {
                        metadata
                            .captures
                            .iter()
                            .any(|capture| capture.binding == *binding)
                    })
            })
        && assembly
            .lexical_environment_projection(body.function)
            .is_ok_and(|projection| {
                projection.inherited.is_empty()
                    && projection.transit.is_empty()
                    && projection.inherited_storage.is_empty()
            })
    {
        facts.reads_captures.clear();
        facts.requires_isolation = true;
    }
    facts
}

pub(super) fn capture_safety(summary: &CaptureFacts) -> SpawnSafetyFact {
    if !summary.reads_captures.is_empty() || !summary.writes_captures.is_empty() {
        SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::MutableLexicalCapture,
        }
    } else if summary.requires_isolation {
        SpawnSafetyFact::RequiresIsolation
    } else {
        SpawnSafetyFact::SpawnSafe
    }
}
