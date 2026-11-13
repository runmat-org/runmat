//! Publication-time foreign closure for one exact selected executable unit.
//! The archived bytes are checked once and retained by the port-free code
//! owner; no project path or adapter configuration is visited at task spawn.

use std::sync::Arc;

use runmat_execution_artifact::archive::ArchiveLimits;
use runmat_execution_artifact::{
    BuildResourceDeclaration, ExecutionBundle, ForeignArtifactClosure, LogicalObject,
};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_vm::SelectedUnitInteropReceipt;

use super::SelectedForeignPreparation;

pub(crate) fn selected_unit_interop_receipt(
    canonical_v3: Arc<[u8]>,
    source_owner: &SelectedProgramSourceOwner,
    prepared: Option<&SelectedForeignPreparation>,
) -> Result<Option<std::rc::Rc<SelectedUnitInteropReceipt>>, String> {
    let admitted =
        runmat_vm::admit_executable_unit(&canonical_v3).map_err(|error| error.to_string())?;
    let interop = &admitted.envelope().manifest.interop;
    if interop == &runmat_types::InteropManifest::empty() && prepared.is_none() {
        return SelectedUnitInteropReceipt::from_bundle(
            None,
            &canonical_v3,
            source_owner,
            ArchiveLimits::default(),
        );
    }
    let prepared = prepared
        .ok_or("selected foreign V3 product has no configured immutable project artifacts")?;
    if prepared.interop() != interop {
        return Err("selected V3 interop differs from configured foreign artifacts".into());
    }
    selected_unit_interop_receipt_with_parts(
        canonical_v3,
        source_owner,
        interop,
        prepared.foreign_objects(),
        prepared.foreign_closures(),
        prepared.policy_object().cloned(),
    )
}

/// Package exactly one compiler-sealed child from an invocation-captured
/// prepared source. Callers select borrowed objects first, then make the sole
/// owned payload copy here for the canonical archive builder.
pub(super) fn selected_unit_interop_receipt_with_parts(
    canonical_v3: Arc<[u8]>,
    source_owner: &SelectedProgramSourceOwner,
    expected_interop: &runmat_types::InteropManifest,
    objects: Vec<LogicalObject>,
    closures: Vec<ForeignArtifactClosure>,
    policy_object: Option<LogicalObject>,
) -> Result<Option<std::rc::Rc<SelectedUnitInteropReceipt>>, String> {
    let admitted =
        runmat_vm::admit_executable_unit(&canonical_v3).map_err(|error| error.to_string())?;
    if &admitted.envelope().manifest.interop != expected_interop {
        return Err("selected V3 interop differs from its prepared foreign selection".into());
    }
    let limits = ArchiveLimits::default();
    let foreign_bytes = objects.iter().try_fold(0usize, |total, object| {
        object.validate().map_err(|error| error.to_string())?;
        total
            .checked_add(object.bytes.len())
            .ok_or_else(|| "selected foreign object bytes overflow".to_string())
    })?;
    if u64::try_from(foreign_bytes).map_or(true, |bytes| bytes > limits.max_total_bytes) {
        return Err("selected foreign objects exceed the archive limit".into());
    }
    let bundle = ExecutionBundle::from_selected_executable_unit_with_policy(
        &canonical_v3,
        objects,
        closures,
        policy_object,
        // The zero-output receipt recipe is packaging metadata, not an
        // execution placement promise. The selected task has its own request.
        BuildResourceDeclaration {
            cpu_millicores: 1000,
            memory_bytes: 1024 * 1024 * 1024,
            scratch_bytes: 1024 * 1024 * 1024,
        },
    )
    .map_err(|error| error.to_string())?;
    SelectedUnitInteropReceipt::from_bundle(Some(bundle), &canonical_v3, source_owner, limits)
}
