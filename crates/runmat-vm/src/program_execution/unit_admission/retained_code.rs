//! Context-free code/source selected from an actual admitted V3 unit. The
//! private conversion consumes the same decoded payload; it never reconstructs
//! executable authority from a name, source id, digest or copied binding.
use super::*;
use runmat_execution::ProgramRevision;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
#[derive(Debug)]
pub(super) struct UnitAdmissionIdentity {
    pub revision: ProgramRevision,
    pub interop: runmat_types::InteropManifest,
}
#[derive(Debug)]
pub(in super::super) struct RetainedExecutableUnit {
    admission: UnitAdmissionIdentity,
    canonical_v3: Arc<[u8]>,
    source_owner: SelectedProgramSourceOwner,
    bytecode: Bytecode,
    registry: FunctionRegistry,
}
impl RuntimeBoundExecutableUnit {
    pub(in super::super) fn into_retained_code(self) -> Result<RetainedExecutableUnit, String> {
        self.validate_runtime_owner(&self.runtime)
            .map_err(str::to_owned)?;
        let admission = self
            .admission
            .ok_or("retained code has no admitted product receipt")?;
        let canonical_v3 = self
            .canonical_v3
            .ok_or("retained code has no canonical V3")?;
        let source_map = self
            .selected_source_map
            .ok_or("retained code has no admitted source map")?;
        let source_owner = self
            .selected_source_owner
            .ok_or("retained code has no issued source owner")?;
        if source_owner.portable_map() != &source_map {
            return Err("retained code differs from its admitted source map".into());
        }
        // Runtime/services/grants are deliberately not moved into this owner.
        // The active worker keeps its separately scoped current contexts; a
        // later retained invocation supplies and binds its own original one.
        Ok(RetainedExecutableUnit {
            admission,
            canonical_v3,
            source_owner,
            bytecode: self.bytecode,
            registry: self.registry,
        })
    }
}
impl RetainedExecutableUnit {
    pub(in super::super) fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }
    pub(in super::super) fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }
    pub(in super::super) fn canonical_v3(&self) -> Option<&Arc<[u8]>> {
        Some(&self.canonical_v3)
    }
    pub(in super::super) fn selected_source_owner(&self) -> Option<&SelectedProgramSourceOwner> {
        Some(&self.source_owner)
    }
    pub(in super::super) fn revision(&self) -> &ProgramRevision {
        &self.admission.revision
    }
    pub(in super::super) fn interop(&self) -> &runmat_types::InteropManifest {
        &self.admission.interop
    }
}
