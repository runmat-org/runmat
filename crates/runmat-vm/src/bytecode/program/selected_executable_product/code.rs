//! Port-free, immutable facts of one fully admitted executable product.
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::executable::{
    selected_unit_content_digest, ExecutableStaticImport, LinkedNativeStaticImport,
    SelectedBodySemanticDigest, SelectedUnitContentDigest,
};
use runmat_execution::{Digest, ExecutableSourceMapV2, FunctionBindingOrigin};

/// Reusable code facts, never a source issuance or an execution grant.
#[derive(Debug)]
pub struct SelectedExecutableCode {
    pub(super) canonical_v3: Arc<[u8]>,
    pub(super) full_digest: Digest,
    pub(super) source_map: ExecutableSourceMapV2,
    pub(super) content_digest: SelectedUnitContentDigest,
    pub(super) program: Digest,
    pub(super) body_fingerprints: HashMap<FunctionBindingOrigin, String>,
    pub(super) body_semantic_digests: HashMap<FunctionBindingOrigin, SelectedBodySemanticDigest>,
    pub(super) static_imports: Vec<ExecutableStaticImport>,
    pub(super) linked_native_imports: Vec<LinkedNativeStaticImport>,
    has_interop: bool,
}

impl SelectedExecutableCode {
    /// Perform all original admission/fingerprint checks once for these bytes.
    pub fn admit(canonical_v3: Arc<[u8]>) -> Result<Rc<Self>, String> {
        let admitted =
            crate::admit_executable_unit(&canonical_v3).map_err(|error| error.to_string())?;
        let interop = &admitted.envelope().manifest.interop;
        let has_interop = !interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty();
        Ok(Rc::new(Self {
            full_digest: Digest::sha256(&canonical_v3),
            source_map: admitted.admitted_source_map().clone(),
            content_digest: selected_unit_content_digest(admitted.envelope())
                .map_err(|error| error.to_string())?,
            program: admitted
                .envelope()
                .manifest
                .identity
                .program
                .identity_digest()
                .map_err(|error| error.to_string())?,
            body_fingerprints: admitted
                .retained_callable_origin_fingerprints()
                .map_err(|error| error.to_string())?,
            body_semantic_digests: admitted
                .admitted_semantic_body_digests()
                .map_err(|error| error.to_string())?,
            static_imports: admitted.static_imports().to_vec(),
            linked_native_imports: admitted.linked_native_imports().to_vec(),
            canonical_v3,
            has_interop,
        }))
    }

    pub fn canonical_v3(&self) -> &Arc<[u8]> {
        &self.canonical_v3
    }

    pub fn requires_interop(&self) -> bool {
        self.has_interop
    }

    pub fn source_map(&self) -> &ExecutableSourceMapV2 {
        &self.source_map
    }

    pub fn program_digest(&self) -> Digest {
        self.program
    }

    pub fn content_digest(&self) -> SelectedUnitContentDigest {
        self.content_digest
    }

    pub fn body_digest(&self, origin: FunctionBindingOrigin) -> Option<SelectedBodySemanticDigest> {
        self.body_semantic_digests.get(&origin).copied()
    }

    pub fn static_imports(&self) -> &[ExecutableStaticImport] {
        &self.static_imports
    }
}
