//! Registered opaque input bytes and exact per-program transfer inventories.
use super::*;
use runmat_execution::value::ValueRef;

impl RemotePoolDriver {
    /// Declare the complete opaque object closure for this exact program request.
    /// Include chunks/descendants, not only argument roots. Unscoped requests retain
    /// catalog-wide semantics; scoped admissions never guess a domain payload's closure.
    pub fn register_program_input_inventory(
        &self,
        program: &ProgramExecutionRequest,
        references: Vec<ValueRef>,
    ) -> NativeExecutionResult<()> {
        for reference in &references {
            if reference.authorization_scope != self.value_scope
                || !self.execution_objects.contains(reference)?
            {
                return Err(NativeExecutionError::Protocol(
                    "program input inventory contains an unregistered or foreign object".into(),
                ));
            }
        }
        self.input_inventories
            .lock()
            .expect("input inventory registry poisoned")
            .register(program, references)
    }

    pub fn register_execution_object(
        &self,
        reference: ValueRef,
        encoded: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        self.execution_objects
            .register(reference, encoded.into(), &self.value_scope)
    }

    pub fn execution_object(
        &self,
        reference: &ValueRef,
    ) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        self.execution_objects.get(reference)
    }

    pub fn execution_object_by_digest(
        &self,
        digest: Digest,
    ) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        self.execution_objects.get_by_digest(digest)
    }
}
