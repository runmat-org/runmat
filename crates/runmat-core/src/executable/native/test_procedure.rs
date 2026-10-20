//! Match discovered test procedures to one body's frozen compiler identity.

use runmat_test::descriptor::{ProcedureDescriptor, ProcedureKind};

use super::{native_product_error, NativeCompilationInput};

impl NativeCompilationInput {
    /// Exact source owner of a local function in this frozen compilation.
    pub fn function_source(
        &self,
        function: runmat_types::FunctionId,
    ) -> Option<&runmat_execution::ExecutableSourceRecordV2> {
        let source = self.mir.functions.get(&function)?.source;
        self.source_map
            .entries
            .get(usize::try_from(source.0).ok()?)
            .filter(|record| record.source_id == source)
    }

    /// Match source ownership, compiler kind/name, and exact HIR-origin byte
    /// span; no caller-supplied ordinal becomes a selection authority.
    pub fn resolve_test_procedure(
        &self,
        procedure: &ProcedureDescriptor,
    ) -> Result<runmat_types::FunctionId, runmat_runtime::RuntimeError> {
        if procedure.source.semantic_path != procedure.semantic_path
            || !matches!(
                procedure.kind,
                ProcedureKind::Function | ProcedureKind::Method
            )
        {
            return Err(native_product_error(
                "paired native test requires a frozen function or method procedure".into(),
            ));
        }
        let name = crate::testing::test_procedure_function_name(procedure)
            .map_err(native_product_error)?;
        let method_name = if procedure.kind == ProcedureKind::Method {
            Some(
                crate::testing::test_procedure_method_name(procedure)
                    .map_err(native_product_error)?,
            )
        } else {
            None
        };
        let mut matches = self.mir.functions.iter().filter_map(|(local, metadata)| {
            let id = *local;
            let source = self.function_source(id)?;
            let kind_matches = match procedure.kind {
                ProcedureKind::Function => {
                    metadata.kind == runmat_hir::FunctionKind::Named
                        && metadata.class_method_owner.is_none()
                        && metadata.enclosing_class.is_none()
                        // HIR discovery includes named local functions, so a
                        // parent is admissible only through this exact span.
                        && metadata.name.0 == name
                }
                ProcedureKind::Method => {
                    metadata.parent.is_none()
                        && metadata.enclosing_class.is_some()
                        && metadata.class_method_owner.as_ref().is_some_and(|owner| {
                            matches!(&metadata.kind, runmat_hir::FunctionKind::ClassMethod { is_static, role }
                                if *is_static == owner.is_static && *role == owner.role)
                                && method_name == Some(owner.method.0.as_str())
                        })
                }
                _ => false,
            };
            (kind_matches
                && source.owner_identity == procedure.source.owner_identity
                && source.relative_path == procedure.source.relative_path
                && usize::try_from(procedure.source.span.start_byte).ok()? == metadata.span.start
                && usize::try_from(procedure.source.span.end_byte).ok()? == metadata.span.end
                && self.mir.bodies.contains_key(local))
            .then_some(id)
        });
        let selected = matches.next().ok_or_else(|| {
            native_product_error("test procedure has no exact frozen compiled body".into())
        })?;
        if matches.next().is_some() {
            return Err(native_product_error(
                "test procedure ambiguously names multiple frozen bodies".into(),
            ));
        }
        Ok(selected)
    }
}
