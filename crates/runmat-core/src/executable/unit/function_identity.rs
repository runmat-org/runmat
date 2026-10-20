//! Checked projection from one immutable product's portable identity to its local body.
use super::ExecutableUnit;
use runmat_execution::FunctionBindingOrigin;
use runmat_types::{CallableIdentity, FunctionId, ProgramFunctionId};

impl ExecutableUnit {
    pub(crate) fn local_function_for_origin(
        &self,
        origin: FunctionBindingOrigin,
    ) -> Result<Option<FunctionId>, String> {
        let program = self
            .revision()
            .program_revision
            .identity_digest()
            .map_err(|e| e.to_string())?;
        if origin.program != program {
            return Ok(None);
        }
        let Some(facts) = self.function_analysis(origin.function) else {
            return Ok(None);
        };
        let Some(CallableIdentity::BoundFunction(local)) = &facts.callable.identity else {
            return Err("portable callable has no exact local self binding".into());
        };
        let body = self
            .functions()
            .get(*local)
            .ok_or("portable callable body is absent")?;
        if body.function != *local
            || body.portable_function != origin.function
            || body.binding_origin != Some(origin)
            || !self.owns_callable(*local)
        {
            return Err("portable callable disagrees with its original body owner".into());
        }
        Ok(Some(*local))
    }

    pub(crate) fn portable_function_for_local(
        &self,
        local: FunctionId,
    ) -> Result<ProgramFunctionId, String> {
        self.analysis()
            .function_for_local(local)
            .map(|facts| facts.function)
            .ok_or_else(|| {
                format!("local function {local:?} has no unique current-product analysis owner")
            })
    }
}
