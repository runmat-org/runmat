use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinOperandPreparation, BuiltinParamArity,
    BuiltinPreparationForm, BuiltinPreparationFormId,
};
use runmat_value::Value;

/// The domain supplies this only after its existing output, compatibility,
/// option and operand admission has succeeded. It is not inferred from values.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinPreparationAdmission {
    pub binding: BuiltinBindingIdentity,
    pub form: BuiltinPreparationFormId,
    pub signature: usize,
    pub input_count: usize,
    pub requested_outputs: Option<usize>,
}

/// Only declared data operands cross the service boundary, in catalog order.
pub struct AdmittedBuiltinPreparation {
    admission: BuiltinPreparationAdmission,
    form: &'static BuiltinPreparationForm,
    operands: Vec<Value>,
}

impl AdmittedBuiltinPreparation {
    pub fn new(
        admission: BuiltinPreparationAdmission,
        operands: Vec<Value>,
    ) -> Result<Self, RuntimeError> {
        let entry =
            runmat_builtins::builtin_catalog_primary_entry_by_name(admission.binding.builtin.name)
                .ok_or_else(|| invalid("preparation requires a canonical builtin identity"))?;
        Self::from_entry(entry, admission, operands)
    }

    pub(super) fn from_entry(
        entry: &'static BuiltinCatalogEntry,
        admission: BuiltinPreparationAdmission,
        operands: Vec<Value>,
    ) -> Result<Self, RuntimeError> {
        if entry.identity != admission.binding.builtin {
            return Err(invalid("preparation binding and catalog identity disagree"));
        }
        if !runmat_builtins::validate_builtin_preparation(entry).is_empty() {
            return Err(invalid("invalid canonical operand preparation declaration"));
        }
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            return Err(invalid(
                "runtime-owned builtin has no shared preparation form",
            ));
        };
        let form = forms
            .iter()
            .find(|form| form.id == admission.form)
            .ok_or_else(|| invalid("unknown canonical preparation form"))?;
        if form.binding_variant != admission.binding.variant
            || !form.signatures.contains(&admission.signature)
        {
            return Err(invalid(
                "admitted binding/signature does not belong to preparation form",
            ));
        }
        if operands.len() != form.data_arguments.len()
            || form
                .data_arguments
                .iter()
                .any(|&position| position >= admission.input_count)
        {
            return Err(invalid(
                "admitted data operands do not match declared argument roles",
            ));
        }
        // A consistency check on the domain's witness, not normal call admission
        // and not a replacement for domain-specific errors or option parsing.
        let signature = &entry.descriptor.signatures[admission.signature];
        let required = signature
            .inputs
            .iter()
            .filter(|p| p.arity == BuiltinParamArity::Required)
            .count();
        let variadic = signature
            .inputs
            .iter()
            .any(|p| p.arity == BuiltinParamArity::Variadic);
        if admission.input_count < required
            || (!variadic && admission.input_count > signature.inputs.len())
        {
            return Err(invalid(
                "admitted input count is inconsistent with descriptor signature",
            ));
        }
        Ok(Self {
            admission,
            form,
            operands,
        })
    }

    pub fn admission(&self) -> BuiltinPreparationAdmission {
        self.admission
    }
    pub fn form(&self) -> &'static BuiltinPreparationForm {
        self.form
    }
    pub fn operands(&self) -> &[Value] {
        &self.operands
    }
    pub fn into_operands(self) -> Vec<Value> {
        self.operands
    }
}

pub(super) fn invalid(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Preparation:InvalidContract")
        .build()
}
