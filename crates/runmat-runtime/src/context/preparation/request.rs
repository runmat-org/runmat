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
    /// Already admitted by the domain. None preserves its default convention;
    /// zero still executes effects. Fixed descriptors can admit padded RunMat
    /// extensions, so output policy and public errors remain domain-owned.
    pub requested_outputs: Option<usize>,
}

/// Only declared data operands cross the service boundary, in catalog order.
pub struct AdmittedBuiltinPreparation {
    admission: BuiltinPreparationAdmission,
    form: &'static BuiltinPreparationForm,
    operands: Vec<Value>,
    numerical_domain: runmat_accelerate_api::NumericalDomainEvidence,
}

impl AdmittedBuiltinPreparation {
    /// Evaluate semantic evidence against this request's immutable operands.
    /// Domain implementations cannot attach a detached cached boolean to a
    /// later call, and services cannot replace these operands in place.
    pub(crate) fn with_numerical_domain(
        mut self,
        classify: impl FnOnce(
            &[Value],
        )
            -> Result<runmat_accelerate_api::NumericalDomainEvidence, RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        use runmat_accelerate_api::NumericalDomainEvidence;
        let evidence = classify(&self.operands)?;
        if !matches!(
            (self.form.operation, evidence),
            (_, NumericalDomainEvidence::Unknown)
                | (
                    runmat_builtins::BuiltinPreparationOperation::RealPower,
                    NumericalDomainEvidence::RealPower(_)
                )
        ) {
            return Err(invalid(
                "numerical evidence does not belong to preparation operation",
            ));
        }
        self.numerical_domain = evidence;
        Ok(self)
    }

    pub fn numerical_domain(&self) -> runmat_accelerate_api::NumericalDomainEvidence {
        self.numerical_domain
    }
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
        // Immutable declaration validation belongs to the existing catalog
        // validation gate. Do not allocate sets or scan every form on each call.
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            return Err(invalid(
                "runtime-owned builtin has no shared preparation form",
            ));
        };
        let mut matching_forms = forms.iter().filter(|form| form.id == admission.form);
        let form = matching_forms
            .next()
            .ok_or_else(|| invalid("unknown canonical preparation form"))?;
        if matching_forms.next().is_some() {
            return Err(invalid("ambiguous canonical preparation form"));
        }
        if form.binding_variant != admission.binding.variant
            || !form.signatures.contains(&admission.signature)
            || entry
                .bindings
                .iter()
                .filter(|binding| binding.variant == admission.binding.variant)
                .count()
                != 1
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
        let signature = entry
            .descriptor
            .signatures
            .get(admission.signature)
            .ok_or_else(|| invalid("admitted signature is absent from the descriptor"))?;
        if form
            .data_arguments
            .iter()
            .any(|&position| position >= signature.inputs.len())
            || !form.data_arguments.windows(2).all(|pair| pair[0] < pair[1])
            || form
                .operation
                .data_arity()
                .is_some_and(|count| count != operands.len())
        {
            return Err(invalid(
                "selected preparation form has inconsistent data roles",
            ));
        }
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
            numerical_domain: Default::default(),
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
