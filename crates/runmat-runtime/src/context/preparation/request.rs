use crate::execution::RootedValues;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinOperandPreparation, BuiltinParamArity,
    BuiltinPreparationForm, BuiltinPreparationFormId, BuiltinPreparationOperation,
    BuiltinPreparationReduction, ReductionNaN, SumOutputClass,
};
use runmat_types::{NumericClass, NumericDomain, SumInvocation};
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
    operands: RootedValues,
    numerical_domain: runmat_accelerate_api::NumericalDomainEvidence,
    sum_reduction: Option<AdmittedSumReduction>,
    compound_body: Option<AdmittedCompoundBody>,
    prepared_completion: Option<super::PreparedCompletionRight>,
}

/// An instruction/publication pair supplied by the actual unary operation
/// owner after normal argument, extension and requested-output admission.
/// Construction remains private to runtime domains; services cannot derive
/// this admission from a name, predicate, provider quote, or output class.
#[derive(Clone, Copy)]
pub struct AdmittedRealUnary {
    operation: runmat_types::ScalarUnaryMath,
    publication: crate::builtins::common::real_unary_preparation::RealUnaryPublication,
}

#[derive(Clone, Copy)]
enum AdmittedCompoundBody {
    RealUnary(AdmittedRealUnary),
    RealPower,
}

impl AdmittedRealUnary {
    pub fn operation(self) -> runmat_types::ScalarUnaryMath {
        self.operation
    }
    pub fn publication(
        self,
    ) -> crate::builtins::common::real_unary_preparation::RealUnaryPublication {
        self.publication
    }
}

/// Already-resolved numerical controls, not a second public option grammar.
/// Axes are zero-based, in-bounds and ordered as the owning reduction executes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedSumReduction {
    pub invocation: SumInvocation,
    pub axes: Vec<usize>,
    pub missing: ReductionNaN,
    pub output: SumOutputClass,
    pub input_class: NumericClass,
    pub input_domain: NumericDomain,
    pub accumulation_class: NumericClass,
    pub publication_class: NumericClass,
}

impl AdmittedBuiltinPreparation {
    pub fn plan_compound(
        &self,
        context: Option<&super::RuntimeContext>,
        provider: &'static dyn runmat_accelerate_api::AccelProvider,
    ) -> crate::BuiltinResult<Option<super::AdmittedBuiltinCompound>> {
        match self.compound_body {
            Some(AdmittedCompoundBody::RealUnary(_)) => {
                super::real_unary::plan(self, context, provider).map(|plan| {
                    plan.map(|plan| super::AdmittedBuiltinCompound::RealUnary(Box::new(plan)))
                })
            }
            Some(AdmittedCompoundBody::RealPower) => {
                if let Some(plan) = super::real_power::plan(self, context, provider)? {
                    Ok(Some(super::AdmittedBuiltinCompound::RealPower(Box::new(
                        plan,
                    ))))
                } else {
                    super::real_power_scalar::plan(self, context, provider).map(|plan| {
                        plan.map(|plan| {
                            super::AdmittedBuiltinCompound::RealPowerScalar(Box::new(plan))
                        })
                    })
                }
            }
            None => Ok(None),
        }
    }
    pub fn host_resource_bound(
        &self,
        context: &super::RuntimeContext,
    ) -> Option<super::AdmittedHostResources> {
        super::host_resources::bound(self, Some(context))
    }
    pub fn prepare_host_numerical_output(
        &self,
        context: &super::RuntimeContext,
    ) -> Result<
        Option<super::PreparedHostNumericalOutput>,
        crate::context::host_allocation::HostAllocationError,
    > {
        let Some(resources) = self.host_resource_bound(context) else {
            return Ok(None);
        };
        super::PreparedHostNumericalOutput::prepare_admitted(context, &resources)
    }
    pub(crate) fn with_real_unary(
        mut self,
        operation: runmat_types::ScalarUnaryMath,
        publication: crate::builtins::common::real_unary_preparation::RealUnaryPublication,
    ) -> Result<Self, RuntimeError> {
        if operation
            .real_predicate()
            .map(BuiltinPreparationOperation::RealUnary)
            != Some(self.form.operation)
        {
            return Err(invalid(
                "unary instruction and admitted preparation form disagree",
            ));
        }
        self.compound_body = Some(AdmittedCompoundBody::RealUnary(AdmittedRealUnary {
            operation,
            publication,
        }));
        Ok(self)
    }

    pub fn real_unary(&self) -> Option<AdmittedRealUnary> {
        match self.compound_body {
            Some(AdmittedCompoundBody::RealUnary(unary)) => Some(unary),
            Some(AdmittedCompoundBody::RealPower) | None => None,
        }
    }

    pub(crate) fn with_real_power(mut self) -> Result<Self, RuntimeError> {
        if self.form.operation != BuiltinPreparationOperation::RealPower {
            return Err(invalid(
                "paired POWER body and admitted preparation form disagree",
            ));
        }
        self.compound_body = Some(AdmittedCompoundBody::RealPower);
        Ok(self)
    }

    pub(super) fn is_real_power(&self) -> bool {
        matches!(self.compound_body, Some(AdmittedCompoundBody::RealPower))
    }
    pub(super) fn with_selected_call(mut self, context: &super::RuntimeContext) -> Self {
        if matches!(
            self.form.operation,
            BuiltinPreparationOperation::RuntimeOwned | BuiltinPreparationOperation::Sink(_)
        ) {
            return self;
        }
        if self.prepared_completion.is_none()
            && super::PreparedHostExecution::supports_form(self.form)
        {
            self.prepared_completion = context.prepared_host_execution().and_then(|origin| {
                origin.claim_right(self.admission.binding, self.admission.requested_outputs)
            });
        }
        self
    }
    /// Opt in only after the domain has parsed and certified its complete
    /// prepared route. The selected binding and output mode are compared to
    /// the dispatcher's non-clone right; absence remains ordinary execution.
    pub(crate) fn with_prepared_call_observation(
        mut self,
        observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    ) -> Self {
        self.prepared_completion = observation.and_then(|observation| {
            observation.prepared_right(self.admission.binding, self.admission.requested_outputs)
        });
        self
    }

    pub fn take_prepared_completion_right(&mut self) -> Option<super::PreparedCompletionRight> {
        self.prepared_completion.take()
    }

    pub fn has_prepared_completion_right(&self) -> bool {
        self.prepared_completion.is_some()
    }

    pub fn with_sum_reduction(
        mut self,
        controls: AdmittedSumReduction,
    ) -> Result<Self, RuntimeError> {
        if self.form.operation
            != BuiltinPreparationOperation::Reduction(BuiltinPreparationReduction::Sum)
        {
            return Err(invalid(
                "sum controls do not belong to preparation operation",
            ));
        }
        if self.admission.binding.builtin
            != runmat_builtins::sum_invocation_entry(controls.invocation).identity
        {
            return Err(invalid(
                "sum invocation and admitted catalog binding disagree",
            ));
        }
        self.sum_reduction = Some(controls);
        Ok(self)
    }

    pub fn sum_reduction(&self) -> Option<&AdmittedSumReduction> {
        self.sum_reduction.as_ref()
    }

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
        let evidence = classify(self.operands.values())?;
        let belongs = match (self.form.operation, evidence) {
            (_, NumericalDomainEvidence::Unknown)
            | (
                runmat_builtins::BuiltinPreparationOperation::RealPower,
                NumericalDomainEvidence::RealPower(_),
            ) => true,
            (
                runmat_builtins::BuiltinPreparationOperation::RealUnary(expected),
                NumericalDomainEvidence::RealUnary { predicate, .. },
            ) => expected == predicate,
            (
                runmat_builtins::BuiltinPreparationOperation::RoundDigits,
                NumericalDomainEvidence::RoundDigits { digits, mode },
            ) => mode == runmat_accelerate_api::RoundDigitsMode::Decimals || digits > 0,
            (
                runmat_builtins::BuiltinPreparationOperation::FloatingRemainder(expected),
                NumericalDomainEvidence::FloatingRemainder(actual),
            ) => expected == actual,
            _ => false,
        };
        if !belongs {
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
        if matches!(form.operation, runmat_builtins::BuiltinPreparationOperation::FloatingCast(class) if !matches!(class, runmat_types::NumericClass::Single | runmat_types::NumericClass::Double))
        {
            return Err(invalid(
                "floating cast preparation requires Single or Double target",
            ));
        }
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
        if matches!(form.operation, runmat_builtins::BuiltinPreparationOperation::ElementwiseNary { data_arity } if data_arity < 2)
        {
            return Err(invalid(
                "N-ary preparation requires at least two data operands",
            ));
        }
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
        let operands = RootedValues::new(operands).map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            admission,
            form,
            operands,
            numerical_domain: Default::default(),
            sum_reduction: None,
            compound_body: None,
            prepared_completion: None,
        })
    }

    pub fn admission(&self) -> BuiltinPreparationAdmission {
        self.admission
    }
    pub fn form(&self) -> &'static BuiltinPreparationForm {
        self.form
    }
    pub fn operands(&self) -> &[Value] {
        self.operands.values()
    }
    /// Transfer the admitted values without retiring their roots. A service
    /// must consume them through the owner's sync/async closure and publish
    /// its replacement carrier before that closure returns.
    pub fn into_operands(self) -> RootedValues {
        self.operands
    }
}

pub(super) fn invalid(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Preparation:InvalidContract")
        .build()
}
