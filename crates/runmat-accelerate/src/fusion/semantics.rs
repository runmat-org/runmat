//! Request-local type resolution. Cached structural plans never own runtime
//! facts or evidence about the contents of a different invocation's operands.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::precision::PrecisionPolicy;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::infer_catalog_call;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    CanonicalFusionDeclaration, FusionAdmissionContext, FusionArithmeticEvidence,
    FusionPowerArithmeticReceipt,
};
use runmat_runtime::value_fact::value_fact;
use runmat_types::{
    infer_binary, infer_unary, CallRequest, NumericClass, NumericDomain, OutputSelection, ValueFact,
};
use runmat_value::Value;
use serde::Serialize;

use super::semantic_bindings::{self, SemanticBinding};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::{FusionCallOrigin, PrimitiveOp, ValueId};

fn exact_scalar_control(
    value: Option<&Value>,
) -> Option<runmat_runtime::builtins::common::spec::FusionScalarControl> {
    use runmat_runtime::builtins::common::spec::FusionScalarControl;
    use runmat_value::NumericScalar;
    match value? {
        Value::Num(value) => Some(FusionScalarControl::DoubleBits(value.to_bits())),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0)? {
            NumericScalar::F32(value) => Some(FusionScalarControl::SingleBits(value.to_bits())),
            NumericScalar::F64(value) => Some(FusionScalarControl::DoubleBits(value.to_bits())),
            _ => None,
        },
        _ => None,
    }
}

#[path = "semantics/constant_identity.rs"]
mod constant_identity;
pub(crate) use constant_identity::exact_constant_digest;
#[path = "semantics/invocation_leaves.rs"]
pub(super) mod invocation_leaves;
use invocation_leaves::FusionInvocationLeaves;
#[path = "semantics/assembly.rs"]
mod assembly;
#[path = "semantics/request_literals.rs"]
mod request_literals;
#[path = "semantics/sign.rs"]
mod sign;
#[path = "semantics/sign_reduction.rs"]
mod sign_reduction;
#[path = "semantics/static_resolution.rs"]
mod static_resolution;
use assembly::{inferred_operation, record_operation, selected_values, source_assembly};

fn validate_arithmetic(
    declaration: CanonicalFusionDeclaration,
    input_ids: &[ValueId],
    arguments: &[ValueFact],
    output_fact: &ValueFact,
    values: &[Option<&Value>],
    admitted: Option<FusionArithmeticEvidence>,
) -> Result<Option<FusionArithmeticEvidence>> {
    let semantic = declaration
        .semantics
        .ok_or_else(|| anyhow!("arithmetic admission lacks selected semantics"))?;
    if admitted.map(|evidence| evidence.kind()) != semantic.required_arithmetic {
        return Err(anyhow!("selected arithmetic authority kind mismatch"));
    }
    if let Some(FusionArithmeticEvidence::RealPower(receipt)) = admitted {
        let [base_id, exponent_id] = input_ids else {
            return Err(anyhow!(
                "power arithmetic receipt has invalid operand topology"
            ));
        };
        if receipt.binding != declaration.binding
            || receipt.revision != semantic.revision
            || (receipt.base_id, receipt.exponent_id) != (*base_id, *exponent_id)
            || arguments[0].numeric().map(|numeric| numeric.class) != Some(receipt.base_class)
            || arguments[1].numeric().map(|numeric| numeric.class) != Some(receipt.exponent_class)
            || output_fact.numeric().map(|numeric| numeric.class) != Some(receipt.output_class)
            || receipt.exponent_scalar != exact_scalar_control(values[1])
            || Some(receipt.mode)
                != runmat_accelerate_api::RealPowerArithmeticMode::for_classes(
                    receipt.base_class,
                    receipt.exponent_class,
                )
            || receipt.output_class != receipt.mode.output_class()
        {
            return Err(anyhow!(
                "power arithmetic receipt differs from selected source"
            ));
        }
    }
    Ok(admitted)
}

fn source_constants(plan: &FusionGroupPlan) -> Result<Vec<(ValueId, Digest)>> {
    let mut constants = plan
        .const_values
        .iter()
        .map(|(&id, value)| {
            constant_identity::exact_constant_digest(value).map(|digest| (id, digest))
        })
        .collect::<Result<Vec<_>>>()?;
    constants.sort_unstable_by_key(|(id, _)| *id);
    Ok(constants)
}

fn source_slot_constants(plan: &FusionGroupPlan) -> Result<Vec<(usize, Digest)>> {
    let mut constants = plan
        .constants
        .iter()
        .map(|(&slot, value)| {
            constant_identity::exact_constant_digest(value).map(|digest| (slot, digest))
        })
        .collect::<Result<Vec<_>>>()?;
    constants.sort_unstable_by_key(|(slot, _)| *slot);
    Ok(constants)
}

#[path = "semantics/source_plan_identity.rs"]
mod source_plan_identity;
pub use source_plan_identity::exact_source_plan_identity;

/// Ordered source topology after the canonical binding capture. Public names
/// never become a second execution identity after this boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
enum SourceOperationIdentity {
    Primitive {
        operation: PrimitiveOp,
        binding: Option<BuiltinBindingIdentity>,
        inputs: Vec<ValueId>,
        output: Option<ValueId>,
    },
    Builtin {
        binding: BuiltinBindingIdentity,
        origin: FusionCallOrigin,
        inputs: Vec<ValueId>,
        output: Option<ValueId>,
    },
}

fn source_operations(
    plan: &FusionGroupPlan,
    snapshots: &[Option<SemanticBinding>],
) -> Result<Vec<SourceOperationIdentity>> {
    plan.operations
        .iter()
        .zip(snapshots)
        .map(|(operation, snapshot)| match operation {
            FusionOp::Primitive { op, inputs, output } => Ok(SourceOperationIdentity::Primitive {
                operation: *op,
                binding: snapshot.map(|binding| binding.declaration.binding),
                inputs: inputs.clone(),
                output: *output,
            }),
            FusionOp::Builtin {
                origin,
                inputs,
                output,
                ..
            } => Ok(SourceOperationIdentity::Builtin {
                binding: snapshot
                    .ok_or_else(|| anyhow!("fusion builtin has no captured binding"))?
                    .declaration
                    .binding,
                origin: *origin,
                inputs: inputs.clone(),
                output: *output,
            }),
        })
        .collect()
}

#[derive(Debug)]
pub(crate) struct ResolvedProgram {
    source_operations: Vec<SourceOperationIdentity>,
    source_input_ids: Vec<ValueId>,
    source_input_set: HashSet<ValueId>,
    source_constants: Vec<(ValueId, Digest)>,
    source_slot_constants: Vec<(usize, Digest)>,
    source_root: Option<ValueId>,
    facts: HashMap<ValueId, ValueFact>,
    source_inputs: Vec<ValueFact>,
    precision_policy: PrecisionPolicy,
    provider_precision: ProviderPrecision,
    /// Selected bindings are part of request identity, not inferred from output
    /// class or a spelling-based expression dispatch table.
    binding_contracts: Vec<SemanticBinding>,
    declarations: HashMap<ValueId, SemanticBinding>,
    evidence: BTreeMap<ValueId, runmat_accelerate_api::NumericalDomainEvidence>,
    sign: BTreeMap<ValueId, runmat_runtime::builtins::common::spec::FusionSignEvidence>,
    arithmetic: BTreeMap<ValueId, FusionArithmeticEvidence>,
    admission_work: BTreeMap<ValueId, runmat_runtime::builtins::common::spec::FusionAdmissionWork>,
}

#[derive(Debug)]
pub(crate) enum FusionResolveFailure {
    Structural(anyhow::Error),
    Admission(anyhow::Error),
    Runtime(Box<runmat_runtime::RuntimeError>),
}

impl std::fmt::Display for FusionResolveFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structural(error) | Self::Admission(error) => {
                std::fmt::Display::fmt(error, formatter)
            }
            Self::Runtime(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for FusionResolveFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) | Self::Admission(error) => Some(error.as_ref()),
            Self::Runtime(error) => Some(error.as_ref()),
        }
    }
}

impl From<anyhow::Error> for FusionResolveFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::Structural(error)
    }
}

impl From<runmat_runtime::RuntimeError> for FusionResolveFailure {
    fn from(error: runmat_runtime::RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}

impl FusionResolveFailure {
    pub(crate) fn into_execution_or_structural(
        self,
    ) -> std::result::Result<anyhow::Error, runmat_accelerate_api::ProviderExecutionError> {
        match self {
            Self::Structural(error) => Ok(error),
            Self::Admission(error) => Err(error.into()),
            Self::Runtime(error) => Err(
                runmat_accelerate_api::ProviderExecutionError::from_source(*error),
            ),
        }
    }
}

impl ResolvedProgram {
    pub async fn resolve(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let inputs: Vec<_> = inputs.iter().collect();
        Self::resolve_borrowed(plan, &inputs, provider_precision, correlation).await
    }

    pub async fn resolve_borrowed(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let bindings = semantic_bindings::capture(plan)?;
        Self::resolve_captured(plan, inputs, provider_precision, correlation, &bindings).await
    }

    pub(crate) async fn resolve_captured(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
        bindings: &[Option<SemanticBinding>],
    ) -> std::result::Result<Self, FusionResolveFailure> {
        Self::resolve_observed(
            plan,
            inputs,
            provider_precision,
            PrecisionPolicy::current(),
            correlation,
            bindings,
        )
        .await
    }

    #[cfg(test)]
    pub async fn resolve_with_policy(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let inputs: Vec<_> = inputs.iter().collect();
        let bindings = semantic_bindings::capture(plan)?;
        Self::resolve_observed(
            plan,
            &inputs,
            provider_precision,
            precision_policy,
            None,
            &bindings,
        )
        .await
    }

    async fn resolve_observed(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
        correlation: Option<crate::placement::PlacementCorrelationId>,
        snapshots: &[Option<SemanticBinding>],
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let (mut program, invocation_leaves) = source_assembly(
            plan,
            inputs,
            provider_precision,
            precision_policy,
            correlation,
            snapshots,
        )?;
        let mut real_ranges = None;
        let mut real_range_index_reported = false;
        for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
            let inferred = inferred_operation(&program, operation, *snapshot, &invocation_leaves)?;
            let values = selected_values(&invocation_leaves, inferred.input_ids);
            let admitted = if let (Some(output), Some(snapshot)) = (inferred.output, snapshot) {
                if let Some(admit) = snapshot
                    .declaration
                    .semantics
                    .and_then(|semantics| semantics.admit)
                {
                    let mut observed = super::admission::AdmissionObservation::new(correlation);
                    let mut operand_evidence = inferred
                        .input_ids
                        .iter()
                        .map(|id| {
                            let numerical_domain = program.evidence.get(id).copied();
                            let sign = program.sign.get(id).copied();
                            if numerical_domain.is_some() || sign.is_some() {
                                Some(
                            runmat_runtime::builtins::common::spec::FusionOperandDomainEvidence {
                                value_id: *id,
                                numerical_domain: numerical_domain.unwrap_or(runmat_accelerate_api::NumericalDomainEvidence::Unknown),
                                sign,
                            }
                                )
                            } else { None }
                        })
                        .collect::<Vec<_>>();
                    if let Some((predicate, internal_error)) = snapshot
                        .declaration
                        .semantics
                        .and_then(|semantics| semantics.required_real_unary)
                    {
                        if real_ranges.is_none() {
                            real_ranges = Some(super::real_domain::RealRangeResolver::new(
                                plan,
                                &invocation_leaves,
                            ));
                        }
                        let resolver = real_ranges.as_mut().expect("just initialized");
                        if !real_range_index_reported {
                            resolver.report_index(
                                &mut observed.work,
                                snapshot.declaration.binding,
                                provider_precision,
                            );
                            real_range_index_reported = true;
                        }
                        if !resolver.is_valid() {
                            return Err(anyhow!(
                                "fusion graph has invalid or duplicate value producers"
                            )
                            .into());
                        }
                        let [operand_id] = inferred.input_ids else {
                            return Err(
                                anyhow!("real unary fusion declaration has invalid arity").into()
                            );
                        };
                        let prior_sign = operand_evidence[0].and_then(|evidence| evidence.sign);
                        operand_evidence[0] = super::real_domain::prove(
                            super::real_domain::RealDomainProofContext {
                                plan,
                                snapshots,
                                facts: &program.facts,
                                binding: snapshot.declaration.binding,
                                internal_error,
                                provider_precision,
                            },
                            resolver,
                            *operand_id,
                            predicate,
                            &mut observed.work,
                        )
                        .await?
                        .map(|mut evidence| {
                            evidence.sign = prior_sign;
                            evidence
                        });
                    }
                    let context = FusionAdmissionContext::new(
                        inferred.input_ids,
                        &values,
                        &inferred.arguments,
                        &operand_evidence,
                        &mut observed.work,
                    )
                    .ok_or_else(|| anyhow!("fusion operand evidence identity mismatch"))?
                    .with_output_fact(&inferred.fact);
                    let result = admit(context)
                        .await
                        .map_err(FusionResolveFailure::Admission)?;
                    let work = observed.completed();
                    let evidence = result.ok_or_else(|| anyhow!(
                        "fusion operation requires unavailable domain evidence: binding={:?}, output={output:?}, inputs={:?}, represented_operands={:?}",
                        snapshot.declaration.binding, inferred.input_ids,
                        values.iter().map(Option::is_some).collect::<Vec<_>>()
                    ))?;
                    Some((evidence, work))
                } else {
                    None
                }
            } else {
                None
            };
            let output = inferred.output;
            let reduction_refinement = if output.is_some()
                && snapshot.is_some_and(|binding| binding.declaration.reduction.is_some())
                && inferred
                    .input_ids
                    .first()
                    .is_some_and(|id| program.sign.contains_key(id))
            {
                let mut observed = super::admission::AdmissionObservation::new(correlation);
                let result = sign_reduction::resolved_sign(
                    &program,
                    &inferred,
                    &values,
                    *snapshot,
                    &mut observed.work,
                )
                .await?;
                let work = observed.completed();
                result.map(|result| (result, work))
            } else {
                None
            };
            let add_refinement = if output.is_some() && sign::applies(operation, *snapshot) {
                let mut observed = super::admission::AdmissionObservation::new(correlation);
                let result = sign::add_sign(
                    &program,
                    operation,
                    &inferred,
                    &values,
                    *snapshot,
                    &mut observed.work,
                )?;
                let work = observed.completed();
                result.map(|result| (result, work))
            } else {
                None
            };
            record_operation(&mut program, inferred, *snapshot, &values, admitted)?;
            if let Some(output) = output {
                if let Some(((sign, kind, storage), work)) = reduction_refinement {
                    assembly::record_sign_refinement(
                        &mut program,
                        output,
                        sign,
                        kind,
                        Some(storage),
                        work,
                    )?;
                }
                if let Some(((sign, kind), work)) = add_refinement {
                    assembly::record_sign_refinement(&mut program, output, sign, kind, None, work)?;
                }
            }
        }
        invocation_leaves.require_live_resident_inputs()?;
        Ok(program)
    }

    pub fn fact(&self, id: ValueId) -> Result<&ValueFact> {
        self.facts
            .get(&id)
            .ok_or_else(|| anyhow!("missing fusion value fact {id}"))
    }

    pub(crate) fn is_invocation_input(&self, id: ValueId) -> bool {
        self.source_input_set.contains(&id)
    }

    pub fn physical_inputs_are_single(&self, indices: &[usize]) -> bool {
        indices.iter().all(|&index| {
            self.source_inputs.get(index).is_some_and(|fact| {
                fact.numeric()
                    .is_some_and(|numeric| numeric.class == NumericClass::Single)
            })
        })
    }

    pub fn declaration(&self, id: ValueId) -> Option<CanonicalFusionDeclaration> {
        self.declarations
            .get(&id)
            .map(|snapshot| snapshot.declaration)
    }
    pub fn power_arithmetic(&self, id: ValueId) -> Option<FusionPowerArithmeticReceipt> {
        match self.arithmetic.get(&id).copied()? {
            FusionArithmeticEvidence::RealPower(receipt) => Some(receipt),
        }
    }
    pub fn exact_single_base_power(
        &self,
        id: ValueId,
    ) -> Option<runmat_accelerate_api::ExactSingleBasePower> {
        use runmat_accelerate_api::{
            ExactSingleBasePower, RealPowerArithmeticMode, RealPowerOperand,
        };
        let receipt = self.power_arithmetic(id)?;
        if receipt.base_class != NumericClass::Single
            || receipt.exponent_class != NumericClass::Double
            || receipt.output_class != NumericClass::Single
            || receipt.mode
                != (RealPowerArithmeticMode::MixedDoubleThenSingle {
                    single_operand: RealPowerOperand::Base,
                })
            || self.fact(receipt.exponent_id).ok()?.shape.element_count() != Some(1)
        {
            return None;
        }
        ExactSingleBasePower::for_control(receipt.exponent_scalar?)
    }

    /// Controls are omitted only if every use is an authenticated exponent.
    /// They remain source inputs and original-bit semantic identity facts.
    pub fn embedded_power_controls(
        &self,
        plan: &FusionGroupPlan,
        outputs: &[ValueId],
    ) -> HashSet<ValueId> {
        let mut controls = plan
            .operations
            .iter()
            .filter_map(|operation| {
                let output = match operation {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                        *output
                    }
                }?;
                self.exact_single_base_power(output)?;
                Some(self.power_arithmetic(output)?.exponent_id)
            })
            .filter(|id| plan.inputs.contains(id) && !outputs.contains(id))
            .collect::<HashSet<_>>();
        for operation in &plan.operations {
            let (inputs, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let exact = output.and_then(|output| self.exact_single_base_power(output));
            for (position, input) in inputs.iter().enumerate() {
                if position != 1 || exact.is_none() {
                    controls.remove(input);
                }
            }
        }
        controls
    }

    pub fn uses_implicit_downcast_for(&self, inputs: &[usize]) -> bool {
        self.precision_policy.permits_double_to_single()
            && self.provider_precision == ProviderPrecision::F32
            && inputs.iter().any(|&index| {
                self.source_inputs.get(index).is_some_and(|fact| {
                    fact.numeric()
                        .is_some_and(|numeric| numeric.class == NumericClass::Double)
                })
            })
    }
    pub fn require_ordinary_arithmetic(
        &self,
        id: ValueId,
        precision: ProviderPrecision,
    ) -> Result<()> {
        match self.arithmetic.get(&id) {
            Some(FusionArithmeticEvidence::RealPower(_))
                if self.exact_single_base_power(id).is_some() =>
            {
                Ok(())
            }
            Some(FusionArithmeticEvidence::RealPower(receipt))
                if matches!(
                    receipt.mode,
                    runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
                ) || (precision == ProviderPrecision::F32
                    && receipt.mode
                        == runmat_accelerate_api::RealPowerArithmeticMode::NativeDouble) =>
            {
                Err(anyhow!(
                    "ordinary fusion cannot execute selected POWER arithmetic"
                ))
            }
            _ => Ok(()),
        }
    }
    pub fn catalog_entry(
        &self,
        id: ValueId,
    ) -> Option<&'static runmat_builtins::BuiltinCatalogEntry> {
        self.declarations.get(&id).map(|snapshot| snapshot.entry)
    }

    pub fn floating_class(&self, id: ValueId) -> Result<NumericClass> {
        let numeric = self
            .fact(id)?
            .numeric()
            .ok_or_else(|| anyhow!("fusion output is not proven numeric"))?;
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return Err(anyhow!(
                "fusion representation requires a non-floating materializer"
            ));
        }
        Ok(numeric.class)
    }

    /// Versioned canonical encoding of typed identities and full facts. The
    /// shader text is additionally hashed by the placement caller, capturing
    /// the selected binding's actual expression lowering, not a function address.
    pub fn cache_identity(&self) -> Result<Digest> {
        let facts: BTreeMap<_, _> = self.facts.iter().collect();
        let declarations = self
            .binding_contracts
            .iter()
            .map(|snapshot| {
                (
                    snapshot.declaration.binding,
                    snapshot.entry.contract,
                    snapshot.entry.placement,
                )
            })
            .collect::<Vec<_>>();
        let revisions: BTreeMap<_, _> = self
            .declarations
            .iter()
            .map(|(id, declaration)| {
                (
                    *id,
                    (
                        declaration.declaration.semantics.map(|semantics| {
                            (
                                semantics.revision,
                                semantics.helpers,
                                semantics.required_arithmetic,
                            )
                        }),
                        declaration
                            .declaration
                            .reduction
                            .map(|reduction| reduction.revision),
                    ),
                )
            })
            .collect();
        #[derive(serde::Serialize)]
        struct AdmissionIdentity<'a> {
            domains: &'a BTreeMap<ValueId, runmat_accelerate_api::NumericalDomainEvidence>,
            signs:
                &'a BTreeMap<ValueId, runmat_runtime::builtins::common::spec::FusionSignEvidence>,
            arithmetic: &'a BTreeMap<ValueId, FusionArithmeticEvidence>,
            work:
                &'a BTreeMap<ValueId, runmat_runtime::builtins::common::spec::FusionAdmissionWork>,
        }
        Ok(Digest::sha256(serde_json::to_vec(&(
            "typed-fusion-v5",
            runmat_builtins::BUILTIN_CATALOG_SCHEMA_VERSION,
            &self.source_operations,
            &self.source_input_ids,
            &self.source_constants,
            &self.source_slot_constants,
            self.source_root,
            declarations,
            facts,
            &self.source_inputs,
            self.precision_policy,
            self.provider_precision,
            revisions,
            AdmissionIdentity {
                domains: &self.evidence,
                signs: &self.sign,
                arithmetic: &self.arithmetic,
                work: &self.admission_work,
            },
        ))?))
    }

    pub fn require_source_plan(&self, plan: &FusionGroupPlan) -> Result<()> {
        let snapshots = semantic_bindings::capture(plan)?;
        if source_operations(plan, &snapshots)? != self.source_operations
            || plan.inputs != self.source_input_ids
            || plan.output != self.source_root
        {
            return Err(anyhow!(
                "fusion source topology differs from admitted program"
            ));
        }
        for (operation, snapshot) in plan.operations.iter().zip(&snapshots) {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(output) = output {
                if let Some(admitted) = self.declarations.get(output) {
                    let selected = snapshot
                        .ok_or_else(|| anyhow!("fusion selected declaration disappeared"))?;
                    if admitted.declaration.binding != selected.declaration.binding
                        || admitted
                            .declaration
                            .semantics
                            .map(|semantic| semantic.revision)
                            != selected
                                .declaration
                                .semantics
                                .map(|semantic| semantic.revision)
                        || admitted
                            .declaration
                            .reduction
                            .map(|reduction| reduction.revision)
                            != selected
                                .declaration
                                .reduction
                                .map(|reduction| reduction.revision)
                    {
                        return Err(anyhow!("fusion selected declaration revision changed"));
                    }
                }
            }
        }
        if source_constants(plan)? != self.source_constants {
            return Err(anyhow!(
                "fusion source constants differ from admitted program"
            ));
        }
        if source_slot_constants(plan)? != self.source_slot_constants {
            return Err(anyhow!(
                "fusion source slot constants differ from admitted program"
            ));
        }
        Ok(())
    }
}

fn primitive_fact(operation: PrimitiveOp, arguments: &[ValueFact]) -> Result<ValueFact> {
    // The host matrix-product owner delegates exactly to elementwise times
    // when either operand is scalar. Other matrix products require their own
    // contraction route: even an outer product adds the term to +0, unlike a
    // bare multiplication for signed zero, and Single uses a wider host sum.
    if operation == PrimitiveOp::Mul
        && !arguments
            .iter()
            .any(|argument| argument.shape.element_count() == Some(1))
    {
        return Err(anyhow!(
            "typed scalar fusion requires a proven scalar matrix-product operand"
        ));
    }
    let operator = match operation {
        // Real operands alone do not establish power's result domain or its
        // exceptional-value semantics. The operation owner must supply proof.
        PrimitiveOp::Pow | PrimitiveOp::ElemPow => {
            return Err(anyhow!(
                "fusion power requires operation-owned domain evidence"
            ));
        }
        _ => operation.operator_kind(),
    };
    let inference = match arguments {
        [operand] => infer_unary(operator, operand),
        [left, right] => infer_binary(operator, left, right),
        _ => return Err(anyhow!("invalid fusion primitive operand count")),
    };
    if !inference.diagnostics.is_empty() {
        return Err(anyhow!("fusion primitive operands are not admitted"));
    }
    Ok(inference.fact)
}

#[cfg(test)]
#[path = "semantics_tests.rs"]
mod tests;
