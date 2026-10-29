//! Request-local type resolution. Cached structural plans never own runtime
//! facts or evidence about the contents of a different invocation's operands.

#[cfg(test)]
use std::collections::HashSet;

use crate::precision::PrecisionPolicy;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::infer_catalog_call;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    CanonicalFusionDeclaration, FusionAdmissionContext, FusionAdmissionWork,
    FusionArithmeticEvidence, FusionPowerArithmeticReceipt,
};
use runmat_runtime::value_fact::value_fact;
use runmat_types::{
    infer_binary, infer_unary, CallRequest, NumericClass, NumericDomain, OutputSelection, ValueFact,
};
use runmat_value::Value;
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};

use super::semantic_bindings::{self, SemanticBinding};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::{FusionCallOrigin, PrimitiveOp, ValueId};

pub(crate) fn exact_scalar_control(
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
pub(crate) use invocation_leaves::{authenticate_resident_input, FloatingFusionEligibility};
#[path = "semantics/fact_index.rs"]
pub(super) mod fact_index;
use fact_index::FactIndex;
#[path = "semantics/slot_map.rs"]
mod slot_map;
use slot_map::SlotMap;

#[path = "semantics/exact_product.rs"]
mod exact_product;
pub(crate) use exact_product::ExactProductReceipt;
#[path = "semantics/assembly.rs"]
mod assembly;
#[path = "semantics/computed_scalars.rs"]
mod computed_scalars;
#[path = "semantics/preparation_resources.rs"]
pub(crate) mod preparation_resources;
#[path = "semantics/request_literals.rs"]
mod request_literals;
use computed_scalars::{ComputedScalars, ScalarEvidenceObservation};
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
    let mut constants = Vec::with_capacity(plan.const_values.len());
    for (&id, value) in &plan.const_values {
        constants.push((id, constant_identity::exact_constant_digest(value)?));
    }
    constants.sort_unstable_by_key(|(id, _)| *id);
    Ok(constants)
}

fn source_slot_constants(plan: &FusionGroupPlan) -> Result<Vec<(usize, Digest)>> {
    let mut constants = Vec::with_capacity(plan.constants.len());
    for (&slot, value) in &plan.constants {
        constants.push((slot, constant_identity::exact_constant_digest(value)?));
    }
    constants.sort_unstable_by_key(|(slot, _)| *slot);
    Ok(constants)
}

#[path = "semantics/source_plan_identity.rs"]
mod source_plan_identity;
pub use source_plan_identity::exact_source_plan_identity;

#[path = "semantics/input_representation.rs"]
mod input_representation;

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
    let mut source = Vec::with_capacity(plan.operations.len());
    for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
        source.push(match operation {
            FusionOp::Primitive { op, inputs, output } => SourceOperationIdentity::Primitive {
                operation: *op,
                binding: snapshot.map(|binding| binding.declaration.binding),
                inputs: inputs.clone(),
                output: *output,
            },
            FusionOp::Builtin {
                origin,
                inputs,
                output,
                ..
            } => SourceOperationIdentity::Builtin {
                binding: snapshot
                    .ok_or_else(|| anyhow!("fusion builtin has no captured binding"))?
                    .declaration
                    .binding,
                origin: *origin,
                inputs: inputs.clone(),
                output: *output,
            },
        });
    }
    Ok(source)
}

#[derive(Debug)]
pub(crate) struct ResolvedProgram {
    source_operations: Vec<SourceOperationIdentity>,
    source_input_ids: Vec<ValueId>,
    source_input_set: Vec<ValueId>,
    source_constants: Vec<(ValueId, Digest)>,
    source_slot_constants: Vec<(usize, Digest)>,
    source_root: Option<ValueId>,
    facts: FactIndex,
    source_inputs: Vec<ValueFact>,
    precision_policy: PrecisionPolicy,
    provider_precision: ProviderPrecision,
    /// Selected bindings are part of request identity, not inferred from output
    /// class or a spelling-based expression dispatch table.
    binding_contracts: Vec<SemanticBinding>,
    declarations: SlotMap<SemanticBinding>,
    evidence: SlotMap<runmat_accelerate_api::NumericalDomainEvidence>,
    sign: SlotMap<runmat_runtime::builtins::common::spec::FusionSignEvidence>,
    arithmetic: SlotMap<FusionArithmeticEvidence>,
    exact_products: SlotMap<ExactProductReceipt>,
    admission_work: SlotMap<runmat_runtime::builtins::common::spec::FusionAdmissionWork>,
}

#[derive(Debug)]
pub(crate) enum FusionResolveFailure {
    Structural(anyhow::Error),
    /// A completed read-only real-domain proof declined selected arithmetic.
    DomainIneligible(BuiltinBindingIdentity),
    Admission(anyhow::Error),
    Runtime(Box<runmat_runtime::RuntimeError>),
    ObservedProbe(Box<FusionResolveFailure>),
}

impl std::fmt::Display for FusionResolveFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structural(error) | Self::Admission(error) => {
                std::fmt::Display::fmt(error, formatter)
            }
            Self::Runtime(error) => std::fmt::Display::fmt(error, formatter),
            Self::DomainIneligible(binding) => write!(
                formatter,
                "selected fusion real domain is ineligible for {binding:?}"
            ),
            Self::ObservedProbe(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for FusionResolveFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) | Self::Admission(error) => Some(error.as_ref()),
            Self::Runtime(error) => Some(error.as_ref()),
            Self::DomainIneligible(_) => None,
            Self::ObservedProbe(error) => Some(error.as_ref()),
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
    pub(crate) fn after_observed_probe(self) -> Self {
        match self {
            Self::DomainIneligible(_) => self,
            other => Self::ObservedProbe(Box::new(other)),
        }
    }

    pub(crate) fn into_execution_or_structural(
        self,
    ) -> std::result::Result<anyhow::Error, runmat_accelerate_api::ProviderExecutionError> {
        match self {
            Self::Structural(error) => Ok(error),
            Self::DomainIneligible(binding) => Ok(anyhow!(
                "selected fusion real domain is ineligible for {binding:?}"
            )),
            Self::Admission(error) => Err(error.into()),
            Self::Runtime(error) => Err(
                runmat_accelerate_api::ProviderExecutionError::from_source(*error),
            ),
            Self::ObservedProbe(error) => Err(
                runmat_accelerate_api::ProviderExecutionError::from_source(*error)
                    .with_unproven_effects(),
            ),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FactResolutionScope {
    Full,
    FiniteOnly,
}

/// One resolution's fact policy and observed provider-effect boundary. A
/// failed proof retains its observed status even before its per-operation
/// admission work can be published into the resolved program.
struct ResolutionObservation {
    facts: FactResolutionScope,
    provider_probe_observed: bool,
}

impl ResolvedProgram {
    pub(crate) async fn resolve_finite_image(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let mut borrowed = Vec::new();
        borrowed
            .try_reserve_exact(inputs.len())
            .map_err(|error| FusionResolveFailure::Admission(error.into()))?;
        borrowed.extend(inputs.iter());
        let bindings = semantic_bindings::capture(plan)?;
        Self::resolve_observed(
            plan,
            &borrowed,
            provider_precision,
            PrecisionPolicy::current(),
            correlation,
            &bindings,
            FactResolutionScope::FiniteOnly,
        )
        .await
    }

    pub async fn resolve(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let mut selected = Vec::with_capacity(inputs.len());
        selected.extend(inputs.iter());
        Self::resolve_borrowed(plan, &selected, provider_precision, correlation).await
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
            FactResolutionScope::Full,
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
            FactResolutionScope::Full,
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
        facts: FactResolutionScope,
    ) -> std::result::Result<Self, FusionResolveFailure> {
        let mut observation = ResolutionObservation {
            facts,
            provider_probe_observed: false,
        };
        let result = Self::resolve_observed_inner(
            plan,
            inputs,
            provider_precision,
            precision_policy,
            correlation,
            snapshots,
            &mut observation,
        )
        .await;
        result.map_err(|error| {
            if observation.provider_probe_observed {
                error.after_observed_probe()
            } else {
                error
            }
        })
    }

    async fn resolve_observed_inner(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
        correlation: Option<crate::placement::PlacementCorrelationId>,
        snapshots: &[Option<SemanticBinding>],
        observation: &mut ResolutionObservation,
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
        let mut computed_scalars = ComputedScalars::from_plan(plan);
        let mut scalar_observation = ScalarEvidenceObservation::new(correlation);
        for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
            let inferred = inferred_operation(
                &program,
                operation,
                *snapshot,
                &invocation_leaves,
                observation.facts == FactResolutionScope::FiniteOnly,
            )?;
            let values = selected_values(&invocation_leaves, &computed_scalars, inferred.input_ids);
            let computed = scalar_observation.evaluate(operation, *snapshot, &values, &inferred);
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
                    let mut real_domain_completed = false;
                    if let Some((predicate, internal_error)) = snapshot
                        .declaration
                        .semantics
                        .and_then(|semantics| semantics.required_real_unary)
                    {
                        if real_ranges.is_none() {
                            real_ranges = Some(super::real_domain::RealRangeResolver::new(
                                plan,
                                &invocation_leaves,
                            )?);
                        }
                        let resolver = real_ranges.as_mut().expect("just initialized");
                        if !real_range_index_reported {
                            resolver.report_index(
                                &mut observed.work,
                                snapshot.declaration.binding,
                                provider_precision,
                            )?;
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
                        operand_evidence[0] = super::real_domain::prove_observed(
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
                            &mut observation.provider_probe_observed,
                        )
                        .await?
                        .map(|mut evidence| {
                            evidence.sign = prior_sign;
                            evidence
                        });
                        real_domain_completed = true;
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
                    let result = admit.call(context).await;
                    observation.provider_probe_observed |= observed.work.provider_effect_observed();
                    let result = result.map_err(FusionResolveFailure::Admission)?;
                    let work = observed.completed();
                    if result.is_none()
                        && real_domain_completed
                        && observation.provider_probe_observed
                    {
                        return Err(FusionResolveFailure::DomainIneligible(
                            snapshot.declaration.binding,
                        ));
                    }
                    let evidence = result.ok_or_else(|| anyhow!(
                        "fusion operation requires unavailable domain evidence: binding={:?}, output={output:?}, inputs={:?}, represented_operands={:?}, operand_facts={:?}, output_fact={:?}",
                        snapshot.declaration.binding, inferred.input_ids,
                        values.iter().map(Option::is_some).collect::<Vec<_>>(),
                        inferred.arguments, inferred.fact
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
                scalar_observation.record(
                    &mut computed_scalars,
                    output,
                    computed,
                    program.facts.get(&output),
                );
            }
        }
        invocation_leaves.require_live_resident_inputs()?;
        scalar_observation.completed();
        Ok(program)
    }

    pub fn fact(&self, id: ValueId) -> Result<&ValueFact> {
        self.facts
            .get(&id)
            .ok_or_else(|| anyhow!("missing fusion value fact {id}"))
    }

    pub(crate) fn observed_provider_probe(&self) -> bool {
        self.admission_work
            .values()
            .any(FusionAdmissionWork::provider_effect_observed)
    }

    pub(crate) fn is_invocation_input(&self, id: ValueId) -> bool {
        self.source_input_set.binary_search(&id).is_ok()
    }

    /// Literal representation follows the same explicit source precision
    /// projection as assembly; an emitter cannot invent a narrowing cast.
    pub(crate) fn represented_constant_class(
        &self,
        id: ValueId,
        original: NumericClass,
    ) -> Result<NumericClass> {
        let effective = self.floating_class(id)?;
        if effective == original
            || (original == NumericClass::Double
                && effective == NumericClass::Single
                && self.precision_policy.permits_double_to_single()
                && self.provider_precision == ProviderPrecision::F32)
        {
            Ok(effective)
        } else {
            Err(anyhow!(
                "constant class differs from admitted source precision"
            ))
        }
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

    /// Controls are omitted only if every use is an authenticated exact scalar
    /// implementation (POWER exponent or binary32-equivalent product).
    /// They remain source inputs and original-bit semantic identity facts.
    #[cfg(test)]
    pub fn embedded_scalar_controls(
        &self,
        plan: &FusionGroupPlan,
        outputs: &[ValueId],
    ) -> HashSet<ValueId> {
        self.embedded_controls_sorted_for_precision(plan, outputs, ProviderPrecision::F32)
            .expect("fusion control geometry is representable")
            .into_iter()
            .collect()
    }

    /// Exact request-sized control IDs for finite elementwise preparation.
    /// The typed producer performs membership checks by binary search; tests
    /// may project this canonical inventory into a set for assertions.
    pub(crate) fn embedded_controls_sorted_for_precision(
        &self,
        plan: &FusionGroupPlan,
        outputs: &[ValueId],
        precision: ProviderPrecision,
    ) -> Result<Vec<ValueId>> {
        let mut controls = Vec::new();
        controls.try_reserve_exact(
            plan.operations
                .len()
                .checked_mul(2)
                .ok_or_else(|| anyhow!("fusion control count overflows"))?,
        )?;
        controls.extend(
            plan.operations
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
                .filter(|id| {
                    (plan.inputs.contains(id) || plan.const_values.contains_key(id))
                        && !outputs.contains(id)
                }),
        );
        if precision == ProviderPrecision::F32 {
            controls.extend(self.exact_products.values().filter_map(|receipt| {
                (!outputs.contains(&receipt.control)).then_some(receipt.control)
            }));
        }
        controls.sort_unstable();
        controls.dedup();
        let exclusion_capacity = plan
            .operations
            .iter()
            .try_fold(0usize, |count, operation| {
                let inputs = match operation {
                    FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
                };
                count.checked_add(inputs.len())
            })
            .ok_or_else(|| anyhow!("fusion control exclusion count overflows"))?;
        let mut exclusions = Vec::new();
        exclusions.try_reserve_exact(exclusion_capacity)?;
        for operation in &plan.operations {
            let (inputs, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let exact = output.and_then(|output| self.exact_single_base_power(output));
            for (position, input) in inputs.iter().enumerate() {
                let product = precision == ProviderPrecision::F32
                    && output
                        .and_then(|id| self.exact_products.get(&id))
                        .is_some_and(|receipt| receipt.control == *input);
                if (position != 1 || exact.is_none()) && !product {
                    exclusions.push(*input);
                }
            }
        }
        exclusions.sort_unstable();
        exclusions.dedup();
        controls.retain(|id| exclusions.binary_search(id).is_err());
        Ok(controls)
    }

    pub(crate) fn exact_product(&self, output: ValueId) -> Option<ExactProductReceipt> {
        self.exact_products.get(&output).copied()
    }

    /// Original words from operation-owned receipts, not a name, a narrowed
    /// physical buffer, or an unevaluated producer's ancestor. Callers first
    /// prove exclusive metadata use through embedded_controls_sorted_for_precision.
    pub(crate) fn embedded_scalar_control_bits(&self, control: ValueId) -> Option<u64> {
        self.exact_products
            .values()
            .find(|receipt| receipt.control == control)
            .map(|receipt| receipt.exact.original_double_bits())
            .or_else(|| {
                self.arithmetic.iter().find_map(|(&output, evidence)| {
                    self.exact_single_base_power(output)?;
                    let FusionArithmeticEvidence::RealPower(receipt) = evidence;
                    if receipt.exponent_id != control {
                        return None;
                    }
                    match receipt.exponent_scalar? {
                        runmat_accelerate_api::PowerScalarControl::DoubleBits(bits) => Some(bits),
                        _ => None,
                    }
                })
            })
    }

    pub(crate) fn require_exact_product_controls(
        &self,
        plan: &FusionGroupPlan,
        inputs: &[Value],
    ) -> Result<()> {
        if self.exact_products.is_empty() {
            return Ok(());
        }
        self.require_source_plan(plan)?;
        if inputs.len() != plan.inputs.len() {
            return Err(anyhow!("exact product input count changed"));
        }
        for receipt in self.exact_products.values() {
            let value = plan
                .inputs
                .iter()
                .position(|&id| id == receipt.control)
                .and_then(|index| inputs.get(index))
                .or_else(|| plan.const_values.get(&receipt.control));
            if exact_scalar_control(value)
                != Some(
                    runmat_runtime::builtins::common::spec::FusionScalarControl::DoubleBits(
                        receipt.exact.original_double_bits(),
                    ),
                )
            {
                return Err(anyhow!("exact product original scalar control changed"));
            }
            let fact = value
                .map(value_fact)
                .ok_or_else(|| anyhow!("exact product control disappeared"))?;
            if fact.shape.element_count() != Some(1) {
                return Err(anyhow!("exact product scalar shape changed"));
            }
        }
        Ok(())
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
        struct DeclarationContracts<'a>(&'a [SemanticBinding]);
        impl Serialize for DeclarationContracts<'_> {
            fn serialize<S: Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
                for snapshot in self.0 {
                    sequence.serialize_element(&(
                        snapshot.declaration.binding,
                        snapshot.entry.contract,
                        snapshot.entry.placement,
                    ))?;
                }
                sequence.end()
            }
        }
        struct Revisions<'a>(&'a SlotMap<SemanticBinding>);
        impl Serialize for Revisions<'_> {
            fn serialize<S: Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(None)?;
                for (id, declaration) in self.0.iter() {
                    map.serialize_entry(
                        id,
                        &(
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
                    )?;
                }
                map.end()
            }
        }
        #[derive(serde::Serialize)]
        struct AdmissionIdentity<'a> {
            domains: &'a SlotMap<runmat_accelerate_api::NumericalDomainEvidence>,
            signs: &'a SlotMap<runmat_runtime::builtins::common::spec::FusionSignEvidence>,
            arithmetic: &'a SlotMap<FusionArithmeticEvidence>,
            exact_products: &'a SlotMap<ExactProductReceipt>,
            work: &'a SlotMap<runmat_runtime::builtins::common::spec::FusionAdmissionWork>,
        }
        Ok(Digest::sha256_serialized(&(
            "typed-fusion-v7",
            runmat_builtins::BUILTIN_CATALOG_SCHEMA_VERSION,
            &self.source_operations,
            &self.source_input_ids,
            &self.source_constants,
            &self.source_slot_constants,
            self.source_root,
            DeclarationContracts(&self.binding_contracts),
            &self.facts,
            &self.source_inputs,
            self.precision_policy,
            self.provider_precision,
            Revisions(&self.declarations),
            AdmissionIdentity {
                domains: &self.evidence,
                signs: &self.sign,
                arithmetic: &self.arithmetic,
                exact_products: &self.exact_products,
                work: &self.admission_work,
            },
        ))?)
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
