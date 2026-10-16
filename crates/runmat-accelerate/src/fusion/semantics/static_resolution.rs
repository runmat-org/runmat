//! Non-suspending selected-source resolution. The operation owner supplies
//! only genuinely static admission; unresolved probes remain on the async path.

use super::*;

impl ResolvedProgram {
    pub(crate) fn resolve_static_captured(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        snapshots: &[Option<SemanticBinding>],
    ) -> std::result::Result<Option<Self>, FusionResolveFailure> {
        Self::resolve_static_source(plan, inputs, provider_precision, snapshots, false)
    }

    /// Ordinary CPU execution consumes original source classes, not the
    /// provider-effective precision projection. Reuse canonical inference and
    /// static admission without acquiring data-dependent sign/domain proofs.
    pub(crate) fn resolve_host_static_captured(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        snapshots: &[Option<SemanticBinding>],
    ) -> std::result::Result<Option<Self>, FusionResolveFailure> {
        Self::resolve_static_source(plan, inputs, provider_precision, snapshots, true)
    }

    fn resolve_static_source(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        snapshots: &[Option<SemanticBinding>],
        host: bool,
    ) -> std::result::Result<Option<Self>, FusionResolveFailure> {
        let (mut program, leaves) = source_assembly(
            plan,
            inputs,
            provider_precision,
            if host {
                PrecisionPolicy::Exact
            } else {
                PrecisionPolicy::current()
            },
            None,
            snapshots,
        )?;
        let mut computed_scalars = ComputedScalars::from_plan(plan);
        for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
            let source_inputs = match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            if snapshot.is_some_and(|binding| binding.declaration.reduction.is_some())
                && source_inputs
                    .first()
                    .is_some_and(|id| program.sign.contains_key(id))
            {
                // The optional owner projection needs asynchronous control
                // preparation. Do not parse MEAN controls on this static pass.
                return Ok(None);
            }
            let inferred = inferred_operation(&program, operation, *snapshot, &leaves, false)?;
            let mut values = selected_values(&leaves, &computed_scalars, inferred.input_ids);
            if host {
                // Static owners may inspect literal scalar controls (including
                // Single scalars), but must not rescan host arrays or read a
                // resident provider merely to price a CPU candidate.
                for value in &mut values {
                    if !matches!(
                        value,
                        Some(Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_))
                    ) && !matches!(value, Some(Value::Tensor(tensor)) if tensor.len() == 1)
                    {
                        *value = None;
                    }
                }
            }
            let computed = ComputedScalars::evaluate(operation, *snapshot, &values, &inferred);
            let admitted = if let (Some(_output), Some(snapshot)) = (inferred.output, snapshot) {
                if let Some(semantics) = snapshot.declaration.semantics {
                    if semantics.admit.is_some() {
                        let Some(admit_static) = semantics.admit_static else {
                            return Ok(None);
                        };
                        let operand_evidence = inferred.input_ids.iter().map(|id| {
                            let numerical_domain = program.evidence.get(id).copied();
                            let sign = program.sign.get(id).copied();
                            if numerical_domain.is_some() || sign.is_some() {
                                Some(runmat_runtime::builtins::common::spec::FusionOperandDomainEvidence {
                                    value_id: *id,
                                    numerical_domain: numerical_domain.unwrap_or(runmat_accelerate_api::NumericalDomainEvidence::Unknown),
                                    sign,
                                })
                            } else { None }
                        }).collect::<Vec<_>>();
                        let mut work =
                            runmat_runtime::builtins::common::spec::FusionAdmissionWork::None;
                        let context = FusionAdmissionContext::new(
                            inferred.input_ids,
                            &values,
                            &inferred.arguments,
                            &operand_evidence,
                            &mut work,
                        )
                        .ok_or_else(|| anyhow!("static fusion operand identity mismatch"))?
                        .with_output_fact(&inferred.fact);
                        admit_static(&context).map_err(FusionResolveFailure::Admission)?
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            if inferred.output.is_some()
                && snapshot.is_some_and(|binding| {
                    binding
                        .declaration
                        .semantics
                        .is_some_and(|semantics| semantics.admit.is_some())
                })
                && admitted.is_none()
            {
                return Ok(None);
            }
            let output = inferred.output;
            let add_refinement = if !host && output.is_some() && sign::applies(operation, *snapshot)
            {
                let mut observed = super::super::admission::AdmissionObservation::new(None);
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
            if let (Some(output), Some(((sign, kind), work))) = (output, add_refinement) {
                assembly::record_sign_refinement(&mut program, output, sign, kind, None, work)?;
            }
            if let Some(output) = output {
                computed_scalars.record(output, computed, program.facts.get(&output));
            }
        }
        leaves.require_live_resident_inputs()?;
        Ok(Some(program))
    }
}
