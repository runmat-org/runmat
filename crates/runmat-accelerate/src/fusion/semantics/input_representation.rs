//! Physical representation at an already selected, explicit conversion boundary.
use super::*;
use runmat_types::{ResidencyFact, ScalarOperation, StorageFact};

/// No source fact changes: this proof only authorizes the existing host upload
/// preparation to materialize this exact leaf as Single before its first use.
pub(crate) struct ExplicitHostSingleInput<'a> {
    owner: &'a ResolvedProgram,
    source: ValueId,
}

impl ExplicitHostSingleInput<'_> {
    pub(crate) fn represented_class_for(
        self,
        owner: &ResolvedProgram,
        source: ValueId,
    ) -> Result<NumericClass> {
        if !std::ptr::eq(self.owner, owner) || source != self.source {
            return Err(anyhow!(
                "explicit input representation belongs to another program or leaf"
            ));
        }
        Ok(NumericClass::Single)
    }
}

impl ResolvedProgram {
    /// Every source use must be a captured canonical unary Cast(Single).
    /// Double arithmetic, publication, scalar controls and resident ownership
    /// are never erased by this physical input representation.
    pub(crate) fn explicit_host_single_input(
        &self,
        plan: &FusionGroupPlan,
        source: ValueId,
    ) -> Option<ExplicitHostSingleInput<'_>> {
        let index = self.source_input_ids.iter().position(|id| *id == source)?;
        let original = self.source_inputs.get(index)?;
        if self.source_input_ids != plan.inputs
            || self.source_root != plan.output
            || self.source_operations.len() != plan.operations.len()
            || self
                .source_input_ids
                .iter()
                .filter(|id| **id == source)
                .count()
                != 1
            || original.residency != ResidencyFact::Host
            || !matches!(original.storage, StorageFact::Dense | StorageFact::Scalar)
            || !original.numeric().is_some_and(|numeric| {
                numeric.class == NumericClass::Double && numeric.domain == NumericDomain::Real
            })
            || plan.output == Some(source)
            || plan
                .materialized_stores
                .iter()
                .any(|store| store.value_id == source)
            || plan.reduction_data == Some(source)
            || plan.reduction_dim == Some(source)
            || plan.const_values.contains_key(&source)
        {
            return None;
        }
        let mut consumed = false;
        for (admitted, operation) in self.source_operations.iter().zip(&plan.operations) {
            // Allocation-free topology check. The enclosing emitter still owns
            // its complete canonical declaration/source-identity validation.
            let (inputs, output) = match (admitted, operation) {
                (
                    SourceOperationIdentity::Primitive {
                        operation: expected,
                        inputs,
                        output,
                        ..
                    },
                    FusionOp::Primitive {
                        op,
                        inputs: actual,
                        output: result,
                    },
                ) if expected == op && inputs == actual && output == result => (inputs, output),
                (
                    SourceOperationIdentity::Builtin {
                        binding,
                        origin,
                        inputs,
                        output,
                    },
                    FusionOp::Builtin {
                        name,
                        origin: actual_origin,
                        inputs: actual,
                        output: result,
                    },
                ) if name == binding.builtin.name
                    && origin == actual_origin
                    && inputs == actual
                    && output == result =>
                {
                    (inputs, output)
                }
                _ => return None,
            };
            if !inputs.contains(&source) {
                continue;
            }
            let output = (*output)?;
            let semantics = self.declarations.get(&output)?.declaration.semantics?;
            if inputs.as_slice() != [source]
                || semantics.scalar_operation != Some(ScalarOperation::Cast(NumericClass::Single))
                || self.floating_class(output).ok()? != NumericClass::Single
            {
                return None;
            }
            consumed = true;
        }
        consumed.then_some(ExplicitHostSingleInput {
            owner: self,
            source,
        })
    }
}
