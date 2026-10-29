//! Original-Runtime first-party control phases for one typed reduction.
use super::*;
use crate::wgsl::host_buffers::{claim_bytes, Claim};
use runmat_execution::resource::ProviderHostMetadataAdmission;

pub(super) fn claim(
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    bytes: u64,
) -> ProviderExecutionResult<Claim> {
    let bytes = usize::try_from(bytes).map_err(ProviderExecutionError::from_source)?;
    claim_bytes(metadata, bytes).map_err(ProviderExecutionError::from_source)
}

pub(super) fn input_rank_upper(mapped: &inputs::Inputs<'_>) -> usize {
    mapped.data.iter().fold(2, |rank, input| {
        rank.max(match input.value {
            Value::Tensor(tensor) => tensor.shape().len(),
            Value::GpuTensor(handle) => handle.shape().len(),
            _ => 2,
        })
    })
}

/// Serialize the exact prior mapping sequence without owning a second Vec.
pub(super) struct DataMapping<'view, 'input> {
    pub data: &'view [ReductionDataInput<'input>],
    pub controls: &'view [ValueId],
}
impl serde::Serialize for DataMapping<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let selected = || {
            self.data
                .iter()
                .filter(|input| !self.controls.contains(&input.value_id))
        };
        let mut sequence = serializer.serialize_seq(Some(selected().count()))?;
        for input in selected() {
            sequence.serialize_element(&(input.value_id, input.request_slot))?;
        }
        sequence.end()
    }
}

fn known_shape(shape: &runmat_types::ShapeFact) -> anyhow::Result<Option<Vec<usize>>> {
    use runmat_types::{DimensionFact, ShapeFact};
    match shape {
        ShapeFact::Scalar => Ok(Some(runmat_value::exact_capacity_vec_from_slice(&[
            1usize, 1,
        ])?)),
        ShapeFact::Shaped { dims } => {
            if dims
                .iter()
                .any(|dimension| !matches!(dimension, DimensionFact::Known(_)))
            {
                return Ok(None);
            }
            let mut result = runmat_value::exact_capacity_vec(dims.len())?;
            for dimension in dims {
                if let DimensionFact::Known(extent) = dimension {
                    result.push(*extent);
                }
            }
            Ok(Some(result))
        }
        // known_dims on a ranked zero-dimensional shape yields an empty,
        // fully known sequence; preserve that distinct existing case.
        ShapeFact::Ranked { rank: 0 } => Ok(Some(Vec::new())),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => Ok(None),
    }
}

pub(super) fn producer_shapes(
    plan: &crate::fusion::FusionGroupPlan,
    program: &ResolvedProgram,
) -> anyhow::Result<Option<Vec<Vec<usize>>>> {
    let mut result = runmat_value::exact_capacity_vec(plan.operations.len())?;
    for operation in &plan.operations {
        let output = match operation {
            crate::fusion::FusionOp::Primitive { output, .. }
            | crate::fusion::FusionOp::Builtin { output, .. } => *output,
        };
        let Some(output) = output else {
            return Ok(None);
        };
        let Ok(fact) = program.fact(output) else {
            return Ok(None);
        };
        let Some(shape) = known_shape(&fact.shape)? else {
            return Ok(None);
        };
        result.push(shape);
    }
    Ok(Some(result))
}

pub(super) fn producer_operand_shapes(
    plan: &crate::fusion::FusionGroupPlan,
    program: &ResolvedProgram,
) -> anyhow::Result<Option<Vec<[Vec<usize>; 2]>>> {
    let mut result = runmat_value::exact_capacity_vec(plan.operations.len())?;
    for operation in &plan.operations {
        let inputs = match operation {
            crate::fusion::FusionOp::Primitive { inputs, .. }
            | crate::fusion::FusionOp::Builtin { inputs, .. } => inputs,
        };
        let [left, right] = inputs.as_slice() else {
            return Ok(None);
        };
        let Ok(left) = program.fact(*left) else {
            return Ok(None);
        };
        let Some(left) = known_shape(&left.shape)? else {
            return Ok(None);
        };
        let Ok(right) = program.fact(*right) else {
            return Ok(None);
        };
        let Some(right) = known_shape(&right.shape)? else {
            return Ok(None);
        };
        result.push([left, right]);
    }
    Ok(Some(result))
}

/// Box destruction precedes refund; a claim hidden inside the erased Prepared
/// object alone could refund before its Box backing is deallocated.
pub(super) struct PreparedControl {
    value: Box<dyn runmat_runtime::builtins::common::spec::PreparedFusionReduction>,
    _claim: Claim,
}
impl PreparedControl {
    pub(super) fn new(
        value: Box<dyn runmat_runtime::builtins::common::spec::PreparedFusionReduction>,
        claim: Claim,
    ) -> Self {
        Self {
            value,
            _claim: claim,
        }
    }
    pub(super) fn as_ref(
        &self,
    ) -> &dyn runmat_runtime::builtins::common::spec::PreparedFusionReduction {
        self.value.as_ref()
    }
}
impl std::ops::Deref for PreparedControl {
    type Target = dyn runmat_runtime::builtins::common::spec::PreparedFusionReduction;
    fn deref(&self) -> &Self::Target {
        self.value.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DimensionFact, ShapeFact};

    #[test]
    fn returned_shape_capacity_and_borrowed_mapping_preserve_existing_projection() {
        let scalar = known_shape(&ShapeFact::Scalar).unwrap().unwrap();
        assert_eq!(scalar, [1, 1]);
        assert_eq!(scalar.capacity(), 2);
        let shaped = known_shape(&ShapeFact::Shaped {
            dims: vec![
                DimensionFact::Known(2),
                DimensionFact::Known(0),
                DimensionFact::Known(7),
            ],
        })
        .unwrap()
        .unwrap();
        assert_eq!(shaped, [2, 0, 7]);
        assert_eq!(shaped.capacity(), 3);
        assert_eq!(
            known_shape(&ShapeFact::Ranked { rank: 0 }).unwrap(),
            Some(Vec::new())
        );
        assert!(known_shape(&ShapeFact::Ranked { rank: 2 })
            .unwrap()
            .is_none());
        assert!(known_shape(&ShapeFact::Unknown).unwrap().is_none());
        assert!(known_shape(&ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown]
        })
        .unwrap()
        .is_none());

        let value = Value::Num(1.0);
        let data = [
            ReductionDataInput {
                value_id: 9,
                request_slot: Some(2),
                value: &value,
            },
            ReductionDataInput {
                value_id: 3,
                request_slot: None,
                value: &value,
            },
            ReductionDataInput {
                value_id: 7,
                request_slot: Some(0),
                value: &value,
            },
        ];
        let view = DataMapping {
            data: &data,
            controls: &[3],
        };
        let original = vec![(9u32, Some(2usize)), (7, Some(0))];
        assert_eq!(
            serde_json::to_vec(&view).unwrap(),
            serde_json::to_vec(&original).unwrap()
        );
        assert_eq!(
            Digest::sha256_serialized(&view).unwrap(),
            Digest::sha256(serde_json::to_vec(&original).unwrap())
        );
    }
}
