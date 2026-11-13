//! Sealed ordinary producer source for legacy reduction geometry. Numerical
//! expressions and helpers come only from the checked typed fusion owner.

use super::{
    semantics::ResolvedProgram,
    typed_producer::{producer_indices, TypedProducer},
    FusionGroupPlan,
};
use crate::graph::ValueId;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{GpuTensorLayout, ProviderPrecision};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::ScalarType;
use runmat_runtime::value_fact::value_fact;
use runmat_value::Value;
use serde::Serialize;
use std::collections::HashMap;

/// Planner-only source geometry/dependency eligibility. It cannot choose
/// literal words, source precision, numerical domain or selected bindings.
pub(crate) fn structural_support(source: &FusionGroupPlan) -> bool {
    let Some(root) = source.reduction_data else {
        return false;
    };
    if source.inputs.is_empty() || source.legacy_reduction_axis().is_none() {
        return false;
    }
    let mut leaves = source
        .inputs
        .iter()
        .chain(source.const_values.keys())
        .copied()
        .collect::<Vec<_>>();
    leaves.sort_unstable();
    leaves.dedup();
    let Ok(selected) = producer_indices(source, &leaves, &[root]) else {
        return false;
    };
    for index in selected.iter() {
        if !super::expression::structural_operation_support(source, &source.operations[index]) {
            return false;
        }
    }
    true
}

pub(crate) struct ReductionProducerEmission {
    root: ValueId,
    precision: ProviderPrecision,
    identity: Digest,
    body: String,
    expression: String,
    preamble: String,
    root_shape: [u32; 2],
    indices: Vec<ReductionInputIndex>,
}

#[derive(Serialize, PartialEq, Eq)]
struct ReductionInputIndex {
    id: ValueId,
    shape: Vec<usize>,
    layout: GpuTensorLayout,
    strides: [u32; 2],
}

fn input_indices(
    source: &FusionGroupPlan,
    inputs: &[Value],
    program: &ResolvedProgram,
    root: ValueId,
) -> Result<([u32; 2], Vec<ReductionInputIndex>)> {
    let dims = program
        .fact(root)?
        .shape
        .known_dims()
        .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
        .ok_or_else(|| anyhow!("selected reduction root has unknown shape"))?;
    // Match the canonical operand-layout projection's rank-2 normalization:
    // a vector is a row, and a scalar occupies one matrix element.
    let (rows, cols) = match dims.as_slice() {
        [] => (1usize, 1usize),
        [length] => (1usize, *length),
        [rows, cols] => (*rows, *cols),
        _ => return Err(anyhow!("legacy reduction geometry has unsupported rank")),
    };
    let root_shape = [u32::try_from(rows)?, u32::try_from(cols)?];
    if inputs.len() != source.inputs.len() {
        return Err(anyhow!("selected reduction input count changed"));
    }
    let indices = source
        .inputs
        .iter()
        .zip(inputs)
        .map(|(&id, value)| {
            let shape = value_fact(value)
                .shape
                .known_dims()
                .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
                .ok_or_else(|| anyhow!("selected reduction input has unknown shape"))?;
            let layout = match value {
                Value::GpuTensor(handle) => handle.view().layout(),
                _ => GpuTensorLayout::Contiguous,
            };
            let transpose = match layout {
                GpuTensorLayout::Transpose(info) => Some(info),
                GpuTensorLayout::Contiguous => None,
            };
            let (_, strides) = crate::backend::fusion_operand_layout::resolve(
                &shape,
                transpose,
                &[rows, cols],
                2,
            )?;
            let [first, second] = strides.as_slice() else {
                return Err(anyhow!("selected reduction indexing rank changed"));
            };
            Ok(ReductionInputIndex {
                id,
                shape,
                layout,
                strides: [*first, *second],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((root_shape, indices))
}

impl ReductionProducerEmission {
    pub(crate) fn new(
        source: &FusionGroupPlan,
        inputs: &[Value],
        program: &ResolvedProgram,
        precision: ProviderPrecision,
    ) -> Result<Self> {
        program.require_source_plan(source)?;
        let root = source
            .output
            .ok_or_else(|| anyhow!("selected reduction producer has no root"))?;
        let (root_shape, indices) = input_indices(source, inputs, program, root)?;
        let scalar = match precision {
            ProviderPrecision::F32 => ScalarType::F32,
            ProviderPrecision::F64 => ScalarType::F64,
        };
        let expressions: HashMap<ValueId, String> = source
            .inputs
            .iter()
            .enumerate()
            .map(|(slot, &id)| {
                (
                    id,
                    if slot == 0 {
                        "v".to_owned()
                    } else {
                        format!("v{slot}")
                    },
                )
            })
            .collect();
        let producer = TypedProducer::emit(source, program, scalar, expressions, Some(&[root]))?;
        let expression = producer.expression(root)?.to_owned();
        let mut preamble = producer.preamble(program)?;
        let index_identity = Digest::sha256(serde_json::to_vec(&(root, root_shape, &indices))?);
        preamble.push_str(&format!("// selected-reduction-indices {index_identity}\n"));
        let identity = program.cache_identity()?;
        Ok(Self {
            root,
            precision,
            identity,
            body: producer.body,
            expression,
            preamble,
            root_shape,
            indices,
        })
    }

    pub(crate) fn require(
        &self,
        root: ValueId,
        precision: ProviderPrecision,
        program: &ResolvedProgram,
        source: &FusionGroupPlan,
        inputs: &[Value],
    ) -> Result<()> {
        let (root_shape, indices) = input_indices(source, inputs, program, root)?;
        if self.root != root
            || self.precision != precision
            || self.identity != program.cache_identity()?
            || self.root_shape != root_shape
            || self.indices != indices
        {
            return Err(anyhow!("selected reduction producer source changed"));
        }
        Ok(())
    }
    pub(crate) fn root(&self) -> ValueId {
        self.root
    }
    pub(crate) fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    pub(crate) fn body(&self) -> &str {
        &self.body
    }
    pub(crate) fn expression(&self) -> &str {
        &self.expression
    }
    pub(crate) fn preamble(&self) -> &str {
        &self.preamble
    }
    pub(crate) fn input_count(&self) -> usize {
        self.indices.len()
    }
    pub(crate) fn index_expression(&self, slot: usize, axis: usize) -> Option<String> {
        let [row_stride, col_stride] = self.indices.get(slot)?.strides;
        let (row, col) = match axis {
            0 => ("r", "col"),
            1 => ("row", "c"),
            _ => return None,
        };
        Some(format!("({row} * {row_stride}u + {col} * {col_stride}u)"))
    }
    pub(crate) fn require_geometry(
        &self,
        axis: usize,
        reduce_len: usize,
        slices: usize,
    ) -> Result<()> {
        let [rows, cols] = self.root_shape;
        let expected = match axis {
            0 => (usize::try_from(rows)?, usize::try_from(cols)?),
            1 => (usize::try_from(cols)?, usize::try_from(rows)?),
            _ => return Err(anyhow!("legacy reduction has no selected axis")),
        };
        if expected != (reduce_len, slices) {
            return Err(anyhow!(
                "selected reduction geometry differs from root shape"
            ));
        }
        Ok(())
    }
}
