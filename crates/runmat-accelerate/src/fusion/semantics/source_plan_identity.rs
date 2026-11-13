//! Exact immutable identity of one reconciled source fusion plan.
//!
//! This projection deliberately names every structural field and bit-encodes
//! floating controls. It is learning identity, never an execution permit.

use anyhow::Result;
use runmat_execution::Digest;
use serde::{ser::SerializeSeq, Serialize, Serializer};

use super::{semantic_bindings, source_constants, source_operations, source_slot_constants};
use crate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionPattern, ImageNormalizePattern,
    ImageScalar,
};
use crate::graph::ValueId;

#[derive(Serialize)]
enum ImageScalarIdentity {
    Constant(u64),
    Value(ValueId),
}

impl From<&ImageScalar> for ImageScalarIdentity {
    fn from(value: &ImageScalar) -> Self {
        match value {
            ImageScalar::Constant(value) => Self::Constant(value.to_bits()),
            ImageScalar::Value(value) => Self::Value(*value),
        }
    }
}

#[derive(Serialize)]
enum PatternIdentity<'a> {
    CenteredGram {
        matrix: ValueId,
        normalization: &'a runmat_accelerate_api::CovNormalization,
    },
    ImageNormalize {
        input: ValueId,
        epsilon: ImageScalarIdentity,
        gain: Option<ImageScalarIdentity>,
        bias: Option<ImageScalarIdentity>,
        gamma: Option<ImageScalarIdentity>,
        clamp_zero: bool,
        clamp_zero_value: Option<ValueId>,
        bound_mean_axes: &'a [(ValueId, runmat_execution::Digest)],
        requires_single_input: bool,
        reduction_routes: &'a Option<Box<crate::fusion::image_reduction::ImageReductionRoutes>>,
    },
    PowerStepNormalize {
        lhs: ValueId,
        rhs: ValueId,
        epsilon: u64,
    },
    ExplainedVariance {
        q: ValueId,
        g: ValueId,
    },
}

fn pattern_identity(pattern: Option<&FusionPattern>) -> Option<PatternIdentity<'_>> {
    pattern.map(|pattern| match pattern {
        FusionPattern::CenteredGram {
            matrix,
            normalization,
        } => PatternIdentity::CenteredGram {
            matrix: *matrix,
            normalization,
        },
        FusionPattern::ImageNormalize(pattern) => {
            let ImageNormalizePattern {
                input,
                epsilon,
                gain,
                bias,
                gamma,
                clamp_zero,
                clamp_zero_value,
                bound_mean_axes,
                requires_single_input,
                reduction_routes,
            } = pattern;
            PatternIdentity::ImageNormalize {
                input: *input,
                epsilon: epsilon.into(),
                gain: gain.as_ref().map(Into::into),
                bias: bias.as_ref().map(Into::into),
                gamma: gamma.as_ref().map(Into::into),
                clamp_zero: *clamp_zero,
                clamp_zero_value: *clamp_zero_value,
                bound_mean_axes: bound_mean_axes.as_slice(),
                requires_single_input: *requires_single_input,
                reduction_routes,
            }
        }
        FusionPattern::PowerStepNormalize { lhs, rhs, epsilon } => {
            PatternIdentity::PowerStepNormalize {
                lhs: *lhs,
                rhs: *rhs,
                epsilon: epsilon.to_bits(),
            }
        }
        FusionPattern::ExplainedVariance { q, g } => {
            PatternIdentity::ExplainedVariance { q: *q, g: *g }
        }
    })
}

#[derive(Serialize)]
enum FlavorIdentity {
    Sum,
    Mean,
    CustomScale(u64),
}

#[derive(Serialize)]
enum AxesIdentity<'a> {
    Unspecified,
    All,
    Explicit(&'a [usize]),
}

struct StoreIdentities<'a>(&'a [crate::fusion::FusionStoreMaterialization]);

impl Serialize for StoreIdentities<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for store in self.0 {
            sequence.serialize_element(&(store.value_id, &store.binding))?;
        }
        sequence.end()
    }
}

/// Exact source-plan identity used only after the VM has reconciled its
/// executable group. Unsupported constants make the region ineligible rather
/// than falling back to a display or pointer hash.
pub fn exact_source_plan_identity(plan: &FusionGroupPlan) -> Result<Digest> {
    let FusionGroupPlan {
        index: _,
        group,
        operations: _,
        inputs,
        stack_pattern,
        constants: _,
        const_values: _,
        materialized_stores,
        output,
        kernel,
        reduction_data,
        reduction_dim,
        reduction_flavor,
        reduction_axes,
        pattern,
    } = plan;
    let FusionGroup {
        id,
        kind,
        nodes,
        shape,
        span,
        pattern: group_pattern,
        stack_layout,
    } = group;
    let FusionKernelSpec {
        kind: kernel_kind,
        supported: kernel_supported,
    } = kernel;

    let snapshots = semantic_bindings::capture(plan)?;
    let operations = source_operations(plan, &snapshots)?;
    let constants = source_constants(plan)?;
    let slot_constants = source_slot_constants(plan)?;
    let flavor = reduction_flavor.map(|flavor| match flavor {
        runmat_accelerate_api::ReductionFlavor::Sum => FlavorIdentity::Sum,
        runmat_accelerate_api::ReductionFlavor::Mean => FlavorIdentity::Mean,
        runmat_accelerate_api::ReductionFlavor::CustomScale(value) => {
            FlavorIdentity::CustomScale(value.to_bits())
        }
    });
    let axes = reduction_axes.as_ref().map(|axes| match axes {
        crate::reduction_meta::ReductionAxes::Unspecified => AxesIdentity::Unspecified,
        crate::reduction_meta::ReductionAxes::All => AxesIdentity::All,
        crate::reduction_meta::ReductionAxes::Explicit(values) => AxesIdentity::Explicit(values),
    });
    Ok(Digest::sha256_serialized(&(
        "runmat.vm-fusion-source-plan.v1",
        (
            id,
            kind,
            nodes,
            shape,
            span,
            pattern_identity(group_pattern.as_ref()),
            stack_layout,
        ),
        operations,
        inputs,
        stack_pattern,
        constants,
        slot_constants,
        StoreIdentities(materialized_stores),
        output,
        (kernel_kind, kernel_supported),
        reduction_data,
        reduction_dim,
        flavor,
        axes,
        pattern_identity(pattern.as_ref()),
    ))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{VarBinding, VarKind};

    #[test]
    fn streamed_store_projection_preserves_legacy_json_bytes() {
        let stores = vec![crate::fusion::FusionStoreMaterialization {
            value_id: 12,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 3,
            },
        }];
        let legacy = stores
            .iter()
            .map(|store| (store.value_id, &store.binding))
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_vec(&StoreIdentities(&stores)).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
    }
}
