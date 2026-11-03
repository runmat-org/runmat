//! Operation-owned host semantics for a prospective fused reduction.
//!
//! This receipt is not provider capability or selected-binding evidence. It
//! deliberately does not bless current resident-path class inconsistencies.
pub use runmat_types::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
    FusionReductionTies,
};
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};
use runmat_value::Value;
use std::{future::Future, pin::Pin};

pub struct FusionReductionContext<'a> {
    pub input: &'a ValueFact,
    pub controls: &'a [Value],
    pub requested_outputs: usize,
}
impl FusionReductionContext<'_> {
    /// This first contract covers real floating data and host-only controls.
    /// Device controls must retain ordinary dispatch, not perform hidden probes
    /// while constructing a supposedly payload-free semantic receipt.
    pub fn real_floating_input(&self) -> crate::BuiltinResult<Option<(NumericClass, Vec<usize>)>> {
        if self.requested_outputs != 1
            || !self.controls.iter().all(|value| {
                matches!(
                    value,
                    Value::Num(_)
                        | Value::Int(_)
                        | Value::String(_)
                        | Value::CharArray(_)
                        | Value::StringArray(_)
                        | Value::Tensor(_)
                )
            })
        {
            return Ok(None);
        }
        let Some(numeric) = self.input.numeric() else {
            return Ok(None);
        };
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return Ok(None);
        }
        // Materialize exactly one final shape, not known_dims' temporary
        // Option vector followed by a second geometrically growing collect.
        let shape = match &self.input.shape {
            runmat_types::ShapeFact::Scalar => {
                let mut shape = reduction_vec(2)?;
                shape.extend([1, 1]);
                shape
            }
            runmat_types::ShapeFact::Shaped { dims } => {
                if !dims
                    .iter()
                    .all(|dim| matches!(dim, runmat_types::DimensionFact::Known(_)))
                {
                    return Ok(None);
                }
                let mut shape = reduction_vec(dims.len())?;
                for dim in dims {
                    if let runmat_types::DimensionFact::Known(value) = dim {
                        shape.push(*value);
                    }
                }
                shape
            }
            _ => return Ok(None),
        };
        if checked_reduction_elements(&shape).is_none() {
            return Ok(None);
        }
        Ok(Some((numeric.class, shape)))
    }
}

pub type FusionReductionFuture<'a> =
    Pin<Box<dyn Future<Output = crate::BuiltinResult<Option<FusionReductionContract>>> + 'a>>;
pub type FusionReductionStagesFuture<'a> =
    Pin<Box<dyn Future<Output = crate::BuiltinResult<Option<Vec<FusionReductionContract>>>> + 'a>>;
pub struct FusionReductionDeclaration {
    pub revision: u32,
    /// Operation fixed by this selected builtin. The resolved contract may
    /// refine axes/classes, but a legacy kernel may never guess its operation
    /// from a name or treat an unsupported extrema as Sum.
    pub operation: FusionReductionOperation,
    pub prepare: FusionReductionPreparer,
    /// Allocation-free bound from the owner's lexical plan or a bounded
    /// rejected scalar-axis parser path. Funding a parser is not mathematical
    /// eligibility. None retains ordinary dispatch and its diagnostics.
    pub resources:
        for<'a> fn(FusionReductionPreparationQuery<'a>) -> Option<FusionReductionPreparationQuote>,
}

pub struct FusionReductionPreparationQuery<'a> {
    pub controls: FusionReductionControls<'a>,
    pub input_rank_upper: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct FusionReductionPreparationQuote {
    pub prepare_scratch_bytes: u64,
    pub prepared_retained_bytes: u64,
    pub resolve_scratch_bytes: u64,
    pub contract_retained_bytes: u64,
}

/// One checked allocation layout, not an allocator-private size estimate.
pub(crate) fn reduction_layout<T>(count: usize) -> Option<u64> {
    u64::try_from(std::alloc::Layout::array::<T>(count).ok()?.size()).ok()
}

/// The existing non-mean public tokens are ASCII and at most sixteen bytes
/// (`ComparisonMethod`). Decode a character row on the stack, then invoke the
/// canonical typed decoder. Unknown/non-ASCII text retains ordinary parsing;
/// no independent spelling table or new string dispatch is introduced here.
pub(crate) fn reduction_control<T>(value: &Value, decode: fn(&str) -> Option<T>) -> Option<T> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => decode(text),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            decode(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
        Value::CharArray(array) if array.rows == 1 => {
            let mut bytes = [0u8; 16];
            let units = array.data();
            let output = bytes.get_mut(..units.len())?;
            for (out, unit) in output.iter_mut().zip(units) {
                *out = u8::try_from(unit.raw()).ok().filter(u8::is_ascii)?;
            }
            decode(std::str::from_utf8(output).ok()?)
        }
        _ => None,
    }
}

/// Borrowed dimension controls are validated without a decoded payload copy.
/// Only the final retained axis vector is materialized, at its exact extent.
#[derive(Clone, Copy)]
pub(crate) struct ReductionDimensions<'a> {
    value: &'a Value,
    pub len: usize,
    zero_is_one: bool,
    scalar_dimension: Option<usize>,
}
impl<'a> ReductionDimensions<'a> {
    pub fn axis_plan(self) -> ReductionAxisPlan<'a> {
        match self.scalar_dimension {
            Some(dimension) => ReductionAxisPlan::Dim(dimension),
            None => ReductionAxisPlan::Dimensions(self),
        }
    }
    pub fn parse(value: &'a Value, mean: bool) -> Option<Self> {
        if !mean && matches!(value, Value::Num(_) | Value::Int(_)) {
            let dimension =
                crate::builtins::common::tensor::scalar_dimension_from_value_typed(value, false)
                    .ok()??;
            return Some(Self {
                value,
                len: 1,
                zero_is_one: false,
                scalar_dimension: Some(dimension),
            });
        }
        let dimensions = crate::builtins::common::tensor::borrowed_host_shape_dimensions(value)?;
        let len = dimensions.len();
        for dimension in dimensions {
            let dimension = dimension.ok()?;
            if dimension == 0 && !(mean && len > 1) {
                return None;
            }
        }
        Some(Self {
            value,
            len,
            zero_is_one: mean && len > 1,
            scalar_dimension: None,
        })
    }
    pub fn materialize(self) -> crate::BuiltinResult<Vec<usize>> {
        let mut values = runmat_value::exact_capacity_vec(self.len).map_err(|source| {
            crate::runtime_error::semantic_error_with_source(
                "ResourceExhausted",
                "reduction axis allocation failed",
                source,
            )
        })?;
        if let Some(dimension) = self.scalar_dimension {
            values.push(dimension);
            return Ok(values);
        }
        let dimensions =
            crate::builtins::common::tensor::borrowed_host_shape_dimensions(self.value)
                .expect("same borrowed host dimension owner");
        for dimension in dimensions {
            let dimension = dimension.expect("lexical plan validated unchanged control");
            values.push(if self.zero_is_one {
                dimension.max(1)
            } else {
                dimension
            });
        }
        Ok(values)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ReductionAxisPlan<'a> {
    Auto,
    Dim(usize),
    Dimensions(ReductionDimensions<'a>),
    All,
}
impl ReductionAxisPlan<'_> {
    pub fn resolved_capacity(self, rank_upper: usize) -> usize {
        match self {
            Self::All => rank_upper.max(1),
            Self::Dimensions(dimensions) if dimensions.len != 0 => dimensions.len,
            _ => 1,
        }
    }
    pub fn retained_len(self) -> usize {
        match self {
            Self::Dimensions(dimensions) => dimensions.len,
            _ => 0,
        }
    }
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
            || matches!(self, Self::Dimensions(dimensions) if dimensions.len == 0)
    }
}

pub(crate) fn reduction_sum_bytes(parts: &[u64]) -> Option<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, &part| total.checked_add(part))
}

/// Recognize a rejected scalar axis using the same typed dimension decoder
/// as the public owners. This only funds their existing parser/error path;
/// it supplies no axis plan, prepared mathematics or kernel eligibility.
pub(crate) fn reduction_rejected_scalar_axis(controls: &[Value]) -> bool {
    matches!(controls, [value @ (Value::Num(_) | Value::Int(_))]
        if crate::builtins::common::tensor::scalar_dimension_from_value_typed(value, false).is_err())
}

pub(crate) fn reduction_vec<T>(count: usize) -> crate::BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "ResourceExhausted",
            "reduction metadata allocation failed",
            source,
        )
    })
}

/// Parsed by the operation owner before producer admission can touch a device.
/// Resolving the final, domain-refined fact is pure and does not parse again.
pub trait PreparedFusionReduction {
    fn resolve(&self, input: &ValueFact) -> crate::BuiltinResult<Option<FusionReductionContract>>;

    /// Carrier certification is independent of any input sign theorem.
    /// Only the prepared operation's ordinary public output owner may opt in.
    fn publication_storage(
        &self,
        _input: &ValueFact,
        _output: &ValueFact,
        _stages: &[FusionReductionContract],
    ) -> Option<StorageFact> {
        None
    }

    /// Ordered arithmetic stages for one public reduction call. The default
    /// covers reductions whose public operation is one physical reduction.
    /// A multi-axis owner may preserve intermediate class and publication
    /// boundaries by returning several contracts from its parsed controls.
    fn resolve_stages(
        &self,
        input: &ValueFact,
    ) -> crate::BuiltinResult<Option<Vec<FusionReductionContract>>> {
        self.resolve(input)
            .map(|contract| contract.map(|stage| vec![stage]))
    }

    /// Optional represented publication and sign theorem. This is deliberately
    /// absent for every owner unless its actual final publication certifies it.
    fn nonnegative_output_storage(
        &self,
        _input: &ValueFact,
        _output: &ValueFact,
        _stages: &[FusionReductionContract],
    ) -> Option<StorageFact> {
        None
    }
}

pub struct FusionReductionControls<'a> {
    pub values: &'a [Value],
    pub requested_outputs: usize,
}
impl FusionReductionControls<'_> {
    pub fn supported(&self) -> bool {
        self.requested_outputs == 1
            && self.values.iter().all(|value| {
                matches!(
                    value,
                    Value::Num(_)
                        | Value::Int(_)
                        | Value::String(_)
                        | Value::CharArray(_)
                        | Value::StringArray(_)
                        | Value::Tensor(_)
                )
            })
    }
}
pub type FusionReductionPrepareFuture<'a> = Pin<
    Box<dyn Future<Output = crate::BuiltinResult<Option<Box<dyn PreparedFusionReduction>>>> + 'a>,
>;
pub type FusionReductionPreparer =
    for<'a> fn(FusionReductionControls<'a>) -> FusionReductionPrepareFuture<'a>;

impl FusionReductionDeclaration {
    pub async fn certify_publication_storage(
        &self,
        context: FusionReductionContext<'_>,
        output: &ValueFact,
    ) -> crate::BuiltinResult<Option<StorageFact>> {
        if context.real_floating_input()?.is_none()
            || !matches!(
                context.input.storage,
                StorageFact::Scalar | StorageFact::Dense
            )
            || matches!(output.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            return Ok(None);
        }
        let Some(prepared) = (self.prepare)(FusionReductionControls {
            values: context.controls,
            requested_outputs: context.requested_outputs,
        })
        .await?
        else {
            return Ok(None);
        };
        let Some(stages) = prepared.resolve_stages(context.input)? else {
            return Ok(None);
        };
        Ok(prepared
            .publication_storage(context.input, output, &stages)
            .filter(|storage| matches!(storage, StorageFact::Scalar | StorageFact::Dense)))
    }

    pub fn resolve<'a>(&'a self, context: FusionReductionContext<'a>) -> FusionReductionFuture<'a> {
        Box::pin(async move {
            // Preserve the standalone resolver's existing unsupported-input gate.
            if context.real_floating_input()?.is_none() {
                return Ok(None);
            }
            let Some(prepared) = (self.prepare)(FusionReductionControls {
                values: context.controls,
                requested_outputs: context.requested_outputs,
            })
            .await?
            else {
                return Ok(None);
            };
            prepared.resolve(context.input)
        })
    }

    pub fn resolve_stages<'a>(
        &'a self,
        context: FusionReductionContext<'a>,
    ) -> FusionReductionStagesFuture<'a> {
        Box::pin(async move {
            if context.real_floating_input()?.is_none() {
                return Ok(None);
            }
            let Some(prepared) = (self.prepare)(FusionReductionControls {
                values: context.controls,
                requested_outputs: context.requested_outputs,
            })
            .await?
            else {
                return Ok(None);
            };
            prepared.resolve_stages(context.input)
        })
    }
}

pub fn checked_reduction_elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_fact::value_fact;

    #[test]
    fn zero_cardinality_precedes_overflow_and_other_overflow_is_unknown() {
        assert_eq!(checked_reduction_elements(&[usize::MAX, 2, 0]), Some(0));
        assert_eq!(checked_reduction_elements(&[usize::MAX, 2]), None);
        assert_eq!(checked_reduction_elements(&[]), Some(1));
    }

    #[test]
    fn contract_input_is_real_floating_single_output_and_does_not_touch_payloads() {
        let fact = value_fact(&Value::Num(3.0));
        let request = FusionReductionContext {
            input: &fact,
            controls: &[],
            requested_outputs: 1,
        };
        assert_eq!(
            request.real_floating_input().unwrap(),
            Some((NumericClass::Double, vec![1, 1]))
        );
        let extra = FusionReductionContext {
            requested_outputs: 2,
            ..request
        };
        assert!(extra.real_floating_input().unwrap().is_none());
        for value in [
            Value::Complex(1.0, 2.0),
            Value::Bool(true),
            Value::String("unsupported".into()),
        ] {
            let fact = value_fact(&value);
            assert!(FusionReductionContext {
                input: &fact,
                controls: &[],
                requested_outputs: 1
            }
            .real_floating_input()
            .unwrap()
            .is_none());
        }
    }
}
