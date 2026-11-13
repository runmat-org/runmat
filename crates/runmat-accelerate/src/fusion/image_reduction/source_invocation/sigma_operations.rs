//! Held source authority for the exact variance-public Add -> direct SQRT.
//! Physical recipes remain separate from the full image arithmetic proof.
use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderElementType, ProviderLayout,
    ProviderRealElementwiseOperation as Operation, ProviderRealElementwiseRequest,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_types::{NumericClass, NumericDomain, RealFloatingAddMode};

#[derive(Clone, Debug)]
pub(super) struct BoundSigmaAddControl {
    source_id: ValueId,
    literal: ScalarLiteral,
    value_digest: Digest,
}

impl BoundSigmaAddControl {
    pub(super) fn literal(&self) -> ScalarLiteral {
        self.literal
    }
}

pub(super) fn freeze_control(
    plan: &FusionGroupPlan,
    inputs: &[Value],
    positions: &ImageValuePositions,
) -> Option<BoundSigmaAddControl> {
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref()? else {
        return None;
    };
    let raw = pattern.reduction_routes.as_ref()?.stage_producers()?.sigma;
    let sqrt_input = plan.operations.iter().find_map(|op| match op {
        FusionOp::Builtin {
            inputs,
            output: Some(id),
            ..
        } if *id == raw && inputs.len() == 1 => Some(inputs[0]),
        _ => None,
    })?;
    let source_id = plan.operations.iter().find_map(|op| match op {
        FusionOp::Primitive {
            op: crate::graph::PrimitiveOp::Add,
            inputs,
            output: Some(id),
        } if *id == sqrt_input && inputs.len() == 2 => Some(inputs[1]),
        _ => None,
    })?;
    let value = positions
        .get(&source_id)
        .and_then(|position| inputs.get(*position))
        .or_else(|| plan.const_values.get(&source_id))?;
    let literal = match super::super::source_output_hooks::scalar(value)? {
        ProviderImageOutputOperand::ScalarF32Bits(bits) => ScalarLiteral::F32Bits(bits),
        ProviderImageOutputOperand::ScalarF64Bits(bits) => ScalarLiteral::F64Bits(bits),
        _ => return None,
    };
    Some(BoundSigmaAddControl {
        source_id,
        literal,
        value_digest: crate::fusion::semantics::exact_constant_digest(value).ok()?,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedSigmaOperations {
    variance_input: u64,
    add_output: u64,
    sqrt_output: u64,
    control: BoundSigmaAddControl,
    mode: RealFloatingAddMode,
    source_fact: ValueFact,
    raw_fact: ValueFact,
    semantic_identity: Digest,
}

impl ImageSourceInvocation {
    pub(crate) fn selected_sigma_operations(&self) -> Option<ImageSelectedSigmaOperations> {
        let moments = self.moment_program.as_ref()?;
        let variance = moments.variance.last()?;
        let raw = self.selected_sigma_raw(&moments.sigma)?;
        let add = self
            .source
            .steps
            .iter()
            .find(|step| u64::from(step.output) == raw.source_input())?;
        let control = self.selected_stage.as_ref()?.sigma_control.as_ref()?;
        if add.inputs.as_slice()
            != [
                ValueId::try_from(variance.public_output).ok()?,
                control.source_id,
            ]
        {
            return None;
        }
        let left = self.source_fact(add.inputs[0])?;
        let right = self.source_fact(control.source_id)?;
        if right.numeric()?.class != control.literal.class()
            || right.shape.element_count() != Some(1)
        {
            return None;
        }
        let mode = runmat_types::real_floating_add_mode(left, right)?;
        if mode == RealFloatingAddMode::MixedDoubleThenSingle {
            return None;
        }
        let output = self.source_fact(add.output)?;
        if output.numeric()?.class != mode.result_class()
            || output.numeric()?.domain != NumericDomain::Real
            || output.shape != left.shape
            || output.numeric() != moments.sigma.raw_fact.numeric()
            || output.shape != moments.sigma.raw_fact.shape
        {
            return None;
        }
        Some(ImageSelectedSigmaOperations {
            variance_input: variance.public_output,
            add_output: raw.source_input(),
            sqrt_output: raw.raw_output(),
            control: control.clone(),
            mode,
            source_fact: left.clone(),
            raw_fact: moments.sigma.raw_fact.clone(),
            semantic_identity: self.semantic_identity,
        })
    }
}

impl ImageSelectedSigmaOperations {
    pub(crate) fn matches_moments(
        &self,
        moments: &runmat_accelerate_api::ProviderImageMomentProgram,
    ) -> bool {
        // Publication storage is certified separately by the prepared MEAN
        // owner. The original source fact may conservatively remain Unknown.
        moments.variance.last().is_some_and(|call| {
            call.public_output == self.variance_input
                && call.public_fact().numeric() == self.source_fact.numeric()
                && call.public_fact().shape == self.source_fact.shape
        }) && (moments.sigma.source_input, moments.sigma.raw_output)
            == (self.add_output, self.sqrt_output)
            && moments.sigma.raw_fact == self.raw_fact
    }
    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }
    pub(crate) fn control(&self) -> ScalarLiteral {
        self.control.literal
    }
    #[cfg(test)]
    pub(crate) fn control_digest(&self) -> Digest {
        self.control.value_digest
    }
    pub(crate) fn source_ids(&self) -> (u64, u64, u64, u64) {
        (
            self.variance_input,
            u64::from(self.control.source_id),
            self.add_output,
            self.sqrt_output,
        )
    }
    #[cfg(test)]
    pub(crate) fn mode(&self) -> RealFloatingAddMode {
        self.mode
    }
    fn matches_carrier(&self, source: &ProviderRepresentation) -> bool {
        let expected = match self.source_fact.numeric().map(|numeric| numeric.class) {
            Some(NumericClass::Single) => ProviderElementType::F32,
            Some(NumericClass::Double) => ProviderElementType::F64,
            _ => return false,
        };
        source.element_type == expected
            && source.storage == ProviderStorage::DenseReal
            && source.layout == ProviderLayout::ColumnMajorContiguous
            && source.residency == ProviderResidency::Device
            && self
                .source_fact
                .shape
                .known_dims()
                .and_then(|dims| {
                    dims.into_iter()
                        .map(|n| u64::try_from(n?).ok())
                        .collect::<Option<Vec<_>>>()
                })
                .as_ref()
                == Some(&source.shape)
    }
    /// Consume a carrier description supplied by the prepared reduction owner;
    /// semantic class/shape facts do not mint a contiguous physical allocation.
    pub(crate) fn add_request(
        &self,
        source: &ProviderRepresentation,
        source_bytes: u64,
    ) -> Option<ProviderRealElementwiseRequest> {
        (self.matches_carrier(source)
            && source
                .checked_byte_len()
                .is_some_and(|bytes| bytes <= source_bytes))
        .then(|| ProviderRealElementwiseRequest {
            operation: Operation::ScalarAdd {
                control: self.control.literal,
                mode: self.mode,
            },
            source: source.clone(),
            source_buffer_bytes: source_bytes,
        })
    }
    pub(crate) fn sqrt_request(
        &self,
        source: &ProviderRepresentation,
        source_bytes: u64,
    ) -> Option<ProviderRealElementwiseRequest> {
        (self.matches_carrier(source)
            && source
                .checked_byte_len()
                .is_some_and(|bytes| bytes <= source_bytes))
        .then(|| ProviderRealElementwiseRequest {
            operation: Operation::Sqrt,
            source: source.clone(),
            source_buffer_bytes: source_bytes,
        })
    }
    pub(crate) fn identity(&self) -> anyhow::Result<Digest> {
        Ok(Digest::sha256_serialized(&(
            "selected-sigma-real-operations-v1",
            self.semantic_identity,
            self.source_ids(),
            self.control.value_digest,
            self.control.literal,
            self.mode.result_class(),
        ))?)
    }
}
