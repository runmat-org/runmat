use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::analysis::{AnalysisStore, ProgramPointFacts};
use runmat_types::{CapabilitySet, EffectSet, ProgramFunctionId, ProgramPointId, ProgramSpan};

pub(super) struct FunctionLoweringContext<'a> {
    pub function: ProgramFunctionId,
    pub source: runmat_types::ProgramSourceId,
    pub analysis: &'a AnalysisStore,
    next_value: u32,
    next_instruction: u32,
    next_safepoint: u32,
}

impl<'a> FunctionLoweringContext<'a> {
    pub fn new(
        function: ProgramFunctionId,
        source: runmat_types::ProgramSourceId,
        analysis: &'a AnalysisStore,
    ) -> Self {
        Self {
            function,
            source,
            analysis,
            next_value: 0,
            next_instruction: 0,
            next_safepoint: 0,
        }
    }

    pub fn value(&mut self) -> NativeValueId {
        let value = NativeValueId(self.next_value);
        self.next_value += 1;
        value
    }

    pub fn instruction(&mut self) -> NativeInstructionId {
        let instruction = NativeInstructionId(self.next_instruction);
        self.next_instruction += 1;
        instruction
    }

    pub fn safepoint(&mut self) -> NativeSafepointId {
        let safepoint = NativeSafepointId(self.next_safepoint);
        self.next_safepoint += 1;
        safepoint
    }

    pub fn point(
        &self,
        block: runmat_mir::BasicBlockId,
        position: usize,
    ) -> NativeCodegenResult<ProgramPointId> {
        Ok(ProgramPointId {
            function: self.function,
            block: u32::try_from(block.0).map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.block_identity",
                    "MIR block identity exceeds the Native IR schema",
                )
                .at_function(self.function)
            })?,
            position: u32::try_from(position).map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.position",
                    "MIR statement position exceeds the Native IR schema",
                )
                .at_function(self.function)
            })?,
        })
    }

    pub fn source_location(
        &self,
        span: runmat_types::Span,
    ) -> NativeCodegenResult<NativeSourceLocation> {
        let start = u64::try_from(span.start).map_err(|_| {
            NativeCodegenError::new("native.lowering.source_span", "source start exceeds u64")
        })?;
        let end = u64::try_from(span.end).map_err(|_| {
            NativeCodegenError::new("native.lowering.source_span", "source end exceeds u64")
        })?;
        if end < start {
            return Err(NativeCodegenError::new(
                "native.lowering.source_span",
                "source span end precedes its start",
            ));
        }
        Ok(NativeSourceLocation {
            source: self.source,
            span: ProgramSpan { start, end },
        })
    }

    pub fn facts(&self, point: ProgramPointId) -> Option<&ProgramPointFacts> {
        self.analysis.facts_at(point)
    }

    pub fn value_type(
        &self,
        point: ProgramPointId,
        local: runmat_mir::MirLocalId,
    ) -> NativeValueType {
        let Ok(local) = u32::try_from(local.0) else {
            return NativeValueType::Generic;
        };
        self.facts(point)
            .and_then(|facts| {
                facts.local(runmat_types::RegionValueId {
                    function: self.function,
                    local,
                })
            })
            .and_then(|fact| fact.fact.clone())
            .map_or(NativeValueType::Generic, |fact| {
                NativeValueType::Analyzed(Box::new(fact))
            })
    }

    pub fn effect_delta(
        &self,
        before: ProgramPointId,
        after: ProgramPointId,
    ) -> (EffectSet, CapabilitySet) {
        let before = self.facts(before);
        let after = self.facts(after);
        let empty_effects = std::collections::BTreeSet::new();
        let empty_capabilities = std::collections::BTreeSet::new();
        let before_effects = before.map_or(&empty_effects, |facts| &facts.effects.0);
        let before_capabilities = before.map_or(&empty_capabilities, |facts| &facts.capabilities.0);
        let effects = after
            .map(|facts| {
                EffectSet(
                    facts
                        .effects
                        .0
                        .difference(before_effects)
                        .copied()
                        .collect(),
                )
            })
            .unwrap_or_default();
        let capabilities = after
            .map(|facts| {
                CapabilitySet(
                    facts
                        .capabilities
                        .0
                        .difference(before_capabilities)
                        .copied()
                        .collect(),
                )
            })
            .unwrap_or_default();
        (effects, capabilities)
    }
}

pub(super) fn frame_state(
    point: ProgramPointId,
    source: NativeSourceLocation,
    locals: &[NativeValueId],
    operands: Vec<NativeValueId>,
    side_effect_epoch: NativeValueId,
) -> NativeFrameState {
    NativeFrameState {
        point,
        source,
        locals: locals
            .iter()
            .enumerate()
            .map(|(local, value)| NativeFrameLocal {
                local: NativeLocalId(local as u32),
                value: *value,
            })
            .collect(),
        operands,
        side_effect_epoch,
    }
}
