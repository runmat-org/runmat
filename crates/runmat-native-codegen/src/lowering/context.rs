use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::analysis::{AnalysisStore, ProgramPointFacts};
use runmat_types::{FunctionId, ProgramPointId, ProgramSpan};

pub(super) struct FunctionLoweringContext<'a> {
    pub function: FunctionId,
    pub portable_function: runmat_types::ProgramFunctionId,
    pub source: runmat_types::ProgramSourceId,
    pub analysis: &'a AnalysisStore,
    next_value: u32,
    next_instruction: u32,
    next_safepoint: u32,
}

impl<'a> FunctionLoweringContext<'a> {
    pub fn new(
        function: FunctionId,
        source: runmat_types::ProgramSourceId,
        analysis: &'a AnalysisStore,
    ) -> NativeCodegenResult<Self> {
        let portable_function = analysis
            .function_for_local(function)
            .ok_or_else(|| {
                NativeCodegenError::new(
                    "native.lowering.function_analysis",
                    "lowered MIR function has no canonical analysis identity",
                )
                .at_function(function)
            })?
            .function;
        Ok(Self {
            function,
            portable_function,
            source,
            analysis,
            next_value: 0,
            next_instruction: 0,
            next_safepoint: 0,
        })
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
            function: self.portable_function,
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
                    function: self.portable_function,
                    local,
                })
            })
            .and_then(|fact| fact.fact.clone())
            .map_or(NativeValueType::Generic, |fact| {
                NativeValueType::Analyzed(Box::new(fact))
            })
    }

    pub fn operation_contract(
        &self,
        site: &runmat_mir::MirOperationSite,
    ) -> NativeCodegenResult<super::operation::OperationRequirements> {
        let contract = self.analysis.operation_contract_at(site).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.operation_contract",
                "lowered MIR site has no canonical operation-local analysis contract",
            )
            .at_point(site.point)
            .for_construct(site.construct)
        })?;
        Ok(super::operation::OperationRequirements {
            effects: contract.effects.clone(),
            capabilities: contract.capabilities.clone(),
        })
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
