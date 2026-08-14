#[cfg(not(target_arch = "wasm32"))]
use runmat_native_codegen::NativeFrameState;
use runmat_native_codegen::NativeLocalId;
use runmat_types::{ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan};
use runmat_value::Value;
use std::collections::BTreeMap;

#[cfg(not(target_arch = "wasm32"))]
use crate::{JitError, JitResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResumeSite {
    pub point: ProgramPointId,
    pub source: ProgramSourceId,
    pub span: ProgramSpan,
    pub phase: runmat_native_codegen::NativeSitePhase,
    pub ordinal: u32,
    pub bytecode_pc: Option<u64>,
    pub side_effect_epoch: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedLocal {
    pub local: NativeLocalId,
    pub value: Option<Value>,
}

/// Executor-neutral values required to resume without replaying an earlier site.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedFrame {
    pub function: ProgramFunctionId,
    pub site: ResumeSite,
    pub locals: Vec<MaterializedLocal>,
    pub operands: Vec<Value>,
    pub supplied_inputs: usize,
    pub requested_outputs: usize,
    pub missing_input_locals: Vec<NativeLocalId>,
    pub global_bindings: BTreeMap<usize, String>,
    pub persistent_bindings: BTreeMap<usize, String>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct NativeMaterializationContext {
    pub phase: runmat_native_codegen::NativeSitePhase,
    pub ordinal: u32,
    pub bytecode_pc: Option<u64>,
    pub supplied_inputs: usize,
    pub requested_outputs: usize,
    pub missing_input_locals: Vec<NativeLocalId>,
    pub global_bindings: BTreeMap<usize, String>,
    pub persistent_bindings: BTreeMap<usize, String>,
}

impl MaterializedFrame {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_native(
        frame: &NativeFrameState,
        context: NativeMaterializationContext,
        resolve: impl Fn(runmat_native_codegen::NativeValueId) -> JitResult<Option<Value>>,
    ) -> JitResult<Self> {
        let locals = frame
            .locals
            .iter()
            .map(|local| {
                Ok(MaterializedLocal {
                    local: local.local,
                    value: resolve(local.value)?,
                })
            })
            .collect::<JitResult<Vec<_>>>()?;
        let operands = frame
            .operands
            .iter()
            .map(|value| {
                resolve(*value)?.ok_or_else(|| {
                    JitError::Host("native deoptimization operand is unmaterialized".into())
                })
            })
            .collect::<JitResult<Vec<_>>>()?;
        Ok(Self {
            function: frame.point.function,
            site: ResumeSite {
                point: frame.point,
                source: frame.source.source,
                span: frame.source.span,
                phase: context.phase,
                ordinal: context.ordinal,
                bytecode_pc: context.bytecode_pc,
                side_effect_epoch: u64::from(frame.side_effect_epoch.0),
            },
            locals,
            operands,
            supplied_inputs: context.supplied_inputs,
            requested_outputs: context.requested_outputs,
            missing_input_locals: context.missing_input_locals,
            global_bindings: context.global_bindings,
            persistent_bindings: context.persistent_bindings,
        })
    }
}
