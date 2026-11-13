//! Request-scoped reduction emission and fault ownership carriers.
use super::super::validated_fusion::ValidatedFusionPipeline;
use super::core::{FrozenAnyPipeline, FrozenAnySnapshot};
use super::typed_std::StdSelected;
use super::*;
use std::cell::Cell;

pub(in crate::backend::wgpu::provider::backend) type AllocationFault<'a> =
    dyn Fn(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()> + 'a;
pub(super) type BeforeBinaryFault<'a> = dyn Fn(bool, &GpuTensorHandle) -> Result<()> + 'a;
pub(super) type SquareRetirementFault<'a> = dyn Fn(&WgpuProvider, &runmat_accelerate_api::GpuTensorWireDescriptor, (usize, u64)) -> Result<()>
    + 'a;

#[derive(Clone, Copy)]
pub(in crate::backend::wgpu::provider::backend) struct SelectedEmission<'a> {
    pub selected: Option<&'a ValidatedFusionPipeline>,
    pub effect: Option<&'a Cell<bool>>,
    pub after_allocation: Option<&'a AllocationFault<'a>>,
}

impl<'a> SelectedEmission<'a> {
    pub(in crate::backend::wgpu::provider::backend) const fn legacy() -> Self {
        Self {
            selected: None,
            effect: None,
            after_allocation: None,
        }
    }
}

pub(super) struct DimSumMeanSelection<'a> {
    pub snapshot: FrozenAnySnapshot,
    pub power_pipeline: Option<&'a FrozenAnyPipeline>,
    pub emission: SelectedEmission<'a>,
}

pub(super) struct StdFormulaPhase<'a> {
    pub selected: Option<&'a StdSelected>,
    pub effect: Option<&'a Cell<bool>>,
}
