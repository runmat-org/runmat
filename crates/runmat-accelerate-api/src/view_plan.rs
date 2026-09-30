//! Pure physical view transitions. These facts do not grant an allocation or
//! execute a provider operation; the owning provider admits the whole chain.

use crate::{GpuTensorLayout, GpuTensorView, NumericElementType, TransposeInfo};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderViewSource {
    /// Start with the authenticated live anchor's exact view.
    Current,
    /// Geometry expected from an earlier admitted operation on the same
    /// provider. This is not a lease or a forged future handle.
    Planned(GpuTensorView),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderViewStep {
    Reshape(Vec<usize>),
    Transpose2D,
    Permute2D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderViewChain {
    pub source: ProviderViewSource,
    pub steps: Vec<ProviderViewStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderViewOutcome {
    pub view: GpuTensorView,
    /// Maximum physical output allocation required by a single materializing
    /// step. Zero means the chain can use views of existing storage.
    pub materialization_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderViewCapability {
    Ready(ProviderViewOutcome),
    UnsupportedOperation,
    ResourceLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderViewPlanError {
    InvalidSource,
    InvalidTransition,
    SizeOverflow,
}

fn elements(shape: &[usize]) -> Result<u64, ProviderViewPlanError> {
    shape.iter().try_fold(1_u64, |count, extent| {
        count
            .checked_mul(u64::try_from(*extent).map_err(|_| ProviderViewPlanError::SizeOverflow)?)
            .ok_or(ProviderViewPlanError::SizeOverflow)
    })
}

impl ProviderViewChain {
    /// Shared shape/layout semantics used by provider capability queries.
    /// Provider-specific storage limits and live-owner checks remain with the
    /// provider. Materialization here follows the existing native 2-D
    /// transpose/reshape/permute contract.
    pub fn analyze(
        &self,
        current: &GpuTensorView,
        element_type: NumericElementType,
    ) -> Result<ProviderViewOutcome, ProviderViewPlanError> {
        let mut view = match &self.source {
            ProviderViewSource::Current => current.clone(),
            ProviderViewSource::Planned(view) => view.clone(),
        };
        if let GpuTensorLayout::Transpose(info) = view.layout() {
            if view.shape() != [info.base_cols, info.base_rows] {
                return Err(ProviderViewPlanError::InvalidSource);
            }
        }
        let mut materialization_bytes = 0;
        for step in &self.steps {
            let count = elements(view.shape())?;
            let bytes = count
                .checked_mul(element_type.element_size() as u64)
                .ok_or(ProviderViewPlanError::SizeOverflow)?;
            match step {
                ProviderViewStep::Reshape(shape) => {
                    if elements(shape)? != count {
                        return Err(ProviderViewPlanError::InvalidTransition);
                    }
                    if matches!(view.layout(), GpuTensorLayout::Transpose(_)) {
                        materialization_bytes = materialization_bytes.max(bytes);
                    }
                    view = view
                        .with_shape(shape.clone())
                        .with_layout(GpuTensorLayout::Contiguous);
                }
                ProviderViewStep::Transpose2D | ProviderViewStep::Permute2D => {
                    let [rows, columns] = view.shape() else {
                        return Err(ProviderViewPlanError::InvalidTransition);
                    };
                    let shape = vec![*columns, *rows];
                    let layout = match (step, view.layout()) {
                        (ProviderViewStep::Transpose2D, GpuTensorLayout::Contiguous) => {
                            GpuTensorLayout::Transpose(TransposeInfo {
                                base_rows: *rows,
                                base_cols: *columns,
                            })
                        }
                        (ProviderViewStep::Transpose2D, GpuTensorLayout::Transpose(_)) => {
                            GpuTensorLayout::Contiguous
                        }
                        (ProviderViewStep::Permute2D, GpuTensorLayout::Contiguous) => {
                            materialization_bytes = materialization_bytes.max(bytes);
                            GpuTensorLayout::Contiguous
                        }
                        (ProviderViewStep::Permute2D, GpuTensorLayout::Transpose(_)) => {
                            GpuTensorLayout::Contiguous
                        }
                        _ => unreachable!("view step is a 2-D permutation"),
                    };
                    view = view.with_shape(shape).with_layout(layout);
                }
            }
        }
        Ok(ProviderViewOutcome {
            view,
            materialization_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_inverse_permutation_is_zero_copy_but_reshape_materializes() {
        let lazy = GpuTensorView::contiguous(vec![2, 3]).with_layout(GpuTensorLayout::Transpose(
            TransposeInfo {
                base_rows: 3,
                base_cols: 2,
            },
        ));
        let inverse = ProviderViewChain {
            source: ProviderViewSource::Current,
            steps: vec![ProviderViewStep::Permute2D],
        }
        .analyze(&lazy, NumericElementType::F32)
        .unwrap();
        assert_eq!(inverse.view.shape(), [3, 2]);
        assert_eq!(inverse.view.layout(), GpuTensorLayout::Contiguous);
        assert_eq!(inverse.materialization_bytes, 0);

        let reshape = ProviderViewChain {
            source: ProviderViewSource::Current,
            steps: vec![ProviderViewStep::Reshape(vec![3, 2])],
        }
        .analyze(&lazy, NumericElementType::F64)
        .unwrap();
        assert_eq!(reshape.materialization_bytes, 48);
    }

    #[test]
    fn invalid_planned_layout_and_changed_count_are_rejected() {
        let current = GpuTensorView::contiguous(vec![2, 3]);
        let invalid = GpuTensorView::contiguous(vec![2, 3]).with_layout(
            GpuTensorLayout::Transpose(TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            }),
        );
        assert_eq!(
            ProviderViewChain {
                source: ProviderViewSource::Planned(invalid),
                steps: vec![ProviderViewStep::Permute2D],
            }
            .analyze(&current, NumericElementType::F32),
            Err(ProviderViewPlanError::InvalidSource)
        );
        assert_eq!(
            ProviderViewChain {
                source: ProviderViewSource::Current,
                steps: vec![ProviderViewStep::Reshape(vec![3, 3])],
            }
            .analyze(&current, NumericElementType::F32),
            Err(ProviderViewPlanError::InvalidTransition)
        );
    }
}
