//! Real retained Householder blocks act independently on complex RHS lanes.
//! One admitted real lane buffer is reused; compact factor DATA stays borrowed.
use super::*;

pub(super) enum RealLanes {
    None,
    F32(Buffer<f32>),
    F64(Buffer<f64>),
}

impl RealLanes {
    pub(super) fn quoted_bytes(
        data: &NumericFactorData,
        complex_rhs: bool,
        count: usize,
    ) -> Result<u64, HostAllocationError> {
        if !complex_rhs {
            return Ok(0);
        }
        match data.storage_class() {
            runmat_value::ExactHostSparseClass::Numeric(runmat_value::NumericDType::F32) => {
                HostTypedWorkspace::<f32>::quoted_bytes(count)
            }
            runmat_value::ExactHostSparseClass::Numeric(runmat_value::NumericDType::F64) => {
                HostTypedWorkspace::<f64>::quoted_bytes(count)
            }
            _ => Ok(0),
        }
    }

    pub(super) async fn new(
        data: &NumericFactorData,
        complex_rhs: bool,
        count: usize,
        receipt: Option<&HostAllocationReceipt>,
        runtime: &RuntimeContext,
    ) -> Result<Self, DenseFactorError> {
        if !complex_rhs {
            return Ok(Self::None);
        }
        match data.storage_class() {
            runmat_value::ExactHostSparseClass::Numeric(runmat_value::NumericDType::F32) => Ok(
                Self::F32(Buffer::<f32>::new(count, receipt, runtime).await?),
            ),
            runmat_value::ExactHostSparseClass::Numeric(runmat_value::NumericDType::F64) => Ok(
                Self::F64(Buffer::<f64>::new(count, receipt, runtime).await?),
            ),
            _ => Ok(Self::None),
        }
    }

    pub(super) fn apply<T: DenseScalar>(
        &mut self,
        qr: &DenseOrthogonalFactorData,
        adjoint: bool,
        mut rhs: MatMut<'_, T::Element>,
        stack: &mut MemStack,
    ) -> Result<(), DenseFactorError> {
        match self {
            Self::None => {
                super::apply::<T>(qr, adjoint, rhs, stack);
                Ok(())
            }
            Self::F32(lane) => {
                apply_lanes::<T, f32>(lane.get_mut(), qr, adjoint, rhs.rb_mut(), stack)
            }
            Self::F64(lane) => {
                apply_lanes::<T, f64>(lane.get_mut(), qr, adjoint, rhs.rb_mut(), stack)
            }
        }
    }
}

fn apply_lanes<T: DenseScalar, R: DenseScalar>(
    lane: &mut [R::Element],
    qr: &DenseOrthogonalFactorData,
    adjoint: bool,
    mut rhs: MatMut<'_, T::Element>,
    mut stack: &mut MemStack,
) -> Result<(), DenseFactorError> {
    let rows = rhs.nrows();
    let columns = rhs.ncols();
    let count = rows
        .checked_mul(columns)
        .ok_or(HostAllocationError::Overflow)?;
    let lane = &mut lane[..count];
    for component in 0..2 {
        for column in 0..columns {
            for row in 0..rows {
                let parts = T::from_element(rhs[(row, column)]).parts();
                lane[column * rows + row] =
                    R::from_parts(if component == 0 { parts.0 } else { parts.1 }, 0.0)
                        .ok_or(DenseFactorError::NonFinite)?
                        .to_factor();
            }
        }
        let real = MatMut::from_column_major_slice_mut(lane, rows, columns);
        super::apply::<R>(qr, adjoint, real, stack.rb_mut());
        for column in 0..columns {
            for row in 0..rows {
                let old = T::from_element(rhs[(row, column)]).parts();
                let new = R::from_element(lane[column * rows + row]).parts().0;
                rhs[(row, column)] = T::from_parts(
                    if component == 0 { new } else { old.0 },
                    if component == 1 { new } else { old.1 },
                )
                .ok_or(DenseFactorError::NonFinite)?
                .to_factor();
            }
        }
    }
    Ok(())
}
