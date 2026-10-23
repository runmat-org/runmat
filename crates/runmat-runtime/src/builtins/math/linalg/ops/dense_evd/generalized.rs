//! Full generalized EVD: native ZGGEV or pinned portable caller-owned GEVD.
//! Both retain homogeneous eigenvalues through normalization and use the
//! same exact-zero, scale-safe public eigenvalue projection.
use super::{workspace::Buffer, DenseEvdError};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationReceipt, HostTypedWorkspace},
    RuntimeContext,
};
use faer::dyn_stack::StackReq;
use num_complex::Complex64;

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
mod native {
    use super::*;
    pub(in crate::builtins::math::linalg::ops::dense_evd) struct Workspace {
        pub beta: Buffer<Complex64>,
        pub real: Buffer<f64>,
        pub original_mass: Option<Buffer<Complex64>>,
        pub work: Option<Buffer<Complex64>>,
    }
    pub(in crate::builtins::math::linalg::ops::dense_evd) fn bytes(
        n: usize,
        left: bool,
    ) -> Result<u64, HostAllocationError> {
        let mut bytes = HostTypedWorkspace::<Complex64>::quoted_bytes(n)?;
        bytes = super::super::add(
            bytes,
            HostTypedWorkspace::<f64>::quoted_bytes(
                n.checked_mul(8).ok_or(HostAllocationError::Overflow)?,
            )?,
        )?;
        if left {
            bytes = super::super::add(
                bytes,
                HostTypedWorkspace::<Complex64>::quoted_bytes(
                    n.checked_mul(n).ok_or(HostAllocationError::Overflow)?,
                )?,
            )?;
        }
        Ok(bytes)
    }
    impl Workspace {
        pub(in crate::builtins::math::linalg::ops::dense_evd) async fn new(
            n: usize,
            left: bool,
            runtime: Option<&RuntimeContext>,
            receipt: Option<&HostAllocationReceipt>,
        ) -> Result<Self, DenseEvdError> {
            Ok(Self {
                beta: Buffer::new(n, runtime, receipt).await?,
                real: Buffer::new(
                    n.checked_mul(8).ok_or(HostAllocationError::Overflow)?,
                    runtime,
                    receipt,
                )
                .await?,
                original_mass: if left {
                    Some(
                        Buffer::new(
                            n.checked_mul(n).ok_or(HostAllocationError::Overflow)?,
                            runtime,
                            receipt,
                        )
                        .await?,
                    )
                } else {
                    None
                },
                work: None,
            })
        }
    }
}

#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
mod portable {
    use super::*;
    pub(in crate::builtins::math::linalg::ops::dense_evd) struct Workspace {
        pub beta: Buffer<Complex64>,
        pub original_mass: Option<Buffer<Complex64>>,
    }
    pub(in crate::builtins::math::linalg::ops::dense_evd) fn bytes(
        n: usize,
        left: bool,
    ) -> Result<u64, HostAllocationError> {
        let mut bytes = HostTypedWorkspace::<Complex64>::quoted_bytes(n)?;
        if left {
            bytes = super::super::add(
                bytes,
                HostTypedWorkspace::<Complex64>::quoted_bytes(
                    n.checked_mul(n).ok_or(HostAllocationError::Overflow)?,
                )?,
            )?;
        }
        Ok(bytes)
    }
    impl Workspace {
        pub(in crate::builtins::math::linalg::ops::dense_evd) async fn new(
            n: usize,
            left: bool,
            runtime: Option<&RuntimeContext>,
            receipt: Option<&HostAllocationReceipt>,
        ) -> Result<Self, DenseEvdError> {
            Ok(Self {
                beta: Buffer::new(n, runtime, receipt).await?,
                original_mass: if left {
                    Some(
                        Buffer::new(
                            n.checked_mul(n).ok_or(HostAllocationError::Overflow)?,
                            runtime,
                            receipt,
                        )
                        .await?,
                    )
                } else {
                    None
                },
            })
        }
    }
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
pub(super) use native::{bytes, Workspace};
#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
pub(super) use portable::{bytes, Workspace};

pub(super) fn requirement(n: usize, left: bool) -> StackReq {
    if n == 0 {
        return StackReq::EMPTY;
    }
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    {
        let _ = (n, left);
        StackReq::EMPTY
    }
    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    {
        super::generalized_kernel::requirement(n, left)
    }
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn zggev(
    owner: &mut super::workspace::Workspace,
    query: bool,
) -> Result<usize, super::NumericalFailure> {
    let n = owner.order;
    if n == 0 {
        return Ok(1);
    }
    let order = i32::try_from(n).map_err(|_| super::NumericalFailure::Dimension)?;
    let state = owner.generalized.as_mut().expect("generalized workspace");
    let mut query_word = [Complex64::new(0.0, 0.0)];
    let mut unused_left = [Complex64::new(0.0, 0.0)];
    let left = owner.left.as_mut().map(Buffer::get_mut);
    let require_left = left.is_some();
    let left = left.unwrap_or(&mut unused_left);
    let work = if query {
        &mut query_word[..]
    } else {
        state.work.as_mut().expect("queried work").get_mut()
    };
    let lwork = if query {
        -1
    } else {
        i32::try_from(work.len()).map_err(|_| super::NumericalFailure::Dimension)?
    };
    let mut info = 0;
    // SAFETY: exact n*n coefficient/right/optional-left arrays, n-element
    // alpha/beta and 8*n real scratch are admitted and initialized. All leading
    // dimensions and work lengths are checked before entering the same ZGGEV.
    unsafe {
        lapack::zggev(
            if require_left { b'V' } else { b'N' },
            b'V',
            order,
            owner.matrix.get_mut(),
            order,
            owner.mass.as_mut().expect("generalized B").get_mut(),
            order,
            owner.values.get_mut(),
            state.beta.get_mut(),
            left,
            if require_left { order } else { 1 },
            owner.right.get_mut(),
            order,
            work,
            lwork,
            state.real.get_mut(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(if query {
            super::NumericalFailure::LapackQuery(info)
        } else {
            super::NumericalFailure::LapackFailure(info)
        });
    }
    if query {
        let requested = query_word[0].re;
        if !requested.is_finite()
            || requested < 1.0
            || requested.fract() != 0.0
            || requested > f64::from(i32::MAX)
        {
            return Err(super::NumericalFailure::Dimension);
        }
        Ok(requested as usize)
    } else {
        Ok(0)
    }
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
pub(super) fn query(
    owner: &mut super::workspace::Workspace,
) -> Result<usize, super::NumericalFailure> {
    if let Some(original) = &mut owner
        .generalized
        .as_mut()
        .expect("generalized")
        .original_mass
    {
        original
            .get_mut()
            .copy_from_slice(owner.mass.as_ref().expect("B").get());
    }
    zggev(owner, true)
}

/// Retain alpha/beta until residual-preserving normalization completes.
/// Tests inspect this internal boundary without a second alpha backing.
pub(super) fn compute_homogeneous(
    owner: &mut super::workspace::Workspace,
) -> Result<(), super::NumericalFailure> {
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    {
        zggev(owner, false)?;
    }
    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    {
        super::generalized_kernel::compute(owner).map_err(super::NumericalFailure::Gevd)?;
    }
    Ok(())
}

pub(super) fn compute(
    owner: &mut super::workspace::Workspace,
) -> Result<(), super::NumericalFailure> {
    if owner.order == 0 {
        return Ok(());
    }
    compute_homogeneous(owner)?;
    let state = owner.generalized.as_ref().expect("generalized workspace");
    if let Some(left) = &mut owner.left {
        super::normalization::left(
            left.get_mut(),
            owner.right.get(),
            owner.order,
            super::normalization::Metric::Pencil {
                mass: state
                    .original_mass
                    .as_ref()
                    .expect("left normalization B")
                    .get(),
                beta: state.beta.get(),
            },
        );
    }
    for (alpha, &beta) in owner.values.get_mut().iter_mut().zip(state.beta.get()) {
        *alpha = super::homogeneous_eigenvalue(*alpha, beta);
    }
    Ok(())
}
