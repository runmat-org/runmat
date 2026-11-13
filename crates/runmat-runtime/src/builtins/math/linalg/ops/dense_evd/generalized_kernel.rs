//! One QZ reduction followed by the shared residual-checked triangular solve.
//! All numerical storage belongs to the original admitted full-EVD owner.
use super::{workspace::Buffer, workspace::Workspace};
use crate::builtins::math::linalg::factor::qz::{
    complex_eigenvectors_in_place, complex_qz_optional_in_place, complex_qz_scratch,
    ComplexPencilView, ComplexQzOptionalView,
};
use faer::{
    col::{ColMut, ColRef},
    dyn_stack::{MemStack, StackReq},
    mat::{AsMatMut, MatMut, MatRef},
};
use num_complex::Complex64;

#[derive(Debug)]
pub(crate) enum GevdFailure {
    Schur(&'static str),
    Eigenvectors(&'static str),
    NonFiniteFactors,
}
impl std::fmt::Display for GevdFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Schur(error) => write!(f, "generalized Schur reduction failed: {error}"),
            Self::Eigenvectors(error) => write!(f, "generalized eigenvector solve failed: {error}"),
            Self::NonFiniteFactors => {
                f.write_str("generalized eigendecomposition produced nonfinite factors")
            }
        }
    }
}
impl std::error::Error for GevdFailure {}

pub(super) fn requirement(n: usize, _left: bool) -> StackReq {
    if n == 0 {
        return StackReq::EMPTY;
    }
    // Reduction and extraction are sequential. Both sides reuse one basis
    // and one product column, without private GEVD norm arrays or a Q/Z copy.
    complex_qz_scratch(n).or(faer::linalg::temp_mat_scratch::<Complex64>(n, 1)
        .and(faer::linalg::temp_mat_scratch::<Complex64>(n, 1)))
}

pub(super) fn compute(owner: &mut Workspace) -> Result<(), GevdFailure> {
    let n = owner.order;
    if n == 0 {
        return Ok(());
    }
    let state = owner.generalized.as_mut().expect("generalized workspace");
    if let Some(original) = &mut state.original_mass {
        original
            .get_mut()
            .copy_from_slice(owner.mass.as_ref().expect("B").get());
    }
    complex_qz_optional_in_place(
        &mut ComplexQzOptionalView {
            aa: MatMut::from_column_major_slice_mut(owner.matrix.get_mut(), n, n),
            bb: MatMut::from_column_major_slice_mut(
                owner.mass.as_mut().expect("B").get_mut(),
                n,
                n,
            ),
            left: owner
                .left
                .as_mut()
                .map(|x| MatMut::from_column_major_slice_mut(x.get_mut(), n, n)),
            right: MatMut::from_column_major_slice_mut(owner.right.get_mut(), n, n),
            alpha: ColMut::from_slice_mut(owner.values.get_mut()),
            beta: ColMut::from_slice_mut(state.beta.get_mut()),
        },
        owner.scratch.get_mut(),
    )
    .map_err(GevdFailure::Schur)?;
    let form = ComplexPencilView {
        aa: MatRef::from_column_major_slice(owner.matrix.get(), n, n),
        bb: MatRef::from_column_major_slice(owner.mass.as_ref().expect("B").get(), n, n),
        alpha: ColRef::from_slice(owner.values.get()),
        beta: ColRef::from_slice(state.beta.get()),
    };
    let stack = MemStack::new(owner.scratch.get_mut());
    let (mut basis_storage, stack) = faer::linalg::temp_mat_zeroed::<Complex64, _, _>(n, 1, stack);
    let (mut product_storage, _) = faer::linalg::temp_mat_zeroed::<Complex64, _, _>(n, 1, stack);
    let basis = basis_storage
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .expect("contiguous scratch")
        .as_slice_mut();
    let product = product_storage
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .expect("contiguous scratch")
        .as_slice_mut();
    complex_eigenvectors_in_place(
        &form,
        MatMut::from_column_major_slice_mut(owner.right.get_mut(), n, n),
        false,
        basis,
        product,
    )
    .map_err(GevdFailure::Eigenvectors)?;
    if let Some(left) = owner.left.as_mut() {
        complex_eigenvectors_in_place(
            &form,
            MatMut::from_column_major_slice_mut(left.get_mut(), n, n),
            true,
            basis,
            product,
        )
        .map_err(GevdFailure::Eigenvectors)?;
    }
    if owner
        .values
        .get()
        .iter()
        .chain(state.beta.get())
        .chain(owner.right.get())
        .chain(owner.left.iter().flat_map(Buffer::get))
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(GevdFailure::NonFiniteFactors);
    }
    Ok(())
}
