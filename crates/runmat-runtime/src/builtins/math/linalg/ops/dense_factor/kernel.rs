use super::*;
fn workspace_requirement<E: ComplexField>(order: usize) -> StackReq {
    let params: Spec<factor::PartialPivLuParams, E> = Default::default();
    StackReq::any_of(&[
        factor::lu_in_place_scratch::<usize, E>(order, order, Par::Seq, params),
        solve::solve_in_place_scratch::<usize, E>(order, 1, Par::Seq),
    ])
}
pub(super) fn quote<T: FaerFactorScalar>(
    order: usize,
) -> Result<(HostAllocationPlan, StackReq), HostAllocationError> {
    let count = order
        .checked_mul(order)
        .ok_or(HostAllocationError::Overflow)?;
    let scratch = workspace_requirement::<T::Element>(order);
    let layout = scratch
        .layout()
        .map_err(|_| HostAllocationError::Overflow)?;
    let mut bytes = 0u64;
    for next in [
        HostTypedWorkspace::<T::Element>::quoted_bytes(count)?,
        HostTypedWorkspace::<usize>::quoted_bytes(order)?,
        HostTypedWorkspace::<usize>::quoted_bytes(order)?,
        HostTypedWorkspace::<T::Element>::quoted_bytes(order)?,
        AlignedHostWorkspace::quoted_bytes(layout)?,
    ] {
        bytes = bytes
            .checked_add(next)
            .ok_or(HostAllocationError::Overflow)?;
    }
    let metadata =
        HostAllocationPlan::retained_metadata_bytes(std::mem::size_of::<DenseFactor<T>>() as u64)?;
    Ok((HostAllocationPlan::new(bytes, metadata)?, scratch))
}

pub(super) async fn unbounded_vec<T: Copy + Default>(
    len: usize,
    runtime: Option<&RuntimeContext>,
) -> Result<Vec<T>, DenseFactorError> {
    let mut work = super::work::Work::new(runtime, None)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DenseFactorError::AllocationFailed)?;
    while values.len() < len {
        values.push(T::default());
        work.tick().await?;
    }
    work.flush().await?;
    Ok(values)
}

pub(super) fn factor_body<E: ComplexField + HostTypedElement>(
    values: &mut [E],
    perm: &mut [usize],
    perm_inv: &mut [usize],
    scratch: &mut [std::mem::MaybeUninit<u8>],
    order: usize,
) {
    let params: Spec<factor::PartialPivLuParams, E> = Default::default();
    let stack = MemStack::new(scratch);
    factor::lu_in_place(
        MatMut::from_column_major_slice_mut(values, order, order),
        perm,
        perm_inv,
        Par::Seq,
        stack,
        params,
    );
}

pub(super) fn solve_body<E: ComplexField + HostTypedElement>(
    lu: &[E],
    perm: &[usize],
    perm_inv: &[usize],
    rhs: &mut [E],
    scratch: &mut [std::mem::MaybeUninit<u8>],
    order: usize,
) {
    let permutation = PermRef::new_checked(perm, perm_inv, order);
    let stack = MemStack::new(scratch);
    solve::solve_in_place(
        MatRef::from_column_major_slice(lu, order, order),
        MatRef::from_column_major_slice(lu, order, order),
        permutation,
        MatMut::from_column_major_slice_mut(rhs, order, 1),
        Par::Seq,
        stack,
    );
}
