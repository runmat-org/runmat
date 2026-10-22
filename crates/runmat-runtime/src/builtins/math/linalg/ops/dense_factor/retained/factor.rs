//! Existing low-level faer kernels over newly admitted canonical dense owners.
use super::storage::{
    index_bytes, matrix_bytes, run_owned, scratch_bytes, DenseScalar, IndexOwner, Scratch,
};
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::SelectedTriangle;
use faer::linalg::lu::partial_pivoting::factor as lu_factor;
use faer::linalg::{cholesky::llt::factor as chol, qr::col_pivoting::factor as qr};
use runmat_value::{ComplexElement, DenseOrthogonalFactorData, FactorPermutation};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg) struct DenseFactorOptions {
    pub kind: NumericFactorKind,
    pub rank_tolerance: Option<f64>,
    pub triangle: Option<SelectedTriangle>,
}
pub(in crate::builtins::math::linalg) async fn factor_dense(
    value: &Value,
    options: DenseFactorOptions,
    runtime: &RuntimeContext,
) -> Result<NumericFactorData, DenseFactorError> {
    if let Some(t) = options.rank_tolerance {
        if t.is_nan() || t < 0.0 {
            return Err(DenseFactorError::InvalidData(
                "rank tolerance must be nonnegative".into(),
            ));
        }
    }
    match value {
        Value::Tensor(t) if t.as_f32_slice().is_some() => {
            factor_typed::<f32>(value, t.shape(), options, runtime).await
        }
        Value::Tensor(t) if t.as_f64_slice().is_some() => {
            factor_typed::<f64>(value, t.shape(), options, runtime).await
        }
        Value::ComplexTensor(t) if t.as_f32_slice().is_some() => {
            factor_typed::<ComplexElement<f32>>(value, t.shape(), options, runtime).await
        }
        Value::ComplexTensor(t) if t.as_f64_slice().is_some() => {
            factor_typed::<ComplexElement<f64>>(value, t.shape(), options, runtime).await
        }
        _ => Err(DenseFactorError::UnsupportedClass),
    }
}
async fn factor_typed<T: DenseScalar>(
    value: &Value,
    shape: &[usize],
    options: DenseFactorOptions,
    runtime: &RuntimeContext,
) -> Result<NumericFactorData, DenseFactorError> {
    if shape.iter().skip(2).any(|d| *d != 1) {
        return Err(DenseFactorError::InvalidShape);
    }
    let m = shape.first().copied().unwrap_or(1);
    let n = shape.get(1).copied().unwrap_or(1);
    if matches!(
        options.kind,
        NumericFactorKind::Lu | NumericFactorKind::Cholesky
    ) && m != n
    {
        return Err(DenseFactorError::InvalidShape);
    }
    if m == 0 || n == 0 {
        let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        return validate(
            NumericFactorCandidate::Empty {
                kind: options.kind,
                rows: m,
                columns: n,
                class: T::storage_class(),
            },
            &mut work::Work::new(Some(runtime), receipt.as_ref())?,
        )
        .await;
    }
    let source = T::source(value).ok_or(DenseFactorError::UnsupportedClass)?;
    if m.checked_mul(n) != Some(source.len()) {
        return Err(DenseFactorError::InvalidShape);
    }
    if matches!(options.kind, NumericFactorKind::Qr | NumericFactorKind::Cod) {
        let first = orthogonal::<T>(m, n, |i| source[i], options.rank_tolerance, runtime).await?;
        let completion =
            if options.kind == NumericFactorKind::Cod && first.rank > 0 && first.rank < n {
                let rank = first.rank;
                let packed = T::elements(&first.packed).expect("native packed factor");
                let second = orthogonal::<T>(
                    n,
                    rank,
                    |i| {
                        let row = i % n;
                        let col = i / n;
                        if row < col {
                            T::default()
                        } else {
                            packed[row * m + col].conjugated()
                        }
                    },
                    Some(0.0),
                    runtime,
                )
                .await?;
                if second.rank != rank {
                    return Err(DenseFactorError::NonFinite);
                }
                Some(second)
            } else {
                None
            };
        let candidate = if options.kind == NumericFactorKind::Qr {
            NumericFactorCandidate::DenseQr(first)
        } else {
            NumericFactorCandidate::DenseCod { first, completion }
        };
        let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        return validate(
            candidate,
            &mut work::Work::new(Some(runtime), receipt.as_ref())?,
        )
        .await;
    }
    square::<T>(source, n, options, runtime).await
}

async fn orthogonal<T: DenseScalar>(
    m: usize,
    n: usize,
    mut read: impl FnMut(usize) -> T,
    tolerance: Option<f64>,
    runtime: &RuntimeContext,
) -> Result<DenseOrthogonalFactorData, DenseFactorError> {
    let k = m.min(n);
    let block = qr::recommended_block_size::<T::Element>(m, n);
    let params: Spec<qr::ColPivQrParams, T::Element> = Default::default();
    let req = qr::qr_in_place_scratch::<usize, T::Element>(m, n, block, Par::Seq, params);
    let output = matrix_bytes::<T>([m, n])?
        .checked_add(matrix_bytes::<T>([block, k])?)
        .and_then(|v| v.checked_add(index_bytes(n).ok()?.checked_mul(2)?))
        .ok_or(HostAllocationError::Overflow)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch_bytes(req)?, output)?,
    )?;
    let mut work = work::Work::new(Some(runtime), receipt.as_ref())?;
    let mut packed = T::allocate([m, n], receipt.as_ref(), runtime).await?;
    let mut all_zero = true;
    for (i, target) in T::elements_mut(&mut packed)
        .expect("fresh native owner")
        .iter_mut()
        .enumerate()
    {
        let v = read(i);
        if !crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseScalar::finite(v) {
            return Err(DenseFactorError::NonFinite);
        }
        *target = v;
        all_zero &= v == T::default();
        work.tick().await?;
    }
    let mut coefficients = T::allocate([block, k], receipt.as_ref(), runtime).await?;
    let mut forward = IndexOwner::new(n, receipt.as_ref(), runtime).await?;
    let mut inverse = IndexOwner::new(n, receipt.as_ref(), runtime).await?;
    if all_zero {
        // faer 0.24's pivoted QR rescales by the largest column norm's
        // reciprocal even when every column is zero. Publish the exact
        // identity-reflector representation rather than form 0 * Inf.
        let values = T::elements_mut(&mut coefficients).expect("unique coefficients");
        for column in 0..k {
            values[column * block + column % block] = T::from_ieee_parts(f64::INFINITY, 0.0);
            work.tick().await?;
        }
        for index in 0..n {
            forward.as_mut_slice()[index] = index;
            inverse.as_mut_slice()[index] = index;
            work.tick().await?;
        }
        work.flush().await?;
        return Ok(DenseOrthogonalFactorData {
            packed,
            coefficients,
            permutation: FactorPermutation {
                forward: forward.finish()?,
                inverse: inverse.finish()?,
            },
            rank: 0,
        });
    }
    let mut scratch = Scratch::new(req, receipt.as_ref(), runtime)?;
    work.flush().await?;
    let (packed, coefficients, forward, inverse) =
        run_owned(runtime, receipt.as_ref(), move || {
            qr::qr_in_place(
                MatMut::from_column_major_slice_mut(
                    T::faer_mut(T::elements_mut(&mut packed).expect("unique packed")),
                    m,
                    n,
                ),
                MatMut::from_column_major_slice_mut(
                    T::faer_mut(T::elements_mut(&mut coefficients).expect("unique coefficients")),
                    block,
                    k,
                ),
                forward.as_mut_slice(),
                inverse.as_mut_slice(),
                Par::Seq,
                MemStack::new(scratch.bytes()),
                Default::default(),
            );
            (packed, coefficients, forward, inverse)
        })
        .await?;
    let values = T::elements(&packed).expect("native packed");
    let mut rank = 0;
    let magnitude = |v: T| {
        let (re, im) = v.parts();
        let scale = re.abs().max(im.abs());
        let unit = if scale == 0.0 {
            0.0
        } else {
            (re / scale).hypot(im / scale)
        };
        (scale, unit)
    };
    let (base, base_unit) = magnitude(values[0]);
    for i in 0..k {
        let (scale, unit) = magnitude(values[i * m + i]);
        let retained = if let Some(tol) = tolerance {
            scale > tol / unit
        } else {
            base != 0.0
                && (scale / base) * (unit / base_unit)
                    > <T as DenseScalar>::epsilon() * (m.max(n) as f64)
        };
        if rank == i && retained {
            rank += 1;
        }
        work.tick().await?;
    }
    work.flush().await?;
    Ok(DenseOrthogonalFactorData {
        packed,
        coefficients,
        permutation: FactorPermutation {
            forward: forward.finish()?,
            inverse: inverse.finish()?,
        },
        rank,
    })
}

async fn square<T: DenseScalar>(
    source: &[T],
    n: usize,
    options: DenseFactorOptions,
    runtime: &RuntimeContext,
) -> Result<NumericFactorData, DenseFactorError> {
    let lu = options.kind == NumericFactorKind::Lu;
    let req = if lu {
        lu_factor::lu_in_place_scratch::<usize, T::Element>(n, n, Par::Seq, Default::default())
    } else {
        chol::cholesky_in_place_scratch::<T::Element>(n, Par::Seq, Default::default())
    };
    let output = matrix_bytes::<T>([n, n])?
        .checked_add(if lu {
            index_bytes(n)?
                .checked_mul(2)
                .ok_or(HostAllocationError::Overflow)?
        } else {
            0
        })
        .ok_or(HostAllocationError::Overflow)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch_bytes(req)?, output)?,
    )?;
    let mut work = work::Work::new(Some(runtime), receipt.as_ref())?;
    let mut packed = T::allocate([n, n], receipt.as_ref(), runtime).await?;
    for (i, target) in T::elements_mut(&mut packed)
        .expect("fresh native owner")
        .iter_mut()
        .enumerate()
    {
        let row = i % n;
        let col = i / n;
        let v = if !lu && row < col {
            T::default()
        } else if !lu && options.triangle == Some(SelectedTriangle::Upper) {
            source[row * n + col].conjugated()
        } else {
            source[i]
        };
        if !lu && row == col && v.parts().1 != 0.0 {
            return Err(DenseFactorError::InvalidData(
                "Cholesky diagonal must be real".into(),
            ));
        }
        if !crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseScalar::finite(v) {
            return Err(DenseFactorError::NonFinite);
        }
        *target = v;
        work.tick().await?;
    }
    let mut permutations = if lu {
        Some((
            IndexOwner::new(n, receipt.as_ref(), runtime).await?,
            IndexOwner::new(n, receipt.as_ref(), runtime).await?,
        ))
    } else {
        None
    };
    let mut scratch = Scratch::new(req, receipt.as_ref(), runtime)?;
    work.flush().await?;
    let (mut packed, permutations) = run_owned(runtime, receipt.as_ref(), move || {
        let a = MatMut::from_column_major_slice_mut(
            T::faer_mut(T::elements_mut(&mut packed).expect("unique packed")),
            n,
            n,
        );
        if let Some((p, inv)) = permutations.as_mut() {
            lu_factor::lu_in_place(
                a,
                p.as_mut_slice(),
                inv.as_mut_slice(),
                Par::Seq,
                MemStack::new(scratch.bytes()),
                Default::default(),
            );
        } else {
            chol::cholesky_in_place(
                a,
                Default::default(),
                Par::Seq,
                MemStack::new(scratch.bytes()),
                Default::default(),
            )
            .map_err(|_| DenseFactorError::NotPositiveDefinite)?;
        }
        Ok::<_, DenseFactorError>((packed, permutations))
    })
    .await??;
    if !lu {
        // The LLT kernel owns only the lower triangle; SIMD updates may
        // overwrite ignored upper lanes. Its complex diagonal is real by
        // definition, even if fused updates leave a finite imaginary residue.
        for (index, value) in T::elements_mut(&mut packed)
            .expect("unique lower")
            .iter_mut()
            .enumerate()
        {
            let row = index % n;
            let column = index / n;
            if row < column {
                *value = T::default();
            } else if row == column {
                let (real, imag) = value.parts();
                if !real.is_finite() || !imag.is_finite() || real <= 0.0 {
                    return Err(DenseFactorError::NotPositiveDefinite);
                }
                *value = T::from_parts(real, 0.0).ok_or(DenseFactorError::NonFinite)?;
            }
            work.tick().await?;
        }
    }
    let candidate = if let Some((p, inv)) = permutations {
        NumericFactorCandidate::DenseLu {
            packed,
            rows: FactorPermutation {
                forward: p.finish()?,
                inverse: inv.finish()?,
            },
        }
    } else {
        NumericFactorCandidate::DenseCholesky { lower: packed }
    };
    validate(candidate, &mut work).await
}
