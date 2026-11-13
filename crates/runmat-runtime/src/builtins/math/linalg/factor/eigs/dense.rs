//! Exact full-EVD subset projection. No all-factor output/regather mirror.
use super::eig::work::Buffer;
use super::*;
use crate::builtins::math::linalg::ops::dense_evd::{Eigenpairs, PreparedEvd, Problem, Vectors};
use crate::context::RuntimeContext;

pub(super) struct Selection {
    slots: Buffer<(usize, usize)>,
    len: usize,
}
impl Selection {
    fn index(&self, i: usize) -> usize {
        self.slots.get()[i].0
    }
}
pub(super) enum Cholesky {
    Identity,
    Permuted(runmat_value::ExactArrayStorage<usize>),
}

// Preserve the old full eig publication boundary without materializing its
// eigenvalue tensor: Single rounding and all-real projection precede subset
// ordering (including close shifts and stable ties).
struct Spectrum<'a> {
    values: &'a [Complex64],
    dtype: NumericDType,
    real: bool,
}
impl Spectrum<'_> {
    fn value(&self, i: usize) -> Complex64 {
        let value = self.values[i];
        let cast = |v: f64| {
            if self.dtype == NumericDType::F32 {
                f64::from(v as f32)
            } else {
                v
            }
        };
        Complex64::new(cast(value.re), if self.real { 0. } else { cast(value.im) })
    }
}
async fn select(
    values: &Spectrum<'_>,
    k: usize,
    sigma: Sigma,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Selection> {
    let n = values.values.len();
    let k = k.min(n);
    let mut slots = Buffer::<(usize, usize)>::new(n, runtime).await?;
    for (i, slot) in slots.get_mut().iter_mut().enumerate() {
        *slot = (i, i);
    }
    let metric = |i: usize| match sigma {
        Sigma::LargestAbs => -values.value(i).norm(),
        Sigma::SmallestAbs => values.value(i).norm(),
        Sigma::LargestReal => -values.value(i).re,
        Sigma::SmallestReal => values.value(i).re,
        Sigma::LargestImag => -values.value(i).im,
        Sigma::SmallestImag => values.value(i).im,
        Sigma::Near(target) => (values.value(i) - target).norm(),
        Sigma::BothEndsReal => values.value(i).re,
        Sigma::BothEndsImag => values.value(i).im,
    };
    // Same stable-source tie policy as the original stable sort, without its
    // hidden temporary allocation. The sort is an indivisible CPU boundary.
    slots.check(runtime)?;
    slots.get_mut().sort_unstable_by(|&(a, oa), &(b, ob)| {
        total_cmp(metric(a), metric(b)).then_with(|| oa.cmp(&ob))
    });
    slots.check(runtime)?;
    if matches!(sigma, Sigma::BothEndsReal | Sigma::BothEndsImag) {
        let mut selected = Buffer::<(usize, usize)>::new(k, runtime).await?;
        let low = k / 2;
        for (i, slot) in selected.get_mut().iter_mut().enumerate() {
            *slot = (
                slots.get()[if i < low { i } else { n - 1 - (i - low) }].0,
                i,
            );
        }
        selected.check(runtime)?;
        selected.get_mut().sort_unstable_by(|&(a, oa), &(b, ob)| {
            let order = if matches!(sigma, Sigma::BothEndsReal) {
                total_cmp(values.value(a).re, values.value(b).re)
            } else {
                total_cmp(values.value(b).im.abs(), values.value(a).im.abs())
                    .then_with(|| total_cmp(values.value(b).im, values.value(a).im))
            };
            order.then_with(|| oa.cmp(&ob))
        });
        selected.check(runtime)?;
        // low + high = k <= n, so the two end ranges are disjoint. The
        // original dedup is therefore a no-op for this exact index domain.
        return Ok(Selection {
            slots: selected,
            len: k,
        });
    }
    Ok(Selection { slots, len: k })
}

fn precision(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(v) if v.numeric_dtype() == NumericDType::F32 => NumericDType::F32,
        Value::ComplexTensor(v) if v.numeric_dtype() == NumericDType::F32 => NumericDType::F32,
        Value::SparseTensor(v) if v.numeric_dtype() == Some(NumericDType::F32) => NumericDType::F32,
        _ => NumericDType::F64,
    }
}

async fn eigenpairs(
    request: &Request,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Eigenpairs> {
    let Some(cholesky) = &request.cholesky else {
        return eig::host::compute(runtime, &request.a, request.b.as_ref(), false, true)
            .await
            .map_err(with_context);
    };
    let b = request
        .b
        .as_ref()
        .expect("Cholesky option requires generalized B");
    let n = matrix_order(&request.a)?;
    let mut prepared = PreparedEvd::new(n, Problem::Generalized, Vectors::Right, runtime)
        .await
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    eig::host::fill(&request.a, &mut prepared, false, runtime)
        .await
        .map_err(with_context)?;
    let mut inverse = Buffer::<usize>::new(n, runtime).await?;
    for (i, slot) in inverse.get_mut().iter_mut().enumerate() {
        *slot = i;
    }
    if let Cholesky::Permuted(permutation) = cholesky {
        for (new, &old) in permutation.iter().enumerate() {
            inverse.get_mut()[old - 1] = new;
        }
    }
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    let mut work = 0u64;
    let mut all_real = true;
    for col in 0..n {
        for row in 0..n {
            let source_row = inverse.get()[row];
            let source_col = inverse.get()[col];
            let mut sum = Complex64::new(0., 0.);
            for k in 0..n {
                sum += eig::host::coefficient(b, k + source_row * n).conj()
                    * eig::host::coefficient(b, k + source_col * n);
                work += 1;
                if work == 4096 {
                    if let Some(budget) = &mut budget {
                        budget
                            .checkpoint(work)
                            .await
                            .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                    }
                    prepared
                        .check_owner(runtime)
                        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
                    work = 0;
                }
            }
            all_real &= sum.im.abs() <= 1e-12;
            prepared.mass_mut().expect("generalized workspace")[row + col * n] = sum;
        }
    }
    if let Some(budget) = &mut budget {
        budget
            .checkpoint(work)
            .await
            .map_err(|e| e.into_runtime_error_in_context(runtime))?;
    }
    inverse.check(runtime)?;
    prepared
        .check_owner(runtime)
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    // Preserve the original Cholesky-to-B language conversion: an entirely
    // nearly-real R'*R is published as real Double before generalized eig.
    if all_real {
        for value in prepared.mass_mut().expect("generalized workspace") {
            value.im = 0.;
        }
    }
    prepared
        .compute(runtime)
        .await
        .map_err(|e| with_context(eig::host::error(runtime, e)))
}

pub(super) async fn compute(
    request: Request,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<SubsetEval> {
    let dtype = if request.b.is_none() {
        precision(&request.a)
    } else if request.cholesky.is_none()
        && precision(&request.a) == NumericDType::F32
        && precision(request.b.as_ref().unwrap()) == NumericDType::F32
    {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let pairs = eigenpairs(&request, runtime).await?;
    let spectrum = Spectrum {
        values: pairs.values(),
        dtype,
        real: pairs
            .values()
            .iter()
            .all(|value| value.im.is_nan() || value.im.abs() <= 1e-12),
    };
    let selected = select(&spectrum, request.k, request.sigma, runtime)
        .await
        .map_err(with_context)?;
    let k = selected.len;
    let n = pairs.order();
    pairs
        .check_owner(runtime)
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    use crate::builtins::math::linalg::ops::dense_evd::publication::{
        matrix_with_projection as matrix, RealProjection,
    };
    let values = matrix(runtime, dtype, k, 1, RealProjection::EigsSubset, |i| {
        spectrum.value(selected.index(i))
    })
    .await
    .map_err(with_context)?;
    pairs
        .check_owner(runtime)
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    let diagonal = matrix(runtime, dtype, k, k, RealProjection::EigsSubset, |i| {
        if i % k == i / k {
            spectrum.value(selected.index(i / k))
        } else {
            Complex64::new(0., 0.)
        }
    })
    .await
    .map_err(with_context)?;
    pairs
        .check_owner(runtime)
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    let right = Spectrum {
        values: pairs.right(),
        dtype,
        real: pairs
            .right()
            .iter()
            .all(|value| value.im.is_nan() || value.im.abs() <= 1e-12),
    };
    let vectors = publish_columns(runtime, dtype, n, k, |row, col| {
        right.value(row + selected.index(col) * n)
    })
    .await?;
    pairs
        .check_owner(runtime)
        .map_err(|e| with_context(eig::host::error(runtime, e)))?;
    selected.slots.check(runtime).map_err(with_context)?;
    Ok(SubsetEval {
        values,
        vectors,
        diagonal,
    })
}

pub(super) async fn publish_columns(
    runtime: Option<&RuntimeContext>,
    dtype: NumericDType,
    rows: usize,
    columns: usize,
    read: impl Fn(usize, usize) -> Complex64,
) -> BuiltinResult<Value> {
    crate::builtins::math::linalg::ops::dense_evd::publication::matrix_with_projection(
        runtime,
        dtype,
        rows,
        columns,
        crate::builtins::math::linalg::ops::dense_evd::publication::RealProjection::EigsSubset,
        |i| read(i % rows, i / rows),
    )
    .await
    .map_err(with_context)
}
