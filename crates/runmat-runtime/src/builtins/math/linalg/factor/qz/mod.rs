//! Generalized Schur factorization of a dense matrix pencil.

mod algorithm;
mod allocation;
mod eigvals;
mod eigvectors;

// Borrowed shared owners, not an additional QZ/eigenvector implementation.
#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
pub(in crate::builtins::math::linalg) use algorithm::{
    complex_qz_optional_in_place, complex_qz_scratch, ComplexQzOptionalView,
};
#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
pub(in crate::builtins::math::linalg) use eigvectors::{
    complex_eigenvectors_in_place, ComplexPencilView,
};
mod ordqz;
#[cfg(target_arch = "wasm32")]
pub(crate) use ordqz::__runmat_wasm_register_builtin_ordqz_builtin;
mod real_eigenvectors;
mod real_reorder;
mod reorder;
pub(crate) mod riccati;
#[cfg(test)]
mod tests;
mod work;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use faer::{c64, Mat};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, ExactHostComplexTensorAllocation, ExactHostTensorAllocation, NumericDType,
    Tensor, Value,
};

use crate::context::RuntimeContext;
use algorithm::{ComplexQz, RealQz};

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("qz")
        .with_identifier("RunMat:qz:InvalidInput")
        .build()
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("qz")
        .with_identifier("RunMat:qz:Numerical")
        .build()
}

fn ordeig_invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("ordeig")
        .with_identifier("RunMat:ordeig:InvalidInput")
        .build()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Complex,
    Real,
}

fn parse_mode(option: Option<&Value>) -> BuiltinResult<Mode> {
    let Some(option) = option else {
        return Ok(Mode::Complex);
    };
    let text = match option {
        Value::String(runmat_value::StringElement::Text(text)) => text.as_str().to_owned(),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned()
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?
        }
        _ => return Err(invalid("qz mode must be 'complex' or 'real'")),
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "complex" => Ok(Mode::Complex),
        "real" => Ok(Mode::Real),
        _ => Err(invalid("qz mode must be 'complex' or 'real'")),
    }
}

fn coefficient_shape(value: &Value) -> BuiltinResult<(usize, NumericDType)> {
    let (shape, dtype) = match value {
        Value::Tensor(t) => (t.shape(), t.numeric_dtype()),
        Value::ComplexTensor(t) => (t.shape(), t.numeric_dtype()),
        Value::GpuTensor(h) => {
            if runmat_accelerate_api::handle_is_logical(h)
                || runmat_accelerate_api::handle_integer_type(h).is_some()
            {
                return Err(invalid("qz requires floating-point resident coefficients"));
            }
            (
                h.shape(),
                match runmat_accelerate_api::handle_precision(h) {
                    Some(runmat_accelerate_api::ProviderPrecision::F32) => NumericDType::F32,
                    Some(runmat_accelerate_api::ProviderPrecision::F64) => NumericDType::F64,
                    None => {
                        return Err(invalid("qz requires floating-point resident coefficients"))
                    }
                },
            )
        }
        Value::Num(_) | Value::Complex(_, _) => return Ok((1, NumericDType::F64)),
        _ => return Err(invalid("qz requires dense single or double matrices")),
    };
    if shape.len() != 2
        || shape[0] != shape[1]
        || !matches!(dtype, NumericDType::F32 | NumericDType::F64)
    {
        return Err(invalid(
            "qz requires square dense single or double matrices",
        ));
    }
    Ok((shape[0], dtype))
}

fn complex_matrix(value: &Value, n: usize) -> BuiltinResult<Mat<c64>> {
    let matrix = match value {
        Value::Tensor(t) => {
            if let Some(values) = t.as_f64_slice() {
                Mat::from_fn(n, n, |i, j| c64::new(values[i + j * n], 0.0))
            } else if let Some(values) = t.as_f32_slice() {
                Mat::from_fn(n, n, |i, j| c64::new(values[i + j * n] as f64, 0.0))
            } else {
                return Err(invalid("qz requires floating-point coefficients"));
            }
        }
        Value::ComplexTensor(t) => {
            if let Some(values) = t.as_f64_slice() {
                Mat::from_fn(n, n, |i, j| {
                    let v = values[i + j * n];
                    c64::new(v.0, v.1)
                })
            } else if let Some(values) = t.as_f32_slice() {
                Mat::from_fn(n, n, |i, j| {
                    let v = values[i + j * n];
                    c64::new(v.0 as f64, v.1 as f64)
                })
            } else {
                return Err(invalid("qz requires floating-point coefficients"));
            }
        }
        Value::Num(v) => Mat::from_fn(1, 1, |_, _| c64::new(*v, 0.0)),
        Value::Complex(re, im) => Mat::from_fn(1, 1, |_, _| c64::new(*re, *im)),
        _ => return Err(invalid("qz requires dense single or double matrices")),
    };
    Ok(matrix)
}

fn real_matrix(value: &Value, n: usize) -> BuiltinResult<Mat<f64>> {
    match value {
        Value::Tensor(t) => {
            if let Some(values) = t.as_f64_slice() {
                Ok(Mat::from_fn(n, n, |i, j| values[i + j * n]))
            } else if let Some(values) = t.as_f32_slice() {
                Ok(Mat::from_fn(n, n, |i, j| values[i + j * n] as f64))
            } else {
                Err(invalid("real QZ requires floating-point coefficients"))
            }
        }
        Value::Num(v) => Ok(Mat::from_fn(1, 1, |_, _| *v)),
        _ => Err(invalid("real QZ requires real matrices")),
    }
}

fn output(
    name: &'static str,
    matrix: &Mat<c64>,
    dtype: NumericDType,
    left_adjoint: bool,
    allocation: Option<(
        &RuntimeContext,
        &crate::context::host_allocation::HostAllocationReceipt,
    )>,
) -> BuiltinResult<Value> {
    let n = matrix.nrows();
    let get = |i, j| {
        let value = if left_adjoint {
            matrix[(j, i)]
        } else {
            matrix[(i, j)]
        };
        if left_adjoint {
            c64::new(value.re, -value.im)
        } else {
            value
        }
    };
    let shape = vec![n, n];
    if let Some((runtime, receipt)) = allocation {
        let tensor = receipt
            .with_output_allocation(runtime, |permit| {
                let mut exact = ExactHostComplexTensorAllocation::new(dtype, &shape, permit)?;
                for j in 0..n {
                    for i in 0..n {
                        let value = get(i, j);
                        exact.set(i + j * n, value.re, value.im)?;
                    }
                }
                exact.into_tensor()
            })
            .map_err(|source| work::admit_error(name, source))?;
        return Ok(Value::ComplexTensor(tensor));
    }
    if dtype == NumericDType::F32 {
        ComplexTensor::from_f32(
            (0..n)
                .flat_map(|j| {
                    (0..n).map(move |i| {
                        let v = get(i, j);
                        (v.re as f32, v.im as f32)
                    })
                })
                .collect(),
            shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|error| numerical(error.to_string()))
    } else {
        ComplexTensor::new(
            (0..n)
                .flat_map(|j| {
                    (0..n).map(move |i| {
                        let v = get(i, j);
                        (v.re, v.im)
                    })
                })
                .collect(),
            shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|error| numerical(error.to_string()))
    }
}

fn publish(
    name: &'static str,
    form: ComplexQz,
    dtype: NumericDType,
    vectors: Option<eigvectors::Eigenvectors>,
    allocation: Option<(
        &RuntimeContext,
        &crate::context::host_allocation::HostAllocationReceipt,
    )>,
) -> BuiltinResult<Value> {
    let count = crate::output_count::current_output_count();
    let requested = count.unwrap_or(1);
    let mut outputs = Vec::with_capacity(requested);
    if requested >= 1 {
        outputs.push(output(name, &form.aa, dtype, false, allocation)?);
    }
    if requested >= 2 {
        outputs.push(output(name, &form.bb, dtype, false, allocation)?);
    }
    if requested >= 3 {
        outputs.push(output(name, &form.left, dtype, true, allocation)?);
    }
    if requested >= 4 {
        outputs.push(output(name, &form.right, dtype, false, allocation)?);
    }
    if let Some(vectors) = vectors {
        if requested >= 5 {
            outputs.push(output(name, &vectors.right, dtype, false, allocation)?);
        }
        if requested >= 6 {
            let left = vectors
                .left
                .ok_or_else(|| numerical("left generalized eigenvectors were not computed"))?;
            outputs.push(output(name, &left, dtype, false, allocation)?);
        }
    }
    match count {
        Some(_) => Ok(Value::output_list(outputs)),
        None => Ok(outputs.remove(0)),
    }
}

fn real_output(
    name: &'static str,
    matrix: &Mat<f64>,
    dtype: NumericDType,
    transpose: bool,
    allocation: Option<(
        &RuntimeContext,
        &crate::context::host_allocation::HostAllocationReceipt,
    )>,
) -> BuiltinResult<Value> {
    let n = matrix.nrows();
    let shape = vec![n, n];
    if let Some((runtime, receipt)) = allocation {
        let tensor = receipt
            .with_output_allocation(runtime, |permit| {
                let mut exact = ExactHostTensorAllocation::new(dtype, &shape, permit)?;
                match dtype {
                    NumericDType::F32 => {
                        let values = exact.as_f32_slice_mut().expect("admitted single output");
                        for j in 0..n {
                            for i in 0..n {
                                values[i + j * n] = if transpose {
                                    matrix[(j, i)]
                                } else {
                                    matrix[(i, j)]
                                } as f32;
                            }
                        }
                    }
                    NumericDType::F64 => {
                        let values = exact.as_f64_slice_mut().expect("admitted double output");
                        for j in 0..n {
                            for i in 0..n {
                                values[i + j * n] = if transpose {
                                    matrix[(j, i)]
                                } else {
                                    matrix[(i, j)]
                                };
                            }
                        }
                    }
                    _ => unreachable!("admitted floating output"),
                }
                exact.into_tensor()
            })
            .map_err(|source| work::admit_error(name, source))?;
        return Ok(Value::Tensor(tensor));
    }
    if dtype == NumericDType::F32 {
        Tensor::from_f32(
            (0..n)
                .flat_map(|j| {
                    (0..n).map(move |i| {
                        if transpose {
                            matrix[(j, i)] as f32
                        } else {
                            matrix[(i, j)] as f32
                        }
                    })
                })
                .collect(),
            shape,
        )
        .map(Value::Tensor)
        .map_err(|error| numerical(error.to_string()))
    } else {
        Tensor::new(
            (0..n)
                .flat_map(|j| {
                    (0..n).map(move |i| {
                        if transpose {
                            matrix[(j, i)]
                        } else {
                            matrix[(i, j)]
                        }
                    })
                })
                .collect(),
            shape,
        )
        .map(Value::Tensor)
        .map_err(|error| numerical(error.to_string()))
    }
}

fn publish_real(
    name: &'static str,
    form: RealQz,
    dtype: NumericDType,
    vectors: Option<real_eigenvectors::RealEigenvectors>,
    allocation: Option<(
        &RuntimeContext,
        &crate::context::host_allocation::HostAllocationReceipt,
    )>,
) -> BuiltinResult<Value> {
    let count = crate::output_count::current_output_count();
    let requested = count.unwrap_or(1);
    let mut outputs = Vec::with_capacity(requested);
    if requested >= 1 {
        outputs.push(real_output(name, &form.aa, dtype, false, allocation)?);
    }
    if requested >= 2 {
        outputs.push(real_output(name, &form.bb, dtype, false, allocation)?);
    }
    if requested >= 3 {
        outputs.push(real_output(name, &form.left, dtype, true, allocation)?);
    }
    if requested >= 4 {
        outputs.push(real_output(name, &form.right, dtype, false, allocation)?);
    }
    if let Some(vectors) = vectors {
        if requested >= 5 {
            outputs.push(real_output(name, &vectors.right, dtype, false, allocation)?);
        }
        if requested >= 6 {
            let left = vectors
                .left
                .ok_or_else(|| numerical("left generalized eigenvectors were not computed"))?;
            outputs.push(real_output(name, &left, dtype, false, allocation)?);
        }
    }
    match count {
        Some(_) => Ok(Value::output_list(outputs)),
        None => Ok(outputs.remove(0)),
    }
}

#[runtime_builtin(
    name = "qz",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::qz",
    context = true
)]
async fn qz_builtin(
    runtime: Option<RuntimeContext>,
    a: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !(1..=2).contains(&rest.len()) {
        return Err(invalid("qz requires a matrix pair and optional mode"));
    }
    let b = &rest[0];
    let mode = parse_mode(rest.get(1))?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 6) {
        return Err(invalid("qz has at most six outputs"));
    }
    let (n, dtype_a) = coefficient_shape(&a)?;
    let (m, dtype_b) = coefficient_shape(b)?;
    if n != m {
        return Err(invalid("qz matrix dimensions must agree"));
    }
    let dtype = if dtype_a == NumericDType::F64 || dtype_b == NumericDType::F64 {
        NumericDType::F64
    } else {
        NumericDType::F32
    };
    if matches!(&a, Value::GpuTensor(_)) || matches!(b, Value::GpuTensor(_)) {
        return Err(invalid(
            "qz resident inputs require a supported provider execution path",
        ));
    }
    if mode == Mode::Real
        && (!matches!(&a, Value::Tensor(_) | Value::Num(_))
            || !matches!(b, Value::Tensor(_) | Value::Num(_)))
    {
        return Err(invalid("real QZ requires real matrices"));
    }
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    let plan = allocation::factor_plan(n, dtype, mode, requested)
        .map_err(|source| work::admit_error("qz", source))?;
    let mut worker_work = work::QzWork::new(runtime.as_ref(), None, "qz");
    let arguments = std::iter::once(&a).chain(rest.iter()).collect::<Vec<_>>();
    if let Some(result) = worker_work
        .try_owned_browser_worker(
            &arguments,
            requested,
            crate::output_count::current_output_count().is_some(),
            plan,
            work::QzOutputRule {
                shape: [n, n],
                dtype,
                domain: match mode {
                    Mode::Real => work::QzOutputDomain::Real,
                    Mode::Complex => work::QzOutputDomain::Complex,
                },
            },
        )
        .await?
    {
        return Ok(result);
    }
    let receipt =
        crate::context::host_allocation::HostAllocationReceipt::admit(runtime.as_ref(), plan)
            .map_err(|source| work::admit_error("qz", source))?;
    let mut work = work::QzWork::new(runtime.as_ref(), receipt.as_ref(), "qz");
    work.checkpoint(0).await?;
    match mode {
        Mode::Complex => {
            let aa = complex_matrix(&a, n)?;
            let bb = complex_matrix(b, n)?;
            work.checkpoint(2).await?;
            let (form, vectors) = work
                .blocking(move || {
                    let form = algorithm::complex_qz(aa, bb)?;
                    let vectors = if requested >= 5 {
                        Some(eigvectors::complex_eigenvectors(&form, requested >= 6)?)
                    } else {
                        None
                    };
                    Ok::<_, &'static str>((form, vectors))
                })
                .await?
                .map_err(numerical)?;
            let result = publish(
                "qz",
                form,
                dtype,
                vectors,
                runtime.as_ref().zip(receipt.as_ref()),
            )?;
            work.checkpoint(u64::try_from(n.saturating_mul(n)).unwrap_or(u64::MAX))
                .await?;
            Ok(result)
        }
        Mode::Real => {
            let aa = real_matrix(&a, n)?;
            let bb = real_matrix(b, n)?;
            work.checkpoint(2).await?;
            let (form, vectors) = work
                .blocking(move || {
                    let form = algorithm::real_qz(aa, bb)?;
                    let vectors = if requested >= 5 {
                        Some(real_eigenvectors::real_eigenvectors(&form, requested >= 6)?)
                    } else {
                        None
                    };
                    Ok::<_, &'static str>((form, vectors))
                })
                .await?
                .map_err(numerical)?;
            let result = publish_real(
                "qz",
                form,
                dtype,
                vectors,
                runtime.as_ref().zip(receipt.as_ref()),
            )?;
            work.checkpoint(u64::try_from(n.saturating_mul(n)).unwrap_or(u64::MAX))
                .await?;
            Ok(result)
        }
    }
}

// The ordinary Schur owner dispatches its two-input form here. There is one
// public `ordeig` registration and one output-count/error authority.
pub(super) async fn generalized_ordeig(
    runtime: Option<&RuntimeContext>,
    aa: &Value,
    bb: &Value,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(ordeig_invalid("ordeig returns one output"));
    }
    let (n, dtype_a) = coefficient_shape(aa).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
    let (m, dtype_b) = coefficient_shape(bb).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
    if n != m {
        return Err(ordeig_invalid(
            "generalized Schur factor dimensions must agree",
        ));
    }
    if matches!(aa, Value::GpuTensor(_)) || matches!(bb, Value::GpuTensor(_)) {
        return Err(ordeig_invalid(
            "two-input ordeig requires host generalized Schur factors",
        ));
    }
    let dtype = if dtype_a == NumericDType::F64 || dtype_b == NumericDType::F64 {
        NumericDType::F64
    } else {
        NumericDType::F32
    };
    let real = matches!(
        (aa, bb),
        (
            Value::Tensor(_) | Value::Num(_),
            Value::Tensor(_) | Value::Num(_)
        )
    );
    let mode = if real { Mode::Real } else { Mode::Complex };
    let requested = crate::output_count::current_output_count() != Some(0);
    let plan = allocation::eigenvalues_plan(n, dtype, mode, requested)
        .map_err(|source| work::admit_error("ordeig", source))?;
    let mut worker_work = work::QzWork::new(runtime, None, "ordeig");
    if let Some(result) = worker_work
        .try_owned_browser_worker(
            &[aa, bb],
            usize::from(requested),
            crate::output_count::current_output_count().is_some(),
            plan,
            work::QzOutputRule {
                shape: [n, 1],
                dtype,
                domain: work::QzOutputDomain::Either,
            },
        )
        .await?
    {
        return Ok(result);
    }
    let receipt = crate::context::host_allocation::HostAllocationReceipt::admit(runtime, plan)
        .map_err(|source| work::admit_error("ordeig", source))?;
    let mut work = work::QzWork::new(runtime, receipt.as_ref(), "ordeig");
    work.checkpoint(0).await?;
    let eigenvalues = if real {
        let a = real_matrix(aa, n).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
        let b = real_matrix(bb, n).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
        work.blocking(move || eigvals::real_schur_values(&a, &b))
            .await?
            .map_err(ordeig_invalid)?
    } else {
        let a = complex_matrix(aa, n).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
        let b = complex_matrix(bb, n).map_err(|e| ordeig_invalid(e.message().to_owned()))?;
        work.blocking(move || eigvals::complex_schur_values(&a, &b))
            .await?
            .map_err(ordeig_invalid)?
    };
    work.checkpoint(u64::try_from(n).unwrap_or(u64::MAX))
        .await?;
    if !requested {
        return Ok(Value::output_list(Vec::new()));
    }
    let allocation = runtime.zip(receipt.as_ref());
    let shape = vec![n, 1];
    let output = if eigenvalues.iter().all(|&(_, im)| im == 0.0) {
        if let Some((runtime, receipt)) = allocation {
            let tensor = receipt
                .with_output_allocation(runtime, |permit| {
                    let mut exact = ExactHostTensorAllocation::new(dtype, &shape, permit)?;
                    match dtype {
                        NumericDType::F32 => {
                            for (slot, &(re, _)) in exact
                                .as_f32_slice_mut()
                                .expect("admitted single")
                                .iter_mut()
                                .zip(&eigenvalues)
                            {
                                *slot = re as f32;
                            }
                        }
                        NumericDType::F64 => {
                            for (slot, &(re, _)) in exact
                                .as_f64_slice_mut()
                                .expect("admitted double")
                                .iter_mut()
                                .zip(&eigenvalues)
                            {
                                *slot = re;
                            }
                        }
                        _ => unreachable!("admitted floating class"),
                    }
                    exact.into_tensor()
                })
                .map_err(|source| work::admit_error("ordeig", source))?;
            return match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::output_list(Vec::new())),
                Some(1) => Ok(Value::output_list(vec![Value::Tensor(tensor)])),
                _ => Ok(Value::Tensor(tensor)),
            };
        }
        if dtype == NumericDType::F32 {
            Tensor::from_f32(
                eigenvalues.into_iter().map(|(re, _)| re as f32).collect(),
                shape,
            )
            .map(Value::Tensor)
            .map_err(|e| ordeig_invalid(e.to_string()))
        } else {
            Tensor::new(eigenvalues.into_iter().map(|(re, _)| re).collect(), shape)
                .map(Value::Tensor)
                .map_err(|e| ordeig_invalid(e.to_string()))
        }
    } else if let Some((runtime, receipt)) = allocation {
        let tensor = receipt
            .with_output_allocation(runtime, |permit| {
                let mut exact = ExactHostComplexTensorAllocation::new(dtype, &shape, permit)?;
                for (index, &(re, im)) in eigenvalues.iter().enumerate() {
                    exact.set(index, re, im)?;
                }
                exact.into_tensor()
            })
            .map_err(|source| work::admit_error("ordeig", source))?;
        Ok(Value::ComplexTensor(tensor))
    } else if dtype == NumericDType::F32 {
        ComplexTensor::from_f32(
            eigenvalues
                .into_iter()
                .map(|(r, i)| (r as f32, i as f32))
                .collect(),
            shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|e| ordeig_invalid(e.to_string()))
    } else {
        ComplexTensor::new(eigenvalues, shape)
            .map(Value::ComplexTensor)
            .map_err(|e| ordeig_invalid(e.to_string()))
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![output?])),
        _ => output,
    }
}
