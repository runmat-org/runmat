//! Reordering a retained complex generalized Schur pencil.

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use faer::{c64, Col, Mat};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Value};

use super::algorithm::{ComplexQz, RealQz};
use super::eigvals::real_schur_homogeneous;
use super::real_reorder::reorder_real_clusters;
use super::reorder::{reorder_clusters, ClusterOrder};
use super::{allocation, work, Mode};
use super::{coefficient_shape, complex_matrix, publish, publish_real, real_matrix};

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("ordqz")
        .with_identifier("RunMat:ordqz:InvalidInput")
        .build()
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("ordqz")
        .with_identifier("RunMat:ordqz:IllConditioned")
        .build()
}

fn numeric_label(value: NumericScalar) -> BuiltinResult<u64> {
    if let Some(integer) = value.into_int_value() {
        return integer
            .try_to_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("ordqz cluster indices must be positive integers"));
    }
    let value = value.materialize_f64();
    if value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value <= (1u64 << 53) as f64 {
        Ok(value as u64)
    } else {
        Err(invalid(
            "ordqz cluster indices must be exactly representable positive integers",
        ))
    }
}

fn inside_unit_disk(alpha: c64, beta: c64) -> BuiltinResult<bool> {
    let scale = alpha
        .re
        .abs()
        .max(alpha.im.abs())
        .max(beta.re.abs())
        .max(beta.im.abs());
    if scale == 0.0 {
        return Err(invalid(
            "ordqz cannot classify an indeterminate generalized eigenvalue",
        ));
    }
    Ok((alpha.re / scale).hypot(alpha.im / scale) < (beta.re / scale).hypot(beta.im / scale))
}

fn left_half_plane(alpha: c64, beta: c64) -> BuiltinResult<bool> {
    let a = alpha.re.abs().max(alpha.im.abs());
    let b = beta.re.abs().max(beta.im.abs());
    if b == 0.0 {
        return if a == 0.0 {
            Err(invalid(
                "ordqz cannot classify an indeterminate generalized eigenvalue",
            ))
        } else {
            // The two-input ordeig owner represents a nonzero/zero pair as
            // positive infinity, which belongs to rhp and udo.
            Ok(false)
        };
    }
    if a == 0.0 {
        return Ok(false);
    }
    let value = (alpha.re / a) * (beta.re / b) + (alpha.im / a) * (beta.im / b);
    Ok(value < 0.0)
}

struct ClusterSelection {
    labels: Vec<u64>,
    order: ClusterOrder,
}

fn selector_labels(selector: &Value, pairs: &[(c64, c64)]) -> BuiltinResult<ClusterSelection> {
    let n = pairs.len();
    if let Value::LogicalArray(array) = selector {
        if array.len() != n
            || (array.shape.len() > 2)
            || (array.shape.len() == 2 && array.shape[0] != 1 && array.shape[1] != 1)
        {
            return Err(invalid(
                "ordqz selection must be a logical vector of pencil order",
            ));
        }
        return Ok(ClusterSelection {
            labels: array
                .data
                .iter()
                .map(|&flag| if flag != 0 { 0 } else { 1 })
                .collect(),
            order: ClusterOrder::SelectedFirst,
        });
    }
    if let Value::Bool(flag) = selector {
        if n != 1 {
            return Err(invalid(
                "ordqz selection length does not match pencil order",
            ));
        }
        return Ok(ClusterSelection {
            labels: vec![if *flag { 0 } else { 1 }],
            order: ClusterOrder::SelectedFirst,
        });
    }
    let keyword = match selector {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(chars) if chars.rows <= 1 => Some(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    };
    if let Some(keyword) = keyword {
        let keyword = keyword.trim().to_ascii_lowercase();
        if !matches!(keyword.as_str(), "lhp" | "rhp" | "udi" | "udo") {
            return Err(invalid("ordqz region must be lhp, rhp, udi, or udo"));
        }
        let labels = (0..n)
            .map(|i| {
                let (alpha, beta) = pairs[i];
                let selected = match keyword.as_str() {
                    "udi" => inside_unit_disk(alpha, beta)?,
                    "udo" => !inside_unit_disk(alpha, beta)?,
                    "lhp" | "rhp" => {
                        let left = left_half_plane(alpha, beta)?;
                        if keyword == "lhp" {
                            left
                        } else {
                            !left
                        }
                    }
                    _ => unreachable!(),
                };
                Ok(if selected { 0 } else { 1 })
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        return Ok(ClusterSelection {
            labels,
            order: ClusterOrder::SelectedFirst,
        });
    }
    if let Value::Tensor(tensor) = selector {
        if tensor.len() != n
            || tensor.shape().len() > 2
            || (tensor.shape().len() == 2 && tensor.shape()[0] != 1 && tensor.shape()[1] != 1)
        {
            return Err(invalid(
                "ordqz cluster labels must be a vector of pencil order",
            ));
        }
        let labels = (0..n)
            .map(|i| {
                numeric_label(
                    tensor
                        .numeric_value_at(i)
                        .ok_or_else(|| invalid("invalid ordqz cluster label"))?,
                )
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        return Ok(ClusterSelection {
            labels,
            order: ClusterOrder::NumericDescending,
        });
    }
    if let Value::Int(integer) = selector {
        if n != 1 {
            return Err(invalid("ordqz cluster length does not match pencil order"));
        }
        return Ok(ClusterSelection {
            labels: vec![integer
                .try_to_u64()
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid("ordqz cluster indices must be positive integers"))?],
            order: ClusterOrder::NumericDescending,
        });
    }
    if let Value::Num(number) = selector {
        if n != 1 {
            return Err(invalid("ordqz cluster length does not match pencil order"));
        }
        return Ok(ClusterSelection {
            labels: vec![numeric_label(NumericScalar::F64(*number))?],
            order: ClusterOrder::NumericDescending,
        });
    }
    Err(invalid(
        "ordqz requires a logical selector, region keyword, or cluster vector",
    ))
}

pub(super) fn pair_labels(labels: &mut [u64], aa: &Mat<f64>, order: ClusterOrder) {
    for i in 0..labels.len().saturating_sub(1) {
        if aa[(i + 1, i)] != 0.0 {
            let combined = order.pair_label(labels[i], labels[i + 1]);
            labels[i] = combined;
            labels[i + 1] = combined;
        }
    }
}

fn validated_factor(value: &Value, n: usize) -> BuiltinResult<Mat<c64>> {
    let (m, _) = coefficient_shape(value).map_err(|error| invalid(error.message().to_owned()))?;
    if m != n {
        return Err(invalid("ordqz factor dimensions must agree"));
    }
    let matrix = complex_matrix(value, n).map_err(|error| invalid(error.message().to_owned()))?;
    if !(0..n).all(|j| {
        (0..n).all(|i| {
            let v = matrix[(i, j)];
            v.re.is_finite() && v.im.is_finite()
        })
    }) {
        return Err(invalid("ordqz factors must be finite"));
    }
    Ok(matrix)
}

#[runtime_builtin(
    name = "ordqz",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::qz",
    context = true
)]
async fn ordqz_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    aa: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() != 4 {
        return Err(invalid("ordqz requires AA, BB, Q, Z, and a selector"));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid("ordqz has at most four outputs"));
    }
    let (n, dtype_a) =
        coefficient_shape(&aa).map_err(|error| invalid(error.message().to_owned()))?;
    let (size_b, dtype_b) =
        coefficient_shape(&rest[0]).map_err(|error| invalid(error.message().to_owned()))?;
    let (size_q, dtype_q) =
        coefficient_shape(&rest[1]).map_err(|error| invalid(error.message().to_owned()))?;
    let (size_z, dtype_z) =
        coefficient_shape(&rest[2]).map_err(|error| invalid(error.message().to_owned()))?;
    if [size_b, size_q, size_z].iter().any(|&size| size != n) {
        return Err(invalid("ordqz factor dimensions must agree"));
    }
    if matches!(&aa, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Err(invalid(
            "ordqz resident inputs require a supported provider execution path",
        ));
    }
    let dtype = if [dtype_a, dtype_b, dtype_q, dtype_z].contains(&NumericDType::F64) {
        NumericDType::F64
    } else {
        NumericDType::F32
    };
    let real = std::iter::once(&aa)
        .chain(rest.view().range(..3).expect("validated ordqz inputs"))
        .all(|value| matches!(value, Value::Tensor(_) | Value::Num(_)));
    let mode = if real { Mode::Real } else { Mode::Complex };
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    let plan = allocation::reorder_plan(n, dtype, mode, requested)
        .map_err(|source| work::admit_error("ordqz", source))?;
    let mut worker_work = work::QzWork::new(runtime.as_ref(), None, "ordqz");
    let arguments = std::iter::once(&aa).chain(rest.iter()).collect::<Vec<_>>();
    if let Some(result) = worker_work
        .try_owned_browser_worker(
            &arguments,
            requested,
            crate::output_count::current_output_count().is_some(),
            plan,
            work::QzOutputRule {
                shape: [n, n],
                dtype,
                domain: if real {
                    work::QzOutputDomain::Real
                } else {
                    work::QzOutputDomain::Complex
                },
            },
        )
        .await?
    {
        return Ok(result);
    }
    let receipt =
        crate::context::host_allocation::HostAllocationReceipt::admit(runtime.as_ref(), plan)
            .map_err(|source| work::admit_error("ordqz", source))?;
    let mut work = work::QzWork::new(runtime.as_ref(), receipt.as_ref(), "ordqz");
    work.checkpoint(0).await?;
    if real {
        let aa = real_matrix(&aa, n).map_err(|error| invalid(error.message().to_owned()))?;
        let bb = real_matrix(&rest[0], n).map_err(|error| invalid(error.message().to_owned()))?;
        let q = real_matrix(&rest[1], n).map_err(|error| invalid(error.message().to_owned()))?;
        let right =
            real_matrix(&rest[2], n).map_err(|error| invalid(error.message().to_owned()))?;
        if [&aa, &bb, &q, &right]
            .into_iter()
            .any(|matrix| (0..n).any(|j| (0..n).any(|i| !matrix[(i, j)].is_finite())))
        {
            return Err(invalid("ordqz factors must be finite"));
        }
        let pairs = real_schur_homogeneous(&aa, &bb).map_err(invalid)?;
        let mut selection = selector_labels(&rest[3], &pairs)?;
        pair_labels(&mut selection.labels, &aa, selection.order);
        work.checkpoint(4).await?;
        let left = Mat::from_fn(n, n, |i, j| q[(j, i)]);
        let mut form = RealQz {
            aa,
            bb,
            left,
            right,
            alpha_re: Col::zeros(n),
            alpha_im: Col::zeros(n),
            beta: Col::zeros(n),
        };
        let form = work
            .blocking(move || {
                reorder_real_clusters(&mut form, &selection.labels, selection.order)?;
                Ok::<_, &'static str>(form)
            })
            .await?
            .map_err(numerical)?;
        let result = publish_real(
            "ordqz",
            form,
            dtype,
            None,
            runtime.as_ref().zip(receipt.as_ref()),
        )?;
        work.checkpoint(u64::try_from(n.saturating_mul(n)).unwrap_or(u64::MAX))
            .await?;
        Ok(result)
    } else {
        let aa = validated_factor(&aa, n)?;
        let bb = validated_factor(&rest[0], n)?;
        let q = validated_factor(&rest[1], n)?;
        let right = validated_factor(&rest[2], n)?;
        let scale = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .fold(0.0_f64, |m, (i, j)| {
                let a = aa[(i, j)];
                let b = bb[(i, j)];
                m.max(a.re.abs())
                    .max(a.im.abs())
                    .max(b.re.abs())
                    .max(b.im.abs())
            });
        let tolerance = 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale;
        for j in 0..n {
            for i in (j + 1)..n {
                if aa[(i, j)].re.hypot(aa[(i, j)].im) > tolerance
                    || bb[(i, j)].re.hypot(bb[(i, j)].im) > tolerance
                {
                    return Err(invalid("ordqz requires triangular complex Schur factors"));
                }
            }
        }
        let pairs = (0..n).map(|i| (aa[(i, i)], bb[(i, i)])).collect::<Vec<_>>();
        let selection = selector_labels(&rest[3], &pairs)?;
        work.checkpoint(4).await?;
        let left = Mat::from_fn(n, n, |i, j| {
            let value = q[(j, i)];
            c64::new(value.re, -value.im)
        });
        let mut form = ComplexQz {
            aa,
            bb,
            left,
            right,
            alpha: Col::zeros(n),
            beta: Col::zeros(n),
        };
        let form = work
            .blocking(move || {
                reorder_clusters(&mut form, &selection.labels, selection.order)?;
                Ok::<_, &'static str>(form)
            })
            .await?
            .map_err(numerical)?;
        let result = publish(
            "ordqz",
            form,
            dtype,
            None,
            runtime.as_ref().zip(receipt.as_ref()),
        )?;
        work.checkpoint(u64::try_from(n.saturating_mul(n)).unwrap_or(u64::MAX))
            .await?;
        Ok(result)
    }
}
