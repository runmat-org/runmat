//! Allocation bounds for the real floating host algorithms shared by canonical
//! numeric implementations. Selection still requires their exact registration
//! and actual host-branch confirmation; these calculators grant no authority.
use crate::builtins::common::spec::{
    FusionHostOutputDomain, FusionHostResourceBound, FusionHostResourceInput,
    FusionHostResourceQuery,
};
use runmat_types::{NumericClass, NumericDomain, OperatorKind};
use runmat_value::NumericDType;

mod envelope;
pub use envelope::{promoting_output, PromotingOutput};
mod prepared;
pub use prepared::scratch_bytes as prepared_scratch_bytes;
pub(crate) use prepared::scratch_for_shapes as prepared_scratch_for_shapes;

fn width(class: NumericClass) -> Option<u64> {
    match class {
        NumericClass::Single => Some(4),
        NumericClass::Double => Some(8),
        _ => None,
    }
}

fn elements(shape: &[usize]) -> Option<u64> {
    shape.iter().try_fold(1_u64, |n, &extent| {
        n.checked_mul(u64::try_from(extent).ok()?)
    })
}

fn metadata(shape: &[usize]) -> Option<u64> {
    u64::try_from(shape.len())
        .ok()?
        .checked_mul(std::mem::size_of::<usize>() as u64)
}

fn payload(input: FusionHostResourceInput<'_>) -> Option<u64> {
    elements(input.shape)?.checked_mul(width(input.class)?)
}

pub fn broadcast_metadata_bytes(rank: usize) -> Option<u64> {
    crate::builtins::common::broadcast::PreparedBroadcastPlan::quoted_bytes(rank).ok()
}

fn dtype(class: NumericClass) -> Option<NumericDType> {
    match class {
        NumericClass::Single => Some(NumericDType::F32),
        NumericClass::Double => Some(NumericDType::F64),
        _ => None,
    }
}

fn retained_output(query: &FusionHostResourceQuery<'_>) -> Option<u64> {
    let dtype = dtype(query.output.class)?;
    let real = || {
        crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
            dtype,
            query.output.shape,
        )
        .ok()
    };
    let complex = || {
        crate::context::host_allocation::HostAllocationPlan::retained_complex_tensor_bytes(
            dtype,
            query.output.shape,
        )
        .ok()
    };
    match query.output_domain {
        FusionHostOutputDomain::Exact(NumericDomain::Real) => real(),
        FusionHostOutputDomain::Exact(NumericDomain::Complex) => complex(),
        FusionHostOutputDomain::RealOrComplex => real()?.checked_add(complex()?),
    }
}

/// Direct maps retain one native output. Consuming exponential maps may clone
/// the COW input, and the argument/shape-to-tensor adapters clone shape metadata.
/// Borrowing trigonometric maps use no more than this same bound.
pub fn real_map(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    if query.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real) {
        return None;
    }
    map_bound(query)
}

/// Plain single/double conversion keeps the input shape but may change its
/// native element width. The conversion consumes the downloaded tensor,
/// retaining at most one COW input copy while constructing one output vector.
pub fn real_cast(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    if query.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real) {
        return None;
    }
    let [input] = query.inputs else { return None };
    if input.shape != query.output.shape {
        return None;
    }
    let output_dtype = match query.output.class {
        NumericClass::Single => NumericDType::F32,
        NumericClass::Double => NumericDType::F64,
        _ => return None,
    };
    let retained_bytes =
        crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
            output_dtype,
            query.output.shape,
        )
        .ok()?;
    Some(FusionHostResourceBound {
        retained_bytes,
        scratch_bytes: payload(*input)?.checked_add(metadata(input.shape)?.checked_mul(2)?)?,
    })
}

fn map_bound(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    let [input] = query.inputs else { return None };
    if input.shape != query.output.shape || input.class != query.output.class {
        return None;
    }
    Some(FusionHostResourceBound {
        retained_bytes: retained_output(query)?,
        scratch_bytes: payload(*input)?.checked_add(metadata(input.shape)?.checked_mul(2)?)?,
    })
}

/// Native promoting maps may consume a COW input and collect native pairs.
/// Real extraction may retain the pair capacity; complex publication retains
/// the pairs directly. The bound covers either branch, not a domain proof.
pub fn real_promoting_map(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    let mut bound = match query.output_domain {
        FusionHostOutputDomain::Exact(NumericDomain::Real | NumericDomain::Complex)
        | FusionHostOutputDomain::RealOrComplex => map_bound(query)?,
    };
    bound.retained_bytes = retained_output(query)?;
    bound.scratch_bytes = bound
        .scratch_bytes
        .checked_add(payload(query.inputs[0])?.checked_mul(2)?)?;
    Some(bound)
}

/// Native binary arithmetic holds COW copies of both inputs plus one output.
/// BroadcastPlan construction, argument shape clones, and the output-shape
/// clone require at most ten vectors of the maximum input/output rank.
pub fn real_binary(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    if query.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real) {
        return None;
    }
    binary_bound(query)
}

fn binary_bound(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    let [left, right] = query.inputs else {
        return None;
    };
    let plan =
        crate::builtins::common::broadcast::BroadcastPlan::new(left.shape, right.shape).ok()?;
    if plan.output_shape() != query.output.shape {
        return None;
    }
    let rank = left
        .shape
        .len()
        .max(right.shape.len())
        .max(query.output.shape.len())
        .max(2);
    let vectors = broadcast_metadata_bytes(rank)?;
    Some(FusionHostResourceBound {
        retained_bytes: retained_output(query)?,
        scratch_bytes: payload(*left)?
            .checked_add(payload(*right)?)?
            .checked_add(vectors)?,
    })
}

/// power::host::real holds native (real,imag) output pairs before extracting
/// the real output. The latter is separately retained, so both buffers count.
pub fn real_power(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    let [left, right] = query.inputs else {
        return None;
    };
    if runmat_accelerate_api::RealPowerArithmeticMode::for_classes(left.class, right.class)?
        .output_class()
        != query.output.class
    {
        return None;
    }
    let mut bound = match query.output_domain {
        FusionHostOutputDomain::Exact(NumericDomain::Real | NumericDomain::Complex)
        | FusionHostOutputDomain::RealOrComplex => binary_bound(query)?,
    };
    bound.retained_bytes = retained_output(query)?;
    bound.scratch_bytes = bound
        .scratch_bytes
        .checked_add(payload(query.output)?.checked_mul(2)?)?;
    Some(bound)
}

/// VM numeric intrinsics do not borrow named builtin authority. Ordinary
/// negation delegates to canonical times(-1,input), while the
/// selected Managed VM branch writes through the same preallocated binary
/// recipe. Matrix multiply is admitted only for its scalar physical route.
pub fn intrinsic(
    operator: OperatorKind,
    query: &FusionHostResourceQuery<'_>,
) -> Option<FusionHostResourceBound> {
    match operator {
        OperatorKind::UnaryPlus => {
            let [input] = query.inputs else { return None };
            if query.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real)
                || input.shape != query.output.shape
                || input.class != query.output.class
                || width(input.class).is_none()
            {
                return None;
            }
            // Unary plus returns the owned input unchanged, so no numerical
            // output is born. The selected caller must separately verify
            // that input ownership survives the VM region boundary.
            Some(FusionHostResourceBound {
                retained_bytes: 0,
                scratch_bytes: 0,
            })
        }
        OperatorKind::UnaryMinus => {
            let [input] = query.inputs else { return None };
            if runmat_types::RealFloatingRecipe::for_classes(
                OperatorKind::ElementwiseMultiply,
                NumericClass::Double,
                input.class,
            )?
            .result_class()
                != query.output.class
            {
                return None;
            }
            let operands = [
                FusionHostResourceInput {
                    shape: &[1, 1],
                    class: NumericClass::Double,
                },
                *input,
            ];
            real_binary(&FusionHostResourceQuery {
                inputs: &operands,
                output: query.output,
                output_domain: query.output_domain,
            })
        }
        OperatorKind::MatrixMultiply
            if query
                .inputs
                .iter()
                .any(|input| elements(input.shape) == Some(1)) =>
        {
            let [left, right] = query.inputs else {
                return None;
            };
            if runmat_types::RealFloatingRecipe::for_classes(
                OperatorKind::ElementwiseMultiply,
                left.class,
                right.class,
            )?
            .result_class()
                != query.output.class
            {
                return None;
            }
            real_binary(query)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::broadcast::PreparedBroadcastPlan;
    use crate::context::host_allocation::HostAllocationPlan;
    use runmat_builtins::BuiltinPreparationOperation;

    #[test]
    fn real_cast_bounds_both_precision_directions_without_aliasing_widths() {
        let shape = [2, 3];
        for (source, target) in [
            (NumericClass::Double, NumericClass::Single),
            (NumericClass::Single, NumericClass::Double),
        ] {
            let input = FusionHostResourceInput {
                shape: &shape,
                class: source,
            };
            let output = FusionHostResourceInput {
                shape: &shape,
                class: target,
            };
            let query = FusionHostResourceQuery {
                inputs: &[input],
                output,
                output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
            };
            let bound = real_cast(&query).expect("native floating cast has an exact bound");
            let dtype = match target {
                NumericClass::Single => NumericDType::F32,
                NumericClass::Double => NumericDType::F64,
                _ => unreachable!(),
            };
            assert_eq!(
                bound.retained_bytes,
                crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
                    dtype, &shape,
                )
                .unwrap()
            );
            assert_eq!(bound.scratch_bytes, payload(input).unwrap() + 32);
            assert_eq!(
                prepared_scratch_bytes(BuiltinPreparationOperation::FloatingCast(target), &query,),
                Some(0)
            );
            assert!(real_map(&query).is_none());
            assert!(real_cast(&FusionHostResourceQuery {
                output_domain: FusionHostOutputDomain::Exact(NumericDomain::Complex),
                ..query
            })
            .is_none());
        }
    }

    #[test]
    fn real_owner_bounds_keep_class_broadcast_pairs_and_overflow_explicit() {
        for class in [NumericClass::Single, NumericClass::Double] {
            for extent in [0, 7] {
                let shape = [1, extent];
                let input = FusionHostResourceInput {
                    shape: &shape,
                    class,
                };
                let unary = FusionHostResourceQuery {
                    inputs: &[input],
                    output: input,
                    output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
                };
                let direct = real_map(&unary).unwrap();
                let promoting = real_promoting_map(&unary).unwrap();
                assert_eq!(
                    prepared_scratch_bytes(BuiltinPreparationOperation::Unary, &unary),
                    Some(0)
                );
                assert_eq!(
                    prepared_scratch_bytes(
                        BuiltinPreparationOperation::RealUnary(
                            runmat_types::RealUnaryPredicate::Nonnegative,
                        ),
                        &unary,
                    ),
                    Some(0)
                );
                let identity = intrinsic(OperatorKind::UnaryPlus, &unary).unwrap();
                assert_eq!(identity.retained_bytes, 0);
                assert_eq!(identity.scratch_bytes, 0);
                let dtype = dtype(class).unwrap();
                let real = HostAllocationPlan::retained_tensor_bytes(dtype, &shape).unwrap();
                let complex =
                    HostAllocationPlan::retained_complex_tensor_bytes(dtype, &shape).unwrap();
                assert_eq!(direct.retained_bytes, real);
                assert_eq!(promoting.retained_bytes, real);
                let complex_bound = real_promoting_map(&FusionHostResourceQuery {
                    output_domain: FusionHostOutputDomain::Exact(NumericDomain::Complex),
                    ..unary
                })
                .unwrap();
                let dual_bound = real_promoting_map(&FusionHostResourceQuery {
                    output_domain: FusionHostOutputDomain::RealOrComplex,
                    ..unary
                })
                .unwrap();
                assert_eq!(complex_bound.retained_bytes, complex);
                assert_eq!(dual_bound.retained_bytes, real + complex);
                assert_eq!(
                    promoting.scratch_bytes - direct.scratch_bytes,
                    payload(input).unwrap() * 2
                );
                let scalar = FusionHostResourceInput {
                    shape: &[1, 1],
                    class: NumericClass::Double,
                };
                let binary = FusionHostResourceQuery {
                    inputs: &[scalar, input],
                    output: input,
                    output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
                };
                let native = real_binary(&binary).unwrap();
                let power = real_power(&binary).unwrap();
                let prepared_broadcast = PreparedBroadcastPlan::quoted_for_shapes(
                    scalar.shape,
                    input.shape,
                    input.shape,
                )
                .unwrap();
                assert_eq!(
                    prepared_scratch_bytes(BuiltinPreparationOperation::ElementwiseBinary, &binary,),
                    Some(prepared_broadcast)
                );
                assert_eq!(
                    prepared_scratch_bytes(BuiltinPreparationOperation::RealPower, &binary),
                    Some(prepared_broadcast)
                );
                assert_eq!(native.retained_bytes, real);
                assert_eq!(power.retained_bytes, real);
                let complex_power = real_power(&FusionHostResourceQuery {
                    output_domain: FusionHostOutputDomain::Exact(NumericDomain::Complex),
                    ..binary
                })
                .unwrap();
                let dual_power = real_power(&FusionHostResourceQuery {
                    output_domain: FusionHostOutputDomain::RealOrComplex,
                    ..binary
                })
                .unwrap();
                assert_eq!(complex_power.retained_bytes, complex);
                assert_eq!(dual_power.retained_bytes, real + complex);
                assert_eq!(
                    power.scratch_bytes - native.scratch_bytes,
                    payload(input).unwrap() * 2
                );
                assert!(native.scratch_bytes >= payload(input).unwrap() + 8);
                assert!(intrinsic(OperatorKind::MatrixMultiply, &binary).is_some());
            }
        }
        let huge = FusionHostResourceInput {
            shape: &[usize::MAX, usize::MAX, 2],
            class: NumericClass::Double,
        };
        assert!(real_map(&FusionHostResourceQuery {
            inputs: &[huge],
            output: huge,
            output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
        })
        .is_none());
        let integer = FusionHostResourceInput {
            shape: &[1, 3],
            class: NumericClass::Int32,
        };
        assert!(real_map(&FusionHostResourceQuery {
            inputs: &[integer],
            output: integer,
            output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
        })
        .is_none());
        let matrix = FusionHostResourceInput {
            shape: &[2, 2],
            class: NumericClass::Single,
        };
        assert!(intrinsic(
            OperatorKind::MatrixMultiply,
            &FusionHostResourceQuery {
                inputs: &[matrix, matrix],
                output: matrix,
                output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
            }
        )
        .is_none());
    }
}
