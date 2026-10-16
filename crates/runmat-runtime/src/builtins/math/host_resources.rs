//! Allocation bounds for the real floating host algorithms shared by canonical
//! numeric implementations. Selection still requires their exact registration
//! and actual host-branch confirmation; these calculators grant no authority.
use crate::builtins::common::spec::{
    FusionHostOutputDomain, FusionHostResourceBound, FusionHostResourceInput,
    FusionHostResourceQuery,
};
use runmat_types::{NumericClass, NumericDomain, OperatorKind};

mod envelope;
pub use envelope::{promoting_output, PromotingOutput};

/// BroadcastPlan's existing typed shape/stride preparation bound. Reused by
/// fusion's pre-selection receipt so the same temporary is not guessed twice.
pub const BROADCAST_METADATA_VECTORS_UPPER: u64 = 10;

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

/// Direct maps retain one native output. Consuming exponential maps may clone
/// the COW input, and the argument/shape-to-tensor adapters clone shape metadata.
/// Borrowing trigonometric maps use no more than this same bound.
pub fn real_map(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    if query.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real) {
        return None;
    }
    map_bound(query)
}

fn map_bound(query: &FusionHostResourceQuery<'_>) -> Option<FusionHostResourceBound> {
    let [input] = query.inputs else { return None };
    if input.shape != query.output.shape || input.class != query.output.class {
        return None;
    }
    Some(FusionHostResourceBound {
        retained_bytes: payload(query.output)?.checked_add(metadata(query.output.shape)?)?,
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
    // Pair -> real collection may reuse the wider pair allocation. Bound
    // returned allocation capacity, not just the real vector's logical len.
    bound.retained_bytes = bound.retained_bytes.checked_add(payload(query.output)?)?;
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
    let vectors = u64::try_from(rank)
        .ok()?
        .checked_mul(std::mem::size_of::<usize>() as u64)?
        .checked_mul(BROADCAST_METADATA_VECTORS_UPPER)?;
    Some(FusionHostResourceBound {
        retained_bytes: payload(query.output)?.checked_add(metadata(query.output.shape)?)?,
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
    // finish_real_power may retain the wider pair allocation after extracting
    // its real components; do not assume collection shrinks its capacity.
    bound.retained_bytes = bound.retained_bytes.checked_add(payload(query.output)?)?;
    bound.scratch_bytes = bound
        .scratch_bytes
        .checked_add(payload(query.output)?.checked_mul(2)?)?;
    Some(bound)
}

/// VM numeric intrinsics do not borrow named builtin authority. Negation's
/// numeric branch delegates to canonical times(-1,input); matrix multiply is
/// admitted only for its existing scalar physical route.
pub fn intrinsic(
    operator: OperatorKind,
    query: &FusionHostResourceQuery<'_>,
) -> Option<FusionHostResourceBound> {
    match operator {
        OperatorKind::UnaryPlus => real_map(query),
        OperatorKind::UnaryMinus => {
            let [input] = query.inputs else { return None };
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
            real_binary(query)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                for domain in [
                    FusionHostOutputDomain::Exact(NumericDomain::Complex),
                    FusionHostOutputDomain::RealOrComplex,
                ] {
                    let alternative = FusionHostResourceQuery {
                        output_domain: domain,
                        ..unary
                    };
                    assert_eq!(real_promoting_map(&alternative), Some(promoting));
                    assert!(real_map(&alternative).is_none());
                }
                assert_eq!(
                    promoting.retained_bytes - direct.retained_bytes,
                    payload(input).unwrap()
                );
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
                for domain in [
                    FusionHostOutputDomain::Exact(NumericDomain::Complex),
                    FusionHostOutputDomain::RealOrComplex,
                ] {
                    let alternative = FusionHostResourceQuery {
                        output_domain: domain,
                        ..binary
                    };
                    assert_eq!(real_power(&alternative), Some(power));
                    assert!(real_binary(&alternative).is_none());
                }
                assert_eq!(
                    power.retained_bytes - native.retained_bytes,
                    payload(input).unwrap()
                );
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
