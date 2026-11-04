//! Reference arithmetic for completed Krylov vector batches.

use runmat_simulation_usage::{CountOverflow, OperationCount};

/// Completion of an admitted arithmetic batch. A prefix is an exact operation
/// count, never an estimate of work that might have run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchCompletion {
    Skipped,
    Complete,
    Prefix(OperationCount),
}

/// Arithmetic domain of a vector kernel, independent of its storage width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticDomain {
    Real,
    Complex,
}

/// Canonical scalar operations in a Krylov recurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarWork {
    Zero,
    Ratio,
    FlexibleRatio,
    Norm,
    RelativeNorm,
    HermitianCheck,
    ComplexBacksolveUpdate,
    ComplexDivide,
    ComplexRotationPair,
    ComplexRotationFinalize,
    /// One real `hypot(re, im)` for a genuinely complex magnitude.
    Hypot,
    /// One reference real multiply or quotient in scalar norm bookkeeping.
    NormScale,
    MinresRotation,
    MinresCheckpoint,
    BiCgBeta {
        domain: ArithmeticDomain,
    },
    BiCgRatio {
        domain: ArithmeticDomain,
    },
    BiCgDivideReal {
        domain: ArithmeticDomain,
    },
    BiCgStepNorm {
        complex_magnitudes: u8,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GivensBranch {
    Zero,
    ZeroFirst,
    General,
}

/// A completed reference batch. Its count is checked before the kernel runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorWork {
    /// One Hermitian dot, with an empty vector costing zero.
    Dot {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Dot terms added to a nonempty accumulator from a preceding batch.
    DotContinue {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Add magnitudes of already-computed dot terms to an existing diagnostic
    /// accumulator. Complex terms with a nonzero imaginary part use `hypot`.
    MagnitudeAccumulate {
        elements: usize,
        complex_entries: usize,
    },
    /// Complete Euclidean norm, including its final square root.
    Norm {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Squared-norm terms, before the final square root.
    NormSquares {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Squared-norm terms added to an existing accumulator.
    NormSquaresContinue {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// The final real square root of a norm.
    NormRoot,
    /// `target = source + beta * target`, with real `beta`.
    Direction {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Multiply each value by a real scalar.
    RealScale {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// `target += real_alpha * source`.
    RealAxpy {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// `(v - oldeps*w1 - delta*w2) * denom`.
    MinresDirection {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// `x += alpha * p; residual -= alpha * product`, with real `alpha`.
    Step {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// `residual = rhs - product`.
    Residual {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Divide each vector entry by a positive real norm.
    Divide {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Fallback normalization divides by both retained norm factors.
    DivideScaled {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// `target += scale * source`, for a real or complex scalar respectively.
    Axpy {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Reference `target += norm * scale * source` with a retained norm scale:
    /// two real multiplies and an add (3), or complex multiply (6), two
    /// real scales, and complex add (10) per active element. Safe-exponent
    /// lowering and zero-component shortcuts do not alter the logical batch.
    AxpyScaled {
        elements: usize,
        domain: ArithmeticDomain,
    },
    /// Classical Gram-Schmidt dots and vector corrections for contiguous rows.
    Projection {
        rows: usize,
        elements: usize,
        domain: ArithmeticDomain,
    },
    HessenbergAdd {
        domain: ArithmeticDomain,
    },
    Givens {
        magnitude_evaluations: u8,
        branch: GivensBranch,
    },
    Scalar(ScalarWork),
}

impl VectorWork {
    pub fn operations(self) -> Result<OperationCount, CountOverflow> {
        let (elements, per_element, initial) = match self {
            Self::Dot { elements, domain } => {
                if elements == 0 {
                    return Ok(OperationCount::ZERO);
                }
                (
                    elements,
                    if domain == ArithmeticDomain::Real {
                        2
                    } else {
                        8
                    },
                    if domain == ArithmeticDomain::Real {
                        1
                    } else {
                        2
                    },
                )
            }
            Self::DotContinue { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    8
                },
                0,
            ),
            Self::MagnitudeAccumulate {
                elements,
                complex_entries,
            } => {
                if complex_entries > elements {
                    return Err(CountOverflow);
                }
                return OperationCount::from_operations(
                    u64::try_from(elements).map_err(|_| CountOverflow)?,
                )
                .checked_add(OperationCount::from_operations(
                    u64::try_from(complex_entries).map_err(|_| CountOverflow)?,
                ));
            }
            Self::Norm { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    4
                },
                0,
            ),
            Self::NormSquares { elements, domain } => {
                if elements == 0 {
                    return Ok(OperationCount::ZERO);
                }
                (
                    elements,
                    if domain == ArithmeticDomain::Real {
                        2
                    } else {
                        4
                    },
                    1,
                )
            }
            Self::NormSquaresContinue { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    4
                },
                0,
            ),
            Self::NormRoot => return Ok(OperationCount::from_operations(1)),
            Self::Direction { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    4
                },
                0,
            ),
            Self::RealScale { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    1
                } else {
                    2
                },
                0,
            ),
            Self::RealAxpy { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    4
                },
                0,
            ),
            Self::MinresDirection { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    5
                } else {
                    10
                },
                0,
            ),
            Self::Step { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    4
                } else {
                    8
                },
                0,
            ),
            Self::Residual { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    1
                } else {
                    2
                },
                0,
            ),
            Self::Divide { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    1
                } else {
                    2
                },
                0,
            ),
            Self::DivideScaled { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    4
                },
                0,
            ),
            Self::Axpy { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    2
                } else {
                    8
                },
                0,
            ),
            Self::AxpyScaled { elements, domain } => (
                elements,
                if domain == ArithmeticDomain::Real {
                    3
                } else {
                    10
                },
                0,
            ),
            Self::Projection {
                rows,
                elements,
                domain,
            } => {
                if rows == 0 || elements == 0 {
                    return Ok(OperationCount::ZERO);
                }
                let n = u64::try_from(elements).map_err(|_| CountOverflow)?;
                let rows = u64::try_from(rows).map_err(|_| CountOverflow)?;
                let per_row =
                    OperationCount::from_operations(if domain == ArithmeticDomain::Real {
                        4
                    } else {
                        16
                    })
                    .checked_mul(n)?
                    .operations()
                    .checked_sub(if domain == ArithmeticDomain::Real {
                        1
                    } else {
                        2
                    })
                    .ok_or(CountOverflow)?;
                return OperationCount::from_operations(per_row).checked_mul(rows);
            }
            Self::HessenbergAdd { domain } => {
                return Ok(OperationCount::from_operations(
                    if domain == ArithmeticDomain::Real {
                        1
                    } else {
                        2
                    },
                ));
            }
            Self::Givens {
                magnitude_evaluations,
                branch,
            } => {
                let branch = match branch {
                    GivensBranch::Zero => 0,
                    GivensBranch::ZeroFirst => 2,
                    GivensBranch::General => 11,
                };
                return Ok(OperationCount::from_operations(
                    u64::from(magnitude_evaluations) + 1 + branch,
                ));
            }
            Self::Scalar(work) => {
                return Ok(OperationCount::from_operations(match work {
                    ScalarWork::Zero => 0,
                    ScalarWork::Ratio
                    | ScalarWork::Norm
                    | ScalarWork::Hypot
                    | ScalarWork::NormScale => 1,
                    ScalarWork::FlexibleRatio
                    | ScalarWork::RelativeNorm
                    | ScalarWork::HermitianCheck => 2,
                    ScalarWork::ComplexBacksolveUpdate => 8,
                    ScalarWork::ComplexDivide => 11,
                    ScalarWork::ComplexRotationPair => 20,
                    ScalarWork::ComplexRotationFinalize => 18,
                    ScalarWork::MinresRotation => 13,
                    ScalarWork::MinresCheckpoint => 2,
                    ScalarWork::BiCgBeta { domain } => {
                        if domain == ArithmeticDomain::Real {
                            3
                        } else {
                            28
                        }
                    }
                    ScalarWork::BiCgRatio { domain } => {
                        if domain == ArithmeticDomain::Real {
                            1
                        } else {
                            11
                        }
                    }
                    ScalarWork::BiCgDivideReal { domain } => {
                        if domain == ArithmeticDomain::Real {
                            1
                        } else {
                            2
                        }
                    }
                    ScalarWork::BiCgStepNorm { complex_magnitudes } => {
                        4 + u64::from(complex_magnitudes)
                    }
                }));
            }
        };
        let elements = u64::try_from(elements).map_err(|_| CountOverflow)?;
        OperationCount::from_operations(per_element)
            .checked_mul(elements)?
            .operations()
            .checked_sub(initial)
            .map(OperationCount::from_operations)
            .ok_or(CountOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_reference_schedule_is_exact_and_checked() {
        for domain in [ArithmeticDomain::Real, ArithmeticDomain::Complex] {
            assert_eq!(
                VectorWork::Dot {
                    elements: 0,
                    domain
                }
                .operations()
                .unwrap(),
                OperationCount::ZERO
            );
            assert_eq!(
                VectorWork::Dot {
                    elements: 3,
                    domain
                }
                .operations()
                .unwrap()
                .operations(),
                if domain == ArithmeticDomain::Real {
                    5
                } else {
                    22
                }
            );
            assert_eq!(
                VectorWork::Step {
                    elements: 3,
                    domain
                }
                .operations()
                .unwrap()
                .operations(),
                if domain == ArithmeticDomain::Real {
                    12
                } else {
                    24
                }
            );
            assert_eq!(
                VectorWork::Norm {
                    elements: 3,
                    domain
                }
                .operations()
                .unwrap()
                .operations(),
                if domain == ArithmeticDomain::Real {
                    6
                } else {
                    12
                }
            );
            let squared = VectorWork::NormSquares {
                elements: 2,
                domain,
            }
            .operations()
            .unwrap()
            .checked_add(
                VectorWork::NormSquaresContinue {
                    elements: 1,
                    domain,
                }
                .operations()
                .unwrap(),
            )
            .unwrap()
            .checked_add(VectorWork::NormRoot.operations().unwrap())
            .unwrap();
            assert_eq!(
                squared,
                VectorWork::Norm {
                    elements: 3,
                    domain
                }
                .operations()
                .unwrap()
            );
        }
        if let Ok(huge) = usize::try_from(u64::MAX) {
            assert!(VectorWork::Dot {
                elements: huge,
                domain: ArithmeticDomain::Real
            }
            .operations()
            .is_err());
        }
    }
}
