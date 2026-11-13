//! Original source differential, not differentiation of rounded H/f.
//! Structural degree-two admission and exact PSD are separate prerequisites.
use super::*;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;
impl Evidence {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn exact_hessian(
        &self,
        columns: usize,
        sense: Sense,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<
        Option<crate::builtins::math::optim::common::conic::convexity::OriginalHessian>,
    > {
        use crate::builtins::math::optim::common::conic::convexity::OriginalHessian;
        let Some(polynomial) = &self.restricted else {
            return Ok(None);
        };
        let mut hessian = OriginalHessian::new(columns, policy, c).await?;
        for (term, weight) in polynomial.entries() {
            if let Monomial::Quadratic(row, column) = term {
                let mut value = if sense == Sense::Maximize {
                    weight.exact.negated()
                } else {
                    weight.exact.clone()
                };
                if row == column {
                    value = value.add(&value, policy, c).await?;
                }
                hessian.add(row, column, value, policy, c).await?;
            }
            c.completed(1).await?;
        }
        Ok(Some(hessian))
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn gradient<'a>(
        &'a self,
        program: &'a Program,
        point: exact::Coordinates<'a>,
        sense: Sense,
        policy: Policy,
        c: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Vec<Rational>>> + 'a {
        let at_origin = matches!(point, exact::Coordinates::Origin);
        let retention = match &self.projection {
            Projection::Affine | Projection::Scalar => exact::Retain::Operands,
            Projection::Squares(_) => exact::Retain::ConstantOperands,
        };
        crate::execution::after_result(
            async move {
                let operands = exact::evaluate_atomic(
                    exact::Source {
                        source: &program.source,
                        layout: &program.layout,
                        facts: &program.facts,
                    },
                    Some(exact::Subset {
                        nodes: &self.selection.nodes,
                        roots: &self.selection.roots,
                    }),
                    point,
                    retention,
                    &self.atoms,
                    policy,
                    c,
                )
                .await?;
                Ok((operands, c))
            },
            move |(operands, c)| {
                self.gradient_from_operands(program, operands, at_origin, sense, policy, c)
            },
        )
    }

    async fn gradient_from_operands(
        &self,
        program: &Program,
        operands: exact::Evaluated,
        at_origin: bool,
        sense: Sense,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Rational>> {
        match &self.projection {
            Projection::Affine | Projection::Scalar => {
                if matches!(self.projection, Projection::Affine) && self.selection.roots.is_empty()
                {
                    let mut zero = runmat_value::exact_capacity_vec(program.layout.columns)
                        .map_err(|e| policy.storage(e))?;
                    for _ in 0..program.layout.columns {
                        zero.push(Rational::zero());
                        c.completed(1).await?;
                    }
                    return Ok(zero);
                }
                let [root] = self.selection.roots.as_slice() else {
                    return Err(
                        policy.invalid("quadratic differential lost its original scalar root")
                    );
                };
                let mut seeds =
                    runmat_value::exact_capacity_vec(1).map_err(|e| policy.storage(e))?;
                seeds.push(exact::Seed {
                    root: *root,
                    row: 0,
                    value: if sense == Sense::Minimize {
                        Rational::from_usize(1)
                    } else {
                        -Rational::from_usize(1)
                    },
                });
                exact::accumulate(program, &operands, seeds, policy, c).await
            }
            Projection::Squares(squares) => {
                let origin = if at_origin {
                    None
                } else {
                    Some(
                        exact::evaluate_selected(
                            exact::Source {
                                source: &program.source,
                                layout: &program.layout,
                                facts: &program.facts,
                            },
                            Some(exact::Subset {
                                nodes: &self.selection.nodes,
                                roots: &self.selection.roots,
                            }),
                            exact::Coordinates::Origin,
                            exact::Retain::ConstantOperands,
                            policy,
                            c,
                        )
                        .await?,
                    )
                };
                let origin = origin.as_ref().unwrap_or(&operands);
                let (mut seeds, _) = squares.seeds(&operands, origin, program, policy, c).await?;
                if sense == Sense::Maximize {
                    for seed in &mut seeds {
                        seed.value = -&seed.value;
                        c.completed(1).await?;
                    }
                }
                exact::accumulate(program, origin, seeds, policy, c).await
            }
        }
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn recession(
        &self,
        program: &Program,
        direction: &[Rational],
        sense: Sense,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if direction.len() != program.layout.columns {
            return Ok(false);
        }
        let origin = self
            .gradient(program, exact::Coordinates::Origin, sense, policy, c)
            .await?;
        let displaced = self
            .gradient(
                program,
                exact::Coordinates::Point(direction),
                sense,
                policy,
                c,
            )
            .await?;
        if origin.len() != direction.len() || displaced.len() != direction.len() {
            return Ok(false);
        }
        let mut slope = Rational::zero();
        for ((base, moved), step) in origin.iter().zip(&displaced).zip(direction) {
            // For an admitted degree-two scalar, grad(d)-grad(0)=H*d.
            // PSD alone is not permission to ignore a small nonzero curvature.
            if !base.compare(moved, policy, c).await?.is_eq() {
                return Ok(false);
            }
            slope = slope
                .add(&base.mul(step, policy, c).await?, policy, c)
                .await?;
            c.completed(1).await?;
        }
        Ok(slope.is_negative())
    }
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn directional_cost(
    gradient: &[Rational],
    point: &[Rational],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Rational> {
    if gradient.len() != point.len() {
        return Err(policy.invalid("quadratic gradient differs from original point layout"));
    }
    let mut value = Rational::zero();
    for (gradient, coordinate) in gradient.iter().zip(point) {
        value = value
            .add(&gradient.mul(coordinate, policy, c).await?, policy, c)
            .await?;
        c.completed(1).await?;
    }
    Ok(value)
}
pub(super) async fn affine_scale(
    program: &Program,
    selected: &selection::Selection,
    sense: Sense,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<super::super::proof::ObjectiveScale> {
    let mut curvature =
        runmat_value::exact_capacity_vec(program.layout.columns).map_err(|e| policy.storage(e))?;
    for _ in 0..program.layout.columns {
        curvature.push(Rational::zero());
        c.completed(1).await?;
    }
    let mut linear = if selected.roots.is_empty() {
        let mut zero = runmat_value::exact_capacity_vec(program.layout.columns)
            .map_err(|e| policy.storage(e))?;
        for _ in 0..program.layout.columns {
            zero.push(Rational::zero());
            c.completed(1).await?;
        }
        zero
    } else {
        let [root] = selected.roots.as_slice() else {
            return Err(policy.invalid("affine quadratic selection lost original scalar root"));
        };
        let origin = exact::evaluate_selected(
            exact::Source {
                source: &program.source,
                layout: &program.layout,
                facts: &program.facts,
            },
            Some(exact::Subset {
                nodes: &selected.nodes,
                roots: &selected.roots,
            }),
            exact::Coordinates::Origin,
            exact::Retain::Operands,
            policy,
            c,
        )
        .await?;
        let mut seeds = runmat_value::exact_capacity_vec(1).map_err(|e| policy.storage(e))?;
        seeds.push(exact::Seed {
            root: *root,
            row: 0,
            value: if sense == Sense::Minimize {
                Rational::from_usize(1)
            } else {
                -Rational::from_usize(1)
            },
        });
        exact::accumulate(program, &origin, seeds, policy, c).await?
    };
    for value in &mut linear {
        *value = value.abs();
        c.completed(1).await?;
    }
    super::super::proof::ObjectiveScale::from_parts(linear, curvature, policy, c).await
}
