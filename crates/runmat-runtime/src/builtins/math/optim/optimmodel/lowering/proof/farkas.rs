//! Exact weighted original rows and bounds. The recovered multiplier vector is
//! used consistently in cone membership, intercept and reverse stationarity.
use super::*;
pub(super) struct Weighted {
    pub gradient: Vec<Rational>,
    pub intercept: Rational,
}
impl<'source> Authority<'source> {
    pub(super) async fn farkas(
        &mut self,
        roles: &[RowRole],
        multipliers: &[Rational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if roles.len() != multipliers.len() {
            return Ok(false);
        }
        for (role, value) in roles.iter().zip(multipliers) {
            if !matches!(role, RowRole::Equality(_)) && value.is_negative() {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        let Some(weighted) = self.weighted_rows(roles, multipliers, cooperation).await? else {
            return Ok(false);
        };
        if !weighted.intercept.is_positive() {
            return Ok(false);
        }
        for value in weighted.gradient {
            if !value.is_zero() {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
    pub(super) fn weighted_rows<'a>(
        &'a mut self,
        roles: &'a [RowRole],
        multipliers: &'a [Rational],
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Option<Weighted>>> + 'a + use<'a, 'source>
    {
        self.weighted_rows_at(roles, multipliers, None, cooperation)
    }
    pub(super) async fn weighted_rows_at(
        &mut self,
        roles: &[RowRole],
        multipliers: &[Rational],
        bounds: Option<BoundValues<'_>>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Weighted>> {
        let policy = self.policy;
        if roles.len() != multipliers.len() {
            return Ok(None);
        }
        self.prepare_origin(cooperation).await?;
        let origin = self.origin.as_ref().expect("original affine intercepts");
        let mut seeds = Vec::new();
        let mut bound_gradient = std::collections::HashMap::<usize, Rational>::new();
        let mut intercept = Rational::zero();
        for (role, weight) in roles.iter().zip(multipliers) {
            if weight.is_zero() {
                continue;
            }
            match *role {
                RowRole::Equality(row) | RowRole::Inequality(row) => {
                    let ranges = if matches!(role, RowRole::Equality(_)) {
                        &self.equalities
                    } else {
                        &self.inequalities
                    };
                    let index = ranges.partition_point(|range| range.end <= row);
                    let Some(range) = ranges.get(index).filter(|range| row >= range.start) else {
                        return Ok(None);
                    };
                    let constraint = &self.formulation.constraints[range.constraint];
                    let local = row - range.start;
                    let Some((left, right)) = self.geometries[range.constraint].indices(local)
                    else {
                        return Ok(None);
                    };
                    let mut pair = [
                        (constraint.left, left, 1i8),
                        (constraint.right, right, -1i8),
                    ];
                    if constraint.relation == Relation::GreaterEqual {
                        pair[0].2 = -1;
                        pair[1].2 = 1;
                    }
                    for (root, source, sign) in pair {
                        let value = if sign < 0 { -weight } else { weight.clone() };
                        let Some(offset) = origin.value(root, source, policy)? else {
                            return Ok(None);
                        };
                        intercept = intercept
                            .add(
                                &value.mul(&offset, policy, cooperation).await?,
                                policy,
                                cooperation,
                            )
                            .await?;
                        seeds.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        seeds.push(exact::Seed {
                            root,
                            row: source,
                            value,
                        });
                    }
                }
                RowRole::Lower(column) | RowRole::Upper(column) => {
                    let Some((variable, row)) = self.program.layout.coordinate(column) else {
                        return Ok(None);
                    };
                    let lower = matches!(role, RowRole::Lower(_));
                    let value = if let Some(bounds) = bounds {
                        (if lower { bounds.lower } else { bounds.upper })
                            .get(column)
                            .copied()
                    } else {
                        variable
                            .view
                            .definition()
                            .bound_value(if lower { Bound::Lower } else { Bound::Upper }, row)
                    };
                    let Some(value) = value else {
                        return Ok(None);
                    };
                    if !value.is_finite() {
                        return Ok(None);
                    }
                    let value = rational(value, policy)?;
                    let contribution = if lower {
                        weight.clone()
                    } else {
                        weight.negated()
                    }
                    .mul(&value, policy, cooperation)
                    .await?;
                    intercept = intercept.add(&contribution, policy, cooperation).await?;
                    bound_gradient
                        .try_reserve(1)
                        .map_err(|e| policy.reserve(e))?;
                    let target = bound_gradient.entry(column).or_insert_with(Rational::zero);
                    *target = target
                        .add(
                            &if lower {
                                weight.negated()
                            } else {
                                weight.clone()
                            },
                            policy,
                            cooperation,
                        )
                        .await?;
                }
            }
            cooperation.completed(1).await?;
        }
        let mut gradient =
            exact::accumulate(self.program, origin, seeds, policy, cooperation).await?;
        for (column, value) in bound_gradient {
            let Some(target) = gradient.get_mut(column) else {
                return Ok(None);
            };
            *target = target.add(&value, policy, cooperation).await?;
            cooperation.completed(1).await?;
        }
        Ok(Some(Weighted {
            gradient,
            intercept,
        }))
    }
}
