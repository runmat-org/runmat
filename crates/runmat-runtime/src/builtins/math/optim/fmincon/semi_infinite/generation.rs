//! Transactional mesh generations for accepted SQP states.
//! A proposal cannot rewrite the generation used by trial/derivative probes.
//! Warm multiplier transport is explicitly not an optimality certificate.
use super::bernstein::invalid;
use super::mesh::{self, Analysis, FamilyId, MeshRevision, PeakId, Samples};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FiniteExtents {
    pub inequality: usize,
    pub equality: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum InequalityRow {
    Finite(usize),
    Peak(PeakId),
}
pub(super) struct Generation {
    pub revision: MeshRevision,
    pub finite: FiniteExtents,
    pub families: Vec<Analysis>,
    pub rows: Vec<InequalityRow>,
}
/// The same immutable admitted row generation is borrowed for an entire
/// major iteration, including finite-difference and line-search probes.
#[derive(Clone)]
pub(super) struct FrozenRows {
    generation: Arc<Generation>,
}
impl FrozenRows {
    pub fn generation(&self) -> &Generation {
        &self.generation
    }
    pub async fn evaluate(
        &self,
        samples: &[Arc<Samples>],
        finite: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        if samples.len() != self.generation.families.len()
            || finite.len() != self.generation.finite.inequality
        {
            return Err(invalid(
                "trial callback changed the frozen constraint family layout",
            ));
        }
        let mut result =
            runmat_value::exact_capacity_vec(self.generation.rows.len()).map_err(|source| {
                crate::build_runtime_error("fseminf: constraint row allocation failed")
                    .with_source(source)
                    .build()
            })?;
        for &value in finite {
            if !value.is_finite() {
                return Err(invalid("finite nonlinear trial constraint is undefined"));
            }
            result.push(value);
            cooperation.completed(1).await?;
        }
        for (family, trial) in self.generation.families.iter().zip(samples) {
            let base = &family.samples;
            if !base.same_domain(trial) {
                return Err(invalid(
                    "trial callback changed the fixed physical family domain",
                ));
            }
            let same_mesh = base.rows == trial.rows
                && base.columns == trial.columns
                && base.spacing.column == trial.spacing.column
                && base.spacing.row == trial.spacing.row;
            for peak in &family.peaks {
                let value = if same_mesh {
                    peak.stencil.evaluate(&trial.values)?
                } else {
                    trial.interpolated_at(peak.location)?
                };
                // Preserve the accepted interpolation-maximization guard
                // within this row generation; remaximize at the next
                // accepted-state refresh before any KKT publication.
                let value = value + (peak.upper - peak.value).max(0.0);
                if !value.is_finite() {
                    return Err(invalid("guarded interpolated peak overflow"));
                }
                result.push(value);
                cooperation.completed(1).await?;
            }
        }
        Ok(result)
    }
}
pub(super) struct WarmMultipliers {
    values: Vec<f64>,
}
impl WarmMultipliers {
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}
pub(super) struct Proposal {
    predecessor: Option<MeshRevision>,
    pub next: Arc<Generation>,
    pub warm: WarmMultipliers,
}
pub(super) struct State {
    families: usize,
    current: Option<Arc<Generation>>,
    next_revision: u64,
}
impl State {
    pub fn new(families: usize) -> BuiltinResult<Self> {
        if families == 0 {
            return Err(invalid("ntheta must be a positive integer"));
        }
        families
            .checked_mul(2)
            .ok_or_else(|| invalid("sampling interval extent overflow"))?;
        Ok(Self {
            families,
            current: None,
            next_revision: 1,
        })
    }
    pub fn family_count(&self) -> usize {
        self.families
    }
    pub fn has_current(&self) -> bool {
        self.current.is_some()
    }
    pub fn probe_intervals(&self) -> BuiltinResult<Vec<f64>> {
        let Some(current) = &self.current else {
            return self.intervals();
        };
        let mut values = runmat_value::exact_capacity_vec(
            self.families
                .checked_mul(2)
                .ok_or_else(|| invalid("sampling interval extent overflow"))?,
        )
        .map_err(|source| {
            crate::build_runtime_error("fseminf: interval allocation failed")
                .with_source(source)
                .build()
        })?;
        for family in &current.families {
            values.push(family.samples.spacing.column);
        }
        for family in &current.families {
            values.push(family.samples.spacing.row.unwrap_or(0.0));
        }
        Ok(values)
    }
    pub fn frozen(&self) -> BuiltinResult<FrozenRows> {
        Ok(FrozenRows {
            generation: self
                .current
                .as_ref()
                .cloned()
                .ok_or_else(|| invalid("sampling generation has not been admitted"))?,
        })
    }
    pub fn intervals(&self) -> BuiltinResult<Vec<f64>> {
        let mut values = runmat_value::exact_capacity_vec(
            self.families
                .checked_mul(2)
                .ok_or_else(|| invalid("sampling interval extent overflow"))?,
        )
        .map_err(|source| {
            crate::build_runtime_error("fseminf: interval allocation failed")
                .with_source(source)
                .build()
        })?;
        // MATLAB column-major ntheta-by-2 sampling matrix. NaN is used only
        // for the original bootstrap request, never as a refined spacing.
        for family in 0..self.families {
            values.push(
                self.current
                    .as_ref()
                    .map_or(f64::NAN, |g| g.families[family].recommended.column),
            );
        }
        for family in 0..self.families {
            values.push(self.current.as_ref().map_or(f64::NAN, |g| {
                g.families[family].recommended.row.unwrap_or(0.0)
            }));
        }
        Ok(values)
    }
    pub async fn propose(
        &self,
        samples: Vec<Arc<Samples>>,
        finite: FiniteExtents,
        previous_multipliers: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Proposal> {
        if samples.len() != self.families {
            return Err(invalid(
                "seminfcon returned a different ntheta sample family count",
            ));
        }
        if self.current.as_ref().is_some_and(|g| g.finite != finite) {
            return Err(invalid(
                "finite nonlinear constraint extents changed across mesh generations",
            ));
        }
        let revision = MeshRevision(self.next_revision);
        let mut families = runmat_value::exact_capacity_vec(self.families).map_err(|source| {
            crate::build_runtime_error("fseminf: mesh generation allocation failed")
                .with_source(source)
                .build()
        })?;
        let mut row_count = finite.inequality;
        for (family, sample) in samples.into_iter().enumerate() {
            if self
                .current
                .as_ref()
                .is_some_and(|g| !g.families[family].samples.same_domain(&sample))
            {
                return Err(invalid(
                    "accepted callback changed a fixed physical family domain",
                ));
            }
            let analysis =
                mesh::analyze(sample, FamilyId(family), revision, tolerance, cooperation).await?;
            row_count = row_count
                .checked_add(analysis.peaks.len())
                .ok_or_else(|| invalid("constraint row count overflow"))?;
            families.push(analysis);
        }
        let mut rows = runmat_value::exact_capacity_vec(row_count).map_err(|source| {
            crate::build_runtime_error("fseminf: row identity allocation failed")
                .with_source(source)
                .build()
        })?;
        for row in 0..finite.inequality {
            rows.push(InequalityRow::Finite(row));
            cooperation.completed(1).await?;
        }
        for family in &families {
            for peak in &family.peaks {
                rows.push(InequalityRow::Peak(peak.id));
                cooperation.completed(1).await?;
            }
        }
        let next = Arc::new(Generation {
            revision,
            finite,
            families,
            rows,
        });
        let warm = remap(
            self.current.as_deref(),
            &next,
            previous_multipliers,
            cooperation,
        )
        .await?;
        Ok(Proposal {
            predecessor: self.current.as_ref().map(|g| g.revision),
            next,
            warm,
        })
    }
    pub fn commit(&mut self, proposal: Proposal) -> BuiltinResult<(FrozenRows, WarmMultipliers)> {
        if proposal.predecessor != self.current.as_ref().map(|g| g.revision)
            || proposal.next.revision != MeshRevision(self.next_revision)
        {
            return Err(invalid(
                "mesh proposal belongs to a stale accepted generation",
            ));
        }
        let successor = self
            .next_revision
            .checked_add(1)
            .ok_or_else(|| invalid("mesh generation identity exhausted"))?;
        self.current = Some(proposal.next);
        self.next_revision = successor;
        Ok((self.frozen()?, proposal.warm))
    }
    /// Check immediately before independent nonlinear KKT/termination
    /// publication. Merely remapped multipliers cannot satisfy this boundary.
    pub fn require_current(
        &self,
        point: &FrozenRows,
        linearization: &FrozenRows,
    ) -> BuiltinResult<()> {
        let Some(current) = &self.current else {
            return Err(invalid("no admitted mesh for optimality checking"));
        };
        if !Arc::ptr_eq(current, &point.generation)
            || !Arc::ptr_eq(current, &linearization.generation)
        {
            return Err(invalid(
                "optimality check refers to a superseded mesh generation",
            ));
        }
        Ok(())
    }
}
async fn remap(
    previous: Option<&Generation>,
    next: &Generation,
    duals: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<WarmMultipliers> {
    if let Some(previous) = previous {
        if previous.rows.len() != duals.len() {
            return Err(invalid(
                "warm multipliers have a different source row layout",
            ));
        }
    } else if !duals.is_empty() {
        return Err(invalid("initial mesh cannot adopt foreign multipliers"));
    }
    let mut values = runmat_value::exact_capacity_vec(next.rows.len()).map_err(|source| {
        crate::build_runtime_error("fseminf: multiplier remap allocation failed")
            .with_source(source)
            .build()
    })?;
    for row in &next.rows {
        let old = previous.and_then(|previous| match row {
            InequalityRow::Finite(index) => previous
                .rows
                .iter()
                .position(|r| *r == InequalityRow::Finite(*index)),
            InequalityRow::Peak(id) => {
                let target = &next.families[id.family.0].peaks;
                let peak = target.iter().find(|p| p.id == *id)?;
                // Reuse only a literal same-family physical parameter point,
                // never its old positional row number or a foreign family.
                let old = previous
                    .families
                    .get(id.family.0)?
                    .peaks
                    .iter()
                    .find(|p| p.location == peak.location)?;
                previous
                    .rows
                    .iter()
                    .position(|r| *r == InequalityRow::Peak(old.id))
            }
        });
        let value = old.map_or(0.0, |index| duals[index]);
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(
                "inequality warm multipliers must be finite and nonnegative",
            ));
        }
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(WarmMultipliers { values })
}
#[cfg(test)]
mod tests {
    use super::super::mesh::Spacing;
    use super::*;
    use futures::executor::block_on;
    fn cooperation() -> ScalarCooperation {
        ScalarCooperation::for_builtin(
            runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" },
            None,
        )
    }
    fn grid(spacing: f64) -> Arc<Samples> {
        let n = (1.0 / spacing).round() as usize + 1;
        Arc::new(
            Samples::new(
                (0..n)
                    .map(|i| -1.0 - (i as f64 * spacing - 0.5).powi(2))
                    .collect(),
                1,
                n,
                Spacing {
                    column: spacing,
                    row: None,
                },
            )
            .unwrap(),
        )
    }
    #[test]
    fn proposed_mesh_cannot_retarget_trials_or_reuse_stale_optimality_rows() {
        let mut state = State::new(1).unwrap();
        assert!(state.intervals().unwrap().iter().all(|x| x.is_nan()));
        let extents = FiniteExtents {
            inequality: 1,
            equality: 0,
        };
        let first =
            block_on(state.propose(vec![grid(0.25)], extents, &[], 1e-6, &mut cooperation()))
                .unwrap();
        let (old, warm) = state.commit(first).unwrap();
        assert_eq!(warm.values().len(), old.generation().rows.len());
        let initial = block_on(old.evaluate(&[grid(0.25)], &[-2.0], &mut cooperation())).unwrap();
        let proposed = block_on(state.propose(
            vec![grid(0.125)],
            extents,
            warm.values(),
            1e-6,
            &mut cooperation(),
        ))
        .unwrap();
        assert!(state.require_current(&old, &old).is_ok());
        assert_eq!(
            block_on(old.evaluate(&[grid(0.25)], &[-2.0], &mut cooperation())).unwrap(),
            initial
        );
        let (new, _) = state.commit(proposed).unwrap();
        assert!(state.require_current(&old, &old).is_err());
        assert!(state.require_current(&new, &old).is_err());
        assert!(state.require_current(&new, &new).is_ok());
        let resampled =
            block_on(old.evaluate(&[grid(0.125)], &[-2.0], &mut cooperation())).unwrap();
        for (actual, expected) in resampled.iter().zip(initial) {
            assert!(
                (actual - expected).abs() < 1e-10,
                "sampling changes do not retarget frozen physical rows"
            );
        }
    }
}
