//! Solve options and reports.

use crate::ids::{ConstraintId, EntityId, VarId};

/// Previously solved values (the `solved { }` cache). Values for variables the system does not
/// have, or that are not free, are ignored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WarmStart {
    pub values: Vec<(VarId, f64)>,
}

impl WarmStart {
    /// Every variable's value from an earlier report.
    pub fn from_report(report: &SolveReport) -> Self {
        Self {
            values: report
                .values
                .iter()
                .enumerate()
                .map(|(i, &v)| (VarId(i as u32), v))
                .collect(),
        }
    }
}

/// Solver settings. The defaults suit millimetre sketches and unit-scale assemblies.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    /// Convergence: every residual's magnitude is at most this.
    pub tolerance: f64,
    /// Relative rank tolerance on the row-normalised Jacobian (DOF and redundancy).
    pub rank_tolerance: f64,
    /// Iteration cap per sub-problem (Newton and LM steps together).
    pub max_iterations: usize,
    /// Solve Dulmage–Mendelsohn blocks one after another (falls back to the whole component
    /// when a block fails).
    pub decompose: bool,
    /// Compute DOF, redundancy and per-entity diagnostics.
    pub diagnose: bool,
    /// Skip per-entity DOF when a component's null-space basis would exceed this many entries.
    pub max_null_space_entries: usize,
    /// Starting values from an earlier solve.
    pub warm_start: Option<WarmStart>,
    /// The warm-start match tolerance (absolute, scaled by `1 + |value|`).
    pub warm_start_tolerance: f64,
    /// Step weight of dragged variables relative to the others (smaller keeps them closer to
    /// the drag target).
    pub drag_weight: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            rank_tolerance: 1e-8,
            max_iterations: 200,
            decompose: true,
            diagnose: true,
            max_null_space_entries: 20_000_000,
            warm_start: None,
            warm_start_tolerance: 1e-9,
            drag_weight: 1e-4,
        }
    }
}

/// Overall outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Converged, no DOF left, no redundant constraints.
    WellDefined,
    /// Converged with DOF left and no redundant constraints.
    UnderDefined,
    /// Converged, but some constraints are redundant (implied by others). DOF may still be
    /// positive (a spatial four-bar is over-defined and has 1 DOF).
    OverDefined,
    /// Some constraints contradict each other; see [`SolveReport::conflicting`].
    Conflicting,
    /// The solver stopped without converging and found no contradiction (a bad start, or a
    /// configuration with no real solution nearby).
    NotConverged,
}

/// A constraint that is linearly dependent on others at the solution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependentConstraint {
    /// The later constraint (in creation order) that adds nothing new.
    pub constraint: ConstraintId,
    /// The earlier constraints it depends on. Empty when it reads only fixed values.
    pub depends_on: Vec<ConstraintId>,
}

/// How a sub-problem was solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Nothing to do: the starting values satisfied every constraint, or there were none.
    None,
    /// Damped Newton (minimum-norm least squares) only.
    Newton,
    /// Newton fell back to Levenberg–Marquardt.
    LevenbergMarquardt,
}

/// Per connected component.
#[derive(Clone, Debug)]
pub struct ComponentReport {
    pub vars: Vec<VarId>,
    pub constraints: Vec<ConstraintId>,
    /// Jacobian rank at the solution.
    pub rank: usize,
    pub dof: usize,
    pub converged: bool,
    pub iterations: usize,
    /// Euclidean norm of the component's residual vector.
    pub residual_norm: f64,
    pub method: Method,
    /// DM blocks solved separately (1 when solved whole).
    pub blocks: usize,
}

/// Per entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityDof {
    pub entity: EntityId,
    /// Remaining DOF of the entity's defining variables (`None` when skipped for size).
    pub dof: Option<usize>,
}

/// Warm-start outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmStartReport {
    /// Cached values that were applied.
    pub applied: usize,
    /// The cached values already satisfied every constraint (no iteration was needed).
    pub cache_satisfied: bool,
    /// The solution equals the cache within tolerance (the cache need not be rewritten).
    pub matches: bool,
    pub max_deviation: f64,
}

/// Drag outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragReport {
    /// Euclidean distance between the dragged variables and their targets after the solve.
    pub distance_to_target: f64,
}

/// A measured reference (driven) dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    pub constraint: ConstraintId,
    pub value: f64,
}

/// The result of [`solve`](crate::solve) or [`drag`](crate::drag).
#[derive(Clone, Debug)]
pub struct SolveReport {
    /// Every variable's value, indexed by [`VarId::index`].
    pub values: Vec<f64>,
    pub status: Status,
    pub converged: bool,
    /// Euclidean norm of all active residuals.
    pub residual_norm: f64,
    /// Largest residual magnitude.
    pub max_residual: f64,
    pub iterations: usize,
    /// Total remaining DOF (free variables minus Jacobian rank).
    pub dof: usize,
    pub components: Vec<ComponentReport>,
    /// Remaining DOF per entity, in entity order.
    pub entity_dof: Vec<EntityDof>,
    /// Per variable: no remaining motion (fixed and driven variables count as determined).
    pub determined: Vec<bool>,
    /// Consistent constraints implied by others.
    pub redundant: Vec<DependentConstraint>,
    /// Constraints that contradict others; `depends_on` names the ones they fight.
    pub conflicting: Vec<DependentConstraint>,
    /// Range limits that ended active (clamped at a bound), with the bound used.
    pub active_limits: Vec<(ConstraintId, f64)>,
    pub measurements: Vec<Measurement>,
    pub warm_start: Option<WarmStartReport>,
    pub drag: Option<DragReport>,
}

impl SolveReport {
    pub fn value(&self, v: VarId) -> f64 {
        self.values[v.index()]
    }
    pub fn entity_dof(&self, e: EntityId) -> Option<usize> {
        self.entity_dof.get(e.index()).and_then(|d| d.dof)
    }
    pub fn measurement(&self, c: ConstraintId) -> Option<f64> {
        self.measurements
            .iter()
            .find(|m| m.constraint == c)
            .map(|m| m.value)
    }
    /// The redundant constraint ids.
    pub fn redundant_ids(&self) -> Vec<ConstraintId> {
        self.redundant.iter().map(|d| d.constraint).collect()
    }
    /// Every constraint involved in a conflict (each conflicting one plus what it fights).
    pub fn conflict_set(&self) -> Vec<ConstraintId> {
        let mut out: Vec<ConstraintId> = self
            .conflicting
            .iter()
            .flat_map(|d| std::iter::once(d.constraint).chain(d.depends_on.iter().copied()))
            .collect();
        out.sort();
        out.dedup();
        out
    }
}
