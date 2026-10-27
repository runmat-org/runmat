//! Weighted original-DAG reverse accumulation. Per-node adjoints contain only
//! active rows; no rational model Jacobian or global-column matrix is formed.
use super::*;
mod binary;
mod difference;
mod dot;
mod edits;
mod gradient;
mod linear;
mod power;
mod products;
mod sparse;
mod step;
use gradient::Gradient;
pub(in crate::builtins::math::optim::optimmodel::lowering) use sparse::SparsePlan;
pub(super) type Weights = HashMap<usize, Rational>;
pub(in crate::builtins::math::optim::optimmodel::lowering) struct Seed {
    pub root: GcHandle,
    pub row: usize,
    pub value: Rational,
}
pub(super) struct State<'a> {
    program: &'a Program,
    pending: HashMap<GcHandle, Weights>,
    gradient: Gradient,
    plan: Option<&'a mut SparsePlan>,
    policy: Policy,
}
impl State<'_> {
    fn active(&self, root: GcHandle) -> BuiltinResult<bool> {
        let fact = self.program.facts.get(&root).ok_or_else(|| {
            self.policy
                .invalid("reverse source has no admitted affine fact")
        })?;
        Ok(fact.degree != array::Degree::Constant)
    }
    fn active_at(&self, root: GcHandle, row: usize) -> BuiltinResult<bool> {
        let fact = self.program.facts.get(&root).ok_or_else(|| {
            self.policy
                .invalid("reverse source has no row dependence fact")
        })?;
        if row >= evaluation::element_count(self.shape(root)?, self.policy)? {
            return Err(self
                .policy
                .invalid("reverse row exceeds original expression shape"));
        }
        Ok(fact.degree.at(row))
    }
    fn shape(&self, root: GcHandle) -> BuiltinResult<&[usize]> {
        self.program
            .source
            .nodes
            .get(&root)
            .map(|node| &node.node().shape[..])
            .ok_or_else(|| self.policy.invalid("reverse source node is unavailable"))
    }
    async fn add(
        &mut self,
        root: GcHandle,
        row: usize,
        value: Rational,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if value.is_zero() || !self.active_at(root, row)? {
            return Ok(());
        }
        if row >= evaluation::element_count(self.shape(root)?, self.policy)? {
            return Err(self
                .policy
                .invalid("reverse row exceeds original expression shape"));
        }
        if !self.pending.contains_key(&root) {
            if let Some(plan) = &mut self.plan {
                plan.enqueue(root, self.policy)?;
            }
        }
        self.pending
            .try_reserve(1)
            .map_err(|e| self.policy.reserve(e))?;
        let weights = self.pending.entry(root).or_default();
        weights.try_reserve(1).map_err(|e| self.policy.reserve(e))?;
        match weights.entry(row) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(value);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                *entry.get_mut() = entry.get().add(&value, self.policy, cooperation).await?;
                if entry.get().is_zero() {
                    entry.remove();
                }
            }
        }
        Ok(())
    }
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn accumulate(
    program: &Program,
    constants: &Evaluated,
    seeds: Vec<Seed>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Rational>> {
    let mut gradient =
        runmat_value::exact_capacity_vec(program.layout.columns).map_err(|e| policy.storage(e))?;
    for _ in 0..program.layout.columns {
        gradient.push(Rational::zero());
        cooperation.completed(1).await?;
    }
    let mut state = State {
        program,
        pending: HashMap::new(),
        gradient: Gradient::Dense(gradient),
        plan: None,
        policy,
    };
    for seed in seeds {
        state
            .add(seed.root, seed.row, seed.value, cooperation)
            .await?;
        cooperation.completed(1).await?;
    }
    for &handle in program.source.order.iter().rev() {
        let Some(weights) = state.pending.remove(&handle) else {
            continue;
        };
        if step::propagate(&mut state, constants, handle, weights, cooperation).await? {
            cooperation.completed(1).await?;
        }
    }
    if !state.pending.is_empty() {
        return Err(policy.invalid("reverse source topology left unresolved adjoints"));
    }
    let Gradient::Dense(gradient) = state.gradient else {
        unreachable!("dense reverse schedule")
    };
    Ok(gradient)
}
