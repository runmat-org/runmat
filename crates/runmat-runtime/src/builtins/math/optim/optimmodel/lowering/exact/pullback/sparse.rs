//! Reusable original topological index and operator facts for selected rows.
//! Entries retain original column/ordinal identities, not Jacobian coefficients.
use super::super::super::{linear::Linear, sparse_rows::SparseRows};
use super::*;
use std::{collections::BinaryHeap, rc::Rc};
pub(super) struct LinearPlan {
    pub plan: Linear,
    pub denominators: Vec<Rational>,
}
pub(in crate::builtins::math::optim::optimmodel::lowering) struct SparsePlan {
    order: HashMap<GcHandle, usize>,
    ready: BinaryHeap<usize>,
    pub(super) linear: HashMap<GcHandle, Rc<LinearPlan>>,
    pub(super) products: HashMap<GcHandle, Rc<Vec<products::Entry>>>,
    rows: HashMap<GcHandle, Rc<SparseRows>>,
}
impl SparsePlan {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn new(
        program: &Program,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut order = HashMap::new();
        order
            .try_reserve(program.source.order.len())
            .map_err(|e| policy.reserve(e))?;
        for (ordinal, &root) in program.source.order.iter().enumerate() {
            if order.insert(root, ordinal).is_some() {
                return Err(policy.invalid("original reverse topology repeats a node"));
            }
            c.completed(1).await?;
        }
        Ok(Self {
            order,
            ready: BinaryHeap::new(),
            linear: HashMap::new(),
            products: HashMap::new(),
            rows: HashMap::new(),
        })
    }
    pub(super) fn enqueue(&mut self, root: GcHandle, policy: Policy) -> BuiltinResult<()> {
        let ordinal = *self
            .order
            .get(&root)
            .ok_or_else(|| policy.invalid("reverse source has no original topological identity"))?;
        self.ready.try_reserve(1).map_err(|e| policy.reserve(e))?;
        self.ready.push(ordinal);
        Ok(())
    }
    pub(super) async fn rows(
        &mut self,
        root: GcHandle,
        matrix: &runmat_value::SparseTensor,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rc<SparseRows>> {
        if let Some(rows) = self.rows.get(&root) {
            return Ok(Rc::clone(rows));
        }
        let rows = Rc::new(SparseRows::prepare(matrix, policy, c).await?);
        self.rows.try_reserve(1).map_err(|e| policy.reserve(e))?;
        self.rows.insert(root, Rc::clone(&rows));
        Ok(rows)
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn row(
        &mut self,
        program: &Program,
        constants: &Evaluated,
        root: GcHandle,
        row: usize,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Weights> {
        if !self.ready.is_empty() {
            return Err(policy.invalid("reverse row started before prior work retired"));
        }
        let mut state = State {
            program,
            pending: HashMap::new(),
            gradient: Gradient::Sparse(HashMap::new()),
            plan: Some(self),
            policy,
        };
        state.add(root, row, Rational::from_usize(1), c).await?;
        while let Some(ordinal) = state
            .plan
            .as_mut()
            .expect("selected reverse plan")
            .ready
            .pop()
        {
            let handle = program.source.order[ordinal];
            let weights = state.pending.remove(&handle).ok_or_else(|| {
                policy.invalid("selected reverse topology repeated completed work")
            })?;
            if step::propagate(&mut state, constants, handle, weights, c).await? {
                c.completed(1).await?;
            }
        }
        if !state.pending.is_empty() {
            return Err(policy.invalid("selected reverse topology left unresolved adjoints"));
        }
        let Gradient::Sparse(gradient) = state.gradient else {
            unreachable!("sparse reverse schedule")
        };
        Ok(gradient)
    }
}
