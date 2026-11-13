//! Refinement deadlines advance only on committed, actual callback points.
//! Modeled SQP probes and supplied observations never advance these clocks.
use super::{
    invalid,
    queue::{Origin, PhaseId},
    sampling::Problem,
};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Refinement {
    NonlinearLocal,
    MultiStart,
    TreeLocal,
    TreeReset,
}
pub(super) struct Request {
    pub kind: Refinement,
    phase: PhaseId,
    points: usize,
    adaptive: usize,
    revision: u64,
}
pub(super) struct Applicability {
    pub problem: Problem,
    pub nonlinear: bool,
    pub linear: bool,
    pub parallel: bool,
}
pub(super) struct Schedule {
    phase: PhaseId,
    variables: usize,
    pair_interval: usize,
    multistart_interval: usize,
    points: usize,
    adaptive: usize,
    local_at: usize,
    multistart_at: usize,
    tree_at: usize,
    tree_adaptive_at: usize,
    revision: u64,
}
impl Schedule {
    pub(super) fn new(variables: usize, phase: PhaseId) -> BuiltinResult<Self> {
        if variables == 0 {
            return Err(invalid(
                "refinement scheduling requires an original variable extent",
            ));
        }
        Ok(Self {
            phase,
            variables,
            pair_interval: variables
                .checked_mul(2)
                .ok_or_else(|| invalid("refinement interval overflow"))?,
            multistart_interval: variables
                .checked_mul(if variables < 15 { 10 } else { 2 })
                .ok_or_else(|| invalid("multistart interval overflow"))?
                .max(50),
            points: 0,
            adaptive: 0,
            local_at: 0,
            multistart_at: 0,
            tree_at: 0,
            tree_adaptive_at: 0,
            revision: 0,
        })
    }
    pub(super) fn completed(&mut self, phase: PhaseId, origin: Origin) -> BuiltinResult<()> {
        // An accepted old-phase call remains a real trial and may improve best,
        // but does not spend a new phase's refinement interval.
        if phase != self.phase {
            return Ok(());
        }
        let points = self
            .points
            .checked_add(1)
            .ok_or_else(|| invalid("refinement point counter overflow"))?;
        let adaptive = self
            .adaptive
            .checked_add(usize::from(origin == Origin::Adaptive))
            .ok_or_else(|| invalid("refinement step counter overflow"))?;
        self.points = points;
        self.adaptive = adaptive;
        Ok(())
    }
    pub(super) fn due(&self, selected: Applicability, before_reset: bool) -> Option<Request> {
        let integer = selected.problem != Problem::Continuous;
        // The nonlinear algorithm explicitly calls Tree instead of its local
        // SQP step when integers are present, including binary variables.
        let tree = integer
            && (selected.nonlinear
                || selected.problem != Problem::Binary
                    && (selected.linear || self.variables >= 15));
        let kind = if tree && before_reset {
            Refinement::TreeReset
        } else if tree
            && (selected.nonlinear && self.points - self.tree_at >= self.pair_interval
                || !selected.nonlinear
                    && self.adaptive - self.tree_adaptive_at >= self.pair_interval
                || selected.parallel && self.adaptive > self.tree_adaptive_at)
        {
            Refinement::TreeLocal
        } else if !integer
            && selected.nonlinear
            && self.adaptive - self.multistart_at >= self.multistart_interval
        {
            Refinement::MultiStart
        } else if !integer
            && selected.nonlinear
            && self.points - self.local_at >= self.pair_interval
        {
            Refinement::NonlinearLocal
        } else {
            return None;
        };
        Some(Request {
            kind,
            phase: self.phase,
            points: self.points,
            adaptive: self.adaptive,
            revision: self.revision,
        })
    }
    /// Call at the real host refinement's start, not when merely asking what
    /// is due. A rejected proposal or a modeled numerical stop must not cause
    /// repeated unbounded solves before the next expensive completion.
    pub(super) fn started(&mut self, request: Request) -> BuiltinResult<()> {
        if request.phase != self.phase
            || request.points != self.points
            || request.adaptive != self.adaptive
            || request.revision != self.revision
        {
            return Err(invalid(
                "refinement request lost its committed trial generation",
            ));
        }
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("refinement request generation overflow"))?;
        match request.kind {
            Refinement::NonlinearLocal => self.local_at = request.points,
            Refinement::MultiStart => {
                self.multistart_at = request.adaptive;
                self.local_at = request.points;
            }
            Refinement::TreeLocal | Refinement::TreeReset => {
                self.tree_at = request.points;
                self.tree_adaptive_at = request.adaptive;
            }
        }
        Ok(())
    }
    pub(super) fn reset(&mut self, phase: PhaseId) -> BuiltinResult<()> {
        *self = Self::new(self.variables, phase)?;
        Ok(())
    }
}
