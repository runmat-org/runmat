use super::GcPlan;
use crate::state::CacheState;

pub fn apply_plan(state: &mut CacheState, plan: &GcPlan) {
    for digest in &plan.delete {
        state.objects.remove(digest);
        state.access.remove(digest);
        state.corruptions.remove(digest);
        state.materializations.remove(digest);
    }
}
