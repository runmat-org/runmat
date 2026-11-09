//! Work counters for Tet10 curved-node optimization. They are reported after every sweep and at
//! the end of elevation, and decide nothing.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Default)]
pub(super) struct ElevationCounters {
    pub sweeps: AtomicU64,
    /// Invalid-element inventories built, and how many element certifications they ran or reused.
    pub inventories: AtomicU64,
    pub inventory_certified: AtomicU64,
    pub inventory_reused: AtomicU64,
    /// Node repairs started, skipped because nothing in the node's star changed since the same
    /// repair declined, ended without a candidate, and moved a node.
    pub repairs: AtomicU64,
    pub repairs_unchanged: AtomicU64,
    pub repairs_without_candidates: AtomicU64,
    pub repair_moves: AtomicU64,
    pub repair_candidates: AtomicU64,
    /// Margin profiles taken, and element margins they computed or reused.
    pub profiles: AtomicU64,
    pub profile_margins: AtomicU64,
    pub profile_margins_reused: AtomicU64,
    pub corner_star_repairs: AtomicU64,
    pub corner_star_moves: AtomicU64,
    pub corner_relocations: AtomicU64,
    pub corner_relocation_moves: AtomicU64,
}

impl ElevationCounters {
    pub fn add(counter: &AtomicU64, amount: u64) {
        counter.fetch_add(amount, Relaxed);
    }

    pub fn report(
        &self,
        stage: &'static str,
        invalid_elements: usize,
        started: std::time::Instant,
    ) {
        let value = |counter: &AtomicU64| counter.load(Relaxed);
        let [certified, certification_work, deepest, work_limited] =
            super::jacobian::SUBDIVISION.snapshot();
        tracing::info!(
            event_code = "meshing.tet10.optimization_counters",
            stage,
            invalid_elements,
            elapsed_s = started.elapsed().as_secs_f64(),
            sweeps = value(&self.sweeps),
            inventories = value(&self.inventories),
            inventory_certified = value(&self.inventory_certified),
            inventory_reused = value(&self.inventory_reused),
            repairs = value(&self.repairs),
            repairs_unchanged = value(&self.repairs_unchanged),
            repairs_without_candidates = value(&self.repairs_without_candidates),
            repair_moves = value(&self.repair_moves),
            repair_candidates = value(&self.repair_candidates),
            profiles = value(&self.profiles),
            profile_margins = value(&self.profile_margins),
            profile_margins_reused = value(&self.profile_margins_reused),
            corner_star_repairs = value(&self.corner_star_repairs),
            corner_star_moves = value(&self.corner_star_moves),
            corner_relocations = value(&self.corner_relocations),
            corner_relocation_moves = value(&self.corner_relocation_moves),
            certified_elements = certified,
            certification_work,
            deepest_subdivision = deepest,
            work_limited_elements = work_limited,
            "Tet10 curved-node optimization counters"
        );
    }
}
