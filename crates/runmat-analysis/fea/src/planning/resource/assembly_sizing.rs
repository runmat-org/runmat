//! Shared assembly scratch reservation for structural resource plans.

// Sparse rows and contribution entries have separate resident-byte accounting. This
// envelope covers the bounded per-worker wave of solid matrices plus the element-local
// quadrature matrices. Multiworker planners multiply the fixed reservation by worker count.
pub(super) const ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
pub(super) const FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER: u64 = 128 * 1024;

#[cfg(test)]
mod tests {
    use crate::{
        assembly::{
            elements::solid::Tetrahedron10Matrix30,
            solver_solid::MAX_SOLID_ELEMENT_CONTRIBUTIONS_PER_WORKER,
        },
        planning::resource::assembly_sizing::{
            ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
        },
    };

    #[test]
    fn reserved_wave_scratch_covers_curved_tet10_matrix_working_set() {
        let matrix = std::mem::size_of::<Tetrahedron10Matrix30>() as u64;
        let retained_per_element = 2 * matrix + 10 * std::mem::size_of::<usize>() as u64 + 128;
        let construction_scratch = 4 * matrix + 6 * 30 * std::mem::size_of::<f64>() as u64;
        for count in 1..=MAX_SOLID_ELEMENT_CONTRIBUTIONS_PER_WORKER as u64 {
            let reservation = FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER
                + count * ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT;
            let working_set = count * retained_per_element + construction_scratch;
            assert!(reservation >= working_set, "{count} curved Tet10 elements");
        }
    }
}
