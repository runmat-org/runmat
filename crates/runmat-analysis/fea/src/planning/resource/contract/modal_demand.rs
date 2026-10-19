//! Resource demand for the monolithic structural modal kernel.

use serde::{Deserialize, Serialize};

use crate::{
    execution::StructuralModalWorkloadIdentity,
    pipeline::structural_linearization::StructuralLinearizationIdentity,
};

use super::{
    Count, PreparedInputResourceFacts, PreparedOutputResourceFacts,
    StructuralOperatorResourceDemand,
};

/// K, M, retained damping, adaptive truncation-boundary probing, and dense Rayleigh-Ritz work.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResourceDemand {
    /// Fresh original constraint construction, rederived by workers.
    pub constraint_compilation_peak_bytes: super::ByteCount,
    /// Additional original local stiffness capture, selection and projected-proof reservation.
    /// Re-derived from original mesh/constraints, reduced support and physical operator family.
    pub original_energy_source_bytes: super::ByteCount,
    pub workload: StructuralModalWorkloadIdentity,
    pub operator: StructuralOperatorResourceDemand,
    /// Exact preload linearization carried by a prestressed-modal task.
    pub prestress_linearization: Option<StructuralLinearizationIdentity>,
    /// Full-space nonzeros imported from the exact geometric-stiffness state, or zero for an
    /// ordinary modal extraction.
    pub geometric_stiffness_nonzero_count: Count,
    /// Reduced nonzeros in the exact symbolic union K + Kg used by the modal solve.
    pub reduced_tangent_nonzero_count_upper_bound: Count,
    pub retained_mode_count_upper_bound: Count,
    pub residual_vector_count: Count,
    pub residual_vector_static_iteration_count_upper_bound: Count,
    pub residual_vector_mass_orthogonalization_passes: Count,
    pub modal_phase_value_count_upper_bound: Count,
    pub modal_output_value_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResponseSpectrumDirectionResourceDemand {
    pub operator: StructuralOperatorResourceDemand,
    pub spectrum_point_count: Count,
    pub mode_count_upper_bound: Count,
    pub correlation_pair_count_upper_bound: Count,
    pub response_vector_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
