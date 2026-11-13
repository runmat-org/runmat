//! Original-field equation closure, not independent spatial accuracy or measured usage.
//! Revalidation evaluates retained cell U/p and every owner-oriented q; it never solves.
use super::{invalid, AnalysisExecutionResult, FluidSteadySolverResult, FluidSteadyWorkload};
use crate::{
    execution::{
        AnalysisWorkloadIdentity, NumericalPlanIdentity, NumericalReceiptIdentity,
        SolverResultIdentity, StepIdentity,
    },
    physics::cfd::canonical::CanonicalFluidReservation,
};
use runmat_canonical_codec::{
    canonical_payload_materialization_resident_bytes_estimate, decode_contract, encode_contract,
    profile_serialized, profile_serialized_shape, CanonicalLimits, SerializedCanonicalProfile,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-fluid-numerical-receipt/v1\0";
const DOMAIN: &str = "fluid-numerical-receipt";
const PROVENANCE: &str = "CpuFp64SharedSparseRightDiagonalGmres";

// Exact fixed-schema streaming-profile law: shared profiler reserves nine bytes per
// value/key/container, plus actual key/string lengths and its maximum envelope.
// Digest's canonical Display is "sha256:" + 64 hex characters. No variable vectors.
const fn keys(keys: &[&str]) -> u64 {
    let mut n = 0;
    let mut bytes = 9; // map container
    while n < keys.len() {
        bytes += 9 + keys[n].len() as u64;
        n += 1;
    }
    bytes
}
const ARTIFACT_KEYS: &[&str] = &[
    "schema_version",
    "identity",
    "source_workload",
    "source_result",
    "receipt",
];
const RECEIPT_KEYS: &[&str] = &[
    "producer",
    "mesh_digest",
    "numerical_plan",
    "numerical_input",
    "authored_authority",
    "provenance",
    "cells",
    "faces",
    "internal_faces",
    "boundary_faces",
    "outer_iterations",
    "linear_iterations",
    "maximum_outer_iterations",
    "maximum_linear_iterations",
    "gmres_restart",
    "linear_relative_tolerance",
    "physical_relative_tolerance",
    "momentum_relative",
    "continuity_relative",
    "flux_constitutive_relative",
    "global_momentum_relative",
    "maximum_cell_momentum_relative",
    "global_boundary_flux_m3_per_s",
    "reservation",
    "preparation_resident_bytes",
];
const RESERVATION_KEYS: &[&str] = &[
    "cells",
    "faces",
    "sparse_entries",
    "resident_bytes_upper_bound",
    "verification_resident_bytes_upper_bound",
    "operator_work_upper_bound",
    "linear_iterations_upper_bound",
];
pub const FLUID_RECEIPT_MAX_ENCODED_BYTES: u64 = 128
    + 9
    + 9
    + 128
    + keys(ARTIFACT_KEYS)
    + 9
    + 3 * (9 + 71)
    + keys(RECEIPT_KEYS)
    + 5 * (9 + 71)
    + 9
    + PROVENANCE.len() as u64
    + 18 * 9
    + keys(RESERVATION_KEYS)
    + RESERVATION_KEYS.len() as u64 * 9;
const LIMITS: CanonicalLimits =
    CanonicalLimits::new(FLUID_RECEIPT_MAX_ENCODED_BYTES as usize, 32, 128, 16);

/// Streaming fixed wire-schema bound, before solving/receipt allocation.
/// Numeric scalars use the shared maximum width; digests/provenance are fixed.
pub fn fluid_receipt_encoding_profile() -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    struct Shape(u8);
    impl Serialize for Shape {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            let names = match self.0 {
                0 => ARTIFACT_KEYS,
                1 => RECEIPT_KEYS,
                _ => RESERVATION_KEYS,
            };
            let mut map = serializer.serialize_map(Some(names.len()))?;
            for name in names {
                match (self.0, *name) {
                    (0, "receipt") => map.serialize_entry(name, &Shape(1))?,
                    (1, "reservation") => map.serialize_entry(name, &Shape(2))?,
                    (0, "identity" | "source_workload" | "source_result")
                    | (
                        1,
                        "producer" | "mesh_digest" | "numerical_plan" | "numerical_input"
                        | "authored_authority",
                    ) => map.serialize_entry(name, &Digest::from_bytes([0; 32]))?,
                    (1, "provenance") => map.serialize_entry(name, PROVENANCE)?,
                    _ => map.serialize_entry(name, &u64::MAX)?,
                }
            }
            map.end()
        }
    }
    Ok(profile_serialized_shape(&Shape(0), LIMITS)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidNumericalReceipt {
    pub producer: StepIdentity,
    pub mesh_digest: Digest,
    pub numerical_plan: NumericalPlanIdentity,
    pub numerical_input: Digest,
    pub authored_authority: Digest,
    pub provenance: FluidSolverProvenance,
    pub cells: u64,
    pub faces: u64,
    pub internal_faces: u64,
    pub boundary_faces: u64,
    pub outer_iterations: usize,
    pub linear_iterations: u64,
    pub maximum_outer_iterations: usize,
    pub maximum_linear_iterations: usize,
    pub gmres_restart: usize,
    pub linear_relative_tolerance: f64,
    pub physical_relative_tolerance: f64,
    pub momentum_relative: f64,
    pub continuity_relative: f64,
    pub flux_constitutive_relative: f64,
    pub global_momentum_relative: f64,
    pub maximum_cell_momentum_relative: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// Signed physical flux, not a dimensionless tolerance or fabricated accuracy.
    pub global_boundary_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, f64>,
    /// Requested analytic bounds, NOT measured allocator/operator usage.
    pub reservation: CanonicalFluidReservation,
    pub preparation_resident_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluidSolverProvenance {
    CpuFp64SharedSparseRightDiagonalGmres,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FluidNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: FluidNumericalReceipt,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: FluidNumericalReceipt,
}
impl FluidNumericalReceiptArtifact {
    pub fn new(
        w: &FluidSteadyWorkload,
        result: &FluidSteadySolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        admit_receipt_materialization(
            w,
            result,
            fluid_receipt_encoding_profile()?.materialization_resident_bytes_estimate,
        )?;
        // This authenticates every original field and independently recomputes its
        // physical/algebraic face/cell closure; self-consistent summaries alone fail.
        Self::new_on(&result.on_source(w, mesh)?, mesh)
    }

    /// The receipt of a result already validated against its source on `mesh`.
    pub fn new_on(
        on: &super::FluidResultOnSource<'_>,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let (w, result) = (on.workload(), on.result());
        admit_receipt_materialization(
            w,
            result,
            fluid_receipt_encoding_profile()?.materialization_resident_bytes_estimate,
        )?;
        let original = result.original();
        let r = &original.residual;
        let cells = original.velocity_m_per_s.canonical_values().len() as u64;
        let faces = original.face_volume_flux_m3_per_s.canonical_values().len() as u64;
        let boundary_faces = mesh.topology.boundary_faces.len() as u64;
        let receipt = FluidNumericalReceipt {
            producer: result.producer_identity(),
            mesh_digest: Digest::from_bytes(mesh.canonical_digest.0),
            numerical_plan: result.numerical_plan_identity(),
            numerical_input: original.numerical_input_identity,
            authored_authority: original.authored_authority_identity,
            provenance: FluidSolverProvenance::CpuFp64SharedSparseRightDiagonalGmres,
            cells,
            faces,
            internal_faces: faces
                .checked_sub(boundary_faces)
                .ok_or_else(|| invalid("fluid receipt face counts"))?,
            boundary_faces,
            outer_iterations: original.outer_iterations,
            linear_iterations: original.linear_iterations,
            maximum_outer_iterations: w.policy.maximum_outer_iterations,
            maximum_linear_iterations: w.policy.maximum_linear_iterations,
            gmres_restart: w.policy.gmres_restart,
            linear_relative_tolerance: w.policy.linear_relative_tolerance,
            physical_relative_tolerance: w.policy.physical_relative_tolerance,
            momentum_relative: r.momentum_relative,
            continuity_relative: r.continuity_relative,
            flux_constitutive_relative: r.flux_constitutive_relative,
            global_momentum_relative: r.global_momentum_relative,
            maximum_cell_momentum_relative: r
                .cell_momentum_relative
                .iter()
                .copied()
                .fold(0.0, f64::max),
            global_boundary_flux_m3_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::VolumeFlowRate,
                _,
            >::new(
                r.global_boundary_flux_m3_per_s.canonical_value()
            ),
            reservation: result.reservation(),
            preparation_resident_bytes: result.preparation_resident_bytes(),
        };
        let source_workload = w.identity()?;
        let source_result = result.identity()?;
        let value = Self {
            schema_version: 1,
            identity: receipt_identity(source_workload, source_result, &receipt)?,
            source_workload,
            source_result,
            receipt,
        };
        value.validate()?;
        Ok(value)
    }
    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.source_workload
    }
    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }
    pub const fn receipt(&self) -> FluidNumericalReceipt {
        self.receipt
    }
    /// Wire integrity only; source authority requires validate_against/validate_for.
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        let within = |v: f64| v.is_finite() && v >= 0.0 && v <= r.physical_relative_tolerance;
        if self.schema_version != 1
            || r.cells == 0
            || r.boundary_faces == 0
            || r.internal_faces.checked_add(r.boundary_faces) != Some(r.faces)
            || r.reservation.cells != r.cells
            || r.reservation.faces != r.faces
            || r.outer_iterations == 0
            || r.outer_iterations > r.maximum_outer_iterations
            || r.linear_iterations > r.reservation.linear_iterations_upper_bound
            || r.maximum_linear_iterations == 0
            || r.gmres_restart == 0
            || !r.linear_relative_tolerance.is_finite()
            || r.linear_relative_tolerance <= 0.0
            || r.linear_relative_tolerance >= 1.0
            || !r.physical_relative_tolerance.is_finite()
            || r.physical_relative_tolerance <= 0.0
            || r.physical_relative_tolerance >= 1.0
            || ![
                r.momentum_relative,
                r.continuity_relative,
                r.flux_constitutive_relative,
                r.global_momentum_relative,
                r.maximum_cell_momentum_relative,
            ]
            .into_iter()
            .all(within)
            || !r
                .global_boundary_flux_m3_per_s
                .canonical_value()
                .is_finite()
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "fluid receipt closure/count/iteration/source integrity",
            ));
        }
        self.encoding_profile()?;
        Ok(())
    }
    pub fn validate_against(
        &self,
        w: &FluidSteadyWorkload,
        r: &FluidSteadySolverResult,
        m: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate_against_on(&r.on_source(w, m)?, m)
    }

    /// [`Self::validate_against`] for a result already validated against its source on `m`.
    pub fn validate_against_on(
        &self,
        on: &super::FluidResultOnSource<'_>,
        m: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new_on(on, m)? {
            return Err(invalid(
                "fluid receipt differs from original workload/result/mesh",
            ));
        }
        Ok(())
    }
    pub fn encoding_profile(&self) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        Ok(profile_serialized(self, LIMITS)?)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(PREFIX, DOMAIN, self, LIMITS)?)
    }
    /// Bounded decode grants integrity only, never authority to a worker/publication.
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let w: Wire = decode_contract(PREFIX, DOMAIN, bytes, LIMITS)?;
        let value = Self {
            schema_version: w.schema_version,
            identity: w.identity,
            source_workload: w.source_workload,
            source_result: w.source_result,
            receipt: w.receipt,
        };
        value.validate()?;
        Ok(value)
    }
    /// Check expected transport digest before decoding, then exact original authority.
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        expected_digest: Digest,
        w: &FluidSteadyWorkload,
        r: &FluidSteadySolverResult,
        m: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        if bytes.len() as u64 > FLUID_RECEIPT_MAX_ENCODED_BYTES
            || Digest::sha256(bytes) != expected_digest
        {
            return Err(invalid("fluid receipt transport digest/size"));
        }
        admit_receipt_materialization(
            w,
            r,
            canonical_payload_materialization_resident_bytes_estimate(bytes)?,
        )?;
        let value = Self::from_canonical_bytes(bytes)?;
        value.validate_against(w, r, m)?;
        Ok(value)
    }
}
fn admit_receipt_materialization(
    w: &FluidSteadyWorkload,
    r: &FluidSteadySolverResult,
    bytes: u64,
) -> AnalysisExecutionResult<()> {
    let peak = r
        .preparation_resident_bytes()
        .checked_add(r.reservation().resident_bytes_upper_bound)
        .and_then(|v| v.checked_add(bytes))
        .ok_or_else(|| invalid("fluid receipt materialization admission overflow"))?;
    if peak > w.resident_ceiling()? {
        return Err(invalid("fluid receipt materialization resident bound"));
    }
    Ok(())
}
fn receipt_identity(
    w: AnalysisWorkloadIdentity,
    r: SolverResultIdentity,
    receipt: &FluidNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    NumericalReceiptIdentity::new(
        r,
        Digest::sha256(encode_contract(
            PREFIX,
            "fluid-numerical-receipt-content",
            &(w, receipt),
            LIMITS,
        )?),
    )
}
#[cfg(test)]
mod tests;
