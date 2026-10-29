//! Exact physical/gauge/energy-work closure, not independent spatial accuracy.
use super::*;
use crate::execution::{NumericalReceiptIdentity, SolverResultIdentity};
use runmat_meshing_core::ElementOrder;
const PREFIX: &[u8] = b"runmat-magnetic-numerical-receipt/v1\0";
pub const MAGNETIC_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const RECEIPT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(MAGNETIC_RECEIPT_MAX_ENCODED_BYTES as usize, 4096, 4096, 32);
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub element_order: ElementOrder,
    pub nodes: usize,
    pub elements: usize,
    pub oriented_edges: usize,
    pub integration_samples: usize,
    pub iterations: u32,
    pub relative_physical_residual: f64,
    pub relative_gauge_residual: f64,
    pub relative_energy_work_error: f64,
    pub solver_relative_tolerance: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub magnetic_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub source_work_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct MagneticNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: MagneticNumericalReceipt,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: MagneticNumericalReceipt,
}
impl TryFrom<ReceiptWire> for MagneticNumericalReceiptArtifact {
    type Error = AnalysisExecutionError;
    fn try_from(w: ReceiptWire) -> AnalysisExecutionResult<Self> {
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
}
impl MagneticNumericalReceiptArtifact {
    pub fn new(
        workload: &MagnetostaticWorkload,
        result: &MagnetostaticSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        let r = result.response();
        let receipt = MagneticNumericalReceipt {
            producer: result.producer(),
            topology_digest: result.mesh_digest(),
            element_order: result.element_order(),
            nodes: mesh.topology.nodes.len(),
            elements: mesh.topology.volume_elements.len(),
            oriented_edges: r.edge_circulations_wb.canonical_values().len(),
            integration_samples: r.sample_count(),
            iterations: r.iterations,
            relative_physical_residual: r.relative_physical_residual,
            relative_gauge_residual: r.relative_gauge_residual,
            relative_energy_work_error: r.relative_energy_work_error,
            solver_relative_tolerance: workload.relative_residual_target,
            magnetic_energy_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(r.magnetic_energy_j.canonical_value()),
            source_work_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(r.source_work_j.canonical_value()),
        };
        let source_workload = workload.identity()?;
        let source_result = result.source_result_identity()?;
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
    pub const fn receipt(&self) -> MagneticNumericalReceipt {
        self.receipt
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        let t = r.solver_relative_tolerance;
        let within = |v: f64, b: f64| v.is_finite() && v >= 0.0 && v <= b;
        let samples = r
            .elements
            .checked_mul(15)
            .ok_or_else(|| invalid("magnetic receipt sample overflow"))?;
        let edges = r
            .elements
            .checked_mul(6)
            .ok_or_else(|| invalid("magnetic receipt edge overflow"))?;
        let dimension = edges
            .checked_add(r.nodes)
            .ok_or_else(|| invalid("magnetic receipt dimension overflow"))?;
        // Direct factorization: one solve plus at most ten refinement steps.
        let _ = (dimension, t);
        let maximum_iterations = 16usize;
        let scale = (2.0 * r.magnetic_energy_j.canonical_value())
            .abs()
            .max(r.source_work_j.canonical_value().abs());
        let closure = if scale > 0.0 {
            (2.0 * r.magnetic_energy_j.canonical_value() - r.source_work_j.canonical_value()).abs()
                / scale
        } else {
            0.0
        };
        if self.schema_version != 1
            || r.nodes == 0
            || r.elements == 0
            || r.oriented_edges == 0
            || r.oriented_edges > edges
            || r.integration_samples != samples
            || r.iterations as usize > maximum_iterations
            || !matches!(r.element_order, ElementOrder::Tet4 | ElementOrder::Tet10)
            || !t.is_finite()
            || t <= 0.0
            || t >= 1.0
            || !within(r.relative_physical_residual, t * 8.0)
            || !within(r.relative_gauge_residual, t * 8.0)
            || !within(r.relative_energy_work_error, t * 16.0)
            || !within(r.magnetic_energy_j.canonical_value(), f64::MAX)
            || !r.source_work_j.canonical_value().is_finite()
            || !closure.is_finite()
            || r.relative_energy_work_error != closure
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "magnetic numerical receipt closure, cardinality or authority is invalid",
            ));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        w: &MagnetostaticWorkload,
        r: &MagnetostaticSolverResult,
        m: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new(w, r, m)? {
            return Err(invalid(
                "magnetic receipt differs from exact workload/result/mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "magnetic-numerical-receipt",
            self,
            RECEIPT_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self =
            decode_contract(PREFIX, "magnetic-numerical-receipt", bytes, RECEIPT_LIMITS)?;
        value.validate()?;
        Ok(value)
    }
}
fn receipt_identity(
    w: AnalysisWorkloadIdentity,
    r: SolverResultIdentity,
    receipt: &MagneticNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    NumericalReceiptIdentity::new(
        r,
        Digest::sha256(encode_contract(
            PREFIX,
            "magnetic-numerical-receipt-content",
            &(w, receipt),
            RECEIPT_LIMITS,
        )?),
    )
}
