//! Original physical acceptance receipt, not an independent discretization certificate.
use super::*;
use crate::execution::{NumericalReceiptIdentity, SolverResultIdentity};
use crate::physics::maxwell::ComplexPhasor;
use runmat_meshing_core::ElementOrder;
const PREFIX: &[u8] = b"runmat-maxwell-numerical-receipt/v2\0";
pub const MAXWELL_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const RECEIPT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(MAXWELL_RECEIPT_MAX_ENCODED_BYTES as usize, 4096, 4096, 32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub element_order: ElementOrder,
    pub nodes: usize,
    pub elements: usize,
    pub oriented_edges: usize,
    pub integration_samples: usize,
    pub angular_frequency_rad_per_s: f64,
    pub relative_physical_residual: f64,
    pub relative_charge_continuity_residual: f64,
    pub relative_complex_power_error: f64,
    pub solver_relative_tolerance: f64,
    pub electric_energy_j: f64,
    pub magnetic_energy_j: f64,
    pub ohmic_power_w: f64,
    pub source_power_va: ComplexPhasor,
    pub reactive_power_var: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct MaxwellNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: MaxwellNumericalReceipt,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: MaxwellNumericalReceipt,
}
impl TryFrom<ReceiptWire> for MaxwellNumericalReceiptArtifact {
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
impl MaxwellNumericalReceiptArtifact {
    pub fn new(
        workload: &MaxwellHarmonicWorkload,
        result: &MaxwellHarmonicSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        let r = result.response();
        let receipt = MaxwellNumericalReceipt {
            producer: result.producer_identity(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            element_order: mesh.resolved_request.element_order,
            nodes: mesh.topology.nodes.len(),
            elements: mesh.topology.volume_elements.len(),
            oriented_edges: r.edge_circulations_v.len(),
            integration_samples: r.samples.len(),
            angular_frequency_rad_per_s: r.angular_frequency_rad_per_s,
            relative_physical_residual: r.relative_physical_residual,
            relative_charge_continuity_residual: r.relative_charge_continuity_residual,
            relative_complex_power_error: r.relative_complex_power_error,
            solver_relative_tolerance: workload.relative_residual_target,
            electric_energy_j: r.electric_energy_j,
            magnetic_energy_j: r.magnetic_energy_j,
            ohmic_power_w: r.ohmic_power_w,
            source_power_va: r.source_power_va,
            reactive_power_var: r.reactive_power_var,
        };
        let source_workload = workload.identity()?;
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
    pub const fn receipt(&self) -> MaxwellNumericalReceipt {
        self.receipt
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        let t = r.solver_relative_tolerance;
        let within = |v: f64, bound: f64| v.is_finite() && v >= 0.0 && v <= bound;
        let samples = r
            .elements
            .checked_mul(15)
            .ok_or_else(|| invalid("Maxwell receipt sample overflow"))?;
        let edges = r
            .elements
            .checked_mul(6)
            .ok_or_else(|| invalid("Maxwell receipt edge overflow"))?;
        let reactive =
            2.0 * r.angular_frequency_rad_per_s * (r.magnetic_energy_j - r.electric_energy_j);
        let predicted = ComplexPhasor::new(r.ohmic_power_w, reactive);
        let scale = predicted.abs().max(r.source_power_va.abs());
        let defect = (r.source_power_va - predicted).abs();
        let closure = if scale > 0.0 { defect / scale } else { defect };
        if self.schema_version != 1
            || r.nodes == 0
            || r.elements == 0
            || r.oriented_edges == 0
            || r.oriented_edges > edges
            || r.integration_samples != samples
            || !r.angular_frequency_rad_per_s.is_finite()
            || r.angular_frequency_rad_per_s <= 0.0
            || !t.is_finite()
            || t <= 0.0
            || t >= 1.0
            || !within(r.relative_physical_residual, 8.0 * t)
            || !within(r.relative_charge_continuity_residual, 16.0 * t)
            || !within(r.relative_complex_power_error, 16.0 * t)
            || !within(r.electric_energy_j, f64::MAX)
            || !within(r.magnetic_energy_j, f64::MAX)
            || !within(r.ohmic_power_w, f64::MAX)
            || !r.source_power_va.is_finite()
            || !r.reactive_power_var.is_finite()
            || r.reactive_power_var != reactive
            || !closure.is_finite()
            || r.relative_complex_power_error != closure
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "Maxwell receipt authority, cardinality or physical balances differ",
            ));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        workload: &MaxwellHarmonicWorkload,
        result: &MaxwellHarmonicSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new(workload, result, mesh)? {
            return Err(invalid(
                "Maxwell receipt differs from original workload/result/mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "maxwell-numerical-receipt",
            self,
            RECEIPT_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self =
            decode_contract(PREFIX, "maxwell-numerical-receipt", bytes, RECEIPT_LIMITS)?;
        value.validate()?;
        Ok(value)
    }
}
fn receipt_identity(
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    receipt: &MaxwellNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    NumericalReceiptIdentity::new(
        result,
        Digest::sha256(encode_contract(
            PREFIX,
            "maxwell-numerical-receipt-content",
            &(workload, receipt),
            RECEIPT_LIMITS,
        )?),
    )
}
