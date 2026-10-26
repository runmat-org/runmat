//! Exact algebraic/flux/energy closure, not an independent spatial accuracy certificate.
use super::*;
use crate::execution::{NumericalReceiptIdentity, SolverResultIdentity};
use crate::physics::electrical::ElectricalConstitutiveResponse;
use runmat_meshing_core::ElementOrder;

const PREFIX: &[u8] = b"runmat-electrical-numerical-receipt/v2\0";
pub const ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const LIMITS: CanonicalLimits = CanonicalLimits::new(
    ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES as usize,
    4096,
    4096,
    32,
);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub equation: StaticElectricalEquation,
    pub element_order: ElementOrder,
    pub nodes: usize,
    pub elements: usize,
    pub iterations: u32,
    pub linear_absolute_residual: f64,
    pub linear_relative_residual: f64,
    pub relative_flux_imbalance: f64,
    pub relative_energy_identity_error: f64,
    pub solver_relative_tolerance: f64,
    pub closure_relative_tolerance: f64,
    /// Physical electrostatic stored energy (J) or conduction Joule power (W),
    /// disambiguated by equation; not an integral of the averaged E squared.
    pub total_constitutive_integral: runmat_types::physical::DimensionedScalar,
    /// Sum of original weak nodal outward flux: C or A, not a local flux density.
    pub net_outward_flux: runmat_types::physical::DimensionedScalar,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct ElectricalNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: ElectricalNumericalReceipt,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: ElectricalNumericalReceipt,
}
impl TryFrom<ReceiptWire> for ElectricalNumericalReceiptArtifact {
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
impl ElectricalNumericalReceiptArtifact {
    pub fn new(
        workload: &ElectricalStaticWorkload,
        result: &ElectricalStaticSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        let r = result.response();
        let total_constitutive_integral = match &r.constitutive {
            ElectricalConstitutiveResponse::Electrostatics { total_energy_j, .. } => {
                total_energy_j.canonical_value()
            }
            ElectricalConstitutiveResponse::SteadyConduction {
                total_joule_power_w,
                ..
            } => total_joule_power_w.canonical_value(),
        };
        let receipt = ElectricalNumericalReceipt {
            producer: result.producer(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            equation: r.equation,
            element_order: result.element_order(),
            nodes: r.ordered_node_ids.len(),
            elements: r.ordered_element_ids.len(),
            iterations: r.iterations,
            linear_absolute_residual: r.linear_absolute_residual,
            linear_relative_residual: r.linear_relative_residual,
            relative_flux_imbalance: r.relative_flux_imbalance,
            relative_energy_identity_error: r.relative_energy_identity_error,
            solver_relative_tolerance: workload.relative_residual_target,
            closure_relative_tolerance: crate::physics::electrical::closure_target(
                workload.relative_residual_target,
            ),
            total_constitutive_integral: runmat_types::physical::Quantity::try_from_dimension(
                r.equation.work_kind().dimension(),
                Some(r.equation.work_kind()),
                total_constitutive_integral,
            )
            .map_err(invalid)?,
            net_outward_flux: runmat_types::physical::Quantity::try_from_descriptor(
                r.nodal_outward_flux.descriptor(),
                r.nodal_outward_flux.canonical_values().iter().sum(),
            )
            .map_err(invalid)?,
        };
        let source_workload = workload.identity()?;
        let source_result = result.source_result_identity()?;
        let value = Self {
            schema_version: 2,
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
    pub const fn receipt(&self) -> ElectricalNumericalReceipt {
        self.receipt
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        r.net_outward_flux
            .require_kind(r.equation.conjugate_flux_kind())
            .map_err(invalid)?;
        r.total_constitutive_integral
            .require_kind(r.equation.work_kind())
            .map_err(invalid)?;
        let within = |value: f64, bound: f64| value.is_finite() && value >= 0.0 && value <= bound;
        if self.schema_version != 2
            || r.nodes == 0
            || r.elements == 0
            || !matches!(r.element_order, ElementOrder::Tet4 | ElementOrder::Tet10)
            || !r.solver_relative_tolerance.is_finite()
            || r.solver_relative_tolerance <= 0.0
            || r.solver_relative_tolerance >= 1.0
            || r.closure_relative_tolerance
                != crate::physics::electrical::closure_target(r.solver_relative_tolerance)
            || !within(r.linear_relative_residual, r.solver_relative_tolerance)
            || !within(r.linear_absolute_residual, f64::MAX)
            || !within(r.relative_flux_imbalance, r.closure_relative_tolerance)
            || !within(
                r.relative_energy_identity_error,
                r.closure_relative_tolerance,
            )
            || !within(r.total_constitutive_integral.canonical_value(), f64::MAX)
            || !r.net_outward_flux.canonical_value().is_finite()
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "electrical numerical receipt closure, authority or identity is invalid",
            ));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        workload: &ElectricalStaticWorkload,
        result: &ElectricalStaticSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new(workload, result, mesh)? {
            return Err(invalid(
                "electrical receipt differs from exact workload/result/mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "electrical-numerical-receipt",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "electrical-numerical-receipt", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
}
fn receipt_identity(
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    receipt: &ElectricalNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    let bytes = encode_contract(
        PREFIX,
        "electrical-numerical-receipt-content",
        &(workload, receipt),
        LIMITS,
    )?;
    NumericalReceiptIdentity::new(result, Digest::sha256(bytes))
}
