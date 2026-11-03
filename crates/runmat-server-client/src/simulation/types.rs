use crate::ids::{OrgId, SimulationLeaseId, SimulationReceiptId, SimulationRunId};
use chrono::{DateTime, Utc};
use runmat_execution::Digest;
use runmat_simulation_usage::{CompletedUsage, MeasurementVersion, OperationCount, OperationTotal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationKind {
    Meshing,
    Solve,
    MeshAndSolve,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationClientKind {
    Desktop,
    Browser,
    Cli,
    Compiled,
    Cluster,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationReceiptOutcome {
    Completed,
    Cancelled,
    Failed,
    Expired,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartSimulationLease {
    pub run_id: SimulationRunId,
    pub attempt: u32,
    pub measurement_version: MeasurementVersion,
    pub runmat_version: String,
    pub kind: SimulationKind,
    pub client: SimulationClientKind,
    pub study_identity: Digest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimate_ops: Option<OperationCount>,
    pub budget_ops: OperationCount,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationLease {
    pub lease_id: SimulationLeaseId,
    pub state: SimulationLeaseState,
    pub granted_ops: OperationCount,
    pub budget_ops: OperationCount,
    pub expires_at: DateTime<Utc>,
    pub heartbeat_seconds: u64,
    pub waived: bool,
}

/// Replaying a start returns the original attempt, including its terminal state.
/// Expiry alone cannot distinguish a recently settled lease from an active one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SimulationLeaseState {
    Active,
    Settled,
    Expired,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportSimulationUsage {
    pub measurement_version: MeasurementVersion,
    pub used_ops: CompletedUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_target_ops: Option<OperationCount>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationGrant {
    pub granted_ops: OperationCount,
    pub budget_ops: OperationCount,
    pub total_ops: OperationTotal,
    pub expires_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettleSimulationLease {
    pub outcome: SimulationOutcome,
    pub measurement_version: MeasurementVersion,
    pub used_ops: CompletedUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationReceipt {
    pub receipt_id: SimulationReceiptId,
    pub lease_id: SimulationLeaseId,
    pub org_id: OrgId,
    pub run_id: SimulationRunId,
    pub attempt: u32,
    pub measurement_version: MeasurementVersion,
    pub kind: SimulationKind,
    pub client: SimulationClientKind,
    pub outcome: SimulationReceiptOutcome,
    pub used_ops: CompletedUsage,
    pub other_detail: Option<String>,
    pub total_ops: OperationTotal,
    pub granted_ops: OperationCount,
    pub budget_ops: OperationCount,
    pub estimate_ops: Option<OperationCount>,
    pub charged_ops: OperationTotal,
    pub waived: bool,
    pub settled_at: DateTime<Utc>,
}
