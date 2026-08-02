use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::descriptor::{FixtureDescriptor, SourceDescriptor, TestDescriptor, TestSelector};
use crate::identity::{FixtureGroupId, SuiteId};
use crate::plan::{FixtureGroupPlan, ProgramRevision, SuitePlan, TestPlan, TestPlanBuilder};
use crate::TestDomainError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestDiscovery {
    pub program_revision: ProgramRevision,
    pub suites: Vec<DiscoveredSuite>,
    #[serde(default)]
    pub pending_materialization: Vec<MaterializationRequest>,
    #[serde(default)]
    pub diagnostics: Vec<DiscoveryDiagnostic>,
}

impl TestDiscovery {
    /// Apply selectors to semantic descriptors after discovery, preserving
    /// stable identities and fixture metadata for every retained suite.
    pub fn select(mut self, selector: &TestSelector) -> Self {
        for suite in &mut self.suites {
            suite.tests.retain(|test| {
                selector.matches(
                    &test.display_name,
                    &test.tags,
                    &test.procedure.source.relative_path,
                )
            });
        }
        self.suites.retain(|suite| !suite.tests.is_empty());
        self
    }

    pub fn into_plan(
        self,
        invocation_identity: impl Into<String>,
    ) -> Result<TestPlan, TestDomainError> {
        if !self.pending_materialization.is_empty() {
            return Err(TestDomainError::InvalidField {
                field: "pending_materialization",
                reason: "discovery must be materialized against the same program revision before planning"
                    .into(),
            });
        }
        let mut builder = TestPlanBuilder::new(self.program_revision, invocation_identity);
        for suite in self.suites {
            builder = builder.add_suite(SuitePlan {
                id: suite.id,
                display_name: suite.display_name,
                fixture_groups: vec![FixtureGroupPlan {
                    id: suite.fixture_group_id,
                    fixtures: suite.fixtures,
                    tests: suite.tests,
                }],
            });
        }
        builder.build()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredSuite {
    pub id: SuiteId,
    pub fixture_group_id: FixtureGroupId,
    pub display_name: String,
    pub source: SourceDescriptor,
    #[serde(default)]
    pub fixtures: Vec<FixtureDescriptor>,
    #[serde(default)]
    pub tests: Vec<TestDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryDiagnostic {
    pub code: String,
    pub message: String,
    pub severity: DiscoveryDiagnosticSeverity,
    pub source: Option<SourceDescriptor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryDiagnosticSeverity {
    Information,
    Warning,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MaterializationRequest {
    pub id: String,
    pub program_revision: ProgramRevision,
    pub kind: MaterializationKind,
    pub semantic_path: String,
    pub source: SourceDescriptor,
    pub expression: String,
    pub limits: MaterializationLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationKind {
    SuiteFactory,
    ParameterExpression,
    FixtureExpression,
    HandleList,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterializationLimits {
    pub max_duration_ms: u32,
    pub max_memory_bytes: u64,
    pub max_steps: u64,
    pub max_values: u32,
    pub max_encoded_bytes: u32,
    pub max_diagnostics: u32,
}

impl Default for MaterializationLimits {
    fn default() -> Self {
        Self {
            max_duration_ms: 5_000,
            max_memory_bytes: 64 * 1024 * 1024,
            max_steps: 10_000_000,
            max_values: 10_000,
            max_encoded_bytes: 4 * 1024 * 1024,
            max_diagnostics: 1_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterializationResponse {
    pub request_id: String,
    pub program_revision: ProgramRevision,
    #[serde(default)]
    pub values: Vec<MaterializedValue>,
    #[serde(default)]
    pub diagnostics: Vec<DiscoveryDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterializedValue {
    pub name: String,
    pub normalized_identity: String,
    pub value: Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MaterializationRecord {
    pub request_id: String,
    pub status: MaterializationStatus,
    pub value_count: u32,
    pub diagnostic_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationStatus {
    Completed,
    Rejected,
    Failed,
}
