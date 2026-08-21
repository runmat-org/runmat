use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
use runmat_test::descriptor::{SourceDescriptor, SourceSpan};
use runmat_test::discovery::{
    materialize_metadata, IsolatedMetadataMaterializer, MaterializationKind, MaterializationLimits,
    MaterializationRequest, MaterializationResponse, MaterializationStatus, MaterializedValue,
};

fn revision(source: &str) -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(source),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
    .with_domain_contribution(
        DomainContribution::new("runmat.test.config", Digest::sha256(b"test")).unwrap(),
    )
    .unwrap()
}

fn request() -> MaterializationRequest {
    MaterializationRequest {
        id: "request-1".into(),
        program_revision: revision("sha256:source-a"),
        kind: MaterializationKind::ParameterExpression,
        semantic_path: "Suite/parameter/Mode".into(),
        source: SourceDescriptor {
            owner_identity: "standalone:project".into(),
            relative_path: "tests/Suite.m".into(),
            semantic_path: "Suite/parameter/Mode".into(),
            span: SourceSpan {
                start_byte: 1,
                end_byte: 2,
                start_line: 1,
                start_column: 2,
                end_line: 1,
                end_column: 3,
            },
        },
        expression: "{'fast','slow'}".into(),
        limits: MaterializationLimits::default(),
    }
}

#[test]
fn accepts_only_bounded_matching_revision_materialization() {
    let request = request();
    let mut materializer = FakeMaterializer {
        revision: request.program_revision.clone(),
    };
    let (responses, records, errors) =
        materialize_metadata(std::slice::from_ref(&request), &mut materializer);
    assert_eq!(responses.len(), 1);
    assert!(errors.is_empty());
    assert_eq!(records[0].status, MaterializationStatus::Completed);
    assert_eq!(records[0].value_count, 2);
}

#[test]
fn rejects_materialization_when_source_revision_changes() {
    let request = request();
    let mut materializer = FakeMaterializer {
        revision: revision("sha256:source-b"),
    };
    let (responses, records, errors) =
        materialize_metadata(std::slice::from_ref(&request), &mut materializer);
    assert!(responses.is_empty());
    assert_eq!(records[0].status, MaterializationStatus::Rejected);
    assert_eq!(errors.len(), 1);
}

struct FakeMaterializer {
    revision: ProgramRevision,
}

impl IsolatedMetadataMaterializer for FakeMaterializer {
    fn materialize(
        &mut self,
        request: &MaterializationRequest,
    ) -> Result<MaterializationResponse, String> {
        Ok(MaterializationResponse {
            request_id: request.id.clone(),
            program_revision: self.revision.clone(),
            values: ["fast", "slow"]
                .into_iter()
                .map(|value| MaterializedValue {
                    name: value.into(),
                    normalized_identity: format!("\"{value}\""),
                    value: serde_json::Value::String(value.into()),
                })
                .collect(),
            diagnostics: Vec::new(),
        })
    }
}
