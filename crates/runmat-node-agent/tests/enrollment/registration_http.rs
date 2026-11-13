use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse as _, Response},
    routing::post,
    Json, Router,
};
use runmat_node_agent::{
    config::ServerEndpoint,
    enrollment::{register_authenticated, CredentialStore, RegistrationTarget},
};
use runmat_server_client::ids::{ClusterId, OrgId};

const ORG: &str = "org_00000000000000000000000000000001";
const CLUSTER: &str = "cls_00000000000000000000000000000002";
const NODE: &str = "node_00000000000000000000000000000003";

type Requests = Arc<Mutex<Vec<serde_json::Value>>>;

async fn registration(
    State(requests): State<Requests>,
    Path((org, cluster)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assert_eq!(org, ORG);
    assert_eq!(cluster, CLUSTER);
    assert_eq!(headers["authorization"], "Bearer account-token");
    registration_response(&requests, body)
}

async fn token_registration(
    State(requests): State<Requests>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assert!(!headers.contains_key("authorization"));
    assert_eq!(body["token"], "bootstrap-token");
    assert_eq!(body["credential"].as_str().unwrap().len(), 43);
    registration_response(&requests, body)
}

fn registration_response(requests: &Requests, body: serde_json::Value) -> Response {
    let mut requests = requests.lock().unwrap();
    requests.push(body.clone());
    if requests.len() == 1 {
        // The caller cannot know whether this request was accepted. Its retry
        // must preserve machine authority rather than generating another node.
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    assert_eq!(requests[0], body);
    Json(serde_json::json!({
        "nodeId": NODE,
        "clusterId": CLUSTER,
        "orgId": ORG,
        "credential": body["credential"],
        "credentialEpoch": 1,
        "leaseEpoch": 1,
        "heartbeatExpiresAt": "2026-11-01T00:00:00Z",
    }))
    .into_response()
}

#[tokio::test]
async fn generated_registration_retries_original_authority_without_persisting_account_key() {
    let requests = Requests::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint =
        ServerEndpoint::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
    let app = Router::new()
        .route("/v1/orgs/:org/clusters/:cluster/nodes", post(registration))
        .with_state(requests.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    let store = CredentialStore::new(&state);
    let target = RegistrationTarget::new(
        endpoint,
        OrgId::parse(ORG).unwrap(),
        ClusterId::parse(CLUSTER).unwrap(),
    );
    let inventory = super::inventory();
    assert!(register_authenticated(
        &store,
        target.clone(),
        inventory.clone(),
        60,
        "account-token"
    )
    .await
    .is_err());
    assert!(state.join("registration.json").is_file());
    assert!(
        !String::from_utf8(std::fs::read(state.join("registration.json")).unwrap())
            .unwrap()
            .contains("account-token")
    );
    let credential = register_authenticated(
        &store,
        target.clone(),
        inventory.clone(),
        60,
        "account-token",
    )
    .await
    .unwrap();
    assert_eq!(credential.node_id, NODE);
    assert_eq!(store.load().unwrap(), credential);
    assert!(!state.join("registration.json").exists());
    assert!(
        !String::from_utf8(std::fs::read(state.join("credential.json")).unwrap())
            .unwrap()
            .contains("account-token")
    );
    // Already registered is a local operation: no new HTTP registration/grant.
    assert_eq!(
        register_authenticated(&store, target, inventory, 60, "not-used")
            .await
            .unwrap(),
        credential
    );
    assert_eq!(requests.lock().unwrap().len(), 2);
    server.abort();
}

#[tokio::test]
async fn generated_token_enrollment_retries_without_persisting_bootstrap_authority() {
    use runmat_node_agent::{enrollment::enroll, service::HttpNodeControlPlane};

    let requests = Requests::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint =
        ServerEndpoint::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
    let app = Router::new()
        .route(
            "/v1/execution/node-enrollments/consume",
            post(token_registration),
        )
        .with_state(requests.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    let store = CredentialStore::new(&state);
    let control = Arc::new(HttpNodeControlPlane::new(endpoint.as_str()).unwrap());
    assert!(enroll(
        control.clone(),
        &store,
        endpoint.clone(),
        "bootstrap-token".into(),
        super::inventory(),
        60
    )
    .await
    .is_err());
    assert!(!std::fs::read_to_string(state.join("registration.json"))
        .unwrap()
        .contains("bootstrap-token"));
    let credential = enroll(
        control,
        &store,
        endpoint,
        "bootstrap-token".into(),
        super::inventory(),
        60,
    )
    .await
    .unwrap();
    assert_eq!(credential.node_id, NODE);
    assert_eq!(store.load().unwrap(), credential);
    assert!(!state.join("registration.json").exists());
    assert!(!std::fs::read_to_string(state.join("credential.json"))
        .unwrap()
        .contains("bootstrap-token"));
    assert_eq!(requests.lock().unwrap().len(), 2);
    server.abort();
}
