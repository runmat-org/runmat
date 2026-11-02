use async_trait::async_trait;
use base64::Engine as _;
use futures_util::StreamExt as _;

use runmat_execution_transport_native::control::{
    AllocationRole, DriverBootstrapCredential, EnrolledNode, EnrollmentRequest, NodeAllocation,
    NodeControlPlane, NodeHeartbeat, NodeStatus, ResourceRequest, RotatedCredential,
    WorkerBootstrapCredential,
};
use runmat_execution_transport_native::{TransportError, TransportResult};
use runmat_server_client::public_api::{self, types};

use crate::server_api::{inventory_to_api, map_error, to_i64, to_u64};

#[derive(Clone)]
pub struct HttpNodeControlPlane {
    client: public_api::Client,
    base_url: String,
    http: reqwest::Client,
}

impl HttpNodeControlPlane {
    pub fn new(base_url: impl Into<String>) -> TransportResult<Self> {
        let base_url = crate::config::ServerEndpoint::parse(&base_url.into())
            .map_err(|error| TransportError::Unavailable(error.to_string()))?
            .as_str()
            .to_owned();
        Ok(Self {
            client: public_api::Client::new(&base_url),
            base_url,
            http: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl NodeControlPlane for HttpNodeControlPlane {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        let response = self
            .client
            .consume_node_enrollment(&types::ConsumeEnrollmentRequest {
                token: request.token,
                credential: request.credential.try_into().map_err(|_| {
                    TransportError::Unavailable(
                        "pending node credential violates the API contract".into(),
                    )
                })?,
                identity_fingerprint: request.identity_fingerprint,
                identity_public_key: encode_bytes(&request.identity_public_key),
                inventory: inventory_to_api(request.inventory)?,
                heartbeat_ttl_seconds: to_i64(request.heartbeat_ttl_seconds, "heartbeat TTL")?,
            })
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(EnrolledNode {
            node_id: response.node_id,
            cluster_id: response.cluster_id,
            org_id: response.org_id,
            credential: response.credential,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
            lease_epoch: to_u64(response.lease_epoch, "lease epoch")?,
        })
    }

    async fn heartbeat(&self, heartbeat: NodeHeartbeat) -> TransportResult<NodeStatus> {
        let response = self
            .client
            .heartbeat(
                &heartbeat.node_id,
                &heartbeat.credential,
                &types::NodeHeartbeatRequest {
                    clock_starts: None,
                    clock_ends: None,
                    clock_release_ack_confirm_ids: None,
                    org_id: heartbeat.org_id,
                    credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
                    inventory: inventory_to_api(heartbeat.inventory)?,
                    heartbeat_ttl_seconds: to_i64(
                        heartbeat.heartbeat_ttl_seconds,
                        "heartbeat TTL",
                    )?,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(NodeStatus {
            state: response.state,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
            lease_epoch: to_u64(response.lease_epoch, "lease epoch")?,
        })
    }

    async fn clock_heartbeat(
        &self,
        heartbeat: NodeHeartbeat,
        starts: Vec<runmat_execution::security::NodeClockStartV1>,
        ends: Vec<runmat_execution::security::NodeClockEndV1>,
        release_ack_confirm_ids: Vec<[u8; 16]>,
    ) -> TransportResult<runmat_execution_transport_native::control::ClockNodeHeartbeatResponseV1>
    {
        use runmat_execution_transport_native::control::MAX_CLOCK_HEARTBEAT_ITEMS;
        if starts.len() > MAX_CLOCK_HEARTBEAT_ITEMS
            || ends.len() > MAX_CLOCK_HEARTBEAT_ITEMS
            || release_ack_confirm_ids.len() > MAX_CLOCK_HEARTBEAT_ITEMS
        {
            return Err(TransportError::Overflow);
        }
        for start in &starts {
            start
                .validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        }
        for end in &ends {
            end.validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        }
        let body = serde_json::json!({
            "orgId": heartbeat.org_id,
            "credentialEpoch": to_i64(heartbeat.credential_epoch, "credential epoch")?,
            "inventory": inventory_to_api(heartbeat.inventory)?,
            "heartbeatTtlSeconds": to_i64(heartbeat.heartbeat_ttl_seconds, "heartbeat TTL")?,
            "clockStarts": starts,
            "clockEnds": ends,
            "clockReleaseAckConfirmIds": release_ack_confirm_ids.iter().map(|id| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(id)).collect::<Vec<_>>(),
        });
        let response = self
            .http
            .post(format!(
                "{}/v1/execution/nodes/{}/heartbeat",
                self.base_url, heartbeat.node_id
            ))
            .header("x-runmat-node-credential", &heartbeat.credential)
            .header("content-type", "application/json")
            .body(
                serde_json::to_vec(&body)
                    .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
            )
            .send()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        if !response.status().is_success() {
            return Err(TransportError::Unavailable(format!(
                "node clock heartbeat returned {}",
                response.status()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| TransportError::Unavailable(error.to_string()))?;
            if bytes
                .len()
                .checked_add(chunk.len())
                .is_none_or(|total| total > 2 * 1024 * 1024)
            {
                return Err(TransportError::FrameTooLarge);
            }
            bytes.extend_from_slice(&chunk);
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Demand {
            start_receipt: runmat_execution::security::NodeClockStartV1,
            demand_challenge: String,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Wire {
            state: String,
            credential_epoch: i64,
            lease_epoch: i64,
            #[serde(default)]
            clock_open_observer_count: u32,
            #[serde(default)]
            clock_demands: Vec<Demand>,
            #[serde(default)]
            clock_acked_start_ids: Vec<String>,
            #[serde(default)]
            clock_acked_end_ids: Vec<String>,
            #[serde(default)]
            clock_acked_release_confirm_ids: Vec<String>,
        }
        let wire: Wire = serde_json::from_slice(&bytes)
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        if wire.clock_demands.len() > MAX_CLOCK_HEARTBEAT_ITEMS
            || wire.clock_acked_start_ids.len() > MAX_CLOCK_HEARTBEAT_ITEMS
            || wire.clock_acked_end_ids.len() > MAX_CLOCK_HEARTBEAT_ITEMS
            || wire.clock_acked_release_confirm_ids.len() > MAX_CLOCK_HEARTBEAT_ITEMS
        {
            return Err(TransportError::Overflow);
        }
        let decode = |value: &str| {
            let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(value)
                .map_err(|_| TransportError::Integrity)?;
            if base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes) != value {
                return Err(TransportError::Integrity);
            }
            bytes.try_into().map_err(|_| TransportError::Integrity)
        };
        Ok(
            runmat_execution_transport_native::control::ClockNodeHeartbeatResponseV1 {
                node: NodeStatus {
                    state: wire.state,
                    credential_epoch: to_u64(wire.credential_epoch, "credential epoch")?,
                    lease_epoch: to_u64(wire.lease_epoch, "lease epoch")?,
                },
                open_observer_count: wire.clock_open_observer_count,
                demands: wire
                    .clock_demands
                    .into_iter()
                    .map(|demand| {
                        demand
                            .start_receipt
                            .validate_shape()
                            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
                        Ok(
                            runmat_execution_transport_native::control::ClockNodeDemandV1 {
                                start_receipt: demand.start_receipt,
                                demand_challenge: decode(&demand.demand_challenge)?,
                            },
                        )
                    })
                    .collect::<TransportResult<_>>()?,
                acked_start_ids: wire
                    .clock_acked_start_ids
                    .iter()
                    .map(|value| decode(value))
                    .collect::<TransportResult<_>>()?,
                acked_end_ids: wire
                    .clock_acked_end_ids
                    .iter()
                    .map(|value| decode(value))
                    .collect::<TransportResult<_>>()?,
                acked_release_confirm_ids: wire
                    .clock_acked_release_confirm_ids
                    .iter()
                    .map(|value| decode(value))
                    .collect::<TransportResult<_>>()?,
            },
        )
    }

    async fn rotate_credential(
        &self,
        heartbeat: &NodeHeartbeat,
    ) -> TransportResult<RotatedCredential> {
        let response = self
            .client
            .rotate_credential(
                &heartbeat.node_id,
                &heartbeat.credential,
                &authority_body(heartbeat)?,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(RotatedCredential {
            credential: response.credential,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
        })
    }

    async fn allocations(&self, heartbeat: &NodeHeartbeat) -> TransportResult<Vec<NodeAllocation>> {
        self.client
            .list_allocations(
                &heartbeat.node_id,
                to_i64(heartbeat.credential_epoch, "credential epoch")?,
                &heartbeat.org_id,
                &heartbeat.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner()
            .allocations
            .into_iter()
            .map(allocation_from_api)
            .collect()
    }

    async fn accept(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()> {
        self.client
            .accept_allocation(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn publish_endpoint_identity(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
        evidence: runmat_execution::security::EndpointIdentityEvidence,
    ) -> TransportResult<()> {
        self.client
            .publish_endpoint_identity(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &types::PublishEndpointIdentityRequest {
                    org_id: heartbeat.org_id.clone(),
                    credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
                    evidence: evidence_to_api(evidence)?,
                },
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn driver_bootstrap(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<DriverBootstrapCredential> {
        let response = self
            .client
            .create_driver_bootstrap(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_bootstrap_error)?
            .into_inner();
        Ok(DriverBootstrapCredential {
            run_id: response.run_id,
            org_id: response.org_id,
            project_id: response.project_id,
            allocation_lease_id: response.allocation_lease_id,
            driver_lease_id: response.driver_lease_id,
            fencing_token: to_u64(response.fencing_token, "driver fencing token")?,
            credential: response.credential,
            expires_at_millis: response.expires_at.timestamp_millis(),
            ready_v1: response.ready_v1.unwrap_or(false),
        })
    }

    async fn worker_bootstrap(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<WorkerBootstrapCredential> {
        let response = self
            .client
            .create_worker_bootstrap(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_bootstrap_error)?
            .into_inner();
        Ok(WorkerBootstrapCredential {
            run_id: response.run_id,
            org_id: response.org_id,
            project_id: response.project_id,
            allocation_lease_id: response.allocation_lease_id,
            allocation_fencing_token: to_u64(
                response.allocation_fencing_token,
                "allocation fencing token",
            )?,
            driver_fencing_token: to_u64(response.driver_fencing_token, "driver fencing token")?,
            endpoint_fingerprint: response.endpoint_fingerprint,
            run_key_envelope: decode_bytes(&response.run_key_envelope)?,
            expires_at_millis: response.expires_at.timestamp_millis(),
            relay_path: response.relay_path,
            relay_protocol: response.relay_protocol,
            relay_ticket: response.relay_ticket,
            ready_v1: response.ready_v1.unwrap_or(false),
        })
    }

    async fn release(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()> {
        self.client
            .release_allocation(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn release_retired_worker(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
        evidence: &runmat_execution::security::WorkerTreeRetirementEvidenceV1,
    ) -> TransportResult<()> {
        evidence
            .validate_shape()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        if evidence.allocation_lease_id != allocation.id
            || evidence.allocation_fencing_token != allocation.fencing_token
        {
            return Err(TransportError::Integrity);
        }
        let mut body = transition_body(heartbeat, allocation)?;
        body.worker_retirement_evidence = Some(retirement_evidence_to_api(evidence)?);
        self.client
            .release_allocation(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &body,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn release_retired_worker_evidence(
        &self,
        heartbeat: &NodeHeartbeat,
        evidence: &runmat_execution::security::WorkerTreeRetirementEvidenceV1,
    ) -> TransportResult<()> {
        evidence
            .validate_shape()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        let mut body = transition_body_for(heartbeat, evidence.allocation_fencing_token)?;
        body.worker_retirement_evidence = Some(retirement_evidence_to_api(evidence)?);
        self.client
            .release_allocation(
                &heartbeat.node_id,
                &evidence.allocation_lease_id,
                &heartbeat.credential,
                &body,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn complete_drain(&self, heartbeat: &NodeHeartbeat) -> TransportResult<()> {
        self.client
            .complete_drain(
                &heartbeat.node_id,
                &heartbeat.credential,
                &authority_body(heartbeat)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }
}

fn authority_body(
    heartbeat: &NodeHeartbeat,
) -> TransportResult<types::RotateNodeCredentialRequest> {
    Ok(types::RotateNodeCredentialRequest {
        org_id: heartbeat.org_id.clone(),
        credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
    })
}

fn transition_body(
    heartbeat: &NodeHeartbeat,
    allocation: &NodeAllocation,
) -> TransportResult<types::TransitionAllocationRequest> {
    transition_body_for(heartbeat, allocation.fencing_token)
}

fn transition_body_for(
    heartbeat: &NodeHeartbeat,
    fencing_token: u64,
) -> TransportResult<types::TransitionAllocationRequest> {
    Ok(types::TransitionAllocationRequest {
        org_id: heartbeat.org_id.clone(),
        credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
        fencing_token: to_i64(fencing_token, "fencing token")?,
        worker_retirement_evidence: None,
    })
}

fn retirement_evidence_to_api(
    evidence: &runmat_execution::security::WorkerTreeRetirementEvidenceV1,
) -> TransportResult<types::WorkerTreeRetirementEvidenceV1> {
    Ok(types::WorkerTreeRetirementEvidenceV1 {
        schema_version: i32::from(evidence.schema_version),
        org_id: evidence.org_id.clone(),
        cluster_id: evidence.cluster_id.clone(),
        node_id: evidence.node_id.clone(),
        run_id: evidence.run_id.clone(),
        allocation_lease_id: evidence.allocation_lease_id.clone(),
        allocation_fencing_token: to_i64(
            evidence.allocation_fencing_token,
            "retirement allocation fence",
        )?,
        driver_fencing_token: to_i64(evidence.driver_fencing_token, "retirement driver fence")?,
        process_id: i64::from(evidence.process_id),
        process_launch_nonce: encoded_wire_bytes(&evidence.process_launch_nonce)?,
        retired_at_unix_millis: to_i64(evidence.retired_at_unix_millis, "retirement time")?,
        clock_domain_nonce: encoded_wire_bytes(&evidence.clock_domain_nonce)?,
        retired_monotonic_ticks: to_i64(
            evidence.retired_monotonic_ticks,
            "retired monotonic ticks",
        )?,
        ticks_per_second: to_i64(evidence.ticks_per_second, "retirement clock frequency")?,
        encrypted_attempt_receipts: evidence
            .encrypted_attempt_receipts
            .iter()
            .map(|receipt| receipt.as_str().to_owned())
            .collect(),
        identity_public_key: encoded_wire_bytes(&evidence.identity_public_key)?,
        identity_fingerprint: evidence.identity_fingerprint.clone(),
        signature: encoded_wire_bytes(&evidence.signature)?,
    })
}

fn allocation_from_api(value: types::AllocationLeaseResponse) -> TransportResult<NodeAllocation> {
    Ok(NodeAllocation {
        id: value.id,
        run_id: value.run_id,
        project_id: value.project_id,
        queue: value.queue,
        resources: ResourceRequest {
            cpu_millicores: to_u64(value.resources.cpu_millicores, "requested CPU")?,
            memory_bytes: to_u64(value.resources.memory_bytes, "requested memory")?,
            scratch_bytes: to_u64(value.resources.scratch_bytes, "requested scratch")?,
            accelerators: value
                .resources
                .accelerators
                .into_iter()
                .map(runmat_server_client::execution::accelerator_request_from_api)
                .collect::<Result<_, _>>()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
            maximum_wall_millis: to_u64(value.resources.maximum_wall_millis, "maximum wall time")?,
        },
        accelerator_devices: value
            .accelerator_devices
            .into_iter()
            .map(runmat_server_client::execution::accelerator_device_from_api)
            .collect::<Result<_, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        role: match value.role.as_str() {
            "driver" => AllocationRole::Driver,
            "worker" => AllocationRole::Worker,
            _ => {
                return Err(TransportError::MalformedFrame(
                    "allocation role is unsupported".into(),
                ))
            }
        },
        state: value.state,
        fencing_token: to_u64(value.fencing_token, "fencing token")?,
        expires_at_millis: value.expires_at.timestamp_millis(),
    })
}

fn evidence_to_api(
    value: runmat_execution::security::EndpointIdentityEvidence,
) -> TransportResult<types::EndpointIdentityEvidenceBody> {
    Ok(types::EndpointIdentityEvidenceBody {
        schema_version: i32::from(value.schema_version),
        org_id: value.org_id,
        cluster_id: value.cluster_id,
        node_id: value.node_id,
        allocation_lease_id: value.allocation_lease_id,
        fencing_token: to_i64(value.fencing_token, "fencing token")?,
        run_identity: value.run_identity,
        identity_public_key: encode_bytes(&value.identity_public_key),
        identity_fingerprint: value.identity_fingerprint,
        recipient: types::EndpointRecipientKeyBody {
            suite: value.recipient.suite,
            public_key: encode_bytes(&value.recipient.public_key),
            fingerprint: value.recipient.fingerprint,
            valid_after_unix_millis: to_i64(
                value.recipient.valid_after_unix_millis,
                "recipient validity",
            )?,
            valid_before_unix_millis: to_i64(
                value.recipient.valid_before_unix_millis,
                "recipient validity",
            )?,
        },
        direct_quic_endpoints: value
            .direct_quic_endpoints
            .into_iter()
            .map(|endpoint| types::DirectQuicEndpointBody {
                authority: endpoint.authority,
                server_name: endpoint.server_name,
                certificate_der: encode_bytes(&endpoint.certificate_der),
                certificate_sha256: endpoint.certificate_sha256,
            })
            .collect(),
        trust_tier: match value.trust_tier {
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted => {
                types::ExecutionTrustTierBody::CustomerTrusted
            }
            runmat_execution::security::ExecutionTrustTier::HostedOrdinary => {
                types::ExecutionTrustTierBody::HostedOrdinary
            }
            runmat_execution::security::ExecutionTrustTier::AttestedConfidential => {
                types::ExecutionTrustTierBody::AttestedConfidential
            }
        },
        attestation_class: value.attestation_class,
        attestation_evidence: value
            .attestation_evidence
            .map(|evidence| encode_bytes(&evidence)),
        issued_at_unix_millis: to_i64(value.issued_at_unix_millis, "evidence issue time")?,
        expires_at_unix_millis: to_i64(value.expires_at_unix_millis, "evidence expiry")?,
        signature: encode_bytes(&value.signature),
    })
}

fn encoded_wire_bytes<T>(value: &[u8]) -> TransportResult<T>
where
    T: TryFrom<String>,
    T::Error: std::fmt::Display,
{
    encode_bytes(value)
        .try_into()
        .map_err(|error: T::Error| TransportError::MalformedFrame(error.to_string()))
}

fn encode_bytes(value: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value)
}

fn decode_bytes(value: &str) -> TransportResult<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| TransportError::Integrity)
}

fn map_bootstrap_error<E: std::fmt::Debug>(error: public_api::Error<E>) -> TransportError {
    if error.status().is_some_and(|status| status.as_u16() == 409) {
        TransportError::NotReady
    } else {
        map_error(error)
    }
}
