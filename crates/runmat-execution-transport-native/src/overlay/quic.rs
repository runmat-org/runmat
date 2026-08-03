use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::Digest as _;

use super::OverlayRoute;
use crate::frame::{FrameLimits, WireFrame};
use crate::{TransportError, TransportResult};

/// Validated direct-route intent. Socket/TLS composition lands with secure
/// delivery in E12; callers cannot silently downgrade to relay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectQuicRoute(pub OverlayRoute);

impl DirectQuicRoute {
    pub fn new(authority: impl Into<String>) -> Option<Self> {
        let authority = authority.into();
        (!authority.trim().is_empty()).then_some(Self(OverlayRoute::DirectQuic { authority }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedQuicEndpoint {
    pub authority: SocketAddr,
    pub server_name: String,
    pub certificate_der: Vec<u8>,
}

impl PinnedQuicEndpoint {
    pub fn validate(&self) -> TransportResult<()> {
        if self.server_name.is_empty()
            || self.server_name.len() > 253
            || !self.server_name.is_ascii()
            || self.certificate_der.is_empty()
            || self.certificate_der.len() > 16 * 1024
        {
            return Err(TransportError::StaleAuthority);
        }
        Ok(())
    }
}

impl TryFrom<&runmat_execution::security::DirectQuicEndpoint> for PinnedQuicEndpoint {
    type Error = TransportError;

    fn try_from(
        endpoint: &runmat_execution::security::DirectQuicEndpoint,
    ) -> Result<Self, Self::Error> {
        let pinned = Self {
            authority: endpoint
                .authority
                .parse()
                .map_err(|_| TransportError::StaleAuthority)?,
            server_name: endpoint.server_name.clone(),
            certificate_der: endpoint.certificate_der.clone(),
        };
        pinned.validate()?;
        let actual = format!("{:x}", sha2::Sha256::digest(&pinned.certificate_der));
        if actual != endpoint.certificate_sha256 {
            return Err(TransportError::StaleAuthority);
        }
        Ok(pinned)
    }
}

pub struct QuicOverlayListener {
    endpoint: Endpoint,
    limits: FrameLimits,
}

impl QuicOverlayListener {
    pub fn bind(
        authority: SocketAddr,
        certificate_chain: Vec<Vec<u8>>,
        private_key_pkcs8: Vec<u8>,
        limits: FrameLimits,
    ) -> TransportResult<Self> {
        if certificate_chain.is_empty() {
            return Err(TransportError::StaleAuthority);
        }
        let certificates = certificate_chain
            .into_iter()
            .map(CertificateDer::from)
            .collect::<Vec<_>>();
        let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(private_key_pkcs8));
        let mut config =
            ServerConfig::with_single_cert(certificates, private_key).map_err(unavailable)?;
        config.transport_config(transport_config(limits));
        let endpoint = Endpoint::server(config, authority).map_err(unavailable)?;
        Ok(Self { endpoint, limits })
    }

    pub fn local_addr(&self) -> TransportResult<SocketAddr> {
        self.endpoint.local_addr().map_err(unavailable)
    }

    pub async fn accept(&self) -> TransportResult<QuicOverlayConnection> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| TransportError::Unavailable("QUIC listener closed".into()))?;
        let connection = incoming.await.map_err(unavailable)?;
        Ok(QuicOverlayConnection {
            connection,
            limits: self.limits,
            _endpoint: None,
        })
    }
}

pub struct QuicOverlayConnection {
    connection: Connection,
    limits: FrameLimits,
    // A client endpoint must outlive its connection.
    _endpoint: Option<Endpoint>,
}

impl QuicOverlayConnection {
    pub async fn connect(
        bind_address: SocketAddr,
        pinned: &PinnedQuicEndpoint,
        limits: FrameLimits,
    ) -> TransportResult<Self> {
        pinned.validate()?;
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(CertificateDer::from(pinned.certificate_der.clone()))
            .map_err(unavailable)?;
        let mut config =
            ClientConfig::with_root_certificates(Arc::new(roots)).map_err(unavailable)?;
        config.transport_config(transport_config(limits));
        let mut endpoint = Endpoint::client(bind_address).map_err(unavailable)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint
            .connect(pinned.authority, &pinned.server_name)
            .map_err(unavailable)?
            .await
            .map_err(unavailable)?;
        Ok(Self {
            connection,
            limits,
            _endpoint: Some(endpoint),
        })
    }

    pub async fn send(&self, frame: &WireFrame) -> TransportResult<()> {
        let bytes = frame.encode(self.limits)?;
        let mut stream = self.connection.open_uni().await.map_err(unavailable)?;
        stream.write_all(&bytes).await.map_err(unavailable)?;
        stream.finish().map_err(unavailable)
    }

    pub async fn receive(&self) -> TransportResult<WireFrame> {
        let mut stream = self.connection.accept_uni().await.map_err(unavailable)?;
        let bytes = stream
            .read_to_end(self.limits.maximum_frame_bytes)
            .await
            .map_err(unavailable)?;
        WireFrame::decode(&bytes, self.limits)
    }

    pub fn close(&self) {
        self.connection.close(0_u32.into(), b"runmat-close");
    }
}

fn transport_config(limits: FrameLimits) -> Arc<quinn::TransportConfig> {
    let mut config = quinn::TransportConfig::default();
    config.max_concurrent_uni_streams(64_u32.into());
    config.max_concurrent_bidi_streams(0_u32.into());
    config.receive_window(
        u32::try_from(limits.maximum_frame_bytes.saturating_mul(64))
            .unwrap_or(u32::MAX)
            .into(),
    );
    Arc::new(config)
}

fn unavailable(error: impl std::fmt::Display) -> TransportError {
    TransportError::Unavailable(error.to_string())
}
