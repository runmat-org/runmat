use super::OverlayRoute;
use crate::frame::{FrameLimits, WireFrame};
use crate::{TransportError, TransportResult};
use futures_util::{SinkExt as _, StreamExt as _};
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::Message;

const DUPLEX_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueRelayRoute(pub OverlayRoute);

impl OpaqueRelayRoute {
    pub fn new(authority: impl Into<String>) -> Option<Self> {
        let authority = authority.into();
        (!authority.trim().is_empty()).then_some(Self(OverlayRoute::OpaqueRelay { authority }))
    }
}

pub struct WebSocketRelayConnection {
    socket: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    limits: FrameLimits,
}

pub struct WebSocketRelayDuplex {
    outbound: tokio::sync::mpsc::Sender<WireFrame>,
    inbound: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<TransportResult<WireFrame>>>,
}

impl WebSocketRelayConnection {
    pub async fn connect_submitter(
        url: &str,
        relay_ticket: &str,
        limits: FrameLimits,
    ) -> TransportResult<Self> {
        if relay_ticket.is_empty()
            || relay_ticket.len() > 256
            || !relay_ticket
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(TransportError::MalformedFrame(
                "relay ticket is malformed".into(),
            ));
        }
        Self::connect(
            url,
            &[(
                "Sec-WebSocket-Protocol".into(),
                format!("runmat-relay-v1, runmat-ticket.{relay_ticket}"),
            )],
            limits,
        )
        .await
    }

    pub async fn connect(
        url: &str,
        headers: &[(String, String)],
        limits: FrameLimits,
    ) -> TransportResult<Self> {
        let mut request = url.into_client_request().map_err(unavailable)?;
        for (name, value) in headers {
            let name = name
                .parse::<tokio_tungstenite::tungstenite::http::HeaderName>()
                .map_err(unavailable)?;
            let value = value
                .parse::<tokio_tungstenite::tungstenite::http::HeaderValue>()
                .map_err(unavailable)?;
            request.headers_mut().insert(name, value);
        }
        let config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
            max_message_size: Some(limits.maximum_frame_bytes),
            max_frame_size: Some(limits.maximum_frame_bytes),
            ..Default::default()
        };
        let (socket, _) =
            tokio_tungstenite::connect_async_with_config(request, Some(config), false)
                .await
                .map_err(unavailable)?;
        Ok(Self { socket, limits })
    }

    pub async fn send(&mut self, frame: &WireFrame) -> TransportResult<()> {
        self.socket
            .send(Message::Binary(frame.encode(self.limits)?))
            .await
            .map_err(unavailable)
    }

    pub async fn receive(&mut self) -> TransportResult<WireFrame> {
        while let Some(message) = self.socket.next().await {
            match message.map_err(unavailable)? {
                Message::Binary(bytes) => return WireFrame::decode(&bytes, self.limits),
                Message::Ping(bytes) => self
                    .socket
                    .send(Message::Pong(bytes))
                    .await
                    .map_err(unavailable)?,
                Message::Close(_) => {
                    return Err(TransportError::Unavailable("relay closed".into()))
                }
                Message::Text(_) | Message::Pong(_) | Message::Frame(_) => {}
            }
        }
        Err(TransportError::Unavailable("relay closed".into()))
    }

    pub async fn close(mut self) -> TransportResult<()> {
        self.socket.close(None).await.map_err(unavailable)
    }

    pub fn into_duplex(self) -> WebSocketRelayDuplex {
        let (outbound, mut outbound_rx) = tokio::sync::mpsc::channel::<WireFrame>(DUPLEX_CAPACITY);
        let (inbound_tx, inbound) = tokio::sync::mpsc::channel(DUPLEX_CAPACITY);
        let limits = self.limits;
        let (mut sink, mut source) = self.socket.split();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    outgoing = outbound_rx.recv() => {
                        let Some(frame) = outgoing else { break; };
                        let encoded = match frame.encode(limits) {
                            Ok(encoded) => encoded,
                            Err(error) => {
                                let _ = inbound_tx.send(Err(error)).await;
                                break;
                            }
                        };
                        if let Err(error) = sink.send(Message::Binary(encoded)).await {
                            let _ = inbound_tx.send(Err(unavailable(error))).await;
                            break;
                        }
                    }
                    incoming = source.next() => {
                        let result = match incoming {
                            Some(Ok(Message::Binary(bytes))) => WireFrame::decode(&bytes, limits),
                            Some(Ok(Message::Ping(bytes))) => {
                                if let Err(error) = sink.send(Message::Pong(bytes)).await {
                                    Err(unavailable(error))
                                } else {
                                    continue;
                                }
                            }
                            Some(Ok(Message::Pong(_))) | Some(Ok(Message::Text(_)))
                            | Some(Ok(Message::Frame(_))) => continue,
                            Some(Ok(Message::Close(_))) | None => {
                                Err(TransportError::Unavailable("relay closed".into()))
                            }
                            Some(Err(error)) => Err(unavailable(error)),
                        };
                        let terminal = result.is_err();
                        if inbound_tx.send(result).await.is_err() || terminal {
                            break;
                        }
                    }
                }
            }
            let _ = sink.close().await;
        });
        WebSocketRelayDuplex {
            outbound,
            inbound: tokio::sync::Mutex::new(inbound),
        }
    }
}

impl WebSocketRelayDuplex {
    pub async fn send(&self, frame: WireFrame) -> TransportResult<()> {
        self.outbound
            .send(frame)
            .await
            .map_err(|_| TransportError::Unavailable("relay closed".into()))
    }

    pub async fn receive(&self) -> TransportResult<WireFrame> {
        self.inbound
            .lock()
            .await
            .recv()
            .await
            .unwrap_or_else(|| Err(TransportError::Unavailable("relay closed".into())))
    }
}

fn unavailable(error: impl std::fmt::Display) -> TransportError {
    TransportError::Unavailable(error.to_string())
}
