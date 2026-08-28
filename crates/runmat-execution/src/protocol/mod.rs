use std::collections::BTreeSet;

use minicbor::{Decoder, Encoder};
use serde::{Deserialize, Serialize};

use crate::schema::{PROTOCOL_MAJOR_V1, PROTOCOL_MINOR_V1};
use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V1: Self = Self {
        major: PROTOCOL_MAJOR_V1,
        minor: PROTOCOL_MINOR_V1,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolLimits {
    pub max_message_bytes: u32,
    pub max_payload_bytes: u32,
    pub max_collection_items: u32,
    pub max_nesting_depth: u16,
}

impl Default for ProtocolLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: 16 * 1024 * 1024,
            max_payload_bytes: 16 * 1024 * 1024 - 128,
            max_collection_items: 1_000_000,
            max_nesting_depth: 64,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolHello {
    pub supported_majors: Vec<u16>,
    pub maximum_minor_by_major: Vec<(u16, u16)>,
    pub implementation: String,
    pub capabilities: BTreeSet<String>,
    pub limits: ProtocolLimits,
}

impl ProtocolHello {
    pub fn v1(
        implementation: impl Into<String>,
        capabilities: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            supported_majors: vec![PROTOCOL_MAJOR_V1],
            maximum_minor_by_major: vec![(PROTOCOL_MAJOR_V1, PROTOCOL_MINOR_V1)],
            implementation: implementation.into(),
            capabilities: capabilities.into_iter().collect(),
            limits: ProtocolLimits::default(),
        }
    }
}

pub fn negotiate(
    left: &ProtocolHello,
    right: &ProtocolHello,
) -> Result<ProtocolVersion, ContractError> {
    let major = left
        .supported_majors
        .iter()
        .filter(|major| right.supported_majors.contains(major))
        .max()
        .copied()
        .ok_or_else(|| ContractError::invalid("protocol", "no shared protocol major"))?;
    let left_minor = minor_for(left, major)?;
    let right_minor = minor_for(right, major)?;
    Ok(ProtocolVersion {
        major,
        minor: left_minor.min(right_minor),
    })
}

fn minor_for(hello: &ProtocolHello, major: u16) -> Result<u16, ContractError> {
    hello
        .maximum_minor_by_major
        .iter()
        .find_map(|(candidate, minor)| (*candidate == major).then_some(*minor))
        .ok_or_else(|| {
            ContractError::invalid(
                "protocol hello",
                format!("major {major} lacks a maximum minor"),
            )
        })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Envelope {
    pub version: ProtocolVersion,
    pub message_kind: u16,
    pub flags: u32,
    pub sequence: u64,
    pub payload: Vec<u8>,
}

impl Envelope {
    pub fn encode(&self, limits: ProtocolLimits) -> Result<Vec<u8>, ContractError> {
        if self.payload.len() > limits.max_payload_bytes as usize {
            return Err(ContractError::Limit {
                field: "protocol payload bytes",
                limit: limits.max_payload_bytes.into(),
            });
        }
        let mut output = Vec::with_capacity(self.payload.len() + 32);
        let mut encoder = Encoder::new(&mut output);
        encoder
            .map(6)
            .and_then(|encoder| encoder.u8(0))
            .and_then(|encoder| encoder.u16(self.version.major))
            .and_then(|encoder| encoder.u8(1))
            .and_then(|encoder| encoder.u16(self.version.minor))
            .and_then(|encoder| encoder.u8(2))
            .and_then(|encoder| encoder.u16(self.message_kind))
            .and_then(|encoder| encoder.u8(3))
            .and_then(|encoder| encoder.u32(self.flags))
            .and_then(|encoder| encoder.u8(4))
            .and_then(|encoder| encoder.u64(self.sequence))
            .and_then(|encoder| encoder.u8(5))
            .and_then(|encoder| encoder.bytes(&self.payload))
            .map_err(protocol_encode_error)?;
        if output.len() > limits.max_message_bytes as usize {
            return Err(ContractError::Limit {
                field: "protocol message bytes",
                limit: limits.max_message_bytes.into(),
            });
        }
        Ok(output)
    }

    pub fn decode(bytes: &[u8], limits: ProtocolLimits) -> Result<Self, ContractError> {
        if bytes.len() > limits.max_message_bytes as usize {
            return Err(ContractError::Limit {
                field: "protocol message bytes",
                limit: limits.max_message_bytes.into(),
            });
        }
        let mut decoder = Decoder::new(bytes);
        let fields = decoder
            .map()
            .map_err(protocol_decode_error)?
            .ok_or_else(|| {
                ContractError::MalformedProtocol("indefinite maps are prohibited".into())
            })?;
        if fields > 64 {
            return Err(ContractError::Limit {
                field: "protocol envelope fields",
                limit: 64,
            });
        }

        let mut major = None;
        let mut minor = None;
        let mut message_kind = None;
        let mut flags = None;
        let mut sequence = None;
        let mut payload = None;
        let mut previous_key = None;

        for _ in 0..fields {
            let key = decoder.u16().map_err(protocol_decode_error)?;
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(ContractError::MalformedProtocol(
                    "envelope keys must be unique and ascending".into(),
                ));
            }
            previous_key = Some(key);
            match key {
                0 => major = Some(decoder.u16().map_err(protocol_decode_error)?),
                1 => minor = Some(decoder.u16().map_err(protocol_decode_error)?),
                2 => message_kind = Some(decoder.u16().map_err(protocol_decode_error)?),
                3 => flags = Some(decoder.u32().map_err(protocol_decode_error)?),
                4 => sequence = Some(decoder.u64().map_err(protocol_decode_error)?),
                5 => {
                    let encoded = decoder.bytes().map_err(protocol_decode_error)?;
                    if encoded.len() > limits.max_payload_bytes as usize {
                        return Err(ContractError::Limit {
                            field: "protocol payload bytes",
                            limit: limits.max_payload_bytes.into(),
                        });
                    }
                    payload = Some(encoded.to_vec());
                }
                _ => decoder.skip().map_err(protocol_decode_error)?,
            }
        }
        if decoder.position() != bytes.len() {
            return Err(ContractError::MalformedProtocol(
                "trailing bytes after envelope".into(),
            ));
        }
        Ok(Self {
            version: ProtocolVersion {
                major: required("major", major)?,
                minor: required("minor", minor)?,
            },
            message_kind: required("message kind", message_kind)?,
            flags: required("flags", flags)?,
            sequence: required("sequence", sequence)?,
            payload: required("payload", payload)?,
        })
    }
}

fn required<T>(field: &'static str, value: Option<T>) -> Result<T, ContractError> {
    value.ok_or_else(|| ContractError::MalformedProtocol(format!("missing {field}")))
}

fn protocol_encode_error<E: std::fmt::Display>(error: E) -> ContractError {
    ContractError::MalformedProtocol(error.to_string())
}

fn protocol_decode_error(error: minicbor::decode::Error) -> ContractError {
    ContractError::MalformedProtocol(error.to_string())
}
