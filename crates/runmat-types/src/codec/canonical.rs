use crate::{ValueFact, RUNMAT_TYPES_SCHEMA};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct FactEnvelope {
    schema: String,
    major: u16,
    minor: u16,
    fact: ValueFact,
}

#[derive(Debug, thiserror::Error)]
pub enum CanonicalCodecError {
    #[error("invalid fact payload: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported fact schema {schema} v{major}.{minor}")]
    UnsupportedSchema {
        schema: String,
        major: u16,
        minor: u16,
    },
}

pub fn encode_canonical(fact: &ValueFact) -> Result<Vec<u8>, CanonicalCodecError> {
    Ok(serde_json::to_vec(&FactEnvelope {
        schema: RUNMAT_TYPES_SCHEMA.name.to_string(),
        major: RUNMAT_TYPES_SCHEMA.major,
        minor: RUNMAT_TYPES_SCHEMA.minor,
        fact: fact.clone(),
    })?)
}

pub fn decode_canonical(bytes: &[u8]) -> Result<ValueFact, CanonicalCodecError> {
    let envelope: FactEnvelope = serde_json::from_slice(bytes)?;
    if envelope.schema != RUNMAT_TYPES_SCHEMA.name
        || envelope.major != RUNMAT_TYPES_SCHEMA.major
        || envelope.minor > RUNMAT_TYPES_SCHEMA.minor
    {
        return Err(CanonicalCodecError::UnsupportedSchema {
            schema: envelope.schema,
            major: envelope.major,
            minor: envelope.minor,
        });
    }
    Ok(envelope.fact)
}
