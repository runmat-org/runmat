use runmat_execution::Digest;
use runmat_types::ValueFact;

/// Exact value representation admitted by a specialized native entrypoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationProfile {
    pub digest: Digest,
    pub facts: Vec<ValueFact>,
}

impl RepresentationProfile {
    pub fn from_facts(facts: Vec<ValueFact>, max_bytes: usize) -> Result<Self, &'static str> {
        if facts.len() > 64 {
            return Err("native representation profile has too many values");
        }
        let encoded = serde_json::to_vec(&facts)
            .map_err(|_| "native representation profile could not be encoded")?;
        if encoded.len() > max_bytes {
            return Err("native representation profile exceeds its byte bound");
        }
        Ok(Self {
            digest: Digest::sha256(encoded),
            facts,
        })
    }
}
