//! Versioned solver records reuse the public RNG state format and validator.

use super::*;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PortableSnapshot {
    algorithm: random::RngAlgorithm,
    seed: Option<u64>,
    words: Vec<u32>,
}

impl PortableSnapshot {
    pub(crate) fn capture(snapshot: &RngSnapshot) -> Self {
        let words = match &snapshot.state {
            random::RandomState::MtSeed(_) | random::RandomState::Mt(_) => {
                let state = snapshot.state.mt_snapshot().expect("twister state");
                let mut words = Vec::with_capacity(625);
                words.extend_from_slice(state.words());
                words.push(state.index() as u32);
                words
            }
            random::RandomState::Lcg(state) => vec![*state as u32, (*state >> 32) as u32],
        };
        Self {
            algorithm: snapshot.algorithm(),
            seed: snapshot.seed,
            words,
        }
    }

    pub(super) fn public_value(mut self) -> BuiltinResult<Value> {
        self.seed = Some(self.seed.unwrap_or(DEFAULT_USER_SEED));
        self.into_value()
    }

    pub(crate) fn restore(
        self,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<RngSnapshot> {
        let _entry = runtime.enter();
        let value = self.into_value()?;
        // This is the canonical class/shape/state and extension validator,
        // not an independent RNG decoder or a replacement stream.
        restore_snapshot_value(&value, Some(runtime))
    }

    fn into_value(self) -> BuiltinResult<Value> {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.fields.insert(
            "Type".to_owned(),
            Value::string(self.algorithm.as_str().to_owned()),
        );
        if let Some(seed) = self.seed {
            let value = if crate::builtins::common::validation::integer_is_exact_f64(
                &IntValue::U64(seed),
            ) {
                Value::Num(seed as f64)
            } else {
                Value::Int(IntValue::U64(seed))
            };
            fields.fields.insert("Seed".to_owned(), value);
        }
        let len = self.words.len();
        let tensor = Tensor::new_integer(IntegerStorage::U32(self.words), vec![len, 1])
            .map_err(|error| rng_internal_error(format!("rng: {error}")))?;
        fields
            .fields
            .insert("State".to_owned(), Value::Tensor(tensor));
        Ok(Value::Struct(fields.finish()))
    }
}
