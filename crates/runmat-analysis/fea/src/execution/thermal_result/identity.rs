//! Shared numerical-identity tuple for retained content and preassembly shape projection.
use super::*;
use crate::{
    admission::AnalysisRunIdentity,
    execution::{StepIdentity, ThermalNumericalIntentIdentity},
};
use serde::Serialize;

// Tuple order is part of numerical identity. Generic borrowed/repeated inventories preserve that
// same wire authority without building hypothetical histories for resource admission.
#[derive(Serialize)]
pub(in crate::execution) struct IdentitySerialization<'a, T, F, R, E, N, L, I, H>(
    pub ThermalNumericalIntentIdentity,
    pub &'a AnalysisRunIdentity,
    pub T,
    pub F,
    pub R,
    pub f64,
    pub E,
    pub f64,
    pub Digest,
    pub StepIdentity,
    pub N,
    pub L,
    pub I,
    pub H,
);

impl ThermalResultMetadata<'_> {
    pub(in crate::execution) fn identity_serialization<'a, F: Serialize + 'a>(
        &'a self,
        fields: F,
    ) -> impl Serialize + 'a {
        IdentitySerialization(
            self.numerical_intent,
            &self.run.identity,
            self.time_points_s,
            fields,
            self.residual_norms,
            self.residual_target,
            self.energy_balances,
            self.reference_temperature_k.canonical_value(),
            self.topology_digest,
            self.producer,
            self.ordered_node_ids,
            self.ordered_element_ids,
            self.initial_state,
            self.nonlinear_evidence,
        )
    }

    pub(in crate::execution) fn identity<F: Serialize>(
        &self,
        fields: F,
    ) -> AnalysisExecutionResult<SolverResultIdentity> {
        // Presentation diagnostics and execution telemetry are not numerical identity inputs.
        let content = encode_contract(
            PREFIX,
            "thermal-solver-result-accepted-intent-field-content",
            &self.identity_serialization(fields),
            LIMITS,
        )?;
        SolverResultIdentity::from_step_result(self.producer, Digest::sha256(content))
    }
}

/// Per-sample field identities. Each sample serializes as a fixed tuple (temperature,
/// gradient, flux, storage rate, then the second-side interface temperature when present);
/// a study with finite-conductance interfaces pairs the samples with the interface node ids,
/// so a study without one keeps exactly its earlier identity.
pub(in crate::execution) struct FieldIdentities<'a, F> {
    pub samples: &'a [F],
    pub interface_node_ids: &'a [u64],
    pub identities: fn(&F) -> Vec<crate::results::field::RecoveredFieldIdentity>,
}

struct Samples<'a, F> {
    samples: &'a [F],
    identities: fn(&F) -> Vec<crate::results::field::RecoveredFieldIdentity>,
}

impl<F> Serialize for Samples<'_, F> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeSeq, SerializeTuple};
        struct Row(Vec<crate::results::field::RecoveredFieldIdentity>);
        impl Serialize for Row {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut tuple = serializer.serialize_tuple(self.0.len())?;
                for identity in &self.0 {
                    tuple.serialize_element(identity)?;
                }
                tuple.end()
            }
        }
        let mut sequence = serializer.serialize_seq(Some(self.samples.len()))?;
        for sample in self.samples {
            sequence.serialize_element(&Row((self.identities)(sample)))?;
        }
        sequence.end()
    }
}

impl<F> Serialize for FieldIdentities<'_, F> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let samples = Samples {
            samples: self.samples,
            identities: self.identities,
        };
        if self.interface_node_ids.is_empty() {
            samples.serialize(serializer)
        } else {
            (samples, self.interface_node_ids).serialize(serializer)
        }
    }
}

pub(in crate::execution) fn manifest_entry_identities(
    entries: &Vec<crate::results::field::RecoveredFieldArtifactEntry>,
) -> Vec<crate::results::field::RecoveredFieldIdentity> {
    entries.iter().map(|entry| entry.field).collect()
}

pub(in crate::execution) fn retained_identities(
    identities: &Vec<crate::results::field::RecoveredFieldIdentity>,
) -> Vec<crate::results::field::RecoveredFieldIdentity> {
    identities.clone()
}
