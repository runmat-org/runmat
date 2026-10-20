//! Derive complete memory-store eviction units under the caller's table locks.
use super::*;
use crate::analysis::storage::retention::groups::{evictions, Unit};

#[derive(Serialize)]
pub(in crate::analysis::storage) struct Eviction {
    pub(in crate::analysis::storage) publication: Option<Digest>,
    pub(in crate::analysis::storage) runs: Vec<String>,
}

/// Ranking facts deliberately exclude physical fields, histories and object payloads.
#[derive(Serialize)]
pub(in crate::analysis::storage) struct RunFacts {
    identity: AnalysisRunIdentity,
    created_at: Option<DateTime<Utc>>,
}

impl RunFacts {
    pub(in crate::analysis::storage) fn from_metadata(
        identity: &AnalysisRunIdentity,
        timestamp: &str,
    ) -> Self {
        Self {
            identity: identity.clone(),
            created_at: parse_recorded_at(timestamp),
        }
    }
}

pub(in crate::analysis::storage) fn facts(record: &PersistedAnalysisArtifact) -> RunFacts {
    let (identity, timestamp) = match record {
        PersistedAnalysisArtifact::Standard(r) => (&r.metadata.run.identity, &r.created_at),
        PersistedAnalysisArtifact::StructuralFrequency(r) => (&r.step.identity, &r.created_at),
        PersistedAnalysisArtifact::AcousticFrequency(r) => (&r.step.identity, &r.created_at),
    };
    RunFacts::from_metadata(identity, timestamp)
}

pub(in crate::analysis::storage) fn inventory_facts(
    runs: &HashMap<String, PersistedAnalysisArtifact>,
) -> HashMap<String, RunFacts> {
    runs.iter()
        .map(|(id, record)| (id.clone(), facts(record)))
        .collect()
}

pub(super) fn plan_memory(
    prepared: &PreparedAnalysisPublication,
    identity: Digest,
    runs: &HashMap<String, PersistedAnalysisArtifact>,
    owners: &HashMap<String, Digest>,
    publications: &HashMap<Digest, Vec<u8>>,
) -> Result<Vec<Eviction>, String> {
    let descriptors = publications
        .iter()
        .map(|(root, bytes)| {
            let (descriptor, _): (PublicationDescriptor, serde::de::IgnoredAny) =
                serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
            Ok((*root, descriptor))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;
    plan_inventory(
        &inventory_facts(&prepared.runs),
        Some(identity),
        &inventory_facts(runs),
        owners,
        &descriptors,
    )
}

pub(in crate::analysis::storage) fn plan_inventory(
    incoming_runs: &HashMap<String, RunFacts>,
    identity: Option<Digest>,
    runs: &HashMap<String, RunFacts>,
    owners: &HashMap<String, Digest>,
    publications: &HashMap<Digest, PublicationDescriptor>,
) -> Result<Vec<Eviction>, String> {
    let limits = super::super::retention::resolved_limits()?;
    let mut incoming = Unit {
        id: identity
            .map(digest_hex)
            .unwrap_or_else(|| "incoming-standalone".into()),
        counts: HashMap::new(),
    };
    for record in incoming_runs.values() {
        *incoming.counts.entry(record.identity.clone()).or_default() += 1;
    }
    let mut ranked = Vec::new();
    for (id, record) in runs {
        ranked.push(FilesystemRunArtifact {
            path: PathBuf::new(),
            identity: record.identity.clone(),
            run_id: id.clone(),
            created_at: record.created_at,
            modified_at: None,
        });
    }
    sort_filesystem_artifacts_newest_first(&mut ranked);
    let mut units: Vec<Unit<AnalysisRunIdentity>> = Vec::new();
    let mut positions = HashMap::new();
    let mut inventory: HashMap<String, Eviction> = HashMap::new();
    for record in ranked {
        let owner = owners.get(&record.run_id).copied();
        if identity.is_some() && owner == identity
            || owner.is_none() && incoming_runs.contains_key(&record.run_id)
        {
            continue;
        }
        if owner.is_some_and(|id| !publications.contains_key(&id)) {
            return Err("retention run lacks its committed publication".into());
        }
        let key = owner
            .map(digest_hex)
            .unwrap_or_else(|| format!("standalone:{}", record.run_id));
        let position = *positions.entry(key.clone()).or_insert_with(|| {
            units.push(Unit {
                id: key.clone(),
                counts: HashMap::new(),
            });
            units.len() - 1
        });
        *units[position].counts.entry(record.identity).or_default() += 1;
        inventory
            .entry(key)
            .or_insert_with(|| Eviction {
                publication: owner,
                runs: Vec::new(),
            })
            .runs
            .push(record.run_id);
    }
    for (root, descriptor) in publications {
        if Some(*root) == identity {
            continue;
        }
        if descriptor.schema_version != 1 || descriptor.identity()? != *root {
            return Err("retention publication descriptor has a foreign identity".into());
        }
        let group = inventory
            .get(&digest_hex(*root))
            .ok_or("retention publication has no child inventory")?;
        if group.runs.iter().cloned().collect::<BTreeSet<_>>()
            != descriptor.records.keys().map(ToOwned::to_owned).collect()
        {
            return Err("retention publication has an incomplete child inventory".into());
        }
    }
    if owners.keys().any(|id| !runs.contains_key(id)) {
        return Err("retention publication owns an absent child".into());
    }
    evictions(limits, &incoming, &units)?
        .into_iter()
        .map(|id| {
            inventory
                .remove(&id)
                .ok_or_else(|| "retention selection is absent from inventory".into())
        })
        .collect()
}
