use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Backend-neutral coverage metric. Only function and statement sites are
/// emitted initially; the remaining variants reserve stable extension points
/// for richer compiler instrumentation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageMetric {
    Function,
    Statement,
    Decision,
    Condition,
    McdcCondition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageBackend {
    Interpreter,
    Jit,
    Wasm,
    NativeAot,
    Gpu,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "support")]
pub enum CoverageBackendSupport {
    Full,
    HostBoundaryOnly { reason: String },
    Unsupported { reason: String },
}

impl CoverageBackend {
    pub fn coverage_support(self) -> CoverageBackendSupport {
        match self {
            Self::Interpreter | Self::Jit | Self::Wasm => CoverageBackendSupport::Full,
            Self::Gpu => CoverageBackendSupport::HostBoundaryOnly {
                reason: "the MATLAB source statement dispatching device work is covered; generated device-kernel internals have no MATLAB source sites".into(),
            },
            Self::NativeAot => CoverageBackendSupport::Unsupported {
                reason: "native AOT execution is not an available RunMat test backend".into(),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageSite {
    pub id: String,
    #[serde(with = "u64_string")]
    pub counter_key: u64,
    pub metric: CoverageMetric,
    pub owner_identity: String,
    pub relative_path: String,
    pub semantic_path: String,
    pub source_id: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub instrumented: bool,
    pub unsupported_reason: Option<String>,
}

/// Counters produced by one worker for one immutable executable unit.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageFragment {
    pub program_revision: String,
    pub plan_revision: String,
    pub sites: Vec<CoverageSite>,
    #[serde(with = "u64_key_map")]
    pub counts: BTreeMap<u64, u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageAggregate {
    pub program_revision: Option<String>,
    pub sites: Vec<CoverageSite>,
    pub counts: BTreeMap<String, u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageSummary {
    pub covered: u64,
    pub instrumented: u64,
    pub unsupported: u64,
}

impl CoverageSummary {
    pub fn percentage(self) -> Option<f64> {
        (self.instrumented != 0).then(|| self.covered as f64 * 100.0 / self.instrumented as f64)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageFileSummary {
    pub owner_identity: String,
    pub relative_path: String,
    pub functions: CoverageSummary,
    pub statements: CoverageSummary,
}

impl CoverageAggregate {
    pub fn count(&self, site: &CoverageSite) -> u64 {
        self.counts.get(&site.id).copied().unwrap_or(0)
    }

    pub fn summary(&self, metric: CoverageMetric) -> CoverageSummary {
        summarize(self.sites.iter().filter(|site| site.metric == metric), self)
    }

    pub fn files(&self) -> Vec<CoverageFileSummary> {
        let mut files = BTreeMap::<(&str, &str), Vec<&CoverageSite>>::new();
        for site in &self.sites {
            files
                .entry((&site.owner_identity, &site.relative_path))
                .or_default()
                .push(site);
        }
        files
            .into_iter()
            .map(
                |((owner_identity, relative_path), sites)| CoverageFileSummary {
                    owner_identity: owner_identity.to_owned(),
                    relative_path: relative_path.to_owned(),
                    functions: summarize(
                        sites
                            .iter()
                            .copied()
                            .filter(|site| site.metric == CoverageMetric::Function),
                        self,
                    ),
                    statements: summarize(
                        sites
                            .iter()
                            .copied()
                            .filter(|site| site.metric == CoverageMetric::Statement),
                        self,
                    ),
                },
            )
            .collect()
    }
}

fn summarize<'a>(
    sites: impl Iterator<Item = &'a CoverageSite>,
    aggregate: &CoverageAggregate,
) -> CoverageSummary {
    let mut summary = CoverageSummary::default();
    for site in sites {
        if site.instrumented {
            summary.instrumented += 1;
            summary.covered += u64::from(aggregate.count(site) != 0);
        } else {
            summary.unsupported += 1;
        }
    }
    summary
}

mod u64_key_map {
    use std::collections::BTreeMap;

    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(counts: &BTreeMap<u64, u64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        counts
            .iter()
            .map(|(key, count)| (key.to_string(), *count))
            .collect::<BTreeMap<_, _>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<u64, u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        BTreeMap::<String, u64>::deserialize(deserializer)?
            .into_iter()
            .map(|(key, count)| {
                key.parse::<u64>()
                    .map(|key| (key, count))
                    .map_err(D::Error::custom)
            })
            .collect()
    }
}

mod u64_string {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse::<u64>()
            .map_err(D::Error::custom)
    }
}
