mod admission;
mod contract;
mod meshing;
mod metamorphic;
mod report;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use contract::{CorpusManifest, FixtureCatalog, FixtureEntry, ScenarioTier};
use report::CaseReport;
pub use report::CorpusReport;
use sha2::{Digest as _, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedTier {
    Small,
    Complete,
}

impl SelectedTier {
    fn includes(self, tier: ScenarioTier) -> bool {
        self == Self::Complete || tier == ScenarioTier::Small
    }

    fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Complete => "complete",
        }
    }
}

pub fn run(
    manifest_path: &Path,
    tier: SelectedTier,
    fixture_filter: Option<&str>,
) -> Result<CorpusReport> {
    let manifest_bytes = std::fs::read(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: CorpusManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if manifest.schema_version != 1 {
        bail!("unsupported meshing integration manifest schema");
    }

    let catalog_path = resolve_from_working_directory(&manifest.fixture_catalog);
    let catalog_bytes = std::fs::read(&catalog_path)
        .with_context(|| format!("failed to read {}", catalog_path.display()))?;
    let catalog: FixtureCatalog = serde_json::from_slice(&catalog_bytes)
        .with_context(|| format!("failed to parse {}", catalog_path.display()))?;
    validate_contract(&manifest, &catalog)?;
    if fixture_filter.is_some_and(|filter| !catalog.entries.iter().any(|entry| entry.id == filter))
    {
        bail!("fixture filter does not name a governed catalog entry");
    }
    let fixtures = catalog
        .entries
        .iter()
        .map(|entry| (entry.id.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let fixture_root = resolve_from_working_directory(&catalog.fixture_root);
    let mut cases = Vec::with_capacity(manifest.cases.len());

    for case in manifest
        .cases
        .iter()
        .filter(|case| fixture_filter.is_none_or(|filter| case.fixture_id == filter))
    {
        let fixture = fixtures[case.fixture_id.as_str()];
        let source = fixture_root.join(&fixture.path);
        let source_bytes = std::fs::read(&source)
            .with_context(|| format!("failed to read fixture {}", source.display()))?;
        let source_sha256 = sha256(&source_bytes);
        let expected_sha256 = decode_sha256(&fixture.sha256)
            .with_context(|| format!("fixture {} has an invalid SHA-256", fixture.id))?;
        if source_sha256 != expected_sha256 {
            bail!(
                "fixture {} digest changed: expected {}, observed {}",
                fixture.id,
                fixture.sha256,
                encode_sha256(source_sha256)
            );
        }

        eprintln!("admission: {}", fixture.id);
        let admission_scenarios = case
            .admission_scenarios
            .iter()
            .map(|scenario| admission::execute(&source, &source_bytes, case.source_units, scenario))
            .collect::<Vec<_>>();
        let admissions = case
            .admission_scenarios
            .iter()
            .map(|scenario| (scenario.id.as_str(), scenario))
            .collect::<BTreeMap<_, _>>();
        let mut mesh_scenarios = Vec::new();
        for scenario in case
            .mesh_scenarios
            .iter()
            .filter(|scenario| tier.includes(scenario.tier))
        {
            eprintln!("mesh: {} / {}", fixture.id, scenario.id);
            mesh_scenarios.push(meshing::execute(
                &source,
                source_sha256,
                case.source_units,
                admissions[scenario.admission.as_str()],
                scenario,
                &manifest.mesh_profiles[scenario.profile.as_str()],
            ));
        }
        cases.push(CaseReport {
            fixture_id: fixture.id.clone(),
            source_sha256: encode_sha256(source_sha256),
            admission_scenarios,
            mesh_scenarios,
        });
    }

    let passed = cases.iter().all(|case| {
        case.admission_scenarios.iter().all(|report| report.passed)
            && case.mesh_scenarios.iter().all(|report| report.passed)
    });
    let metamorphic_relations = if fixture_filter.is_none() {
        metamorphic::evaluate(&manifest.metamorphic_relations, &cases)
    } else {
        Vec::new()
    };
    let passed = passed && metamorphic_relations.iter().all(|relation| relation.passed);
    Ok(CorpusReport {
        schema_version: 1,
        manifest_sha256: encode_sha256(sha256(&manifest_bytes)),
        fixture_catalog_sha256: encode_sha256(sha256(&catalog_bytes)),
        selected_tier: tier.name().into(),
        fixture_filter: fixture_filter.map(str::to_owned),
        cases,
        metamorphic_relations,
        passed,
    })
}

fn validate_contract(manifest: &CorpusManifest, catalog: &FixtureCatalog) -> Result<()> {
    if catalog.schema_version != 1 || catalog.corpus_revision == 0 {
        bail!("unsupported or unversioned fixture catalog");
    }
    if manifest.mesh_profiles.is_empty() {
        bail!("integration manifest has no governed mesh profiles");
    }
    for (name, profile) in &manifest.mesh_profiles {
        if name.is_empty() || profile.repeats == 0 || profile.partition_layouts.is_empty() {
            bail!("mesh profile {name:?} has invalid repeats or partition layouts");
        }
        if profile
            .partition_layouts
            .iter()
            .any(|layout| layout.edges == 0 || layout.faces == 0)
        {
            bail!("mesh profile {name} contains a zero partition bound");
        }
    }
    let fixture_ids = unique_ids(
        catalog.entries.iter().map(|entry| entry.id.as_str()),
        "fixture catalog",
    )?;
    let case_ids = unique_ids(
        manifest.cases.iter().map(|case| case.fixture_id.as_str()),
        "integration manifest",
    )?;
    if fixture_ids != case_ids {
        let missing = fixture_ids.difference(&case_ids).collect::<Vec<_>>();
        let unknown = case_ids.difference(&fixture_ids).collect::<Vec<_>>();
        bail!(
            "integration manifest must cover every fixture exactly once; missing={missing:?}, unknown={unknown:?}"
        );
    }
    for fixture in &catalog.entries {
        validate_fixture_metadata(fixture)?;
    }
    for case in &manifest.cases {
        if case.admission_scenarios.is_empty() {
            bail!("fixture {} has no admission scenarios", case.fixture_id);
        }
        let admission_ids = unique_ids(
            case.admission_scenarios
                .iter()
                .map(|scenario| scenario.id.as_str()),
            "fixture admission scenarios",
        )?;
        unique_ids(
            case.mesh_scenarios
                .iter()
                .map(|scenario| scenario.id.as_str()),
            "fixture mesh scenarios",
        )?;
        for scenario in &case.mesh_scenarios {
            if !admission_ids.contains(scenario.admission.as_str()) {
                bail!(
                    "fixture {} mesh scenario {} refers to unknown admission scenario {}",
                    case.fixture_id,
                    scenario.id,
                    scenario.admission
                );
            }
            let Some(profile) = manifest.mesh_profiles.get(&scenario.profile) else {
                bail!(
                    "fixture {} mesh scenario {} refers to unknown profile {}",
                    case.fixture_id,
                    scenario.id,
                    scenario.profile
                );
            };
            let _ = profile;
        }
    }
    unique_ids(
        manifest
            .metamorphic_relations
            .iter()
            .map(|relation| relation.id.as_str()),
        "metamorphic relations",
    )?;
    for relation in &manifest.metamorphic_relations {
        if relation.comparisons.is_empty()
            || !relation.volume_relative_tolerance.is_finite()
            || relation.volume_relative_tolerance < 0.0
            || !relation.quality_relative_tolerance.is_finite()
            || relation.quality_relative_tolerance < 0.0
        {
            bail!("metamorphic relation {} has invalid bounds", relation.id);
        }
        for reference in std::iter::once(&relation.reference).chain(&relation.comparisons) {
            let Some(case) = manifest
                .cases
                .iter()
                .find(|case| case.fixture_id == reference.fixture_id)
            else {
                bail!(
                    "metamorphic relation {} refers to unknown fixture {}",
                    relation.id,
                    reference.fixture_id
                );
            };
            let Some(scenario) = case
                .mesh_scenarios
                .iter()
                .find(|scenario| scenario.id == reference.scenario_id)
            else {
                bail!(
                    "metamorphic relation {} refers to unknown scenario {}/{}",
                    relation.id,
                    reference.fixture_id,
                    reference.scenario_id
                );
            };
            if scenario.tier != ScenarioTier::Small {
                bail!(
                    "metamorphic relation {} must use small-tier scenarios so it is never silently skipped",
                    relation.id
                );
            }
        }
    }
    Ok(())
}

fn validate_fixture_metadata(fixture: &FixtureEntry) -> Result<()> {
    if fixture.path.is_empty()
        || fixture.tier.is_empty()
        || fixture.format.is_empty()
        || fixture.features.is_empty()
        || !fixture.provenance.is_object()
        || !fixture.expected.is_object()
        || !fixture.test.is_object()
    {
        bail!("fixture {} has incomplete governed metadata", fixture.id);
    }
    let _ = fixture.supported_input;
    Ok(())
}

fn unique_ids<'a>(ids: impl Iterator<Item = &'a str>, label: &str) -> Result<BTreeSet<&'a str>> {
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.is_empty() || !unique.insert(id) {
            bail!("{label} contains an empty or duplicate identifier");
        }
    }
    Ok(unique)
}

fn resolve_from_working_directory(path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .expect("current directory must remain available")
            .join(path)
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn decode_sha256(encoded: &str) -> Result<[u8; 32]> {
    if encoded.len() != 64 {
        bail!("SHA-256 must contain exactly 64 hexadecimal characters");
    }
    let mut decoded = [0_u8; 32];
    for (index, pair) in encoded.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).context("invalid hexadecimal digit")?;
        let low = hex_nibble(pair[1]).context("invalid hexadecimal digit")?;
        decoded[index] = (high << 4) | low;
    }
    Ok(decoded)
}

fn encode_sha256(bytes: [u8; 32]) -> String {
    use std::fmt::Write;

    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
