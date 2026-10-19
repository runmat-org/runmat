//! FEA acceptance obligations and discoverable witnesses, not production physics.
//! Register integrity never certifies that a witness passed or proves its claims.
#[cfg(feature = "docs-export")]
pub mod export;
pub mod requirement;
mod requirements;

pub fn requirements() -> impl Iterator<Item = &'static requirement::Requirement> {
    requirements::ALL.iter().flat_map(|domain| domain.iter())
}

/// Deterministic tab-separated discovery inventory. No row represents a passed gate.
/// Tabs, newlines, carriage returns and backslashes inside fields are escaped.
/// Required envelopes and claims are separate so partial linkage cannot imply proof.
pub fn write_inventory(mut output: impl std::io::Write) -> std::io::Result<()> {
    use requirement::WitnessLocator;
    fn row(out: &mut impl std::io::Write, fields: &[&str]) -> std::io::Result<()> {
        for (index, field) in fields.iter().enumerate() {
            if index != 0 {
                out.write_all(b"\t")?;
            }
            for character in field.chars() {
                match character {
                    '\\' => out.write_all(b"\\\\")?,
                    '\t' => out.write_all(b"\\t")?,
                    '\n' => out.write_all(b"\\n")?,
                    '\r' => out.write_all(b"\\r")?,
                    _ => {
                        let mut bytes = [0; 4];
                        out.write_all(character.encode_utf8(&mut bytes).as_bytes())?;
                    }
                }
            }
        }
        out.write_all(b"\n")
    }
    row(&mut output, &["inventory", "1", "discovery_only"])?;
    let mut witnesses = std::collections::BTreeSet::new();
    for requirement in requirements() {
        row(
            &mut output,
            &[
                "requirement",
                requirement.id,
                requirement.title,
                requirement.acceptance,
                &format!("{:?}", requirement.stages),
            ],
        )?;
        for owner in requirement.owners {
            row(&mut output, &["owner", requirement.id, owner])?;
        }
        for dependency in requirement.dependencies {
            row(&mut output, &["dependency", requirement.id, dependency])?;
        }
        for obligation in requirement.obligations {
            row(
                &mut output,
                &[
                    "obligation",
                    requirement.id,
                    obligation.id,
                    obligation.criterion,
                    &format!("{:?}", obligation.evidence_kind),
                ],
            )?;
            for envelope in obligation.required_envelopes {
                row(
                    &mut output,
                    &[
                        "required",
                        requirement.id,
                        obligation.id,
                        &format!("{:?}", envelope.formulation),
                        &format!("{:?}", envelope.mesh),
                        &format!("{:?}", envelope.execution),
                    ],
                )?;
            }
        }
        for witness in requirement.witnesses {
            if witnesses.insert(witness.id) {
                match witness.locator {
                    WitnessLocator::CargoTest {
                        package,
                        target,
                        name,
                        source,
                    } => row(
                        &mut output,
                        &[
                            "witness",
                            witness.id,
                            "cargo",
                            package,
                            &format!("{:?}", target),
                            name,
                            source,
                        ],
                    )?,
                    WitnessLocator::GovernanceCheck { source, check } => row(
                        &mut output,
                        &["witness", witness.id, "governance", source, check],
                    )?,
                }
            }
            for claim in witness.claims {
                for envelope in claim.envelopes {
                    row(
                        &mut output,
                        &[
                            "claim",
                            requirement.id,
                            claim.obligation,
                            witness.id,
                            &format!("{:?}", envelope.formulation),
                            &format!("{:?}", envelope.mesh),
                            &format!("{:?}", envelope.execution),
                            claim.boundary,
                        ],
                    )?;
                }
            }
        }
    }
    Ok(())
}
