use crate::{requirement::validate, requirements};
use serde::Serialize;

#[derive(Serialize)]
struct RegisterExport<'a> {
    schema_version: u16,
    posture: &'static str,
    requirements: Vec<&'a crate::requirement::Requirement>,
}

pub fn write_requirements(output: impl std::io::Write) -> Result<(), Box<dyn std::error::Error>> {
    validate(requirements()).map_err(|error| format!("Invalid register: {error:?}"))?;
    let register = RegisterExport {
        schema_version: 1,
        posture: "discovery_only",
        requirements: requirements().collect(),
    };
    serde_json::to_writer_pretty(output, &register)?;
    Ok(())
}
