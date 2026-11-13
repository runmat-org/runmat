//! Organization discovery and terminal selection; authorization remains Server-owned.
use std::io::{IsTerminal as _, Write as _};

use anyhow::{bail, Context as _, Result};
use runmat_server_client::{
    auth::{configured_public_org_id, map_public_error, RemoteConfig},
    ids::OrgId,
    public_api,
};
use uuid::Uuid;

struct Candidate {
    id: OrgId,
    name: String,
}

enum Selection {
    Organization(OrgId),
    NextPage,
}

pub(super) async fn resolve(
    client: &public_api::Client,
    config: &RemoteConfig,
    explicit: Option<Uuid>,
    machine_output: bool,
) -> Result<OrgId> {
    if let Some(org) = configured_public_org_id(config, explicit)? {
        return Ok(org);
    }
    let interactive =
        !machine_output && std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let mut cursor: Option<String> = None;
    loop {
        let page = client
            .list_orgs(cursor.as_deref(), Some(100))
            .await
            .map_err(map_public_error)?
            .into_inner();
        let candidates = page
            .items
            .into_iter()
            .map(|org| {
                Ok(Candidate {
                    id: OrgId::try_from(org.id)
                        .context("Server returned an invalid organization identity")?,
                    name: org.name,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if cursor.is_none() && candidates.len() == 1 && page.next_cursor.is_none() {
            return Ok(candidates.into_iter().next().expect("one candidate").id);
        }
        if candidates.is_empty() && page.next_cursor.is_none() {
            bail!("No authorized organization is available for these credentials");
        }
        if !interactive {
            bail!("Organization is ambiguous; select one with --org, RUNMAT_ORG_ID, or runmat login --org <id>");
        }
        if page.next_cursor.is_some() && page.next_cursor == cursor {
            bail!("Server returned a non-advancing organization cursor");
        }
        let has_next = page.next_cursor.is_some();
        match tokio::task::spawn_blocking(move || choose(candidates, has_next))
            .await
            .context("organization selection task failed")??
        {
            Selection::Organization(id) => return Ok(id),
            Selection::NextPage => cursor = page.next_cursor,
        }
    }
}

fn choose(candidates: Vec<Candidate>, has_next: bool) -> Result<Selection> {
    eprintln!("Select an organization (or use --org to select it explicitly):");
    for (index, org) in candidates.iter().enumerate() {
        eprintln!("  {}. {} ({})", index + 1, org.name.escape_debug(), org.id);
    }
    if has_next {
        eprintln!("  0. Next page");
    }
    eprint!("Organization number: ");
    std::io::stderr().flush()?;
    let mut line = String::new();
    // Bound terminal input; a pasted document must not become retained control state.
    use std::io::{BufRead as _, Read as _};
    std::io::stdin().lock().take(1025).read_line(&mut line)?;
    if line.len() > 1024 {
        bail!("Organization selection exceeds 1024 bytes");
    }
    selection(
        candidates,
        has_next,
        line.trim()
            .parse()
            .context("Enter a listed organization number")?,
    )
}

fn selection(mut candidates: Vec<Candidate>, has_next: bool, number: usize) -> Result<Selection> {
    if number == 0 && has_next {
        return Ok(Selection::NextPage);
    }
    let index = number
        .checked_sub(1)
        .filter(|index| *index < candidates.len())
        .context("Organization selection is not in the displayed page")?;
    Ok(Selection::Organization(candidates.swap_remove(index).id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_never_treats_a_page_control_or_invalid_number_as_an_identity() {
        let candidate = || Candidate {
            id: OrgId::from_uuid(Uuid::from_u128(1)),
            name: "Example".into(),
        };
        assert!(matches!(
            selection(vec![candidate()], true, 0).unwrap(),
            Selection::NextPage
        ));
        assert!(selection(vec![candidate()], false, 0).is_err());
        assert!(selection(vec![candidate()], true, 2).is_err());
        assert!(
            matches!(selection(vec![candidate()], false, 1).unwrap(), Selection::Organization(id) if id == candidate().id)
        );
    }
}
