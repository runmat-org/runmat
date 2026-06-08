use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use uuid::Uuid;

pub async fn git_clone(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    if directory.exists() && fs::read_dir(directory)?.next().is_some() {
        anyhow::bail!("Destination directory is not empty")
    }
    fs::create_dir_all(directory)?;
    run_git(directory, &["init"]).context("Failed to init git repo")?;
    let stream = fetch_git_stream(server_url, token, project_id).await?;
    run_git_with_input(directory, &["fast-import"], &stream)
        .context("Failed to import snapshot history")?;
    run_git(directory, &["checkout", "-f", "main"]).context("Failed to checkout")?;
    run_git(
        directory,
        &["config", "runmat.server", server_url.trim_end_matches('/')],
    )
    .ok();
    run_git(
        directory,
        &["config", "runmat.project", &project_id.to_string()],
    )
    .ok();
    println!("Cloned snapshots into {}", directory.display());
    Ok(())
}

pub async fn git_pull(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    let stream = fetch_git_stream(server_url, token, project_id).await?;
    run_git_with_input(directory, &["fast-import"], &stream)
        .context("Failed to import snapshot history")?;
    run_git(directory, &["checkout", "-f", "main"]).context("Failed to checkout")?;
    println!("Pulled snapshots into {}", directory.display());
    Ok(())
}

pub async fn git_push(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    let branches = run_git_capture(
        directory,
        &["for-each-ref", "refs/heads", "--format=%(refname)"],
    )
    .context("Failed to list branches")?;
    let branch_list = branches
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if branch_list.len() != 1 || branch_list[0] != "refs/heads/main" {
        anyhow::bail!("Only refs/heads/main is supported for git push")
    }
    let export = run_git_capture(
        directory,
        &["fast-export", "--full-tree", "--all", "--show-original-ids"],
    )
    .context("Failed to export git history")?;
    let client = reqwest::Client::new();
    let url = format!(
        "{}/v1/projects/{}/fs/git/receive",
        server_url.trim_end_matches('/'),
        project_id
    );
    let response = client
        .post(url)
        .header("authorization", format!("Bearer {}", token))
        .header("content-type", "application/octet-stream")
        .body(export)
        .send()
        .await
        .context("Failed to push git history")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Git push failed ({status}): {text}")
    }
    println!("Pushed git history to project {project_id}");
    Ok(())
}

async fn fetch_git_stream(server_url: &str, token: &str, project_id: Uuid) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/v1/projects/{}/fs/git/upload",
        server_url.trim_end_matches('/'),
        project_id
    );
    let response = client
        .get(url)
        .header("authorization", format!("Bearer {}", token))
        .send()
        .await
        .context("Failed to download git export")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Git export failed ({status}): {text}")
    }
    Ok(response.bytes().await?.to_vec())
}

fn run_git(directory: &PathBuf, args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .current_dir(directory)
        .args(args)
        .status()
        .context("Failed to run git")?;
    if !status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(())
}

fn run_git_with_input(directory: &PathBuf, args: &[&str], input: &[u8]) -> Result<()> {
    let mut child = Command::new("git")
        .current_dir(directory)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to run git")?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input)?;
    }
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(())
}

fn run_git_capture(directory: &PathBuf, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(args)
        .output()
        .context("Failed to run git")?;
    if !output.status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
