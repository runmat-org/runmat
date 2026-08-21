use runmat_package::{
    plan_git_acquisition, validate_git_acquisition, ContentDigest, GitAcquisitionIntent,
    GitAcquisitionPolicy, GitCommitId, GitLockAction, GitRepositoryUrl, GitSelector, GitSourceId,
    NormalizedRelativePath,
};

fn repository() -> GitRepositoryUrl {
    GitRepositoryUrl::new("https://example.com/acme/project.git").unwrap()
}

fn subdir() -> NormalizedRelativePath {
    NormalizedRelativePath::new("package").unwrap()
}

fn locked() -> GitSourceId {
    GitSourceId::new(
        repository().as_str(),
        "0123456789abcdef0123456789abcdef01234567"
            .parse::<GitCommitId>()
            .unwrap(),
        subdir(),
        ContentDigest::sha256("tree"),
    )
    .unwrap()
}

fn mutable_selector() -> GitSelector {
    GitSelector::Branch {
        value: "main".to_string(),
    }
}

#[test]
fn normal_execution_uses_exact_lock_but_can_fill_a_missing_cache() {
    let locked = locked();
    let plan = plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Execute,
        GitAcquisitionPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        plan.selector,
        GitSelector::Rev {
            value: locked.commit.hex.clone()
        }
    );
    assert!(plan.allow_network);
    assert_eq!(plan.expected, Some(locked));
    assert_eq!(plan.lock_action, GitLockAction::Preserve);
}

#[test]
fn frozen_is_locked_offline_and_never_mutates() {
    let locked = locked();
    let plan = plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Execute,
        GitAcquisitionPolicy {
            frozen: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!plan.allow_network);
    assert_eq!(plan.lock_action, GitLockAction::Preserve);
    assert!(plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        None,
        GitAcquisitionIntent::Execute,
        GitAcquisitionPolicy {
            frozen: true,
            ..Default::default()
        },
    )
    .is_err());
    assert!(plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Update,
        GitAcquisitionPolicy {
            frozen: true,
            ..Default::default()
        },
    )
    .is_err());
}

#[test]
fn locked_fetch_can_download_exact_content_but_cannot_change_lock() {
    let locked = locked();
    let plan = plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Fetch,
        GitAcquisitionPolicy {
            locked: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(plan.allow_network);
    assert_eq!(plan.lock_action, GitLockAction::Preserve);
    assert!(plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Update,
        GitAcquisitionPolicy {
            locked: true,
            ..Default::default()
        },
    )
    .is_err());
}

#[test]
fn offline_update_uses_only_cached_mutable_ref_and_replaces_lock() {
    let plan = plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked()),
        GitAcquisitionIntent::Update,
        GitAcquisitionPolicy {
            offline: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(plan.selector, mutable_selector());
    assert!(!plan.allow_network);
    assert_eq!(plan.lock_action, GitLockAction::Replace);
    assert!(plan.expected.is_none());
}

#[test]
fn provider_result_is_bound_to_locked_identity() {
    let locked = locked();
    let plan = plan_git_acquisition(
        repository(),
        mutable_selector(),
        subdir(),
        Some(&locked),
        GitAcquisitionIntent::Execute,
        GitAcquisitionPolicy::default(),
    )
    .unwrap();
    validate_git_acquisition(&plan, &locked).unwrap();
    let mut wrong = locked;
    wrong.tree_digest = ContentDigest::sha256("different");
    assert!(validate_git_acquisition(&plan, &wrong).is_err());
}
