# Sourcery AI Code Review - Response & Instructions

**Review Date:** 2025-12-20  
**Tool:** [Sourcery.ai](https://sourcery.ai/) (Automated code quality analysis)

## How to Interact with Sourcery

### Configuration
- Sourcery runs automatically on all PRs (if enabled in repo)
- Comments are left directly on the PR
- To respond: reply in PR comments or address issues in new commits

### This Review's Findings

Sourcery found **3 issues** in the REPL PR. Below is our response to each.

---

## Issue 1: Plot Window Threading Bug ❌ (Not Our Code)

**Location:** `crates/runmat-plot/src/gui/single_window_manager.rs:33-42`

**Issue:** `WINDOW_ACTIVE` never reset if thread spawn fails

**Status:** **NOT IN OUR PR** — This appears to be from a different commit in the branch history. Verify this is not part of the REPL changes.

**Action:** 
- [ ] Check if this is unrelated code
- [ ] If related: Fix by resetting `WINDOW_ACTIVE` on spawn failure
- [ ] If unrelated: Note in PR comments that this is pre-existing

---

## Issue 2: Ctrl+C Behavior Inconsistency ⚠️ (Cross-crate, Acknowledged)

**Location:** Comparison between `runmat` and `runmat-repl` REPLs

**Issue:** 
- `runmat` REPL (main binary): Ctrl+C → **exit process** (`break`)
- `runmat-repl` (crate): Ctrl+C → **return to prompt** (intended per spec)

**Finding:** Behavior should be consistent and documented

**Our Response:**
This is **intentional in the crate**, but we acknowledge the main `runmat` binary may differ. 

**Action Plan:**
- [x] Document in spec that `runmat-repl` crate treats Ctrl+C as non-exiting interrupt
- [x] Add comment in code explaining the design choice
- [ ] **TODO for future PR:** Align main `runmat` binary behavior with spec (separate PR, out of scope)
- [ ] Add note to `docs/repl-spec.md` section 4 (Deviations)

**Code Fix:**
Update `crates/runmat-repl/src/main.rs` line ~108:
```rust
Err(ReadlineError::Interrupted) => {
    // Ctrl+C: return to prompt (non-exiting interrupt)
    // This differs from main runmat binary which exits
    // See docs/repl-spec.md for design rationale
    continue;
}
```

---

## Issue 3: Command Parser Doesn't Handle Paths with Spaces ✅ (Real Bug, Easy Fix)

**Location:** `crates/runmat-repl/src/commands.rs` — `parse_and_execute()` function

**Issue:**
```rust
// Current (broken):
let parts: Vec<&str> = trimmed.split_whitespace().collect();
// ...
cmd_cd(if parts.len() > 1 { parts[1] } else { "." })
// Problem: "cd My Folder" becomes ["cd", "My", "Folder"]
//          only parts[1]="My" is used, ignores "Folder"
```

**Fix:**
```rust
// Better:
let mut parts = trimmed.split_whitespace();
let cmd = parts.next();
let rest: String = parts.collect::<Vec<_>>().join(" ");

match cmd {
    Some("cd") => cmd_cd(if rest.is_empty() { "." } else { &rest }),
    Some("dir") | Some("ls") => cmd_dir_ls(if rest.is_empty() { "." } else { &rest }),
    // etc.
}
```

**Action:** 
- [ ] Implement the fix above
- [ ] Add test case: `cmd_cd_with_spaces` → `cd My Folder`
- [ ] Test on Windows (backslash paths)

---

## Issue 4: Typo in Documentation ✅ (Minor, Easy Fix)

**Location:** `docs/REPL_PROGRESS.md` line 279

**Issue:** "buslines" should be "rustyline"

**Current:**
```markdown
3. Tab completion (requires buslines library)
```

**Fix:**
```markdown
3. Tab completion (via [rustyline](https://crates.io/crates/rustyline) completion API)
```

---

## Response Checklist

- [ ] Issue 1: Verify plot window code is not in scope, document in PR comment
- [ ] Issue 2: Add code comment explaining Ctrl+C non-exiting design
- [ ] Issue 3: Fix command parser to handle spaces in paths
- [ ] Issue 3: Add test for `cd "My Folder"`
- [ ] Issue 4: Fix typo in REPL_PROGRESS.md
- [ ] Push fix commits to same PR branch
- [ ] Reply to Sourcery with "Fixed in commits [hash]"

---

## How to Reply to Sourcery

In the PR, click **Reply** on each Sourcery comment:

```
✅ Fixed in commit [hash]: [brief description]

Example:
✅ Fixed in commit 7a2c3b4: commands.rs now joins remaining tokens for paths with spaces
```

Or reply once with:
```
Thank you for the review! I've addressed issues 2, 3, and 4 in follow-up commits. 
Issue 1 (plot window) appears to be pre-existing and outside this PR's scope—will verify.
```

---

## For Future PRs

**To improve Sourcery reviews:**
1. Run `cargo clippy` locally before pushing (catches many issues early)
2. Add docstrings to new public functions
3. Keep commits small and focused (easier for review tools to parse)
4. Link issues in commit messages (e.g., `Fixes issue 2: cmd parser...`)

---

*Last updated: 2025-12-20*
