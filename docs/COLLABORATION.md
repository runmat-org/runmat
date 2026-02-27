# Collaboration & Teams

RunMat lets you work with your team on shared projects. When someone saves a file, everyone else sees the update in real time. No shared drives, no emailing attachments, no "which version is current?" — one project, one source of truth, live for everyone who has access.

This page covers how organizations and projects work, how to invite teammates, what roles and permissions mean, and how real-time sync keeps everyone in sync. It also touches on API keys for scripts and CI, and on enterprise single sign-on for larger teams.

---

## Organizations and projects

### The mental model

- **Organizations** are your team boundary. Billing, members, and policy live at the org level. You can belong to multiple orgs (e.g. your company and a partner org).
- **Projects** live inside an org. Each project is a shared workspace: its own file tree, version history, and snapshot chain. When you collaborate, you're collaborating inside a project.

```
Organization (Acme Corp)
├── Org members ─────── owner, admin, member, viewer
│
├── Project: Research
│   └── Project members ── alice (write), bob (read)
│
├── Project: Shared Models
│   └── Project members ── alice (write), carol (write), dave (read)
│
└── Project: Published Results
    └── Project members ── everyone (read)
```

You add people to the org, then grant them access to specific projects. That way you can have a "Research" project that only some people see, and a "Shared Models" project that the whole team uses.

### Switching context

From the CLI you choose which org and project you're talking to:

```sh
runmat login
runmat org list
runmat project list --org <org-id>
runmat project select <project-id>
```

The desktop app and API use the same model: you pick an org, then a project. Your last-used org and project are remembered so you don't have to reselect every time.

---

## Inviting teammates

### Add people to your org

You can add people in two ways:

1. **Invite by email** — You send an invite to one or more email addresses with a role (e.g. member). If they already have a RunMat account, they're added to the org immediately. If not, they get a link to sign up and join. Invites respect your org's seat limit; if you're at capacity, you'll need to change your plan before adding more people.
2. **Add an existing user** — If you know their RunMat user id (e.g. from another org), you can add them directly with a role. Same permission needed: only org owners and admins can add or remove members.

### Add people to a project

Being in the org doesn't automatically give access to every project. Org owners and admins (and project members with the right to manage the project) grant **project-level access**:

- **Read** — Can open files, view history, and see the project. No edits.
- **Write** — Can read and edit files, create snapshots, and run code against the project.

So you might add your whole team to the org as members, then give only the core modeling team write access to the "Sensitivity analysis" project, and give everyone else read-only access to "Published results."

### Where to do it

From the RunMat Cloud UI you can manage org members and project members in the relevant settings screens. From the API, membership and invite endpoints let you automate onboarding or sync with your own directory.

---

## Roles and permissions

### Organization roles

| Role     | What they can do |
|----------|------------------|
| **Owner** | Billing, SSO/SCIM setup, org policy, and everything admins can do. There is always at least one owner. |
| **Admin** | Add and remove org members, create and delete projects, assign project roles. Cannot change billing or SSO. |
| **Member** | Use the org and any project they've been given access to. Default role for new invites. |
| **Viewer** | Read-only at the org level (e.g. view project list and usage). Use project roles to control file access. |

### Project roles

| Role    | What they can do |
|---------|------------------|
| **Read**  | Open and read files, view version history and snapshots. No edits. |
| **Write** | Everything in Read, plus edit files, create/restore snapshots, and run code (e.g. LLM, remote run) in the project. |

Project access is explicit: your org role does not by itself give you access to any project. An admin must add you to each project with a read or write role.

### Quick reference

| Area                      | Owner | Admin | Member | Viewer |
|---------------------------|-------|-------|--------|--------|
| Billing & org settings    | Yes   | No    | No     | No     |
| SSO / SCIM                | Yes   | No    | No     | No     |
| Manage org members        | Yes   | Yes   | No     | No     |
| Create / delete projects  | Yes   | Yes   | No     | No     |
| Assign project access     | Yes   | Yes   | No     | No     |
| Use a project (read/write) | Yes*  | Yes*  | Yes*   | Yes*   |

\*Only if they have been granted a project role (read or write) on that project.

---

## Real-time sync

When a teammate saves a file or creates a snapshot, you don't have to refresh or re-open the project. RunMat pushes those changes to you as they happen.

### How it works

- The server sends **events** over a single long-lived connection (Server-Sent Events). File changes, membership changes, and usage updates all flow over this stream.
- Your client (desktop app or a custom integration) keeps a **cursor** so that if you disconnect and reconnect, it can replay only what you missed. No need to re-download the whole project.
- Updates are **coalesced** over a short window so that a burst of edits doesn't make the UI jitter. You see a consistent, up-to-date view.

So: one person saves `main.m`, and everyone else with the project open sees the new version appear. No polling, no "sync" button. Same for the file tree, history, and project membership — they stay in sync automatically.

### How this compares

| | Traditional (shared drive, email, or manual sync) | RunMat |
|---|--------------------------------------------------|--------|
| **Seeing others' changes** | Refresh, re-open, or re-download | Automatic; changes stream in |
| **Conflict handling** | "Who has the latest?" / overwrites | One version per path; history shows who changed what |
| **Offline** | Often no collaboration until back online | Reconnect and replay from cursor; no merge conflicts |
| **Audit** | File timestamps or separate versioning | Every change has actor, timestamp, and content hash |

Real-time sync applies to the **project view** (files, history, snapshots) and to the **org view** (project list, member list, usage). So whether you're looking at a single project or at the org dashboard, you're seeing live data.

---

## Service accounts and API keys

For scripts, CI pipelines, and headless workloads, you don't want to log in as a human every time. RunMat supports **service accounts** and **API keys** scoped to an org (and optionally to a single project).

- **Service account** — A synthetic user that represents a bot or a pipeline. You create it in the org, then attach one or more API keys to it.
- **API key** — A long-lived secret (e.g. `rk.prod.abc123.<secret>`) that authenticates as that service account. Keys can have an expiration date and can be revoked immediately.

Use API keys when you run `runmat remote run` from CI, when you call the RunMat API from your own code, or when you need a fixed identity for automation. Restrict keys to a single project when the job only needs access to one project — that way a leaked key can't touch the rest of the org.

For full CLI and env var details, see the [CLI Reference](/docs/cli).

---

## Enterprise: SSO and provisioning

Larger teams often want to use their existing identity provider (IdP) and to manage users and groups in one place.

### Single sign-on (SSO)

RunMat supports **SAML** and **OIDC**. An org owner (or IT) configures the connection: your IdP's metadata, domains, and optional group claims. RunMat gives you a few verification steps (e.g. a DNS TXT record or a well-known URL) to prove you control the domain. Once that's done, users with that domain log in through your IdP instead of a separate RunMat password. No duplicate accounts.

### User and group provisioning (SCIM)

When SSO is enabled, RunMat can expose **SCIM 2.0** endpoints so your IdP (or a provisioning tool) can create, update, and deactivate users, and sync group membership. Groups can be mapped to org roles (e.g. "RunMat Admins" → org admin). That way when someone is added to the admins group in your directory, they become an admin in RunMat automatically. Membership changes are reflected in real time for anyone with the org or project open.

If your organization uses SSO and SCIM, your IT or security team will typically own the setup; you just log in with your usual credentials and see the projects you've been given access to.

---

## CLI workflows at a glance

Authenticate and list orgs and projects:

```sh
runmat login
runmat org list
runmat project list --org <org-id>
runmat project select <project-id>
```

Work with the remote filesystem (requires a selected project and appropriate role):

```sh
runmat remote run /script.m
runmat fs ls /data
runmat fs read /data/example.mat --output example.mat
runmat fs write /data/example.mat ./example.mat
```

For version history and snapshots in a project, see the [Filesystem](/docs/filesystem) and [Versioning & History](/docs/versioning) docs. For environment variables and API key usage in scripts, see the [CLI Reference](/docs/cli).

---

## Summary

- **Orgs** are your team; **projects** are shared workspaces inside an org. You add people to the org, then grant them read or write access per project.
- **Invites** can be by email (with sign-up links for new users) or by adding an existing user. Seat limits apply.
- **Roles** at the org level (owner, admin, member, viewer) control who manages the org and its projects; **project roles** (read, write) control who can see and edit each project's files.
- **Real-time sync** streams file and membership changes to everyone with the project or org open. Reconnect replays from a cursor; no manual refresh.
- **API keys** and **service accounts** support CI and headless use; **SSO and SCIM** let enterprises use their IdP and directory for login and provisioning.
