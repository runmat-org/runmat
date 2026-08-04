# runmat-execution-artifact

Canonical, filesystem-independent execution bundles and exact program artifacts.
The crate consumes an already frozen package graph; it does not resolve packages,
schedule work, execute programs, authorize tenants, or expose physical paths.

Schema-v2 bundles carry the existing frozen-project handoff with its storage paths rebound to canonical logical source objects. A native or browser worker validates the graph, source catalog, program revision, object digests, and callable inventory before rebasing that same handoff onto a private host-owned materialization. Workers never resolve package locators, fetch Git/project/registry sources, or receive package credentials.
