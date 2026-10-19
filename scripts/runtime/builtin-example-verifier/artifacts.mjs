// @ts-check

import { lstatSync, readFileSync, readdirSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { compareUtf8, digestObject, sha256 } from "./identity.mjs";
import { requiredArtifactRoles } from "./product-contracts.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";

export const ARTIFACT_SCHEMA = "runmat.builtin-examples.artifacts.v2";
export const ARTIFACT_LAYOUTS = Object.freeze(["recursive-tree", "exact-files"]);

export function buildArtifactManifest(fields) {
    const manifest = {
        schema: ARTIFACT_SCHEMA,
        product: fields.product,
        artifactProfile: fields.artifactProfile,
        sourceRevision: fields.sourceRevision,
        producerRevision: fields.producerRevision ?? null,
        layout: structuredClone(fields.layout),
        entrypoints: [...fields.entrypoints].sort((left, right) => compareUtf8(left.role, right.role)),
        files: [...fields.files].sort((left, right) => compareUtf8(left.path, right.path)),
        artifactManifestDigest: ""
    };
    manifest.artifactManifestDigest = digestObject(manifest, ["artifactManifestDigest"]);
    return manifest;
}

// Exact-file manifests are limited to profiles whose package contract names every
// relevant file. Native packages use the recursive-tree producer so adjacent
// runtime libraries contribute to the same product identity.
export function buildArtifactManifestFromFiles(fields) {
    if (fields.product !== "browser-wasm" || fields.artifactProfile !== "web") {
        throw new Error("Exact-file artifact production is only valid for browser-wasm/web");
    }
    const manifestPath = resolve(fields.manifestPath);
    const base = dirname(manifestPath);
    const entrypoints = [];
    const files = [];
    for (const [role, source] of Object.entries(fields.files)) {
        const absolute = resolve(source);
        const path = canonicalRelativePath(relative(base, absolute).split(sep).join("/"), `artifact path for ${role}`);
        resolveRelativeArtifactPath(base, path);
        const stat = lstatSync(absolute);
        if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Artifact source for ${role} is not a regular file`);
        entrypoints.push({ role, path });
        files.push(fileRecord(path, absolute, stat));
    }
    const manifest = buildArtifactManifest({
        product: fields.product,
        artifactProfile: fields.artifactProfile,
        sourceRevision: fields.sourceRevision,
        producerRevision: fields.producerRevision ?? null,
        layout: { kind: "exact-files" },
        entrypoints,
        files
    });
    validateArtifactManifest(manifest, { sourceRevision: fields.sourceRevision, verifyFiles: true, manifestPath });
    return manifest;
}

export function buildArtifactManifestFromTree(fields) {
    const manifestPath = resolve(fields.manifestPath);
    const base = dirname(manifestPath);
    const root = resolve(fields.root);
    const relativeRoot = relative(base, root).split(sep).join("/");
    const rootPath = canonicalRootPath(relativeRoot || ".");
    resolveRelativeRoot(base, rootPath);
    const files = collectTreeFiles(root, manifestPath);
    const entrypoints = Object.entries(fields.entrypoints).map(([role, path]) => ({
        role,
        path: canonicalRelativePath(String(path).split(sep).join("/"), `artifact entrypoint path for ${role}`)
    }));
    const manifest = buildArtifactManifest({
        product: fields.product,
        artifactProfile: fields.artifactProfile,
        sourceRevision: fields.sourceRevision,
        producerRevision: fields.producerRevision ?? null,
        layout: { kind: "recursive-tree", root: rootPath },
        entrypoints,
        files
    });
    validateArtifactManifest(manifest, { sourceRevision: fields.sourceRevision, verifyFiles: true, manifestPath });
    return manifest;
}

export function validateArtifactManifest(manifest, options = {}) {
    if (!manifest || manifest.schema !== ARTIFACT_SCHEMA) throw new Error("Unsupported builtin example artifact manifest schema");
    exactKeys(manifest, ["schema", "product", "artifactProfile", "sourceRevision", "producerRevision", "layout", "entrypoints", "files", "artifactManifestDigest"], "artifact manifest");
    enumValue(manifest.product, ["native-cli", "browser-wasm", "desktop-native"], "artifact product");
    gitRevision(manifest.sourceRevision, "artifact source revision");
    if (manifest.producerRevision !== null) gitRevision(manifest.producerRevision, "artifact producer revision");
    if ((manifest.product === "desktop-native") !== (manifest.producerRevision !== null)) {
        throw new Error("Only desktop-native artifacts require a producer revision");
    }
    digest(manifest.artifactManifestDigest, "artifact manifest digest");
    if (digestObject(manifest, ["artifactManifestDigest"]) !== manifest.artifactManifestDigest) throw new Error("Artifact manifest digest mismatch");
    validateLayout(manifest);
    if (!Array.isArray(manifest.entrypoints) || !Array.isArray(manifest.files)) throw new Error("Artifact manifest entrypoints and files must be arrays");

    const roles = new Set();
    const entrypointPaths = new Set();
    for (const entrypoint of manifest.entrypoints) {
        exactKeys(entrypoint, ["role", "path"], `artifact entrypoint ${entrypoint?.role ?? "unknown"}`);
        if (typeof entrypoint.role !== "string" || !entrypoint.role || roles.has(entrypoint.role)) throw new Error(`Invalid or duplicate artifact role: ${entrypoint?.role}`);
        roles.add(entrypoint.role);
        const path = canonicalRelativePath(entrypoint.path, `artifact entrypoint path for ${entrypoint.role}`);
        if (entrypointPaths.has(path)) throw new Error(`Duplicate artifact entrypoint path: ${path}`);
        entrypointPaths.add(path);
    }
    const orderedEntrypoints = [...manifest.entrypoints].sort((left, right) => compareUtf8(left.role, right.role));
    if (JSON.stringify(orderedEntrypoints) !== JSON.stringify(manifest.entrypoints)) throw new Error("Artifact entrypoints are not in canonical role order");

    const filePaths = new Set();
    for (const file of manifest.files) {
        exactKeys(file, ["path", "size", "sha256"], `artifact file ${file?.path ?? "unknown"}`);
        const path = canonicalRelativePath(file.path, "artifact file path");
        if (filePaths.has(path)) throw new Error(`Duplicate artifact file path: ${path}`);
        filePaths.add(path);
        integer(file.size, `artifact size for ${path}`);
        digest(file.sha256, `artifact digest for ${path}`);
    }
    const orderedFiles = [...manifest.files].sort((left, right) => compareUtf8(left.path, right.path));
    if (JSON.stringify(orderedFiles) !== JSON.stringify(manifest.files)) throw new Error("Artifact files are not in canonical path order");
    for (const path of entrypointPaths) if (!filePaths.has(path)) throw new Error(`Artifact entrypoint is absent from its file closure: ${path}`);

    const requiredRoles = requiredArtifactRoles(manifest.product, manifest.artifactProfile);
    for (const role of roles) if (!requiredRoles.includes(role)) throw new Error(`Artifact role ${role} is outside the ${manifest.product}/${manifest.artifactProfile} contract`);
    if (JSON.stringify([...roles].sort(compareUtf8)) !== JSON.stringify([...requiredRoles].sort(compareUtf8))) throw new Error(`Artifact manifest does not exactly implement ${manifest.product}/${manifest.artifactProfile}`);
    if (manifest.layout.kind === "exact-files" && filePaths.size !== entrypointPaths.size) throw new Error("Exact-file artifact manifests cannot contain untyped package files");
    if (options.sourceRevision && manifest.sourceRevision !== options.sourceRevision) throw new Error("Artifact source revision does not match the execution plan");
    if (options.producerRevision && manifest.producerRevision !== options.producerRevision) throw new Error("Artifact producer revision does not match the expected Desktop source");
    if (options.verifyFiles) verifyArtifactFiles(manifest, options.manifestPath);
    return manifest;
}

export function verifyArtifactFiles(manifest, manifestPath) {
    if (typeof manifestPath !== "string" || !manifestPath.trim()) throw new Error("Artifact file verification requires the manifest path");
    const base = dirname(resolve(manifestPath));
    const root = manifest.layout.kind === "recursive-tree" ? resolveRelativeRoot(base, manifest.layout.root) : base;
    if (manifest.layout.kind === "recursive-tree") {
        const actual = collectTreeFiles(root, resolve(manifestPath));
        if (JSON.stringify(actual) !== JSON.stringify(manifest.files)) throw new Error("Artifact recursive tree does not match its manifest");
    } else {
        for (const file of manifest.files) verifyFile(resolveRelativeArtifactPath(base, file.path), file);
    }
}

export function resolveArtifactPath(manifest, manifestPath, role) {
    const entrypoint = manifest.entrypoints.find((candidate) => candidate.role === role);
    if (!entrypoint) throw new Error(`Artifact role is absent: ${role}`);
    const base = manifestPath ? dirname(resolve(manifestPath)) : process.cwd();
    const root = manifest.layout.kind === "recursive-tree" ? resolveRelativeRoot(base, manifest.layout.root) : base;
    return resolveRelativeArtifactPath(root, entrypoint.path);
}

function validateLayout(manifest) {
    if (!manifest.layout || typeof manifest.layout !== "object" || Array.isArray(manifest.layout)) throw new Error("Artifact layout must be an object");
    enumValue(manifest.layout.kind, ARTIFACT_LAYOUTS, "artifact layout kind");
    if (manifest.layout.kind === "recursive-tree") {
        exactKeys(manifest.layout, ["kind", "root"], "recursive artifact layout");
        canonicalRootPath(manifest.layout.root);
        const recursiveProfile = (manifest.product === "native-cli" && manifest.artifactProfile === "embedded-aot")
            || (manifest.product === "desktop-native" && manifest.artifactProfile === "desktop-host");
        if (!recursiveProfile) throw new Error("Recursive-tree artifact layout is invalid for this product profile");
    } else {
        exactKeys(manifest.layout, ["kind"], "exact-file artifact layout");
        if (manifest.product !== "browser-wasm" || manifest.artifactProfile !== "web") throw new Error("Exact-file artifact layout is only valid for browser-wasm/web");
    }
}

function collectTreeFiles(root, excludedPath = null) {
    const rootStat = lstatSync(root);
    if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) throw new Error("Artifact product root must be a real directory");
    const files = [];
    const visit = (directory, prefix) => {
        for (const entry of readdirSync(directory, { withFileTypes: true })) {
            const path = prefix ? `${prefix}/${entry.name}` : entry.name;
            const absolute = resolve(directory, entry.name);
            if (excludedPath && absolute === excludedPath) continue;
            if (entry.isSymbolicLink()) throw new Error(`Artifact tree contains a symbolic link: ${path}`);
            if (entry.isDirectory()) visit(absolute, path);
            else if (entry.isFile()) files.push(fileRecord(canonicalRelativePath(path, "artifact file path"), absolute));
            else throw new Error(`Artifact tree contains a non-regular entry: ${path}`);
        }
    };
    visit(root, "");
    return files.sort((left, right) => compareUtf8(left.path, right.path));
}

function canonicalRootPath(path) {
    if (path === ".") return path;
    return canonicalRelativePath(path, "artifact product root");
}

function resolveRelativeRoot(base, path) {
    if (path === ".") return resolve(base);
    return resolveRelativeArtifactPath(base, path);
}

function fileRecord(path, absolute, knownStat = null) {
    const stat = knownStat ?? lstatSync(absolute);
    return { path, size: stat.size, sha256: sha256(readFileSync(absolute)) };
}

function verifyFile(path, expected) {
    const stat = lstatSync(path);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== expected.size) throw new Error(`Artifact size mismatch for ${expected.path}`);
    if (sha256(readFileSync(path)) !== expected.sha256) throw new Error(`Artifact digest mismatch for ${expected.path}`);
}

function canonicalRelativePath(path, label) {
    if (typeof path !== "string" || !path || path !== path.trim() || isAbsolute(path) || path.includes("\\")) throw new Error(`${label} must be a canonical relative path`);
    const segments = path.split("/");
    if (segments.some((segment) => !segment || segment === "." || segment === "..")) throw new Error(`${label} must be a canonical relative path`);
    return path;
}

function resolveRelativeArtifactPath(base, path) {
    canonicalRelativePath(path, "artifact path");
    const resolved = resolve(base, path);
    if (resolved !== base && !resolved.startsWith(`${base}${sep}`)) throw new Error(`Artifact path escapes its manifest directory: ${path}`);
    return resolved;
}
