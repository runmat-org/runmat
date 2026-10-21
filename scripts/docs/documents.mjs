import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { inputFile } from "./manifest.mjs";
export const readJson = (file) => JSON.parse(readFileSync(file, "utf8"));
export function copyInput(source, output, sourcePath, outputPath = sourcePath) {
    const target = join(output, outputPath);
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(inputFile(source, sourcePath), target);
}
export function collectDocuments(source, output, validate) {
    const catalog = readJson(join(source, "docs/catalog.json"));
    validate("document", catalog);
    const ids = new Set();
    for (const document of catalog.documents) {
        if (ids.has(document.id))
            throw new Error(`Duplicate document: ${document.id}`);
        ids.add(document.id);
        copyInput(source, output, document.path, "documents/" + document.path);
    }
    writeFileSync(join(output, "documents.json"), JSON.stringify(catalog, null, 2) + "\n");
}
/** Builtin reference pages come from the one builtin documentation producer
 * (`export_builtin_documentation`), which merges catalog-owned documentation with the
 * remaining legacy sidecars. Copying `docs/builtins/reference` alone would drop every
 * builtin whose documentation has moved into the catalog. */
export function collectBuiltinReferences(documentationPath, output) {
    const exported = readJson(documentationPath);
    if (exported.schema_version !== 2 || !Array.isArray(exported.builtins))
        throw new Error("Unsupported builtin documentation export schema");
    const names = new Set();
    mkdirSync(join(output, "builtins/reference"), { recursive: true });
    for (const document of exported.builtins) {
        const name = document.module_stem;
        if (typeof name !== "string" || !/^[A-Za-z0-9_.]+$/.test(name))
            throw new Error(`Invalid builtin documentation name: ${name}`);
        if (names.has(name.toLowerCase()))
            throw new Error(`Duplicate builtin documentation: ${name}`);
        names.add(name.toLowerCase());
        writeFileSync(join(output, "builtins/reference", name + ".json"), JSON.stringify(document, null, 2) + "\n");
    }
    return names.size;
}
