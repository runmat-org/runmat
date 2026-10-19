import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
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
    for (const name of readdirSync(join(source, "docs/builtins/reference")).sort()) {
        if (name.endsWith(".json"))
            copyInput(source, output, "docs/builtins/reference/" + name, "builtins/reference/" + name);
    }
}
