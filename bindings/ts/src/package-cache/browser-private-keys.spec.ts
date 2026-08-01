import { afterEach, describe, expect, it } from "vitest";

import {
  createAndRegisterBrowserRecipientKey,
  createBrowserPrivatePackageKeyStore,
  generateBrowserRecipientKey,
  revokeAndRemoveBrowserRecipientKey
} from "./browser-private-keys.js";

const databases: string[] = [];

afterEach(async () => {
  await Promise.all(
    databases.splice(0).map(
      (name) =>
        new Promise<void>((resolve) => {
          const deletion = indexedDB.deleteDatabase(name);
          deletion.onsuccess = () => resolve();
          deletion.onerror = () => resolve();
          deletion.onblocked = () => resolve();
        })
    )
  );
});

describe("browser private package key store", () => {
  it("persists only a non-extractable CryptoKey and supplies it by envelope ID", async () => {
    const database = `runmat-private-keys-${crypto.randomUUID()}`;
    databases.push(database);
    const generated = await generateBrowserRecipientKey();
    expect(generated.privateKey.extractable).toBe(false);
    expect(generated.publicKey).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(generated.fingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);

    const store = await createBrowserPrivatePackageKeyStore(database);
    await store.put(
      "https://packages.runmat.test",
      "pkr_test",
      generated.privateKey
    );
    const loaded = await store.provider().privateKeyForEnvelope({
      registry: "https://packages.runmat.test",
      namespace: "acme",
      name: "private",
      version: "1.0.0",
      recipientKeyId: "pkr_test",
      recipientKeyFingerprint: generated.fingerprint
    });
    expect(loaded).toBeInstanceOf(CryptoKey);
    expect(loaded?.extractable).toBe(false);

    await store.remove("https://packages.runmat.test", "pkr_test");
    expect(
      await store.get("https://packages.runmat.test", "pkr_test")
    ).toBeNull();
    store.close();
  });

  it("registers atomically and removes the local key only after server revocation", async () => {
    const database = `runmat-private-keys-${crypto.randomUUID()}`;
    databases.push(database);
    const store = await createBrowserPrivatePackageKeyStore(database);
    const revoked: string[] = [];
    const registration = await createAndRegisterBrowserRecipientKey(
      store,
      "https://packages.runmat.test",
      async (publicKey) => {
        const bytes = Uint8Array.from(
          atob(publicKey.replaceAll("-", "+").replaceAll("_", "/")),
          (character) => character.charCodeAt(0)
        );
        const digest = new Uint8Array(
          await crypto.subtle.digest("SHA-256", bytes)
        );
        return {
          id: "pkr_registered",
          fingerprint: `sha256:${Array.from(digest, (byte) =>
            byte.toString(16).padStart(2, "0")
          ).join("")}`
        };
      },
      async (id) => {
        revoked.push(id);
      }
    );
    expect(registration.id).toBe("pkr_registered");
    expect(
      await store.get("https://packages.runmat.test", registration.id)
    ).not.toBeNull();

    await revokeAndRemoveBrowserRecipientKey(
      store,
      "https://packages.runmat.test",
      registration.id,
      async (id) => {
        revoked.push(id);
      }
    );
    expect(revoked).toEqual(["pkr_registered"]);
    expect(
      await store.get("https://packages.runmat.test", registration.id)
    ).toBeNull();
    store.close();
  });

  it("revokes a mismatched registration without retaining private material", async () => {
    const database = `runmat-private-keys-${crypto.randomUUID()}`;
    databases.push(database);
    const store = await createBrowserPrivatePackageKeyStore(database);
    const revoked: string[] = [];
    await expect(
      createAndRegisterBrowserRecipientKey(
        store,
        "https://packages.runmat.test",
        async () => ({
          id: "pkr_mismatch",
          fingerprint: `sha256:${"0".repeat(64)}`
        }),
        async (id) => {
          revoked.push(id);
        }
      )
    ).rejects.toThrow("does not match");
    expect(revoked).toEqual(["pkr_mismatch"]);
    expect(
      await store.get("https://packages.runmat.test", "pkr_mismatch")
    ).toBeNull();
    store.close();
  });
});
