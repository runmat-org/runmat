import type {
  BrowserPrivatePackageKeyProvider,
  PrivatePackageEnvelopeContext
} from "./private-artifact.js";

const DATABASE_NAME = "runmat-private-package-keys";
const DATABASE_VERSION = 1;
const STORE_NAME = "recipient-keys";

export interface GeneratedBrowserRecipientKey {
  privateKey: CryptoKey;
  publicKey: string;
  fingerprint: string;
}

interface StoredRecipientKey {
  id: string;
  registry: string;
  recipientKeyId: string;
  privateKey: CryptoKey;
  createdAt: number;
}

export interface BrowserPrivatePackageKeyStore {
  put(registry: string, recipientKeyId: string, privateKey: CryptoKey): Promise<void>;
  get(registry: string, recipientKeyId: string): Promise<CryptoKey | null>;
  remove(registry: string, recipientKeyId: string): Promise<void>;
  provider(): BrowserPrivatePackageKeyProvider;
  close(): void;
}

export interface BrowserRecipientKeyRegistration {
  id: string;
  fingerprint: string;
}

export type RegisterBrowserRecipientKey = (
  publicKey: string
) => Promise<BrowserRecipientKeyRegistration>;

export type RevokeBrowserRecipientKey = (recipientKeyId: string) => Promise<void>;

export async function generateBrowserRecipientKey(): Promise<GeneratedBrowserRecipientKey> {
  const pair = (await subtle().generateKey(
    { name: "ECDH", namedCurve: "P-256" },
    false,
    ["deriveBits"]
  )) as CryptoKeyPair;
  validatePrivateKey(pair.privateKey);
  const publicBytes = new Uint8Array(await subtle().exportKey("raw", pair.publicKey));
  try {
    return {
      privateKey: pair.privateKey,
      publicKey: base64Url(publicBytes),
      fingerprint: await sha256(publicBytes)
    };
  } finally {
    publicBytes.fill(0);
  }
}

export async function createBrowserPrivatePackageKeyStore(
  databaseName = DATABASE_NAME
): Promise<BrowserPrivatePackageKeyStore> {
  if (!globalThis.indexedDB) {
    throw new Error("IndexedDB is required for browser private package keys");
  }
  const database = await openDatabase(databaseName);
  const get = async (registry: string, recipientKeyId: string): Promise<CryptoKey | null> => {
    validateRegistry(registry);
    validateRecipientKeyId(recipientKeyId);
    const value = await request<StoredRecipientKey | undefined>(
      database
        .transaction(STORE_NAME, "readonly")
        .objectStore(STORE_NAME)
        .get(storageId(registry, recipientKeyId))
    );
    if (!value) return null;
    validatePrivateKey(value.privateKey);
    return value.privateKey;
  };
  return {
    async put(registry, recipientKeyId, privateKey) {
      validateRegistry(registry);
      validateRecipientKeyId(recipientKeyId);
      validatePrivateKey(privateKey);
      await request(
        database
          .transaction(STORE_NAME, "readwrite")
          .objectStore(STORE_NAME)
          .put({
            id: storageId(registry, recipientKeyId),
            registry,
            recipientKeyId,
            privateKey,
            createdAt: Date.now()
          } satisfies StoredRecipientKey)
      );
    },
    get,
    async remove(registry, recipientKeyId) {
      validateRegistry(registry);
      validateRecipientKeyId(recipientKeyId);
      await request(
        database
          .transaction(STORE_NAME, "readwrite")
          .objectStore(STORE_NAME)
          .delete(storageId(registry, recipientKeyId))
      );
    },
    provider() {
      return {
        privateKeyForEnvelope: (context: PrivatePackageEnvelopeContext) =>
          get(context.registry, context.recipientKeyId)
      };
    },
    close() {
      database.close();
    }
  };
}

export async function createAndRegisterBrowserRecipientKey(
  store: BrowserPrivatePackageKeyStore,
  registry: string,
  register: RegisterBrowserRecipientKey,
  revoke: RevokeBrowserRecipientKey
): Promise<BrowserRecipientKeyRegistration> {
  validateRegistry(registry);
  const generated = await generateBrowserRecipientKey();
  const registration = await register(generated.publicKey);
  validateRecipientKeyId(registration.id);
  if (registration.fingerprint !== generated.fingerprint) {
    await bestEffortRevoke(revoke, registration.id);
    throw new Error("recipient key fingerprint returned by the registry does not match");
  }
  try {
    await store.put(registry, registration.id, generated.privateKey);
  } catch (cause) {
    await bestEffortRevoke(revoke, registration.id);
    throw cause;
  }
  return registration;
}

export async function revokeAndRemoveBrowserRecipientKey(
  store: BrowserPrivatePackageKeyStore,
  registry: string,
  recipientKeyId: string,
  revoke: RevokeBrowserRecipientKey
): Promise<void> {
  validateRegistry(registry);
  validateRecipientKeyId(recipientKeyId);
  await revoke(recipientKeyId);
  await store.remove(registry, recipientKeyId);
}

function openDatabase(name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(name, DATABASE_VERSION);
    open.onupgradeneeded = () => {
      const database = open.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    open.onerror = () => reject(open.error ?? new Error("private key database open failed"));
    open.onblocked = () => reject(new Error("private key database upgrade is blocked"));
    open.onsuccess = () => resolve(open.result);
  });
}

function request<T = IDBValidKey>(value: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    value.onsuccess = () => resolve(value.result);
    value.onerror = () =>
      reject(value.error ?? new Error("private key database request failed"));
  });
}

function storageId(registry: string, recipientKeyId: string): string {
  return `${new URL(registry).origin}\u0000${recipientKeyId}`;
}

function validateRegistry(value: string): void {
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("private package key registry must be a credential-free HTTPS URL");
  }
}

function validateRecipientKeyId(value: string): void {
  if (!value || value.length > 128) {
    throw new Error("private package recipient key ID is invalid");
  }
}

function validatePrivateKey(value: CryptoKey): void {
  const algorithm = value.algorithm as EcKeyAlgorithm;
  if (
    value.type !== "private" ||
    value.extractable ||
    algorithm.name !== "ECDH" ||
    algorithm.namedCurve !== "P-256" ||
    !value.usages.includes("deriveBits")
  ) {
    throw new Error(
      "private package keys must be non-extractable P-256 ECDH keys with deriveBits usage"
    );
  }
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await subtle().digest("SHA-256", bytes.slice().buffer));
  return `sha256:${Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function base64Url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

function subtle(): SubtleCrypto {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto is required for browser private package keys");
  }
  return globalThis.crypto.subtle;
}

async function bestEffortRevoke(
  revoke: RevokeBrowserRecipientKey,
  recipientKeyId: string
): Promise<void> {
  try {
    await revoke(recipientKeyId);
  } catch {
    // Preserve the primary registration/storage integrity error.
  }
}
