/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import posthog from 'posthog-js'

const POSTHOG_KEY = process.env.NEXT_PUBLIC_POSTHOG_KEY || '';
const POSTHOG_HOST = process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com';
const GA_MEASUREMENT_ID = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID || '';

if (POSTHOG_KEY) {
    posthog.init(POSTHOG_KEY, {
        api_host: POSTHOG_HOST,
        autocapture: true,
        capture_pageview: true,
        capture_pageleave: true,
    });
}

function parseGaClientIdFromCookie(): string | undefined {
    if (typeof document === 'undefined') return undefined;
    const match = document.cookie.match(/(?:^|; )_ga=([^;]+)/);
    if (!match) return undefined;
    try {
        const raw = decodeURIComponent(match[1]);
        const parts = raw.split('.');
        if (parts.length >= 4) {
            return `${parts[parts.length - 2]}.${parts[parts.length - 1]}`;
        }
    } catch {
        // ignore
    }
    return undefined;
}

function getGaClientIdFromGtag(): Promise<string | undefined> {
    return new Promise((resolve) => {
        if (typeof window === 'undefined' || !GA_MEASUREMENT_ID) {
            resolve(undefined);
            return;
        }
        try {
            if (typeof (window as any).gtag === 'function') {
                (window as any).gtag('get', GA_MEASUREMENT_ID, 'client_id', (clientId: string | undefined) => {
                    resolve(clientId || undefined);
                });
            } else {
                resolve(undefined);
            }
        } catch {
            resolve(undefined);
        }
    });
}

if (typeof window !== 'undefined') {
    let completed = false;
    const tryAlias = (gaClientId?: string) => {
        if (completed) return;
        if (!gaClientId) return;
        const currentDistinctId = posthog.get_distinct_id();
        if (currentDistinctId !== gaClientId) {
            posthog.alias(gaClientId, currentDistinctId);
            posthog.identify(gaClientId);
        } else {
            posthog.identify(gaClientId);
        }
        posthog.register({ ga_client_id: gaClientId });
        completed = true;
    };

    const cookieId = parseGaClientIdFromCookie();
    if (cookieId) {
        tryAlias(cookieId);
    }

    getGaClientIdFromGtag().then((cid) => tryAlias(cid)).catch(() => undefined);

    const onGaClientId = (e: Event) => {
        tryAlias((e as CustomEvent<string>).detail);
    };

    window.addEventListener('ga_client_id', onGaClientId, { once: true });
}