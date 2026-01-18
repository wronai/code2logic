// Types for re-export tests

export type Result<T> = { ok: true; value: T } | { ok: false; error: string };

export interface User {
    id: string;
    name: string;
}
