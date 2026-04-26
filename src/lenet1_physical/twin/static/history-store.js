export function createHistoryStore() {
    const cache = new Map();

    async function refreshList() {
        const res = await fetch("/history");
        if (!res.ok) throw new Error(`GET /history failed: ${res.status}`);
        const list = await res.json();
        return list;
    }

    async function getRecord(id) {
        if (cache.has(id)) return cache.get(id);
        const res = await fetch(`/history/${id}`);
        if (!res.ok) throw new Error(`GET /history/${id} failed: ${res.status}`);
        const record = await res.json();
        cache.set(id, record);
        return record;
    }

    return { refreshList, getRecord };
}
