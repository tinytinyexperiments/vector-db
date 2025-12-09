use serde::Serialize;

use crate::index::HnswIndex;

#[derive(Debug, Serialize)]
pub struct HealthReport {
    pub ok: bool,
    pub reason: String,
    pub size: usize,
}

pub fn basic_index_health(index: &HnswIndex) -> HealthReport {
    if index.is_empty() {
        return HealthReport {
            ok: true,
            reason: "index-empty".to_string(),
            size: 0,
        };
    }

    // For now, assume that if we can query dim and len without panicking, it's fine.
    HealthReport {
        ok: true,
        reason: "ok".to_string(),
        size: index.len(),
    }
}


