use std::path::PathBuf;

use self_healing_vector_db::{EngineConfig, SelfHealingVectorDb};
use tempfile::tempdir;

#[test]
fn health_on_fresh_engine_reports_empty() {
    let dim = 384;
    let tmp_dir = tempdir().expect("tempdir");
    let db_path: PathBuf = tmp_dir.path().join("vectors.sqlite");

    let cfg = EngineConfig {
        dim,
        storage_path: db_path,
        hnsw_max_elements: 10_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let engine =
        SelfHealingVectorDb::new(cfg, None).expect("engine created");

    let health = engine.health();
    assert!(health.ok, "fresh engine should be healthy");
    assert_eq!(health.reason, "index-empty");
    assert_eq!(health.size, 0);
}

#[test]
fn search_on_empty_engine_returns_no_results() {
    let dim = 384;
    let tmp_dir = tempdir().expect("tempdir");
    let db_path: PathBuf = tmp_dir.path().join("vectors.sqlite");

    let cfg = EngineConfig {
        dim,
        storage_path: db_path,
        hnsw_max_elements: 10_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let engine =
        SelfHealingVectorDb::new(cfg, None).expect("engine created");

    let query = vec![0.0_f32; dim];
    let results = engine.search(&query, 10).expect("search should not error");
    assert!(
        results.is_empty(),
        "search on an empty index should yield no results"
    );
}


