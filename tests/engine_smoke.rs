use std::path::PathBuf;

use self_healing_vector_db::{EngineConfig, SelfHealingVectorDb};
use tempfile::tempdir;

fn make_flat_vectors(dim: usize) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
    // v1: [1, 0, 0, ..., 0]
    let mut v1 = vec![0.0_f32; dim];
    v1[0] = 1.0;

    // v2: [0, 1, 0, ..., 0]
    let mut v2 = vec![0.0_f32; dim];
    v2[1] = 1.0;

    let ids = vec![1_i64, 2_i64];
    let mut flat = Vec::with_capacity(dim * 2);
    flat.extend_from_slice(&v1);
    flat.extend_from_slice(&v2);

    (ids, flat, v1)
}

#[test]
fn basic_add_search_and_self_heal() {
    let dim = 384;
    let tmp_dir = tempdir().expect("tempdir");
    let db_path: PathBuf = tmp_dir.path().join("vectors.sqlite");

    let cfg = EngineConfig {
        dim,
        storage_path: db_path.clone(),
        hnsw_max_elements: 10_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let (ids, flat, query_vec) = make_flat_vectors(dim);

    // First engine: add and search.
    let mut engine =
        SelfHealingVectorDb::new(cfg.clone(), None).expect("engine created");
    engine
        .add_vectors(&ids, &flat)
        .expect("add_vectors should succeed");

    let results = engine
        .search(&query_vec, 2)
        .expect("search should succeed");
    assert!(!results.is_empty(), "expected at least one neighbor");
    assert_eq!(results[0].id, 1, "nearest neighbor should be id 1");

    let health = engine.health();
    assert_eq!(health.size, 2, "health should report 2 vectors");

    // Drop engine to simulate a crash / restart.
    drop(engine);

    // New engine bootstraps from the same SQLite file (self-healing).
    let engine2 =
        SelfHealingVectorDb::new(cfg, None).expect("engine recreated");
    let health2 = engine2.health();
    assert_eq!(
        health2.size, 2,
        "recreated engine should see the same 2 vectors"
    );

    let results2 = engine2
        .search(&query_vec, 2)
        .expect("search after restart should succeed");
    assert!(!results2.is_empty(), "expected at least one neighbor after restart");
    assert_eq!(results2[0].id, 1, "nearest neighbor after restart should still be id 1");
}


