-- Apply with: gcloud spanner databases ddl update DATABASE --instance=INSTANCE --ddl-file=schema.sql
-- (strip these comment lines first if your gcloud version rejects them)

-- LangGraph checkpoints: one row per step of the ReAct loop.
CREATE TABLE checkpoints (
  thread_id            STRING(256) NOT NULL,
  checkpoint_ns        STRING(MAX) NOT NULL,
  checkpoint_id        STRING(64)  NOT NULL,
  parent_checkpoint_id STRING(64),
  checkpoint_type      STRING(64)  NOT NULL,
  checkpoint           BYTES(MAX)  NOT NULL,
  metadata             JSON,
  created_at           TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true),
) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id);

-- Writes made by tasks that have not been folded into a checkpoint yet.
CREATE TABLE checkpoint_writes (
  thread_id     STRING(256) NOT NULL,
  checkpoint_ns STRING(MAX) NOT NULL,
  checkpoint_id STRING(64)  NOT NULL,
  task_id       STRING(64)  NOT NULL,
  idx           INT64       NOT NULL,
  task_path     STRING(MAX) NOT NULL,
  channel       STRING(MAX) NOT NULL,
  value_type    STRING(64)  NOT NULL,
  value         BYTES(MAX),
) PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx);

-- LangGraph store items (long-term memory). Namespaces are stored dot-joined.
CREATE TABLE store_items (
  namespace  STRING(MAX) NOT NULL,
  item_key   STRING(MAX) NOT NULL,
  value      JSON        NOT NULL,
  created_at TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true),
  updated_at TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true),
) PRIMARY KEY (namespace, item_key);

-- The final answer of every run.
CREATE TABLE agent_results (
  run_id      STRING(64)  NOT NULL,
  thread_id   STRING(256) NOT NULL,
  agent       STRING(64)  NOT NULL,
  customer_id STRING(256) NOT NULL,
  status      STRING(16)  NOT NULL,
  answer      STRING(MAX) NOT NULL,
  error       STRING(MAX),
  created_at  TIMESTAMP   NOT NULL,
) PRIMARY KEY (run_id);

CREATE INDEX agent_results_by_thread ON agent_results (thread_id, created_at DESC);
