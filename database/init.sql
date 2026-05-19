-- =============================================================================
-- database/init.sql
-- Full schema for Personalized Learning Path Generator
-- Run once on a fresh PostgreSQL instance (handled by docker-compose on first boot)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";        -- gen_random_uuid(), crypt()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";         -- trigram indexes for text search

-- ---------------------------------------------------------------------------
-- Enum types
-- ---------------------------------------------------------------------------

CREATE TYPE difficulty_level AS ENUM ('beginner', 'intermediate', 'advanced');

CREATE TYPE question_type AS ENUM ('mcq', 'theory', 'coding');

CREATE TYPE session_status AS ENUM (
    'generating',
    'ready',
    'in_progress',
    'grading',
    'completed',
    'abandoned',
    'error'
);

CREATE TYPE vark_style AS ENUM ('visual', 'auditory', 'reading', 'kinesthetic');

CREATE TYPE storage_backend AS ENUM ('s3', 'minio', 'local');

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255)    NOT NULL UNIQUE,
    username        VARCHAR(80)     NOT NULL UNIQUE,
    password_hash   VARCHAR(255)    NOT NULL,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    is_verified     BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email    ON users (email);
CREATE INDEX idx_users_username ON users (username);

COMMENT ON TABLE  users                IS 'Registered learner accounts';
COMMENT ON COLUMN users.password_hash  IS 'bcrypt hash — never store plaintext';

-- ---------------------------------------------------------------------------
-- documents  (uploaded PDFs / reference material)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    filename        VARCHAR(255)    NOT NULL,
    storage_key     VARCHAR(512)    NOT NULL,           -- S3 / MinIO object key
    backend         storage_backend NOT NULL DEFAULT 'minio',
    file_size_bytes BIGINT,
    mime_type       VARCHAR(100)    DEFAULT 'application/pdf',
    status          VARCHAR(40)     NOT NULL DEFAULT 'uploaded',
                                    -- uploaded | processing | indexed | error
    error_message   TEXT,
    faiss_index_id  VARCHAR(255),   -- FAISS/ChromaDB collection identifier
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_user_id ON documents (user_id);
CREATE INDEX idx_documents_status  ON documents (status);

COMMENT ON TABLE documents IS 'Uploaded reference PDFs ingested by the ML pipeline';

-- ---------------------------------------------------------------------------
-- quiz_sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID             NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    doc_id          UUID             REFERENCES documents (id) ON DELETE SET NULL,
    subject         VARCHAR(120)     NOT NULL,
    difficulty      difficulty_level NOT NULL DEFAULT 'intermediate',
    question_count  SMALLINT         NOT NULL DEFAULT 10
                                     CHECK (question_count BETWEEN 3 AND 50),
    question_types  TEXT[]           NOT NULL DEFAULT '{mcq}',
                                     -- stored as Postgres array of question_type values
    status          session_status   NOT NULL DEFAULT 'generating',
    score           NUMERIC(5, 2),   -- 0.00 – 100.00
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id    ON quiz_sessions (user_id);
CREATE INDEX idx_sessions_status     ON quiz_sessions (status);
CREATE INDEX idx_sessions_created_at ON quiz_sessions (created_at DESC);

COMMENT ON TABLE  quiz_sessions         IS 'One row per quiz attempt';
COMMENT ON COLUMN quiz_sessions.score   IS 'Weighted aggregate score 0-100 after grading';

-- ---------------------------------------------------------------------------
-- questions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS questions (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID            NOT NULL REFERENCES quiz_sessions (id) ON DELETE CASCADE,
    position        SMALLINT        NOT NULL DEFAULT 0,   -- ordering within session
    q_type          question_type   NOT NULL DEFAULT 'mcq',
    topic           VARCHAR(120),
    text            TEXT            NOT NULL,
    options         JSONB,          -- ["A. ...", "B. ...", "C. ...", "D. ..."] for MCQ
    correct_answer  TEXT            NOT NULL,
    explanation     TEXT,           -- shown post-submission
    difficulty      difficulty_level NOT NULL DEFAULT 'intermediate',
    code_stub       TEXT,           -- starter code for coding questions
    language        VARCHAR(40),    -- python | javascript | java | cpp
    test_cases      JSONB,          -- [{"stdin": "4\n", "expected_output": "16"}]
    doc_chunk_ref   VARCHAR(255),   -- reference back to the source FAISS chunk
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_questions_session_id ON questions (session_id, position);
CREATE INDEX idx_questions_topic      ON questions (topic);

COMMENT ON TABLE  questions            IS 'Generated questions belonging to a session';
COMMENT ON COLUMN questions.options    IS 'MCQ answer options stored as JSON array';
COMMENT ON COLUMN questions.test_cases IS 'Coding question test cases stored as JSON array';

-- ---------------------------------------------------------------------------
-- responses  (student answers)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS responses (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID            NOT NULL REFERENCES quiz_sessions (id) ON DELETE CASCADE,
    question_id     UUID            NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    user_id         UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    selected_answer TEXT,           -- MCQ letter / theory free text
    code_answer     TEXT,           -- submitted source code
    code_output     TEXT,           -- captured stdout from sandbox
    sandbox_passed  BOOLEAN,        -- NULL if not a coding question
    sandbox_error   TEXT,
    is_correct      BOOLEAN,        -- set by grader after submission
    partial_score   NUMERIC(5, 2),  -- 0-100; for theory cosine-sim grading
    time_taken_ms   INTEGER         NOT NULL DEFAULT 0,
    submitted_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (session_id, question_id)   -- one answer per question per session
);

CREATE INDEX idx_responses_session_id  ON responses (session_id);
CREATE INDEX idx_responses_question_id ON responses (question_id);
CREATE INDEX idx_responses_user_id     ON responses (user_id);
CREATE INDEX idx_responses_is_correct  ON responses (is_correct);

COMMENT ON TABLE  responses               IS 'Student answers for each question';
COMMENT ON COLUMN responses.partial_score IS 'Theory answers graded 0-100 via cosine similarity';

-- ---------------------------------------------------------------------------
-- vark_profiles  (one row per session completion — history preserved)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vark_profiles (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    session_id      UUID        REFERENCES quiz_sessions (id) ON DELETE SET NULL,
    visual          SMALLINT    NOT NULL DEFAULT 25 CHECK (visual      BETWEEN 0 AND 100),
    auditory        SMALLINT    NOT NULL DEFAULT 25 CHECK (auditory    BETWEEN 0 AND 100),
    reading         SMALLINT    NOT NULL DEFAULT 25 CHECK (reading     BETWEEN 0 AND 100),
    kinesthetic     SMALLINT    NOT NULL DEFAULT 25 CHECK (kinesthetic BETWEEN 0 AND 100),
    dominant_style  vark_style  NOT NULL DEFAULT 'reading',
    classifier_ver  VARCHAR(20) NOT NULL DEFAULT 'rule_v1',  -- rule_v1 | xgb_v2 etc.
    raw_signals     JSONB,      -- time_taken, skip_ratio, type_accuracy etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_vark_user_id    ON vark_profiles (user_id);
CREATE INDEX idx_vark_session_id ON vark_profiles (session_id);
CREATE INDEX idx_vark_created_at ON vark_profiles (user_id, created_at DESC);

COMMENT ON TABLE  vark_profiles              IS 'VARK learning-style snapshot per session';
COMMENT ON COLUMN vark_profiles.raw_signals  IS 'Feature vector used by the ML classifier';

-- ---------------------------------------------------------------------------
-- refresh_tokens  (JWT refresh token tracking)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    jti         VARCHAR(36) NOT NULL UNIQUE,   -- JWT ID claim
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX idx_refresh_tokens_jti     ON refresh_tokens (jti);

COMMENT ON TABLE refresh_tokens IS 'Persisted refresh tokens for rotation & revocation';

-- ---------------------------------------------------------------------------
-- updated_at auto-update trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON quiz_sessions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();