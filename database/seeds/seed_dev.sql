-- =============================================================================
-- database/seeds/seed_dev.sql
-- Sample data for local development and manual QA
-- Run AFTER init.sql:
--   psql -U plp_user -d plp_dev -f database/seeds/seed_dev.sql
-- =============================================================================

-- Safety net: only run in dev / test databases
DO $$
BEGIN
    IF current_database() NOT IN ('plp_dev', 'plp_test') THEN
        RAISE EXCEPTION
            'seed_dev.sql must only run against plp_dev or plp_test, not "%"',
            current_database();
    END IF;
END
$$;

-- Wrap everything in a transaction so a mid-seed failure leaves no partial state
BEGIN;

-- ---------------------------------------------------------------------------
-- Truncate existing seed data (cascade handles FK children)
-- ---------------------------------------------------------------------------
TRUNCATE TABLE
    refresh_tokens,
    vark_profiles,
    responses,
    questions,
    quiz_sessions,
    documents,
    users
CASCADE;

-- ---------------------------------------------------------------------------
-- Users
-- Two learners + one admin-style power user
-- Passwords are bcrypt hashes of the plaintext shown in comments
-- (generated with: python -c "from bcrypt import hashpw, gensalt; print(hashpw(b'password123', gensalt()).decode())")
-- ---------------------------------------------------------------------------
INSERT INTO users (id, email, username, password_hash, is_active, is_verified)
VALUES
    -- password: password123
    ('11111111-1111-1111-1111-111111111111',
     'alice@example.com', 'alice',
     '$2b$12$KIXtL5Uf6z9u3y1v0w2e3OQa4Rfg5Shi6Tjk7Uvl8Wm9Xno0Yop1Z',
     TRUE, TRUE),

    -- password: password123
    ('22222222-2222-2222-2222-222222222222',
     'bob@example.com', 'bob',
     '$2b$12$KIXtL5Uf6z9u3y1v0w2e3OQa4Rfg5Shi6Tjk7Uvl8Wm9Xno0Yop1Z',
     TRUE, TRUE),

    -- password: password123
    ('33333333-3333-3333-3333-333333333333',
     'charlie@example.com', 'charlie',
     '$2b$12$KIXtL5Uf6z9u3y1v0w2e3OQa4Rfg5Shi6Tjk7Uvl8Wm9Xno0Yop1Z',
     TRUE, FALSE);   -- unverified — useful for testing verification flow

-- ---------------------------------------------------------------------------
-- Documents
-- ---------------------------------------------------------------------------
INSERT INTO documents (id, user_id, filename, storage_key, backend, file_size_bytes, status)
VALUES
    ('aaaa0001-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
     '11111111-1111-1111-1111-111111111111',
     'dsa_notes.pdf',
     'uploads/alice/dsa_notes.pdf',
     'minio', 204800, 'indexed'),

    ('aaaa0002-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
     '22222222-2222-2222-2222-222222222222',
     'python_basics.pdf',
     'uploads/bob/python_basics.pdf',
     'minio', 102400, 'indexed'),

    ('aaaa0003-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
     '11111111-1111-1111-1111-111111111111',
     'ml_fundamentals.pdf',
     'uploads/alice/ml_fundamentals.pdf',
     'minio', 512000, 'processing');   -- intentionally still processing

-- ---------------------------------------------------------------------------
-- Quiz sessions
-- ---------------------------------------------------------------------------
INSERT INTO quiz_sessions
    (id, user_id, doc_id, subject, difficulty, question_count,
     question_types, status, score, started_at, completed_at)
VALUES
    -- Session 1: Alice — completed DSA quiz
    ('sess0001-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'aaaa0001-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
     'Data Structures & Algorithms', 'intermediate', 5,
     '{mcq,coding}', 'completed', 80.00,
     NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour'),

    -- Session 2: Alice — theory session, still in progress
    ('sess0002-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     NULL,
     'Machine Learning Basics', 'beginner', 5,
     '{mcq,theory}', 'in_progress', NULL,
     NOW() - INTERVAL '20 minutes', NULL),

    -- Session 3: Bob — completed Python quiz
    ('sess0003-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'aaaa0002-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
     'Python Programming', 'beginner', 5,
     '{mcq}', 'completed', 60.00,
     NOW() - INTERVAL '3 hours', NOW() - INTERVAL '2 hours 30 minutes'),

    -- Session 4: Bob — abandoned session (useful for testing abandoned-state UI)
    ('sess0004-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     NULL,
     'System Design', 'advanced', 10,
     '{mcq,theory}', 'abandoned', NULL,
     NOW() - INTERVAL '1 day', NULL),

    -- Session 5: Charlie — still generating (testing polling)
    ('sess0005-0000-0000-0000-000000000000',
     '33333333-3333-3333-3333-333333333333',
     NULL,
     'Operating Systems', 'intermediate', 5,
     '{mcq}', 'generating', NULL,
     NULL, NULL);

-- ---------------------------------------------------------------------------
-- Questions  (for session 1 and session 3 only — completed sessions)
-- ---------------------------------------------------------------------------

-- Session 1 — DSA (MCQ + Coding)
INSERT INTO questions
    (id, session_id, position, q_type, topic, text,
     options, correct_answer, explanation, difficulty, code_stub, language, test_cases)
VALUES
    -- Q1 MCQ
    ('ques0001-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000', 1, 'mcq',
     'Binary Search', 'What is the time complexity of binary search on a sorted array of n elements?',
     '["A. O(n)", "B. O(log n)", "C. O(n log n)", "D. O(1)"]',
     'B',
     'Binary search halves the search space each step, giving O(log n).',
     'intermediate', NULL, NULL, NULL),

    -- Q2 MCQ
    ('ques0002-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000', 2, 'mcq',
     'Linked Lists', 'Which operation is O(1) in a singly linked list?',
     '["A. Search by value", "B. Insert at tail (no tail pointer)", "C. Insert at head", "D. Delete by value"]',
     'C',
     'Inserting at the head only requires updating the head pointer — constant time.',
     'beginner', NULL, NULL, NULL),

    -- Q3 MCQ
    ('ques0003-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000', 3, 'mcq',
     'Sorting', 'Which sorting algorithm has the best worst-case time complexity?',
     '["A. Quick Sort", "B. Bubble Sort", "C. Merge Sort", "D. Selection Sort"]',
     'C',
     'Merge Sort guarantees O(n log n) in all cases; Quick Sort degrades to O(n²) in the worst case.',
     'intermediate', NULL, NULL, NULL),

    -- Q4 Coding
    ('ques0004-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000', 4, 'coding',
     'Recursion',
     'Write a function that returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).',
     NULL,
     'def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)',
     'A recursive solution is simple; memoisation improves it to O(n).',
     'intermediate',
     'def fib(n):\n    # your code here\n    pass',
     'python',
     '[{"stdin": "0\n", "expected_output": "0"}, {"stdin": "5\n", "expected_output": "5"}, {"stdin": "10\n", "expected_output": "55"}]'),

    -- Q5 Coding
    ('ques0005-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000', 5, 'coding',
     'Arrays',
     'Read N from stdin, then read N integers on the next line. Print their sum.',
     NULL,
     'n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums))',
     'A simple linear scan accumulates the total in O(n).',
     'beginner',
     'n = int(input())\n# read n integers and print their sum',
     'python',
     '[{"stdin": "3\n1 2 3\n", "expected_output": "6"}, {"stdin": "5\n10 20 30 40 50\n", "expected_output": "150"}]');

-- Session 3 — Python MCQ
INSERT INTO questions
    (id, session_id, position, q_type, topic, text,
     options, correct_answer, explanation, difficulty)
VALUES
    ('ques0006-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000', 1, 'mcq',
     'Python Basics', 'What does len([1, 2, 3]) return?',
     '["A. 2", "B. 3", "C. 4", "D. None"]',
     'B', 'len() counts elements; the list has 3.', 'beginner'),

    ('ques0007-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000', 2, 'mcq',
     'Python Basics', 'Which keyword is used to define a function in Python?',
     '["A. func", "B. function", "C. def", "D. lambda"]',
     'C', 'def is the keyword for defining named functions.', 'beginner'),

    ('ques0008-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000', 3, 'mcq',
     'Data Types', 'What is the output of type(3.14)?',
     '["A. <class ''int''>", "B. <class ''str''>", "C. <class ''float''>", "D. <class ''decimal''>"]',
     'C', '3.14 is a floating-point literal; Python identifies it as float.', 'beginner'),

    ('ques0009-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000', 4, 'mcq',
     'Control Flow', 'How do you start an infinite loop in Python?',
     '["A. for(;;)", "B. loop:", "C. while True:", "D. repeat:"]',
     'C', 'while True: is the idiomatic Python infinite loop.', 'beginner'),

    ('ques0010-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000', 5, 'mcq',
     'String Methods', 'What does "hello".upper() return?',
     '["A. hello", "B. Hello", "C. HELLO", "D. None"]',
     'C', '.upper() converts every character to uppercase.', 'beginner');

-- ---------------------------------------------------------------------------
-- Responses  (answers for session 1 — alice scored 80/100)
-- ---------------------------------------------------------------------------
INSERT INTO responses
    (id, session_id, question_id, user_id,
     selected_answer, code_answer, code_output,
     sandbox_passed, is_correct, partial_score, time_taken_ms)
VALUES
    -- Q1 correct
    ('resp0001-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000',
     'ques0001-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'B', NULL, NULL, NULL, TRUE, 100.00, 12000),

    -- Q2 correct
    ('resp0002-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000',
     'ques0002-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'C', NULL, NULL, NULL, TRUE, 100.00, 8000),

    -- Q3 WRONG (chose A — good for testing explain endpoint)
    ('resp0003-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000',
     'ques0003-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'A', NULL, NULL, NULL, FALSE, 0.00, 20000),

    -- Q4 coding correct
    ('resp0004-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000',
     'ques0004-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     NULL,
     'def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)',
     '0\n5\n55',
     TRUE, TRUE, 100.00, 45000),

    -- Q5 coding WRONG (off-by-one error)
    ('resp0005-0000-0000-0000-000000000000',
     'sess0001-0000-0000-0000-000000000000',
     'ques0005-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     NULL,
     'n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums) - 1)',
     '5\n149',
     FALSE, FALSE, 0.00, 60000);

-- Responses for session 3 — bob scored 60/100 (3 correct, 2 wrong)
INSERT INTO responses
    (id, session_id, question_id, user_id,
     selected_answer, is_correct, partial_score, time_taken_ms)
VALUES
    ('resp0006-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000',
     'ques0006-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'B', TRUE, 100.00, 5000),

    ('resp0007-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000',
     'ques0007-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'A', FALSE, 0.00, 9000),     -- chose "func"

    ('resp0008-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000',
     'ques0008-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'C', TRUE, 100.00, 7000),

    ('resp0009-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000',
     'ques0009-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'C', TRUE, 100.00, 6000),

    ('resp0010-0000-0000-0000-000000000000',
     'sess0003-0000-0000-0000-000000000000',
     'ques0010-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'B', FALSE, 0.00, 11000);    -- chose "Hello" instead of "HELLO"

-- ---------------------------------------------------------------------------
-- VARK profiles  (one per completed session)
-- ---------------------------------------------------------------------------
INSERT INTO vark_profiles
    (id, user_id, session_id, visual, auditory, reading, kinesthetic,
     dominant_style, classifier_ver, raw_signals)
VALUES
    -- Alice after session 1: strong kinesthetic (coding questions, fast on code)
    ('vark0001-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'sess0001-0000-0000-0000-000000000000',
     20, 15, 25, 40, 'kinesthetic', 'rule_v1',
     '{"skip_ratio": 0.0, "coding_accuracy": 0.5, "mcq_accuracy": 0.67, "avg_time_ms": 29000}'),

    -- Bob after session 3: reading/visual profile (MCQ, no coding)
    ('vark0002-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'sess0003-0000-0000-0000-000000000000',
     35, 10, 40, 15, 'reading', 'rule_v1',
     '{"skip_ratio": 0.0, "coding_accuracy": null, "mcq_accuracy": 0.6, "avg_time_ms": 7600}');

-- ---------------------------------------------------------------------------
-- Refresh tokens (two valid, one revoked)
-- ---------------------------------------------------------------------------
INSERT INTO refresh_tokens (id, user_id, jti, issued_at, expires_at, revoked)
VALUES
    ('rtok0001-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'alice-jti-0001-dev-seed-placeholder-1',
     NOW(), NOW() + INTERVAL '30 days', FALSE),

    ('rtok0002-0000-0000-0000-000000000000',
     '22222222-2222-2222-2222-222222222222',
     'bob-jti-0002-dev-seed-placeholder-11',
     NOW(), NOW() + INTERVAL '30 days', FALSE),

    -- Revoked token (tests the blocklist flow)
    ('rtok0003-0000-0000-0000-000000000000',
     '11111111-1111-1111-1111-111111111111',
     'alice-jti-0003-dev-seed-revoked-xyzz',
     NOW() - INTERVAL '5 days', NOW() + INTERVAL '25 days', TRUE);

COMMIT;

-- Quick sanity check
SELECT 'users'         AS tbl, COUNT(*) FROM users
UNION ALL
SELECT 'documents',            COUNT(*) FROM documents
UNION ALL
SELECT 'quiz_sessions',        COUNT(*) FROM quiz_sessions
UNION ALL
SELECT 'questions',            COUNT(*) FROM questions
UNION ALL
SELECT 'responses',            COUNT(*) FROM responses
UNION ALL
SELECT 'vark_profiles',        COUNT(*) FROM vark_profiles
UNION ALL
SELECT 'refresh_tokens',       COUNT(*) FROM refresh_tokens;