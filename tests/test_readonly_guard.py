import pytest

from core.utils.readonly_guard import UnsafeQueryError, check_read_only

ALLOWED = [
    "SELECT name, update_date FROM employees WHERE dept = 'Sales'",
    "select count(*) from orders;",
    "WITH t AS (SELECT id FROM a) SELECT * FROM t",
    "SELECT 'DROP TABLE users' AS note FROM dual",  # keyword inside a string literal
    "SELECT `delete` FROM audit_log",  # keyword as a quoted identifier
    "SELECT id FROM a -- delete later\n WHERE x = 1",
    "SHOW TABLES",
    "DESCRIBE employees",
    "EXPLAIN SELECT * FROM orders",
]

REFUSED = [
    "DROP TABLE employees",
    "UPDATE employees SET salary = 0",
    "DELETE FROM orders",
    "INSERT INTO a VALUES (1)",
    "SELECT * FROM a; DROP TABLE a",
    "WITH d AS (DELETE FROM a RETURNING *) SELECT * FROM d",
    "SELECT * FROM users INTO OUTFILE '/tmp/x'",
    "SELECT 1 /*!50000 , (SELECT 1) */",
    "SET GLOBAL max_connections = 1",
    "GRANT ALL ON *.* TO 'x'",
    "TRUNCATE orders",
    "CALL purge_all()",
    "EXPLAIN ANALYZE DELETE FROM orders",
    "",
    "   ",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_read_only_queries_pass(sql):
    assert check_read_only(sql) == sql


@pytest.mark.parametrize("sql", REFUSED)
def test_writes_and_tricks_are_refused(sql):
    with pytest.raises(UnsafeQueryError):
        check_read_only(sql)
