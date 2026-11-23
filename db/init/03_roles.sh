#!/usr/bin/env bash
# Creates the least-privilege, read-only role the application connects with.
# Executed by the official postgres image on first start (after 01/02 SQL files).
# Note: the postgres entrypoint may *source* this file (if it is not executable),
# so avoid `set -u` and `exit`, which would leak into the entrypoint shell.
set -eo pipefail

: "${READONLY_USER:=text2sql_reader}"
: "${READONLY_PASSWORD:?READONLY_PASSWORD must be set}"
: "${QUERY_TIMEOUT_SECONDS:=15}"

# psql variables (:"name" for identifiers, :'name' for literals) quote the values
# safely, so passwords with special characters cannot break or inject into the SQL.
psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v ro_user="$READONLY_USER" \
     -v ro_pass="$READONLY_PASSWORD" \
     -v db_name="$POSTGRES_DB" \
     -v stmt_timeout="${QUERY_TIMEOUT_SECONDS}s" <<'EOSQL'

CREATE ROLE :"ro_user" WITH LOGIN PASSWORD :'ro_pass'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 20;

-- Server-side guard rails that apply even if the app is misconfigured.
ALTER ROLE :"ro_user" SET default_transaction_read_only = on;
ALTER ROLE :"ro_user" SET statement_timeout = :'stmt_timeout';
ALTER ROLE :"ro_user" SET idle_in_transaction_session_timeout = '60s';
ALTER ROLE :"ro_user" SET lock_timeout = '5s';

-- Nobody but the owner can create objects in public.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE :"db_name" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"db_name" TO :"ro_user";

-- SELECT-only access to the analytics schema.
GRANT USAGE ON SCHEMA public TO :"ro_user";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"ro_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO :"ro_user";

-- No access at all to the sensitive schema.
REVOKE ALL ON SCHEMA internal FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA internal FROM PUBLIC;
EOSQL

echo "Read-only role '$READONLY_USER' created."
