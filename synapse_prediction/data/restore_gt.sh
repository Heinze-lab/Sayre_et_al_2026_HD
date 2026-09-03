#!/usr/bin/env bash
# Restore the synapse ground-truth graph for NO_cube1 into local MongoDB.
# Training reads the GT (pre/post-synaptic nodes + pairing edges) from MongoDB;
# this loads the small dump committed in ground_truth/ into a local database
# named 'synapses_megalopta_NO1_cube_gb'.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${SYN_DB_HOST:-mongodb://localhost:27017}"

echo ">> Restoring GT db 'synapses_megalopta_NO1_cube_gb' into $HOST ..."
mongorestore --uri="$HOST" --gzip --drop \
    --archive="$HERE/ground_truth/synapses_megalopta_NO1_cube_gb.archive.gz"
echo "Done. (570 nodes / 285 synapse edges)"
