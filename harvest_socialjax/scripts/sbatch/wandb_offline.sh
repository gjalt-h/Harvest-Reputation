#!/bin/bash
# W&B helpers for Snellius sbatch jobs.
#
# Default behavior (when WANDB_API_KEY is set): upgrade WANDB_MODE offline->online
# so runs stream to wandb.ai within minutes. Offline + periodic `wandb sync` is
# kept as a fallback (WANDB_FORCE_OFFLINE=1) but is unreliable mid-run on Snellius
# (sync often prints "... done" without creating the cloud run).
#
# Usage in sbatch (after export WANDB_DIR=...):
#   source "${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"
#   wandb_offline_job_start "$WANDB_DIR"    # prefer online; else trap + sync
#   python train.py ...
#   wandb_offline_job_finish "$WANDB_DIR"   # stop loop / final sync if offline

_WANDB_PERIODIC_SYNC_PID=""
_WANDB_PERIODIC_SYNC_DIR=""

# Set WANDB_ENTITY to your W&B username/team before submitting jobs.
# WANDB_PROJECT defaults to the reputation project; override per experiment.
: "${WANDB_PROJECT:=harvest_ippo_rep}"

wandb_prepare_sync_env() {
    # Cluster nodes often set TMPDIR=/scratch-local/... which breaks wandb-core
    # port-file startup (ServicePollForTokenError). Prefer a durable home cache
    # for the sync process only.
    if [ -z "${WANDB_ENTITY:-}" ]; then
        echo "[wandb] WARNING: WANDB_ENTITY is unset; set it to your W&B user/team" >&2
    else
        export WANDB_ENTITY
    fi
    export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
    export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
    local tmp="${HOME}/.cache/wandb_tmp"
    mkdir -p "$tmp" 2>/dev/null || tmp="/tmp"
    case "${TMPDIR:-}" in
        /scratch-local/*|/scratch-node/*|"")
            export TMPDIR="$tmp"
            ;;
    esac
    export TMPDIR="${TMPDIR:-$tmp}"
    mkdir -p "$TMPDIR" 2>/dev/null || true
}

wandb_find_offline_runs() {
    local wandb_dir="${1:?WANDB_DIR required}"
    [ -d "$wandb_dir" ] || return 0
    # Prefer nested wandb/offline-run-* (wandb.init(dir=WANDB_DIR)); also accept
    # a bare offline-run-* directly under WANDB_DIR.
    find "$wandb_dir" -type d \( -name 'offline-run-*' -o -name 'run-*' \) 2>/dev/null \
        | grep -E '/(offline-)?run-' \
        | sort
}

wandb_sync_offline_runs() {
    local wandb_dir="${1:?WANDB_DIR required}"
    local run_dir
    local any=0
    local ok=0
    local sync_log
    # Continues: append into the existing cloud run instead of overwriting history.
    local -a sync_extra=()
    if [ -n "${WANDB_RESUME_ID:-}" ]; then
        sync_extra=(--id "${WANDB_RESUME_ID}" --append)
    fi
    if ! command -v wandb >/dev/null 2>&1; then
        echo "[wandb] WARNING: wandb CLI not on PATH; cannot sync $wandb_dir" >&2
        return 1
    fi
    wandb_prepare_sync_env
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "[wandb] WARNING: WANDB_API_KEY unset; sync will fail for $wandb_dir" >&2
    fi
    while IFS= read -r run_dir; do
        [ -z "$run_dir" ] && continue
        # Skip completely empty dirs (pre-init).
        if [ -z "$(ls -A "$run_dir" 2>/dev/null)" ]; then
            continue
        fi
        any=1
        echo "[wandb] syncing $run_dir → ${WANDB_ENTITY}/${WANDB_PROJECT}${WANDB_RESUME_ID:+ (append id=$WANDB_RESUME_ID)} ..."
        echo "[wandb]   TMPDIR=$TMPDIR WANDB__SERVICE_WAIT=$WANDB__SERVICE_WAIT"
        sync_log=$(mktemp "${TMPDIR}/wandb_sync_XXXXXX.log" 2>/dev/null || mktemp)
        # Explicit entity/project — empty entity previously printed wandb.ai//project
        # and syncs often never landed in harvest_ippo_rep.
        # --include-synced: prior failed/partial syncs mark local dirs "synced"
        # and a bare `wandb sync` then prints "... done" without creating the run.
        if wandb sync \
            --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" \
            --include-synced \
            "${sync_extra[@]}" \
            "$run_dir" >"$sync_log" 2>&1
        then
            # wandb sync can exit 0 even when the service thread fails or the
            # cloud run was never created; treat those as failure.
            if grep -qiE 'ServicePollForTokenError|Failed to read port|Traceback|Could not find run|run .* not found|ERROR Error while calling' "$sync_log"; then
                echo "[wandb] WARNING: sync reported success but log has errors for $run_dir:" >&2
                tail -n 40 "$sync_log" >&2 || true
                rm -f "$sync_log"
            elif ! grep -qE "https://wandb\.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/" "$sync_log"; then
                echo "[wandb] WARNING: sync OK but no ${WANDB_ENTITY}/${WANDB_PROJECT} URL in log for $run_dir:" >&2
                tail -n 40 "$sync_log" >&2 || true
                rm -f "$sync_log"
            else
                ok=1
                grep -E 'https://wandb\.ai/' "$sync_log" | tail -n 3 || true
                echo "[wandb] synced $run_dir → https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/$(basename "$run_dir" | sed -E 's/^(offline-)?run-[0-9_]+-//')"
                rm -f "$sync_log"
            fi
        else
            echo "[wandb] WARNING: sync failed for $run_dir (exit=$?)" >&2
            tail -n 40 "$sync_log" >&2 || true
            rm -f "$sync_log"
        fi
    done < <(wandb_find_offline_runs "$wandb_dir")
    if [ "$any" -eq 0 ]; then
        echo "[wandb] no offline-run-* / run-* under $wandb_dir yet"
        return 0
    fi
    if [ "$ok" -eq 0 ]; then
        return 1
    fi
    return 0
}

wandb_latest_run_id() {
    local wandb_dir="${1:?WANDB_DIR required}"
    local run_dir
    run_dir=$(wandb_find_offline_runs "$wandb_dir" | tail -1)
    if [ -n "$run_dir" ]; then
        basename "$run_dir" | sed -E 's/^(offline-)?run-[0-9_]+-//'
    fi
}

_wandb_on_job_exit() {
    wandb_offline_job_finish "${_WANDB_PERIODIC_SYNC_DIR:-}"
}

wandb_prefer_online_when_keyed() {
    # Offline + `wandb sync` on Snellius often prints "... done" without creating
    # the cloud run (especially mid-training). When an API key is available,
    # stream online so runs appear on wandb.ai within minutes.
    # Override: WANDB_FORCE_OFFLINE=1
    if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_FORCE_OFFLINE:-0}" != "1" ]; then
        if [ "${WANDB_MODE:-offline}" = "offline" ]; then
            echo "[wandb] WANDB_MODE offline->online (API key present; set WANDB_FORCE_OFFLINE=1 to keep offline sync)"
            export WANDB_MODE=online
        fi
    fi
}

wandb_offline_job_start() {
    local wandb_dir="${1:?WANDB_DIR required}"
    local interval_sec="${2:-${WANDB_SYNC_INTERVAL:-300}}"
    # First sync sooner so runs appear on wandb.ai during long JIT / early train.
    local first_delay_sec="${3:-${WANDB_SYNC_FIRST_DELAY:-60}}"
    _WANDB_PERIODIC_SYNC_DIR="$wandb_dir"
    mkdir -p "$wandb_dir"
    wandb_prepare_sync_env
    wandb_prefer_online_when_keyed

    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "[wandb] WARNING: WANDB_API_KEY is empty — offline sync cannot upload" >&2
        echo "[wandb]   export WANDB_API_KEY=... in ~/.bashrc or the sbatch env" >&2
    else
        echo "[wandb] WANDB_API_KEY is set (len=${#WANDB_API_KEY})"
    fi
    echo "[wandb] mode=${WANDB_MODE:-?} entity=${WANDB_ENTITY} project=${WANDB_PROJECT} TMPDIR=${TMPDIR}"
    if ! command -v wandb >/dev/null 2>&1; then
        echo "[wandb] WARNING: wandb CLI missing on PATH after conda activate" >&2
    else
        echo "[wandb] wandb CLI: $(command -v wandb)"
    fi

    if [ "${WANDB_MODE:-offline}" != "offline" ]; then
        echo "[wandb] online/streaming mode — no offline sync loop (runs appear live on wandb.ai)"
        return 0
    fi

    trap _wandb_on_job_exit EXIT TERM INT
    (
        # Inherit key / project / durable TMPDIR for the background syncer.
        export WANDB_API_KEY="${WANDB_API_KEY:-}"
        export WANDB_ENTITY="${WANDB_ENTITY:-}"
        export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
        export WANDB_RESUME_ID="${WANDB_RESUME_ID:-}"
        export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
        export TMPDIR="${TMPDIR:-}"
        sleep "$first_delay_sec"
        wandb_sync_offline_runs "$wandb_dir" || true
        while true; do
            sleep "$interval_sec"
            wandb_sync_offline_runs "$wandb_dir" || true
        done
    ) &
    _WANDB_PERIODIC_SYNC_PID=$!
    echo "[wandb] offline mode: first sync in ${first_delay_sec}s, then every ${interval_sec}s (pid=$_WANDB_PERIODIC_SYNC_PID)"
    echo "[wandb] offline dir=$wandb_dir (nested wandb/offline-run-*)"
}

wandb_offline_job_finish() {
    local wandb_dir="${1:-${_WANDB_PERIODIC_SYNC_DIR:-}}"
    trap - EXIT TERM INT
    if [ "${WANDB_MODE:-offline}" != "offline" ]; then
        echo "[wandb] mode=${WANDB_MODE} — skip final offline sync"
        return 0
    fi
    if [ -n "${_WANDB_PERIODIC_SYNC_PID:-}" ] && kill -0 "$_WANDB_PERIODIC_SYNC_PID" 2>/dev/null; then
        kill "$_WANDB_PERIODIC_SYNC_PID" 2>/dev/null || true
        wait "$_WANDB_PERIODIC_SYNC_PID" 2>/dev/null || true
    fi
    _WANDB_PERIODIC_SYNC_PID=""
    if [ -n "$wandb_dir" ]; then
        wandb_sync_offline_runs "$wandb_dir" || true
        WANDB_RUN_ID=$(wandb_latest_run_id "$wandb_dir")
        if [ -n "${WANDB_RUN_ID:-}" ]; then
            export WANDB_RUN_ID
            echo "WANDB_RUN_ID=$WANDB_RUN_ID"
            echo "[wandb] view: https://wandb.ai/${WANDB_ENTITY:-ENTITY}/${WANDB_PROJECT:-harvest_ippo_rep}/runs/${WANDB_RUN_ID}"
        fi
    fi
}
