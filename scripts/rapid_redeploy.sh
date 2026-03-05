#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rapid_redeploy.sh --user-id <telegram_user_id> [options]

Options:
  --user-id <id>       Telegram user_id to delete from DynamoDB session table
  --stack-name <name>  CloudFormation stack name (default: delhi-scheme-saathi)
  --region <region>    AWS region (default: ap-south-1)
  --config-env <env>   samconfig environment (default: default)
  --table-name <name>  DynamoDB table name (optional; auto-resolved from stack output)
  --skip-build         Skip `sam build`
  --skip-deploy        Skip `sam deploy`
  --no-clean           Do not remove `.aws-sam` before build
  --no-health          Skip final /health curl check
  -h, --help           Show this help

Environment:
  TG_USER_ID           Optional fallback for --user-id
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

STACK_NAME="delhi-scheme-saathi"
REGION="ap-south-1"
CONFIG_ENV="default"
TABLE_NAME=""
USER_ID="${TG_USER_ID:-}"
SKIP_BUILD=0
SKIP_DEPLOY=0
NO_CLEAN=0
NO_HEALTH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-id)
      USER_ID="${2:-}"
      shift 2
      ;;
    --stack-name)
      STACK_NAME="${2:-}"
      shift 2
      ;;
    --region)
      REGION="${2:-}"
      shift 2
      ;;
    --config-env)
      CONFIG_ENV="${2:-}"
      shift 2
      ;;
    --table-name)
      TABLE_NAME="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --no-clean)
      NO_CLEAN=1
      shift
      ;;
    --no-health)
      NO_HEALTH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${USER_ID}" ]]; then
  echo "Error: --user-id is required (or set TG_USER_ID)." >&2
  usage
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${NO_CLEAN}" -eq 0 ]]; then
  log "Cleaning previous SAM build artifacts"
  rm -rf .aws-sam
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  log "Building SAM application"
  sam build -t sam-template.yaml --use-container
else
  log "Skipping build"
fi

if [[ "${SKIP_DEPLOY}" -eq 0 ]]; then
  log "Deploying stack ${STACK_NAME} (${REGION})"
  sam deploy \
    -t .aws-sam/build/template.yaml \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --config-env "${CONFIG_ENV}" \
    --resolve-s3 \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset
else
  log "Skipping deploy"
fi

if [[ -z "${TABLE_NAME}" ]]; then
  log "Resolving SessionTableName from stack outputs"
  TABLE_NAME="$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='SessionTableName'].OutputValue | [0]" \
    --output text)"
fi

if [[ -z "${TABLE_NAME}" || "${TABLE_NAME}" == "None" ]]; then
  echo "Error: Could not resolve SessionTableName. Pass --table-name explicitly." >&2
  exit 1
fi

KEY_JSON="$(printf '{"user_id":{"S":"%s"}}' "${USER_ID}")"

log "Deleting session for user_id=${USER_ID} from table=${TABLE_NAME}"
aws dynamodb delete-item \
  --table-name "${TABLE_NAME}" \
  --key "${KEY_JSON}" \
  --region "${REGION}"

REMAINING_ITEM="$(aws dynamodb get-item \
  --table-name "${TABLE_NAME}" \
  --key "${KEY_JSON}" \
  --region "${REGION}" \
  --query "Item" \
  --output text)"

if [[ "${REMAINING_ITEM}" == "None" || -z "${REMAINING_ITEM}" ]]; then
  log "Session delete verified"
else
  log "Warning: session row still exists (check key/table)."
fi

if [[ "${NO_HEALTH}" -eq 0 ]]; then
  API_ENDPOINT="$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue | [0]" \
    --output text)"

  if [[ -n "${API_ENDPOINT}" && "${API_ENDPOINT}" != "None" ]]; then
    HEALTH_URL="${API_ENDPOINT%/}/health"
    log "Health check: ${HEALTH_URL}"
    curl -sS "${HEALTH_URL}"
    printf '\n'
  else
    log "Skipping health check (ApiEndpoint not found)"
  fi
fi

log "Rapid redeploy flow completed"
