#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_FILE="sam-template.yaml"
BUILD_TEMPLATE=".aws-sam/build/template.yaml"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rapid_redeploy.sh --user-id <telegram_user_id> [options]

Options:
  --user-id <id>       Telegram user_id to delete from DynamoDB session table
  --stack-name <name>  CloudFormation stack name (default: delhi-scheme-saathi)
  --region <region>    AWS region (default: ap-south-1)
  --profile <name>     AWS profile (default: AWS_PROFILE / AWS_DEFAULT_PROFILE)
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

die() {
  echo "Error: $*" >&2
  exit 1
}

require_arg() {
  local flag="$1"
  local value="${2-}"

  if [[ -z "${value}" || "${value}" == --* ]]; then
    die "${flag} requires a value."
  fi
}

require_command() {
  local cmd="$1"
  local help_text="$2"

  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "Required command '${cmd}' not found. ${help_text}"
  fi
}

check_docker_access() {
  if ! docker info >/dev/null 2>&1; then
    die "Docker daemon is not reachable. Start Docker and ensure this shell can access /var/run/docker.sock."
  fi
}

check_aws_auth() {
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    if [[ -n "${PROFILE}" ]]; then
      die "AWS authentication failed for profile '${PROFILE}'. Run 'aws sso login --profile ${PROFILE} --use-device-code' and retry."
    fi
    die "AWS authentication failed. Export AWS_PROFILE or configure default credentials, then retry."
  fi
}

stack_exists() {
  aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    >/dev/null 2>&1
}

STACK_NAME=""
REGION=""
PROFILE=""
CONFIG_ENV=""
TABLE_NAME=""
USER_ID="${TG_USER_ID:-}"
SKIP_BUILD=0
SKIP_DEPLOY=0
NO_CLEAN=0
NO_HEALTH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-id)
      require_arg "$1" "${2-}"
      USER_ID="${2:-}"
      shift 2
      ;;
    --stack-name)
      require_arg "$1" "${2-}"
      STACK_NAME="${2:-}"
      shift 2
      ;;
    --region)
      require_arg "$1" "${2-}"
      REGION="${2:-}"
      shift 2
      ;;
    --profile)
      require_arg "$1" "${2-}"
      PROFILE="${2:-}"
      shift 2
      ;;
    --config-env)
      require_arg "$1" "${2-}"
      CONFIG_ENV="${2:-}"
      shift 2
      ;;
    --table-name)
      require_arg "$1" "${2-}"
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

STACK_NAME="${STACK_NAME:-delhi-scheme-saathi}"
REGION="${REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-south-1}}}"
PROFILE="${PROFILE:-${AWS_PROFILE:-${AWS_DEFAULT_PROFILE:-}}}"
CONFIG_ENV="${CONFIG_ENV:-default}"

export AWS_REGION="${REGION}"
export AWS_DEFAULT_REGION="${REGION}"
if [[ -n "${PROFILE}" ]]; then
  export AWS_PROFILE="${PROFILE}"
  export AWS_DEFAULT_PROFILE="${PROFILE}"
fi

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
  die "Template file '${TEMPLATE_FILE}' not found in ${PROJECT_ROOT}."
fi
if [[ ! -f "samconfig.toml" ]]; then
  die "samconfig.toml not found. rapid_redeploy expects the repo's SAM config to exist."
fi

if [[ "${SKIP_BUILD}" -eq 0 || "${SKIP_DEPLOY}" -eq 0 ]]; then
  require_command "sam" "Install AWS SAM CLI and ensure it is on PATH."
fi
require_command "aws" "Install AWS CLI and configure credentials or AWS SSO."
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  require_command "docker" "Docker is required for 'sam build --use-container'."
  check_docker_access
fi
if [[ "${NO_HEALTH}" -eq 0 ]]; then
  require_command "curl" "curl is required for the final health check."
fi

check_aws_auth

if [[ "${SKIP_DEPLOY}" -eq 0 && ! stack_exists ]]; then
  die "Stack '${STACK_NAME}' does not exist in region '${REGION}'. rapid_redeploy is intended for redeploying an existing stack; perform the initial SAM deploy separately with the required parameter overrides."
fi

if [[ "${NO_CLEAN}" -eq 0 && "${SKIP_BUILD}" -eq 0 ]]; then
  log "Cleaning previous SAM build artifacts"
  rm -rf .aws-sam
elif [[ "${NO_CLEAN}" -eq 0 ]]; then
  log "Skipping clean because --skip-build requires existing build artifacts"
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  log "Building SAM application"
  sam build -t "${TEMPLATE_FILE}" --use-container
else
  log "Skipping build"
fi

if [[ ! -f "${BUILD_TEMPLATE}" ]]; then
  die "Built template '${BUILD_TEMPLATE}' not found. Re-run without --skip-build."
fi

if [[ "${SKIP_DEPLOY}" -eq 0 ]]; then
  log "Deploying stack ${STACK_NAME} (${REGION})"
  sam deploy \
    -t "${BUILD_TEMPLATE}" \
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
    curl -fsS "${HEALTH_URL}"
    printf '\n'
  else
    log "Skipping health check (ApiEndpoint not found)"
  fi
fi

log "Rapid redeploy flow completed"
