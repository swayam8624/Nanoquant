#!/usr/bin/env bash
set -euo pipefail

root="$(pwd)"

say() { printf "\n==> %s\n" "$*"; }
move_dir() { # move_dir <src> <dst_dir>
  local src="$1"; local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    say "Moving dir  $src  ->  $dst/"
    mv "$src" "$dst/"
  else
    say "Skip (no dir): $src"
  fi
}
move_file() { # move_file <src> <dst_dir>
  local src="$1"; local dst="$2"
  if [ -f "$src" ]; then
    mkdir -p "$dst"
    say "Moving file $src  ->  $dst/"
    mv "$src" "$dst/"
  else
    say "Skip (no file): $src"
  fi
}

# Ensure target roots
mkdir -p public-repo private_repo

# If you accidentally created 'private-repo', fold it into 'private_repo'
if [ -d "private-repo" ]; then
  say "Consolidating private-repo/ -> private_repo/"
  rsync -a "private-repo/" "private_repo/"
  rm -rf "private-repo"
fi

say "Public: move clean, non-sensitive code & docs"
# Already present in your tree; put them under public-repo/
move_dir "nanoquant_public"        "public-repo"
move_dir "vibe-quant-studio-public" "public-repo"

# Public docs & guides
move_file "README.md"                "public-repo"
move_file "CLOUD_DEPLOYMENT_GUIDE.md" "public-repo"
move_file "REPOSITORY_STRUCTURE.md"  "public-repo"

# Public documentation & charts if they contain no secrets
move_dir "docs"    "public-repo"
move_dir "charts"  "public-repo"

# Public static assets (safe images/icons). Skip if you know it has private logos or keys.
move_dir "assets"  "public-repo"

# Keep Makefile public (useful for building OSS)
move_file "Makefile" "public-repo"

say "Private: move internal engines, builds, artifacts, and scripts"
# Internal cores & packages
move_dir "nanoquant-core"         "private_repo"
move_dir "nanoquants"             "private_repo"
move_dir "nanoquant.egg-info"     "private_repo"

# Build & dist artifacts
move_dir "build"                  "private_repo"
move_dir "build_simple"           "private_repo"
move_dir "dist"                   "private_repo"
move_dir "dist_simple"            "private_repo"
move_file "NanoQuant.spec"        "private_repo"

# Local backups & housekeeping
move_dir "backup"                 "private_repo"
move_file "cleanup.sh"            "private_repo"
move_file "push_repositories.sh"  "private_repo"

# If you later recreate any model/test dirs, keep them private:
for d in \
  real_test_compressed_models test_compressed_models test_compression_fix_output \
  tuned_models temp_models local_models test_results tests test_local_model test_nanoquants
do
  move_dir "$d" "private_repo"
done

say "Summary:"
printf "\n-- public-repo (top-level) --\n"
ls -la "public-repo" || true
printf "\n-- private_repo (top-level) --\n"
ls -la "private_repo" || true

say "Done. Review above lists before committing."
