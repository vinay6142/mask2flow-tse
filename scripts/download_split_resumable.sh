#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --time=12:00:00
#SBATCH --job-name=dl_split
#SBATCH --output=scripts/logs/dl_split_%j.out

# Mask2Flow-TSE -- RESUMABLE single-split LibriSpeech download.
#
# WHY this exists alongside scripts/download_librispeech_full.sh: that script
# goes through torchaudio.datasets.LIBRISPEECH, which fetches via
# torch.hub.download_url_to_file -- a temp file with NO resume support. On this
# cluster the openslr link runs at 0.2-0.8 MB/s, so train-clean-360 (21.5GB)
# takes 7-15h and does not reliably fit a 12h job. Job 11219 lost 7.5GB to a
# "Connection reset by peer" and job 11223 was still at 42% after 4h42m; in both
# cases a restart begins again from zero.
#
# wget -c resumes a partial file, so every resubmission makes progress and a
# timeout costs only the time, never the bytes. Same approach that let
# scripts/download_wham.sh survive a mid-job kill.
#
# Submit:  sbatch scripts/download_split_resumable.sh train-clean-360
#          sbatch scripts/download_split_resumable.sh            # defaults below

set -eo pipefail

SPLIT="${1:-train-clean-360}"
RAW_DIR="data/raw"
LS_DIR="$RAW_DIR/LibriSpeech"
URL="https://www.openslr.org/resources/12/${SPLIT}.tar.gz"
TARBALL="$RAW_DIR/${SPLIT}.tar.gz"

# Published LibriSpeech utterance counts. Existence alone is NOT a completeness
# test: job 11224 ran out of disk mid-extract and left a partial train-clean-360
# behind, which every existence check in this project (download.py,
# librispeech.py:184, and the earlier version of this script) would have read as
# "done" -- training silently on a fraction of the split.
expected_flac() {
    case "$1" in
        train-clean-100) echo 28539 ;;
        train-clean-360) echo 104014 ;;
        train-other-500) echo 148688 ;;
        dev-clean)       echo 2703 ;;
        dev-other)       echo 2864 ;;
        test-clean)      echo 2620 ;;
        test-other)      echo 2939 ;;
        *)               echo 0 ;;
    esac
}
WANT_FLAC=$(expected_flac "$SPLIT")

if [ -d "$LS_DIR/$SPLIT" ]; then
    n=$(find "$LS_DIR/$SPLIT" -name '*.flac' 2>/dev/null | wc -l)
    if [ "$WANT_FLAC" -gt 0 ] && [ "$n" -lt "$WANT_FLAC" ]; then
        echo "ERROR: $LS_DIR/$SPLIT exists but is INCOMPLETE: $n flac, expected $WANT_FLAC."
        echo "Almost certainly a half-finished extraction (job 11224 hit ENOSPC)."
        echo "Delete it and resubmit -- the verified tarball is reused, so this"
        echo "costs only extraction time, not another download:"
        echo "    rm -rf $LS_DIR/$SPLIT"
        exit 1
    fi
    echo "=== $LS_DIR/$SPLIT already complete ($n flac). Nothing to do. ==="
    exit 0
fi

echo "=== Disk before ==="
df -h . || true
echo ""
echo "=== Target: $SPLIT ==="
echo "  url     : $URL"
echo "  tarball : $TARBALL"
if [ -f "$TARBALL" ]; then
    echo "  resuming from $(du -h "$TARBALL" | cut -f1) already on disk"
else
    echo "  starting fresh"
fi
echo ""

mkdir -p "$RAW_DIR"

# -c resume, --tries=0 retry forever, --read-timeout so a stalled socket is
# dropped and retried instead of hanging until the job's time limit, and
# --waitretry to back off rather than hammer openslr.
echo "=== Downloading (resumable) ==="
# NOTE: no `-O`. With -O, wget -c cannot resume correctly (it treats the output
# as a stream to rewrite); the default naming plus -P is what makes -c work.
# --tries=0 retries forever, --read-timeout drops a stalled socket instead of
# hanging until the job's time limit, --waitretry backs off rather than
# hammering openslr. If the job is killed mid-transfer, the partial file stays
# and the next submission continues from that byte offset.
set +e
wget -c --tries=0 --read-timeout=30 --waitretry=10 --progress=dot:giga \
     -P "$RAW_DIR" "$URL"
WGET_RC=$?
set -e

if [ "$WGET_RC" -ne 0 ]; then
    echo ""
    echo "wget exited $WGET_RC -- transfer INCOMPLETE (likely the job's time"
    echo "limit, or the link dropping as it did in jobs 11219/11223)."
    echo "The partial file is kept at $TARBALL; resubmit this script and it"
    echo "resumes from where it stopped. Nothing was extracted."
    du -h "$TARBALL" 2>/dev/null | sed 's/^/  have: /'
    exit 1
fi

echo ""
echo "=== Download finished. Verifying the archive before extracting ==="
# A truncated tarball extracts partially and then the split dir EXISTS, which
# every downstream existence check reads as "done" -- the trap that silently
# broke Libri2Mix generation in job 10817. Verify BEFORE extracting.
#
# Checksums are torchaudio's own (torchaudio/datasets/librispeech.py). Job
# 11223 is how we have 360's: torchaudio rejected its truncated download with
# `invalid hash value (expected 146a5649..., got 0c2f3a09...)`, which is the
# authoritative value for the complete archive.
expected_sha() {
    case "$1" in
        train-clean-360) echo "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf" ;;
        *)               echo "" ;;
    esac
}
EXPECTED=$(expected_sha "$SPLIT")

if [ -n "$EXPECTED" ]; then
    echo "  computing sha256 (a few minutes for a 20GB file)..."
    GOT=$(sha256sum "$TARBALL" | cut -d' ' -f1)
    if [ "$GOT" != "$EXPECTED" ]; then
        echo "ERROR: checksum mismatch for $TARBALL"
        echo "  expected $EXPECTED"
        echo "  got      $GOT"
        echo ""
        echo "wget reported a complete transfer, so this is CORRUPTION, not a"
        echo "partial file -- resuming would not repair it. Delete the tarball"
        echo "and resubmit to start it over:"
        echo "    rm $TARBALL && sbatch \$0 $SPLIT"
        echo "Nothing was extracted, so no downstream check can mistake this"
        echo "for a completed split."
        exit 1
    fi
    echo "  sha256 OK"
else
    echo "  no known checksum for $SPLIT -- falling back to a tar integrity test"
    if ! tar -tzf "$TARBALL" > /dev/null 2>&1; then
        echo "ERROR: $TARBALL is incomplete or corrupt (tar -tzf failed)."
        echo "Not extracted. Resubmit -- wget -c resumes the remaining bytes."
        exit 1
    fi
    echo "  archive OK"
fi

echo ""
# Job 11224 downloaded and verified 21.5GB successfully, then ENOSPC'd during
# extraction because other users consumed the shared volume during the 3h
# transfer -- leaving a partial split dir. Check space at the moment of
# extraction, not just at job start.
TAR_GB=$(du -BG "$TARBALL" 2>/dev/null | cut -f1 | tr -dc '0-9')
NEED_EXTRACT_GB=$(( ${TAR_GB:-22} + 3 ))   # flac is already compressed; ~1:1, + margin
AVAIL_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
echo "=== Space check before extracting: have ${AVAIL_GB}GB, need ~${NEED_EXTRACT_GB}GB ==="
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt "$NEED_EXTRACT_GB" ]; then
    echo "ERROR: not enough room to extract. NOT extracting, so no partial"
    echo "split dir is created and nothing downstream can mistake this for done."
    echo "The verified tarball is kept -- free space and resubmit, and it will"
    echo "extract without re-downloading. Safe to reclaim, all already extracted:"
    echo "  data/raw/train-other-500.tar.gz (~28GB), train-clean-100.tar.gz (~6GB),"
    echo "  dev-clean/dev-other/test-clean/test-other .tar.gz (~1.3GB total),"
    echo "  data/raw/wham_noise (~36GB, Libri2Mix already generated from it)."
    exit 1
fi

echo "=== Extracting into $RAW_DIR (tar contains LibriSpeech/$SPLIT/...) ==="
# Remove a partial dir from a previous failed extract so tar starts clean.
rm -rf "${LS_DIR:?}/${SPLIT:?}"
# --no-same-owner/--no-same-permissions: this extracts ~104k small files onto a
# shared NFS mount as a non-root user, and tar's attempts to restore ownership
# and modes are extra metadata round-trips that NFS can reject. Job 11228 failed
# with 361 "Cannot close: Input/output error" + 30 "Bad file descriptor" with
# 501GB free and a checksum-verified tarball -- a storage-layer fault, not space.
# Dropping those operations removes a class of failure and speeds extraction;
# it does not fix a genuinely sick fileserver, so if this recurs it is one for
# the HPC admins (same category as the munge daemon failure in job 10790).
if ! tar -xzf "$TARBALL" -C "$RAW_DIR" --no-same-owner --no-same-permissions; then
    echo "ERROR: extraction failed (see tar errors above)."
    echo "Removing the partial directory so it cannot be mistaken for a"
    echo "complete split by download.py / librispeech.py / this script."
    rm -rf "${LS_DIR:?}/${SPLIT:?}"
    exit 1
fi

n=$(find "$LS_DIR/$SPLIT" -name '*.flac' 2>/dev/null | wc -l)
echo "  extracted: $n flac files (expected $WANT_FLAC)"
if [ "$WANT_FLAC" -gt 0 ] && [ "$n" -lt "$WANT_FLAC" ]; then
    echo "ERROR: extraction came up short. Removing the partial directory."
    rm -rf "${LS_DIR:?}/${SPLIT:?}"
    exit 1
fi

echo ""
echo "=== Disk after ==="
df -h . || true
echo ""
echo "The tarball is KEPT at $TARBALL so a re-extract needs no re-download."
echo "Delete it once the flac count above looks right and space is needed."
