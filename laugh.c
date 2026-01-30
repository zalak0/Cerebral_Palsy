#include "laugh.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* =========================
   Configuration (matches Python intent)
   ========================= */

// RMS parameters (equivalent to librosa RMS with n_fft/hop)
#define FRAME_LEN 1024     // n_fft
#define HOP 256            // hop length
#define MAX_RMS_FRAMES 240 // ring size (~3.8s at 16kHz with hop=256)

// Classification window length (how many recent seconds to use)
#define WINDOW_SEC 3.0f

// Peak detection configuration
#define PROM_WIN_SEC 0.50f // prominence search window (seconds)
#define MIN_PEAK_DIST 2    // minimum peak distance in RMS frames
#define MAX_PEAKS 90

// Laugh rules
#define FAST_GAP_SEC 0.35f
#define MIN_CLUSTER_PEAKS 3
#define MIN_CLUSTER_COUNT 2
#define LONG_RUN_PEAKS 4
#define MIN_TOTAL_PEAKS 6

// Cry burst-shape rules (envelope shape constraints)
#define BURST_WIN_SEC 1.5f
#define FLOOR_TOL 0.25f
#define RISE_THR 0.18f
#define FALL_THR 0.18f
#define UP_RATIO_THR 0.55f
#define DOWN_RATIO_THR 0.55f

#define EPS 1e-9f

/* =========================
   Helpers: sorting & stats
   ========================= */

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/*
  Compute ~10th percentile by sorting a small array and taking an index.
  Note: numpy percentile may use interpolation; this is a close approximation.
*/
static float percentile10(const float *x, int n)
{
    if (n <= 0)
        return 0.0f;
    float tmp[MAX_RMS_FRAMES];
    int m = (n < MAX_RMS_FRAMES) ? n : MAX_RMS_FRAMES;
    for (int i = 0; i < m; i++)
        tmp[i] = x[i];
    qsort(tmp, m, sizeof(float), cmp_float);
    int idx = (int)floorf(0.10f * (m - 1));
    if (idx < 0)
        idx = 0;
    if (idx > m - 1)
        idx = m - 1;
    return tmp[idx];
}

/* Median of a small float array (sort then pick mid). */
static float median_of(const float *x, int n)
{
    if (n <= 0)
        return 0.0f;
    float tmp[120];
    int m = (n < 120) ? n : 120;
    for (int i = 0; i < m; i++)
        tmp[i] = x[i];
    qsort(tmp, m, sizeof(float), cmp_float);
    if (m & 1)
        return tmp[m / 2];
    return 0.5f * (tmp[m / 2 - 1] + tmp[m / 2]);
}

/* =========================
   Internal: push RMS into ring buffer
   ========================= */

static void push_rms(CL_State *st, float rms)
{
    // Optional smoothing to reduce jitter (EMA).
    // If you want "no smoothing" like pure RMS, set alpha=1.0f.
    const float alpha = 0.25f;

    if (!st->ema_init)
    {
        st->ema = rms;
        st->ema_init = true;
    }
    else
        st->ema = alpha * rms + (1.0f - alpha) * st->ema;

    rms = st->ema;

    st->rms_ring[st->rms_widx] = rms;
    st->rms_widx++;
    if (st->rms_widx >= MAX_RMS_FRAMES)
        st->rms_widx = 0;
    if (st->rms_count < MAX_RMS_FRAMES)
        st->rms_count++;

    st->new_rms = true;
}

/* =========================
   Internal: feed one sample energy into RMS sliding window
   ========================= */

/*
  We maintain a circular buffer of sample energies (x^2) for last FRAME_LEN samples.
  sum_energy holds the running sum, so we can compute RMS quickly.
*/
static void feed_sample_energy(CL_State *st, int64_t energy)
{
    int64_t old = st->energy_ring[st->samp_idx];
    st->energy_ring[st->samp_idx] = energy;
    st->sum_energy += (energy - old);

    st->samp_idx++;
    if (st->samp_idx >= FRAME_LEN)
        st->samp_idx = 0;

    // Every HOP samples, we output one RMS value
    st->hop_count++;
    if (st->hop_count >= HOP)
    {
        st->hop_count = 0;

        double meanE = (double)st->sum_energy / (double)FRAME_LEN;
        // Normalize assuming int16 input amplitude, convert to approx [0..1] scale
        float rms = (float)(sqrt(meanE) / 32768.0);
        push_rms(st, rms);
    }
}

/* =========================
   Internal: copy latest RMS frames into a linear window (time order)
   ========================= */

static int get_latest_rms_window(const CL_State *st, float *out, int out_cap, float *out_duration_sec)
{
    if (st->rms_count <= 0)
        return 0;

    float frame_sec = (float)HOP / (float)st->sr;
    int need = (int)ceilf(WINDOW_SEC / frame_sec);
    if (need > out_cap)
        need = out_cap;

    int useN = (st->rms_count < need) ? st->rms_count : need;

    // Compute start index in ring buffer
    int start = st->rms_widx - useN;
    while (start < 0)
        start += MAX_RMS_FRAMES;

    // Copy sequentially into out[]
    for (int i = 0; i < useN; i++)
    {
        int idx = start + i;
        if (idx >= MAX_RMS_FRAMES)
            idx -= MAX_RMS_FRAMES;
        out[i] = st->rms_ring[idx];
    }

    if (out_duration_sec)
        *out_duration_sec = useN * frame_sec;
    return useN;
}

/* =========================
   Peak detection (simple find_peaks approximation)
   - local maxima + height threshold + prominence threshold + min distance
   ========================= */

/*
  Simple prominence approximation:
  For each candidate peak i, search within +/- PROM_WIN_SEC for left min and right min,
  then define prominence as:
      prom = peak - max(leftMin, rightMin)
  This is not a perfect replica of scipy.find_peaks prominence, but works well for envelope logic.
*/
static int find_peaks_simple(const float *rms, int N, float frame_sec,
                             float height_thr, float prom_thr,
                             int *peaks, int maxPeaks)
{
    int w = (int)floorf(PROM_WIN_SEC / frame_sec);
    if (w < 2)
        w = 2;

    int k = 0;
    int last = -999999;

    for (int i = 1; i < N - 1; i++)
    {
        if (i - last < MIN_PEAK_DIST)
            continue;

        // local maximum check
        bool isLocalMax = (rms[i] > rms[i - 1]) && (rms[i] >= rms[i + 1]);
        if (!isLocalMax)
            continue;

        if (rms[i] < height_thr)
            continue;

        int l0 = i - w;
        if (l0 < 0)
            l0 = 0;
        int r0 = i + w;
        if (r0 > N - 1)
            r0 = N - 1;

        float leftMin = rms[i];
        for (int j = l0; j <= i; j++)
            if (rms[j] < leftMin)
                leftMin = rms[j];

        float rightMin = rms[i];
        for (int j = i; j <= r0; j++)
            if (rms[j] < rightMin)
                rightMin = rms[j];

        float prom = rms[i] - (leftMin > rightMin ? leftMin : rightMin);
        if (prom < prom_thr)
            continue;

        peaks[k++] = i;
        last = i;
        if (k >= maxPeaks)
            break;
    }

    return k;
}

/* =========================
   Cry burst-shape check (envelope shape around a peak)
   ========================= */

/*
  For a peak pk, find:
  - left valley (minimum) within BURST_WIN_SEC left side
  - right valley (minimum) within BURST_WIN_SEC right side
  Then enforce:
  - rise and fall amplitudes are large enough
  - both valleys return close to noise floor
  - left side mostly rising (diff > 0)
  - right side mostly falling (diff < 0)
*/
static bool burst_shape_ok(const float *rms, int N, int pk,
                           float frame_sec, float noise_floor, float dyn)
{
    int w = (int)floorf(BURST_WIN_SEC / frame_sec);
    int l0 = pk - w;
    if (l0 < 0)
        l0 = 0;
    int r0 = pk + w;
    if (r0 > N - 1)
        r0 = N - 1;

    if ((pk - l0) < 5 || (r0 - pk) < 5)
        return false;

    // left valley
    int li = l0;
    float leftMin = rms[l0];
    for (int i = l0; i < pk; i++)
    {
        if (rms[i] < leftMin)
        {
            leftMin = rms[i];
            li = i;
        }
    }

    // right valley
    int ri = pk;
    float rightMin = rms[pk];
    for (int i = pk; i <= r0; i++)
    {
        if (rms[i] < rightMin)
        {
            rightMin = rms[i];
            ri = i;
        }
    }

    float peak = rms[pk];
    float rise = peak - leftMin;
    float fall = peak - rightMin;

    // amplitude gate
    if (rise < RISE_THR * dyn || fall < FALL_THR * dyn)
        return false;

    // must return near floor
    float floor_limit = noise_floor + FLOOR_TOL * dyn;
    if (leftMin > floor_limit || rightMin > floor_limit)
        return false;

    // rising ratio on left side
    int upN = pk - li;
    int upPos = 0;
    for (int i = li; i < pk; i++)
        if ((rms[i + 1] - rms[i]) > 0)
            upPos++;
    float up_ratio = (upN > 0) ? ((float)upPos / (float)upN) : 0.0f;

    // falling ratio on right side
    int downN = ri - pk;
    int downNeg = 0;
    for (int i = pk; i < ri; i++)
        if ((rms[i + 1] - rms[i]) < 0)
            downNeg++;
    float down_ratio = (downN > 0) ? ((float)downNeg / (float)downN) : 0.0f;

    return (up_ratio >= UP_RATIO_THR) && (down_ratio >= DOWN_RATIO_THR);
}

/* =========================
   Public API
   ========================= */

void cl_init(CL_State *st, int sample_rate)
{
    if (!st)
        return;
    memset(st, 0, sizeof(*st));
    st->sr = sample_rate;
}

void cl_feed_mono_block(CL_State *st, const int16_t *mono, int n)
{
    if (!st || !mono || n <= 0)
        return;
    for (int i = 0; i < n; i++)
    {
        int32_t s = (int32_t)mono[i];
        int64_t e = (int64_t)s * (int64_t)s;
        feed_sample_energy(st, e);
    }
}

void cl_feed_stereo_block(CL_State *st, const int16_t *left, const int16_t *right, int n)
{
    if (!st || !left || !right || n <= 0)
        return;
    for (int i = 0; i < n; i++)
    {
        int32_t l = (int32_t)left[i];
        int32_t r = (int32_t)right[i];
        // energy fusion prevents phase-cancel issues of (L+R)/2
        int64_t e = (((int64_t)l * l) + ((int64_t)r * r)) / 2;
        feed_sample_energy(st, e);
    }
}

bool cl_has_new_rms(const CL_State *st)
{
    return st ? st->new_rms : false;
}

void cl_clear_new_rms(CL_State *st)
{
    if (!st)
        return;
    st->new_rms = false;
}

CL_Label cl_classify_latest(const CL_State *st, CL_Debug *debug)
{
    if (!st)
        return CL_UNKNOWN;

    float rmsWin[MAX_RMS_FRAMES];
    float duration_sec = 0.0f;
    int useN = get_latest_rms_window(st, rmsWin, MAX_RMS_FRAMES, &duration_sec);
    if (useN < 10)
        return CL_UNKNOWN;

    float frame_sec = (float)HOP / (float)st->sr;

    // floor + dynamic range
    float noise_floor = percentile10(rmsWin, useN);
    float peak_max = 0.0f;
    for (int i = 0; i < useN; i++)
        if (rmsWin[i] > peak_max)
            peak_max = rmsWin[i];
    float dyn = peak_max - noise_floor;
    if (dyn < EPS)
        dyn = EPS;

    // peak thresholds (adaptive)
    float height_thr = noise_floor + 0.2f * dyn;
    float prom_thr = 0.1f * dyn;

    // detect peaks
    int peaks[MAX_PEAKS];
    int numPeaks = find_peaks_simple(rmsWin, useN, frame_sec, height_thr, prom_thr, peaks, MAX_PEAKS);

    /* ===== Stage 1: LAUGH ===== */
    float peaks_per_sec = (duration_sec > 1e-6f) ? ((float)numPeaks / duration_sec) : 0.0f;

    float fast_repeat_ratio = 0.0f;
    float median_interval = 999.0f;
    float cv_fast = 10.0f;
    int max_fast_run = 1;
    int cluster_count = 0;

    if (numPeaks >= 3)
    {
        int m = numPeaks - 1;
        float intervals[MAX_PEAKS];
        bool shortMask[MAX_PEAKS];

        int shortCnt = 0;
        float shortIntervals[MAX_PEAKS];
        int shortN = 0;

        for (int i = 0; i < m; i++)
        {
            intervals[i] = (peaks[i + 1] - peaks[i]) * frame_sec;
            shortMask[i] = (intervals[i] < FAST_GAP_SEC);
            if (shortMask[i])
            {
                shortCnt++;
                shortIntervals[shortN++] = intervals[i];
            }
        }

        fast_repeat_ratio = (m > 0) ? ((float)shortCnt / (float)m) : 0.0f;
        median_interval = median_of(intervals, m);

        // CV of short intervals (stability metric)
        if (shortN >= 2)
        {
            float mean = 0.0f;
            for (int i = 0; i < shortN; i++)
                mean += shortIntervals[i];
            mean /= (float)shortN;

            float var = 0.0f;
            for (int i = 0; i < shortN; i++)
            {
                float d = shortIntervals[i] - mean;
                var += d * d;
            }
            var /= (float)shortN;
            float sd = sqrtf(var);
            cv_fast = sd / (mean + EPS);
        }
        else
        {
            cv_fast = 10.0f;
        }

        // longest run of "fast gaps" (consecutive close peaks)
        int cur = 1;
        max_fast_run = 1;
        for (int i = 0; i < m; i++)
        {
            if (shortMask[i])
            {
                cur++;
                if (cur > max_fast_run)
                    max_fast_run = cur;
            }
            else
            {
                cur = 1;
            }
        }

        // count clusters (each True-run indicates a cluster of close peaks)
        cluster_count = 0;
        int i = 0;
        while (i < m)
        {
            if (shortMask[i])
            {
                int runStart = i;
                while (i < m && shortMask[i])
                    i++;
                int runLen = i - runStart;
                int clusterPeaks = runLen + 1;
                if (clusterPeaks >= MIN_CLUSTER_PEAKS)
                    cluster_count++;
            }
            else
            {
                i++;
            }
        }
    }

    bool old_laugh_rule =
        (peaks_per_sec >= 2.0f) &&
        (fast_repeat_ratio >= 0.55f) &&
        ((cv_fast <= 0.35f) || (max_fast_run >= LONG_RUN_PEAKS) || (median_interval <= 0.30f));

    bool cluster_laugh_rule =
        (numPeaks >= MIN_TOTAL_PEAKS) &&
        (cluster_count >= MIN_CLUSTER_COUNT) &&
        (max_fast_run >= MIN_CLUSTER_PEAKS) &&
        (fast_repeat_ratio >= 0.40f);

    bool is_laugh = old_laugh_rule || cluster_laugh_rule;

    /* ===== Stage 2: CRY ===== */
    int cry_good = 0;
    int cry_checked = 0;
    float burst_ratio = 0.0f;
    bool is_cry = false;

    if (!is_laugh && numPeaks >= 1)
    {
        for (int i = 0; i < numPeaks; i++)
        {
            int pk = peaks[i];
            bool ok = burst_shape_ok(rmsWin, useN, pk, frame_sec, noise_floor, dyn);
            cry_checked++;
            if (ok)
                cry_good++;
        }

        burst_ratio = (cry_checked > 0) ? ((float)cry_good / (float)cry_checked) : 0.0f;

        // If only a few peaks exist, be more permissive
        if (numPeaks <= 3)
            is_cry = (cry_good >= 1) && (burst_ratio >= 0.34f);
        else
            is_cry = (cry_good >= 2) && (burst_ratio >= 0.60f);
    }

    CL_Label label = CL_UNKNOWN;
    if (is_laugh)
        label = CL_LAUGH;
    else if (is_cry)
        label = CL_CRY;

    // Fill debug metrics if requested
    if (debug)
    {
        debug->duration_sec = duration_sec;
        debug->noise_floor = noise_floor;
        debug->peak_max = peak_max;
        debug->dyn = dyn;
        debug->num_peaks = numPeaks;

        debug->peaks_per_sec = peaks_per_sec;
        debug->fast_repeat_ratio = fast_repeat_ratio;
        debug->median_interval = median_interval;
        debug->cv_fast = cv_fast;
        debug->max_fast_run = max_fast_run;
        debug->cluster_count = cluster_count;

        debug->cry_good_peaks = cry_good;
        debug->cry_checked_peaks = cry_checked;
        debug->cry_burst_ratio = burst_ratio;
    }

    return label;
}
