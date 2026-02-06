#include "laugh.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* =========================
   Configuration
   ========================= */
#define FRAME_LEN 1024
#define HOP 256
#define MAX_RMS_FRAMES 240

#define WINDOW_SEC 3.0f

#define PROM_WIN_SEC 0.50f
#define MIN_PEAK_DIST 2
#define MAX_PEAKS 90

// Laugh rules
#define FAST_GAP_SEC 0.35f
#define MIN_CLUSTER_PEAKS 3
#define MIN_CLUSTER_COUNT 2
#define LONG_RUN_PEAKS 4
#define MIN_TOTAL_PEAKS 6

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
   RMS ring push (EMA smoothing)
   ========================= */
static void push_rms(CL_State *st, float rms)
{
    const float alpha = 0.25f; // smoothing factor

    if (!st->ema_init)
    {
        st->ema = rms;
        st->ema_init = true;
    }
    else
    {
        st->ema = alpha * rms + (1.0f - alpha) * st->ema;
    }

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
   Feed one sample energy into sliding RMS window
   ========================= */
static void feed_sample_energy(CL_State *st, int64_t energy)
{
    int64_t old = st->energy_ring[st->samp_idx];
    st->energy_ring[st->samp_idx] = energy;
    st->sum_energy += (energy - old);

    st->samp_idx++;
    if (st->samp_idx >= FRAME_LEN)
        st->samp_idx = 0;

    st->hop_count++;
    if (st->hop_count >= HOP)
    {
        st->hop_count = 0;

        double meanE = (double)st->sum_energy / (double)FRAME_LEN;
        float rms = (float)(sqrt(meanE) / 32768.0); // normalized RMS ~[0..1]
        push_rms(st, rms);
    }
}

/* =========================
   Copy latest RMS frames (time order) to linear buffer
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

    int start = st->rms_widx - useN;
    while (start < 0)
        start += MAX_RMS_FRAMES;

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
   ========================= */
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

    float noise_floor = percentile10(rmsWin, useN);
    float peak_max = 0.0f;
    for (int i = 0; i < useN; i++)
        if (rmsWin[i] > peak_max)
            peak_max = rmsWin[i];

    float dyn = peak_max - noise_floor;
    if (dyn < EPS)
        dyn = EPS;

    float height_thr = noise_floor + 0.2f * dyn;
    float prom_thr = 0.1f * dyn;

    int peaks[MAX_PEAKS];
    int numPeaks = find_peaks_simple(rmsWin, useN, frame_sec, height_thr, prom_thr, peaks, MAX_PEAKS);

    // ===== Laugh features =====
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

        // CV of short intervals
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

        // longest run of close peaks
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

        // count clusters
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
    }

    return is_laugh ? CL_LAUGH : CL_UNKNOWN;
}
