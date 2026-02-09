#include "laugh.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPS 1e-9f

// ====================== Helpers ======================
static inline float clampf(float x, float a, float b)
{
    return (x < a) ? a : ((x > b) ? b : x);
}

static inline float fmaxf_safe(float a, float b) { return (a > b) ? a : b; }

static void hann_window(float *w, int n)
{
    for (int i = 0; i < n; i++)
    {
        w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
}

// ---- FFT radix2 (in-place) ----
static void bit_reverse_reorder(float *re, float *im, int n)
{
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        if (i < j)
        {
            float tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            float ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
        int m = n >> 1;
        while (m >= 1 && j >= m)
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

static void fft_radix2(float *re, float *im, int n)
{
    bit_reverse_reorder(re, im, n);
    for (int len = 2; len <= n; len <<= 1)
    {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wlen_re = cosf(ang);
        float wlen_im = sinf(ang);
        for (int i = 0; i < n; i += len)
        {
            float w_re = 1.0f, w_im = 0.0f;
            int half = len >> 1;
            for (int j = 0; j < half; j++)
            {
                int u = i + j;
                int v = u + half;

                float vr = re[v] * w_re - im[v] * w_im;
                float vi = re[v] * w_im + im[v] * w_re;

                float ur = re[u];
                float ui = im[u];

                re[u] = ur + vr;
                im[u] = ui + vi;
                re[v] = ur - vr;
                im[v] = ui - vi;

                float nw_re = w_re * wlen_re - w_im * wlen_im;
                float nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

// ---- quickselect percentile ----
static void swapf(float *a, float *b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

static int partition(float *arr, int left, int right, int pivot)
{
    float pv = arr[pivot];
    swapf(&arr[pivot], &arr[right]);
    int store = left;
    for (int i = left; i < right; i++)
    {
        if (arr[i] < pv)
        {
            swapf(&arr[store], &arr[i]);
            store++;
        }
    }
    swapf(&arr[right], &arr[store]);
    return store;
}

static float quickselect_k(float *arr, int n, int k)
{
    int left = 0, right = n - 1;
    while (1)
    {
        if (left == right)
            return arr[left];
        int pivot = (left + right) / 2;
        pivot = partition(arr, left, right, pivot);
        if (k == pivot)
            return arr[k];
        if (k < pivot)
            right = pivot - 1;
        else
            left = pivot + 1;
    }
}

static float percentile_copy(const float *x, int n, float p, float *tmp, int tmp_cap)
{
    if (n <= 0)
        return 0.0f;
    if (n > tmp_cap)
        n = tmp_cap;
    memcpy(tmp, x, (size_t)n * sizeof(float));
    int k = (int)lrintf(clampf(p, 0, 1) * (float)(n - 1));
    if (k < 0)
        k = 0;
    if (k > n - 1)
        k = n - 1;
    return quickselect_k(tmp, n, k);
}

static float median_copy(const float *x, int n, float *tmp, int tmp_cap)
{
    return percentile_copy(x, n, 0.5f, tmp, tmp_cap);
}

// ====================== History unfold ======================
static void unfold_hist(CL_State *st)
{
    int n = st->hist_len;
    int head = st->hist_head;
    int m = CL_MAX_FRAMES;
    for (int i = 0; i < n; i++)
    {
        int idx = head - n + i;
        while (idx < 0)
            idx += m;
        idx %= m;
        st->rms_tmp[i] = st->rms_hist[idx];
        st->zcr_tmp[i] = st->zcr_hist[idx];
        st->flat_tmp[i] = st->flat_hist[idx];
        st->cent_tmp[i] = st->cent_hist[idx];
    }
}

// ====================== Feature compute per frame ======================
static void compute_features_frame(CL_State *st,
                                   const float *frame,
                                   float *out_rms,
                                   float *out_zcr,
                                   float *out_flat,
                                   float *out_cent)
{
    const int N = st->n_fft;
    const int half = N / 2;

    // RMS
    double e = 0.0;
    for (int i = 0; i < N; i++)
    {
        double v = frame[i];
        e += v * v;
    }
    float rms = (float)sqrt(e / (double)N);

    // ZCR
    int zc = 0;
    float prev = frame[0];
    for (int i = 1; i < N; i++)
    {
        float cur = frame[i];
        if ((prev >= 0 && cur < 0) || (prev < 0 && cur >= 0))
            zc++;
        prev = cur;
    }
    float zcr = (float)zc / (float)(N - 1);

    // FFT input with Hann
    for (int i = 0; i < N; i++)
    {
        st->fft_re[i] = frame[i] * st->hann[i];
        st->fft_im[i] = 0.0f;
    }
    fft_radix2(st->fft_re, st->fft_im, N);

    // mag bins
    int bins = half + 1;
    double log_sum = 0.0, mag_sum = 0.0;
    double cent_num = 0.0, cent_den = 0.0;
    for (int k = 0; k < bins; k++)
    {
        float re = st->fft_re[k];
        float im = st->fft_im[k];
        float mag = sqrtf(re * re + im * im) + EPS;
        log_sum += log((double)mag);
        mag_sum += (double)mag;

        double f = (double)k * (double)st->sr / (double)N;
        cent_num += f * (double)mag;
        cent_den += (double)mag;
    }

    double gmean = exp(log_sum / (double)bins);
    double amean = mag_sum / (double)bins;
    float flat = (float)(gmean / (amean + 1e-12));
    float cent = (cent_den > 0.0) ? (float)(cent_num / cent_den) : 0.0f;

    *out_rms = rms;
    *out_zcr = zcr;
    *out_flat = flat;
    *out_cent = cent;
}

// ====================== find_peaks (RMS) ======================
static int find_peaks_rms(const float *rms, int n,
                          float height_thr,
                          float prom_thr,
                          int distance,
                          int *out_idx, int out_cap)
{
    int cnt = 0;
    int last_pk = -999999;

    for (int i = 1; i < n - 1; i++)
    {
        if (!(rms[i] > rms[i - 1] && rms[i] > rms[i + 1]))
            continue;
        if (rms[i] < height_thr)
            continue;
        if (i - last_pk < distance)
            continue;

        // prominence approx (similar to scipy-ish)
        float left_min = rms[i];
        for (int j = i - 1; j >= 0; j--)
        {
            if (rms[j] > rms[i])
                break;
            if (rms[j] < left_min)
                left_min = rms[j];
        }
        float right_min = rms[i];
        for (int j = i + 1; j < n; j++)
        {
            if (rms[j] > rms[i])
                break;
            if (rms[j] < right_min)
                right_min = rms[j];
        }
        float base = fmaxf(left_min, right_min);
        float prom = rms[i] - base;
        if (prom < prom_thr)
            continue;

        if (cnt < out_cap)
        {
            out_idx[cnt++] = i;
            last_pk = i;
        }
        else
            break;
    }
    return cnt;
}

// ====================== mask runs count ======================
static int count_good_runs(const bool *mask, int n, int min_len)
{
    int seg_count = 0;
    int i = 0;
    while (i < n)
    {
        if (!mask[i])
        {
            i++;
            continue;
        }
        int s = i;
        while (i < n && mask[i])
            i++;
        int e = i;
        if ((e - s) >= min_len)
            seg_count++;
    }
    return seg_count;
}

// ====================== HUM ======================
static bool is_hum_tonal_bursts(CL_State *st,
                                const float *rms, const float *flat, const float *zcr,
                                int n,
                                float noise_floor, float dyn,
                                CL_Debug *dbg)
{
    const float voiced_k = 0.18f;
    const float flat_thr = 0.28f;
    const float zcr_thr = 0.14f;
    const float min_voiced_seg_sec = 0.12f;
    const int min_seg_count = 2;
    const float tonal_ratio_thr = 0.60f;

    if (n < 10)
        return false;

    float thr = noise_floor + voiced_k * dyn;

    int voiced_cnt = 0;
    int tonal_cnt = 0;
    for (int i = 0; i < n; i++)
    {
        bool v = (rms[i] > thr);
        st->voiced_tmp[i] = v;
        if (v)
        {
            voiced_cnt++;
            if (flat[i] < flat_thr && zcr[i] < zcr_thr)
                tonal_cnt++;
        }
    }

    float voiced_ratio = (float)voiced_cnt / (float)n;
    float tonal_ratio = (voiced_cnt > 0) ? ((float)tonal_cnt / (float)voiced_cnt) : 0.0f;

    int min_frames = (int)ceilf(min_voiced_seg_sec / st->frame_sec);
    if (min_frames < 1)
        min_frames = 1;
    int seg_count = count_good_runs(st->voiced_tmp, n, min_frames);

    // zcr median (voiced)
    float tmpz[CL_MAX_FRAMES];
    int zn = 0;
    for (int i = 0; i < n; i++)
        if (st->voiced_tmp[i])
            tmpz[zn++] = zcr[i];
    float zcr_med = (zn > 0) ? median_copy(tmpz, zn, st->rms_tmp, CL_MAX_FRAMES) : 1.0f;

    if (dbg)
    {
        dbg->hum_voiced_ratio = voiced_ratio;
        dbg->hum_tonal_ratio = tonal_ratio;
        dbg->hum_zcr_med = zcr_med;
        dbg->hum_seg_count = seg_count;
    }

    return (tonal_ratio >= tonal_ratio_thr) && (seg_count >= min_seg_count);
}

// ====================== CRY burst shape (RMS only) ======================
static bool burst_shape_ok(const float *rms, int n,
                           int pk,
                           float noise_floor,
                           float frame_sec,
                           float dyn)
{
    const float win_sec = 1.5f;
    const float floor_tol = 0.25f;
    const float rise_thr = 0.18f;
    const float fall_thr = 0.18f;
    const float up_ratio_thr = 0.55f;
    const float down_ratio_thr = 0.55f;

    int w = (int)ceilf(win_sec / frame_sec);
    int l0 = pk - w;
    if (l0 < 0)
        l0 = 0;
    int r0 = pk + w;
    if (r0 > n - 1)
        r0 = n - 1;

    if (pk - l0 < 5 || r0 - pk < 5)
        return false;

    int li = l0;
    float left_min = rms[l0];
    for (int i = l0; i < pk; i++)
    {
        if (rms[i] < left_min)
        {
            left_min = rms[i];
            li = i;
        }
    }

    int ri = pk;
    float right_min = rms[pk];
    for (int i = pk; i <= r0; i++)
    {
        if (rms[i] < right_min)
        {
            right_min = rms[i];
            ri = i;
        }
    }

    float peak = rms[pk];
    float rise = peak - left_min;
    float fall = peak - right_min;

    if (rise < rise_thr * dyn || fall < fall_thr * dyn)
        return false;

    float floor_limit = noise_floor + floor_tol * dyn;
    if (left_min > floor_limit || right_min > floor_limit)
        return false;

    int up_cnt = 0, up_total = 0;
    for (int i = li; i < pk; i++)
    {
        float d = rms[i + 1] - rms[i];
        up_total++;
        if (d > 0)
            up_cnt++;
    }
    int down_cnt = 0, down_total = 0;
    for (int i = pk; i < ri; i++)
    {
        float d = rms[i + 1] - rms[i];
        down_total++;
        if (d < 0)
            down_cnt++;
    }
    float up_ratio = (up_total > 0) ? ((float)up_cnt / (float)up_total) : 0.0f;
    float down_ratio = (down_total > 0) ? ((float)down_cnt / (float)down_total) : 0.0f;

    return (up_ratio >= up_ratio_thr) && (down_ratio >= down_ratio_thr);
}

// ====================== TALK pattern (挡板用，方案2不输出) ======================
static bool is_talk_pattern(CL_State *st,
                            const float *rms, const float *flat, const float *cent,
                            const int *peaks, int peak_n,
                            int n_frames,
                            float noise_floor, float dyn,
                            CL_Debug *dbg)
{
    const float min_syllable_gap = 0.08f;
    const float max_syllable_gap = 0.55f;
    const float regularity_cv_thr = 0.65f;
    const int min_peaks = 3;
    const int max_peaks = 15;
    const float envelope_smoothness_thr = 0.12f;
    const float speech_centroid_lo = 600.0f;
    const float speech_centroid_hi = 2800.0f;
    const float harmonicity_thr = 0.35f;
    const float floor_return_ratio_thr = 0.40f;

    if (peak_n < min_peaks || peak_n > max_peaks)
        return false;

    // intervals
    float intervals[CL_MAX_PEAKS];
    int int_n = 0;
    for (int i = 0; i < peak_n - 1; i++)
    {
        float dt = (float)(peaks[i + 1] - peaks[i]) * st->frame_sec;
        if (dt >= min_syllable_gap && dt <= max_syllable_gap)
            intervals[int_n++] = dt;
    }
    if (int_n < 2)
        return false;

    double sum = 0, sum2 = 0;
    for (int i = 0; i < int_n; i++)
    {
        sum += intervals[i];
        sum2 += intervals[i] * intervals[i];
    }
    float mean = (float)(sum / (double)int_n);
    float var = (float)(sum2 / (double)int_n - mean * mean);
    if (var < 0)
        var = 0;
    float std = sqrtf(var);
    float cv = std / (mean + EPS);
    if (cv > regularity_cv_thr)
        return false;

    // smoothness: mean abs(second diff) on normalized rms
    float rmax = 0.0f;
    for (int i = 0; i < n_frames; i++)
        if (rms[i] > rmax)
            rmax = rms[i];
    float inv = (rmax > EPS) ? (1.0f / rmax) : 1.0f;

    double s2 = 0.0;
    int c2 = 0;
    for (int i = 0; i < n_frames - 2; i++)
    {
        float x0 = rms[i] * inv, x1 = rms[i + 1] * inv, x2 = rms[i + 2] * inv;
        float d2 = x2 - 2.0f * x1 + x0;
        s2 += fabs((double)d2);
        c2++;
    }
    float smoothness = (c2 > 0) ? (float)(s2 / (double)c2) : 1.0f;
    if (smoothness > envelope_smoothness_thr)
        return false;

    // floor return ratio
    float floor_thr = noise_floor + 0.20f * dyn;
    int pairs = peak_n - 1;
    int floor_returns = 0;
    for (int i = 0; i < pairs; i++)
    {
        int a = peaks[i], b = peaks[i + 1];
        bool ok = false;
        for (int k = a; k < b; k++)
        {
            if (rms[k] <= floor_thr)
            {
                ok = true;
                break;
            }
        }
        if (ok)
            floor_returns++;
    }
    float floor_return_ratio = (pairs > 0) ? ((float)floor_returns / (float)pairs) : 0.0f;
    if (floor_return_ratio < floor_return_ratio_thr)
        return false;

    // harmonicity = 1 - median(flatness)
    float flat_med = median_copy(flat, n_frames, st->rms_tmp, CL_MAX_FRAMES);
    float harmonicity = 1.0f - flat_med;
    if (harmonicity < harmonicity_thr)
        return false;

    float cent_med = median_copy(cent, n_frames, st->rms_tmp, CL_MAX_FRAMES);
    if (!(cent_med >= speech_centroid_lo && cent_med <= speech_centroid_hi))
        return false;

    if (dbg)
    {
        dbg->talk_interval_mean = mean;
        dbg->talk_interval_cv = cv;
        dbg->talk_smoothness = smoothness;
        dbg->talk_floor_return_ratio = floor_return_ratio;
        dbg->talk_harmonicity = harmonicity;
        dbg->talk_centroid_median = cent_med;
    }
    return true;
}

// ====================== LAUGH rules ======================
static bool is_laugh_rule(CL_State *st,
                          const int *peaks, int peak_n,
                          float duration_sec,
                          CL_Debug *dbg)
{
    const float FAST_GAP_SEC = 0.35f;
    const int MIN_CLUSTER_PEAKS = 3;
    const int MIN_CLUSTER_COUNT = 2;
    const int LONG_RUN_PEAKS = 4;
    const int MIN_TOTAL_PEAKS = 6;

    float peaks_per_sec = (duration_sec > 1e-6f) ? ((float)peak_n / duration_sec) : 0.0f;

    float fast_repeat_ratio = 0.0f;
    float median_interval = 999.0f;
    float cv_fast = 10.0f;
    int max_fast_run = 1;
    int cluster_count = 0;

    if (peak_n >= 3)
    {
        int int_n = peak_n - 1;
        float intervals[CL_MAX_PEAKS];
        bool short_mask[CL_MAX_PEAKS];
        int short_n = 0;

        for (int i = 0; i < int_n; i++)
        {
            float dt = (float)(peaks[i + 1] - peaks[i]) * st->frame_sec;
            intervals[i] = dt;
            short_mask[i] = (dt < FAST_GAP_SEC);
            if (short_mask[i])
                short_n++;
        }
        fast_repeat_ratio = (float)short_n / (float)int_n;

        // median interval
        median_interval = median_copy(intervals, int_n, st->rms_tmp, CL_MAX_FRAMES);

        // CV of short intervals
        if (short_n >= 2)
        {
            float tmp[CL_MAX_PEAKS];
            int k = 0;
            for (int i = 0; i < int_n; i++)
                if (short_mask[i])
                    tmp[k++] = intervals[i];
            double sum = 0, sum2 = 0;
            for (int i = 0; i < k; i++)
            {
                sum += tmp[i];
                sum2 += tmp[i] * tmp[i];
            }
            float mean = (float)(sum / (double)k);
            float var = (float)(sum2 / (double)k - mean * mean);
            if (var < 0)
                var = 0;
            float std = sqrtf(var);
            cv_fast = std / (mean + EPS);
        }

        // max fast run
        int cur = 1;
        max_fast_run = 1;
        for (int i = 0; i < int_n; i++)
        {
            if (short_mask[i])
            {
                cur++;
                if (cur > max_fast_run)
                    max_fast_run = cur;
            }
            else
                cur = 1;
        }

        // cluster count
        int i = 0;
        while (i < int_n)
        {
            if (short_mask[i])
            {
                int s = i;
                while (i < int_n && short_mask[i])
                    i++;
                int run_len = i - s;
                int cluster_peaks = run_len + 1;
                if (cluster_peaks >= MIN_CLUSTER_PEAKS)
                    cluster_count++;
            }
            else
                i++;
        }
    }

    bool old_laugh_rule =
        (peaks_per_sec >= 2.0f) &&
        (fast_repeat_ratio >= 0.55f) &&
        ((cv_fast <= 0.35f) || (max_fast_run >= LONG_RUN_PEAKS) || (median_interval <= 0.30f));

    bool cluster_laugh_rule =
        (peak_n >= MIN_TOTAL_PEAKS) &&
        (cluster_count >= MIN_CLUSTER_COUNT) &&
        (max_fast_run >= MIN_CLUSTER_PEAKS) &&
        (fast_repeat_ratio >= 0.40f);

    if (dbg)
    {
        dbg->peaks_per_sec = peaks_per_sec;
        dbg->fast_repeat_ratio = fast_repeat_ratio;
        dbg->median_interval = median_interval;
        dbg->cv_fast = cv_fast;
        dbg->max_fast_run = max_fast_run;
        dbg->cluster_count = cluster_count;
    }

    return old_laugh_rule || cluster_laugh_rule;
}

// ====================== API ======================
const char *cl_label_str(CL_Label lab)
{
    switch (lab)
    {
    case CL_HUM:
        return "hum";
    case CL_LAUGH:
        return "laugh";
    case CL_CRY:
        return "cry";
    default:
        return "unknown";
    }
}

void cl_init(CL_State *st, int sample_rate)
{
    if (!st)
        return;
    memset(st, 0, sizeof(*st));
    st->sr = (sample_rate > 0) ? sample_rate : CL_SR_DEFAULT;
    st->hop = CL_HOP;
    st->n_fft = CL_NFFT;
    st->frame_sec = (float)st->hop / (float)st->sr;
    hann_window(st->hann, st->n_fft);
}

void cl_reset(CL_State *st)
{
    if (!st)
        return;
    int sr = st->sr;
    cl_init(st, sr);
}

static inline float pcm16_to_f32(int16_t v)
{
    return (float)v / 32768.0f;
}

bool cl_feed_mono_block(CL_State *st, const int16_t *mono, int n)
{
    if (!st || !mono || n <= 0)
        return false;

    bool produced = false;

    for (int i = 0; i < n; i++)
    {
        float x = pcm16_to_f32(mono[i]);

        st->frame_buf[st->fb_fill++] = x;
        if (st->fb_fill > st->n_fft)
        {
            // keep last NFFT samples
            memmove(&st->frame_buf[0], &st->frame_buf[1], (size_t)(st->n_fft - 1) * sizeof(float));
            st->frame_buf[st->n_fft - 1] = x;
            st->fb_fill = st->n_fft;
        }

        st->hop_acc++;

        if (st->fb_fill >= st->n_fft && st->hop_acc >= st->hop)
        {
            float rms, zcr, flat, cent;
            compute_features_frame(st, st->frame_buf, &rms, &zcr, &flat, &cent);

            int w = st->hist_head;
            st->rms_hist[w] = rms;
            st->zcr_hist[w] = zcr;
            st->flat_hist[w] = flat;
            st->cent_hist[w] = cent;

            st->hist_head = (st->hist_head + 1) % CL_MAX_FRAMES;
            if (st->hist_len < CL_MAX_FRAMES)
                st->hist_len++;

            // slide by hop
            memmove(&st->frame_buf[0],
                    &st->frame_buf[st->hop],
                    (size_t)(st->n_fft - st->hop) * sizeof(float));
            st->fb_fill = st->n_fft - st->hop;
            st->hop_acc = 0;

            produced = true;
        }
    }

    return produced;
}

CL_Label cl_classify_latest(CL_State *st, CL_Debug *dbg)
{
    if (!st)
        return CL_UNKNOWN;
    if (dbg)
        memset(dbg, 0, sizeof(*dbg));
    if (st->hist_len < 10)
        return CL_UNKNOWN;

    unfold_hist(st);
    int n = st->hist_len;

    float duration_sec = (float)n * st->frame_sec;

    // noise floor = 10th percentile of RMS
    float noise_floor = percentile_copy(st->rms_tmp, n, 0.10f, st->zcr_tmp, CL_MAX_FRAMES);

    float peak_max = 0.0f;
    for (int i = 0; i < n; i++)
        if (st->rms_tmp[i] > peak_max)
            peak_max = st->rms_tmp[i];
    float dyn = fmaxf_safe(peak_max - noise_floor, EPS);

    // peaks from RMS
    float height_thr = noise_floor + 0.2f * dyn;
    float prom_thr = 0.1f * dyn;
    int peak_n = find_peaks_rms(st->rms_tmp, n, height_thr, prom_thr, 2, st->peaks, CL_MAX_PEAKS);

    if (dbg)
    {
        dbg->duration_sec = duration_sec;
        dbg->noise_floor = noise_floor;
        dbg->peak_max = peak_max;
        dbg->dyn = dyn;
        dbg->num_peaks = peak_n;
    }

    // ========= Decision order =========
    // 方案2：保留 talk 检测当挡板，但 talk 不输出，只返回 UNKNOWN
    // hum > laugh > talk挡板(->unknown) > cry > unknown

    bool hum = is_hum_tonal_bursts(st, st->rms_tmp, st->flat_tmp, st->zcr_tmp, n, noise_floor, dyn, dbg);
    if (hum)
        return CL_HUM;

    bool laugh = is_laugh_rule(st, st->peaks, peak_n, duration_sec, dbg);
    if (laugh)
        return CL_LAUGH;

    bool talk = is_talk_pattern(st, st->rms_tmp, st->flat_tmp, st->cent_tmp,
                                st->peaks, peak_n, n, noise_floor, dyn, dbg);
    if (talk)
    {
        if (dbg)
            dbg->talk_blocked = 1;
        return CL_UNKNOWN; // ✅ 关键：talk 只挡住误判，不输出 talk
    }

    int cry_good = 0, cry_checked = 0;
    for (int i = 0; i < peak_n; i++)
    {
        int pk = st->peaks[i];
        bool ok = burst_shape_ok(st->rms_tmp, n, pk, noise_floor, st->frame_sec, dyn);
        cry_checked++;
        if (ok)
            cry_good++;
    }
    float burst_ratio = (cry_checked > 0) ? ((float)cry_good / (float)cry_checked) : 0.0f;

    bool cry = false;
    if (peak_n <= 3)
        cry = (cry_good >= 1) && (burst_ratio >= 0.34f);
    else
        cry = (cry_good >= 2) && (burst_ratio >= 0.60f);

    if (dbg)
    {
        dbg->cry_good_peaks = cry_good;
        dbg->cry_checked_peaks = cry_checked;
        dbg->cry_burst_ratio = burst_ratio;
    }

    return cry ? CL_CRY : CL_UNKNOWN;
}
