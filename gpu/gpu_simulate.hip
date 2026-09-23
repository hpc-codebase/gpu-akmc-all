
#include "gpu_simulate.h"
#include <cstdlib>
#include <mpi.h>

#include <hiprand/hiprand_kernel.h>
// recb 随机游走保险丝：限制的是循环尝试次数，不是成功跳跃次数。
#ifndef KMC_RECB_MAX_ITERATIONS
#define KMC_RECB_MAX_ITERATIONS 100000
#endif

// #define const int MAX_PROBE_STEPS = 3

// Global Variables
__device__ static GPUHashSet *g_MoRe_Hash;
__device__ static GPUHashSet *g_MoMo_Hash;
__device__ static GPUHashSet *g_Re_Hash;
__device__ static GPUHashSet *g_V_Hash;
__device__ static dev_meta *g_meta;
__device__ static GPUHashSet *g_Busy_Set;
__device__ static GPUHashSet *g_surface_Hash;
__device__ static GPUHashSet *g_ghost_Hash;
__device__ static GPUHashSet *g_comm_ghost_bucket0;
__device__ static GPUHashSet *g_comm_ghost_bucket1;
__device__ static GPUHashSet *g_comm_ghost_bucket2;
__device__ static GPUHashSet *g_comm_ghost_bucket3;
__device__ static GPUHashSet *g_comm_ghost_bucket4;
__device__ static GPUHashSet *g_comm_ghost_bucket5;
__device__ static GPUHashSet *g_comm_ghost_bucket6;
__device__ static comm::Region<comm::_type_lattice_size> *g_comm_ghost_regions;
__device__ static GPUHashSet **g_comm_surface_buckets_ptr;
__device__ static CommBox *g_comm_surface_boxes;
__device__ static int *g_comm_surface_offsets;
__device__ static long int  *g_Pair_Atoms;
__device__ unsigned long long int* g_Global_Bitmap_Lock;//哈希表的全局位图锁
__device__ static hiprandStatePhilox4_32_10_t *g_rng_states;//每个 walker 线程一条独立持久的随机数流
// roster/hash 差值排查：validate_roster_all 本次 rebuild 清除的陈旧 slot 数
// （区分"误清"还是"compact 漏 key"：清除数≈差值→validate 清；清除数≪差值→compact 漏）
// [0]=validate_cleared [1]=validate_cleared_ghost [2]=hash_keys_before_compact [3]=roster_after_compact
// A sector is stable only after a complete recb pass moves no pair. Stable
// sectors reconsider only walkers whose 1NN shell intersects the latest KMC
// event; communication/rebuild invalidates all sectors.
__device__ static unsigned int g_recb_sector_stable[8];
__device__ static unsigned int g_recb_pairs_changed;
__device__ static unsigned int g_recb_was_full_pass;
__device__ static long int g_recb_event_from = -1;
__device__ static long int g_recb_event_to = -1;
__device__ static unsigned int g_recb_event_valid;
// event-local correctness audit: [0]=excluded pairs checked,
// [1]=excluded but no longer captured (must never be silently skipped).
__device__ static unsigned int g_recb_local_audit[2];
__device__ static unsigned int g_validate_counters[4];

// 哈希统计统一由 KMC_RECB_HASH_STATS=1 控制。
// 下标：[0]MoRe [1]MoMo [2]Re [3]V [4]Busy [5]ghost [6]surface。
static constexpr int RECB_HASH_TABLE_COUNT = 7;
static constexpr int RECB_WALKER_HIST_BINS = 11;
__device__ static unsigned long long g_hash_query[RECB_HASH_TABLE_COUNT];
__device__ static unsigned long long g_hash_update[RECB_HASH_TABLE_COUNT];
__device__ static bool g_hash_stats_on = false;
// 每张表独立的最长探测链（桶数）；仅统计模式开启时更新。
__device__ static unsigned int g_probe_max[RECB_HASH_TABLE_COUNT];
// Re probe 到达深度计数，以及 MoRe 的低开销 dirty 标记。
__device__ static unsigned long long g_re_probe_reach[7];
// 完整 walker 尝试次数直方图；仅 KMC_RECB_HASH_STATS=1 时由终止线程提交一次。
// bins: 0, 1-7, 8-31, 32-63, 64-127, 128-255, 256-511,
//       512-1023, 1024-2047, 2048-4095, >=4096.
__device__ static unsigned long long g_recb_walker_attempt_hist[RECB_WALKER_HIST_BINS];
// wave 长尾估算：[sum_attempts, max*active_lanes, max*waveSize, active_waves]。
__device__ static unsigned long long g_recb_wave_tail[4];
__device__ static unsigned int g_recb_walker_attempt_max;
__device__ static unsigned int g_recb_wave_size;
__device__ static unsigned long long g_recb_extra_rng_draws;
__device__ static unsigned int g_more_dirty = 0;
// [MoMo, MoRe, Re] × {empty, tombstone, live}
__device__ static unsigned int g_hash_occ[9];

static long int arr_size;
// pair_atoms_size is the fixed device allocation capacity.  The compact live
// roster occupies [0, pair_roster_scan_size); keeping these values separate
// avoids scanning the reserved half of the allocation on every ReCB call.
static long int pair_atoms_size;
static long int pair_roster_scan_size;

// Correctness-only conflict-bias histogram.  Each eligible physical move is
// classified four ways (direction/op/local Re/local SIA) and gets exactly one
// terminal outcome.  This lets the final run summary show whether hardware
// contention rejects particular physical environments disproportionately.
enum RecbStratumOutcome {
  RECB_ELIGIBLE = 0,
  RECB_LOCK_FAIL,
  RECB_SRC_CHANGED,
  RECB_DST_CHANGED,
  RECB_TX_FAIL,
  RECB_SUCCESS,
  RECB_STRATUM_OUTCOME_COUNT
};
static constexpr int RECB_DIRECTION_STRATA = 14;
static constexpr int RECB_OP_STRATA = 4;
static constexpr int RECB_NUM_RE_STRATA = 9;   // exact 1NN count: 0..8
static constexpr int RECB_NUM_SIA_STRATA = 3;  // 0, 1, >=2
static constexpr int RECB_STRATUM_COUNT =
    RECB_DIRECTION_STRATA + RECB_OP_STRATA +
    RECB_NUM_RE_STRATA + RECB_NUM_SIA_STRATA;

struct RecbVerifyStats {
  unsigned long long attempts;
  unsigned long long successful_hops;
  unsigned int max_successful_hops;
  unsigned int multi_hop_pairs;
  unsigned int captured_no_move;
  unsigned int sector_pairs;
  unsigned int captured_walks;
  unsigned int exited_walks;
  unsigned int capped_walks; // 触发 KMC_RECB_MAX_ITERATIONS 保险丝的 walker 数（正常应恒为 0）
  unsigned int stale_source_walkers;
  // 排查 roster/hash 不守恒用：把 stale 按触发场景细分。
  // 期望满足 stale_source_walkers == stale_at_start + stale_src_changed。
  unsigned int stale_at_start;      // walker 开头检查发现 slot 指向类型非 MoMo/MoRe
  unsigned int stale_src_changed;   // DCL 二次校验发现源类型被并发修改
  unsigned int stale_start_ghost;   // 开头触发且源 id 落在 ghost 区（跨扇区通信嫌疑）
  unsigned int stale_start_is_re;   // 开头触发且实际类型是 Re（否则是 Mo/V）
  unsigned int stale_start_is_v;    // 开头触发且实际类型是 V 空位（空位事件移动间隙对嫌疑）
  unsigned int stale_debug_seq;     // device printf 限流计数（每次 recb 清零，只打印前几个 stale）
  unsigned int unclassified_walkers;
  unsigned long long target_mo;
  unsigned long long target_re;
  unsigned long long target_other;
  unsigned long long lock_failures;
  unsigned long long src_check_failures;
  unsigned long long dst_mo_check_failures;
  unsigned long long dst_re_check_failures;
  unsigned long long dst_other_check_failures;
  unsigned long long transaction_failures;
  unsigned long long op_attempts[4];
  unsigned long long op_successes[4];
  unsigned long long op_failures[4];
  unsigned long long direction_counts[14];
  unsigned long long strata[RECB_STRATUM_COUNT][RECB_STRATUM_OUTCOME_COUNT];
  unsigned int roster_type_mismatches;
  unsigned int roster_live_count;
  unsigned int hash_pair_count;
};

namespace cg = cooperative_groups;
//这些都是GPU内部的全局变量
namespace gpuSimulate {

    // device variable
    double *d_sum_rates;
    double *d_config;
    dev_event *d_event;

    dev_Vacancy *deviceVacancyInfo;
    dev_nnLattice *deviceNeighbourInfo;

    // host variable
    _type_lattice_count currentVacCount;
    double h_config[2];

    dev_atom *atoms;
    dev_atom *d_atoms;
    dev_atom *ret_atoms;
    dev_Vacancy *hostVacancyInfo;
    dev_nnLattice *hostNeighbourInfo;

    int blockSize;
    int numBlocks;
    _type_lattice_count vacHostArrayCapacity;
    _type_lattice_count HostAtomCapacity;

    long int *d_vac_all;
    long int *d_vac_sector;
    int *d_vac_count_all;
    int *d_vac_count_sector;
    int vac_capacity_all;
    int vac_capacity_sector;

    double *sum_rates;
    dev_event* h_event;

    _type_lattice_id to_x;
    _type_lattice_id to_y;
    _type_lattice_id to_z;

    LatticesList *lattice_list;
    comm::ColoredDomain *p_domain;


    //下面的是recb的命名空间

    dev_meta meta;
    HIPHashSet *MoRe_Hash;
    HIPHashSet *MoMo_Hash;
    HIPHashSet *Re_Hash;
    HIPHashSet *V_Hash;
    HIPHashSet *Busy_Set;
    HIPHashSet *ghost_Hash;
    HIPHashSet *surface_Hash;
    HIPHashSet *comm_ghost_bucket0;
    HIPHashSet *comm_ghost_bucket1;
    HIPHashSet *comm_ghost_bucket2;
    HIPHashSet *comm_ghost_bucket3;
    HIPHashSet *comm_ghost_bucket4;
    HIPHashSet *comm_ghost_bucket5;
    HIPHashSet *comm_ghost_bucket6;
    comm::Region<comm::_type_lattice_size> *d_comm_ghost_regions;
    int comm_ghost_capacity;
    HIPHashSet *comm_surface_buckets[24];
    GPUHashSet **d_comm_surface_bucket_ptrs;
    CommBox *d_comm_surface_boxes;
    int *d_comm_surface_offsets;
    int comm_surface_capacity;
    int comm_surface_boxes_cap;
    int comm_surface_offsets_cap;

    long int *Pair_Atoms;
    long int *d_Pair_Atoms;
    dev_meta *d_meta;
    RecbVerifyStats *d_recb_verify_stats = nullptr;
    unsigned long long *d_recb_walker_lengths = nullptr; // hash stats 专用：[extra_rng:32|attempts:32]
    hiprandStatePhilox4_32_10_t *d_rng_states = nullptr;
    long int rng_states_count = 0;
    unsigned int *d_recb_work_head = nullptr;
    long int *d_sector_slots = nullptr;   // recb 前筛出的本扇区 walker 槽位号（g_Pair_Atoms 下标）
    int *d_sector_slot_count = nullptr;   // 本扇区 walker 数
    long int *d_more_refresh_keys = nullptr;       // MoRe 原地刷新时暂存全部 LIVE key
    unsigned int *d_more_refresh_counts = nullptr; // empty/tombstone/live/reinsert_fail
    unsigned int h_more_cached_counts[3] = {0, 0, 0};
    bool h_more_cache_initialized = false;

    unsigned long long int* d_Global_Bitmap_Lock = nullptr;
    size_t bitmap_lock_bytes = 0;
    GpuCommMaintenanceProfiler comm_gpu_maintenance_profiler;

    long int *d_ghost_list;
    long int *d_surface_list;
    long int *d_ghost_list_type;
    long int *d_surface_list_type;
    int *d_ghost_count;
    int *d_surface_count;
    int ghost_capacity;
    int surface_capacity;

    // 只保留两类运行时开关：
    //   KMC_RECB_CORRECTNESS_DEBUG=1  正确性/不变量/全表核验
    //   KMC_RECB_HASH_STATS=1         哈希访问量、分类 probe、占用与性能计时
    // Logging remains split into correctness and hash-statistics categories.
    // KMC_RECB_EVENT_LOCAL is an algorithm A/B switch, not a logging switch.
    static bool recbHashStatsEnabled() {
      static int cached = -1;
      if (cached != -1) return cached != 0;
      const char* v = std::getenv("KMC_RECB_HASH_STATS");
      cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
      return cached != 0;
    }

    static bool recbCorrectnessDebugEnabled() {
      static int cached = -1;
      if (cached != -1) return cached != 0;
      const char* v = std::getenv("KMC_RECB_CORRECTNESS_DEBUG");
      cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
      return cached != 0;
    }

    // Algorithm A/B switch. The default baseline uses full passes;
    // KMC_RECB_EVENT_LOCAL=1 explicitly enables the locality optimization.
    static bool recbEventLocalEnabled() {
      static int cached = -1;
      if (cached != -1) return cached != 0;
      const char* v = std::getenv("KMC_RECB_EVENT_LOCAL");
      // The original->GPU baseline is a full pass. Event-local scheduling is
      // an explicitly enabled optimization, not the default baseline.
      cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
      return cached != 0;
    }

    static bool recbPhaseTimingEnabled() {
      static int cached = -1;
      if (cached != -1) return cached != 0;
      const char* v = std::getenv("KMC_RECB_PHASE_TIMING");
      cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
      return cached != 0;
    }

    enum RecbPhaseEvent {
      RECB_PHASE_BEGIN = 0,
      RECB_PHASE_AFTER_FILTER,
      RECB_PHASE_AFTER_COUNT_D2H,
      RECB_PHASE_AFTER_WALKER,
      RECB_PHASE_AFTER_BOUNDARY_COMPACT,
      RECB_PHASE_AFTER_REFRESH,
      RECB_PHASE_AFTER_BOUNDARY_TYPE,
      RECB_PHASE_AFTER_COPY,
      RECB_PHASE_AFTER_CLEAR,
      RECB_PHASE_EVENT_COUNT
    };

    static hipEvent_t recb_phase_events[RECB_PHASE_EVENT_COUNT]{};
    static bool recb_phase_events_inited = false;
    static double recb_phase_ms[RECB_PHASE_EVENT_COUNT - 1]{};
    static unsigned long long recb_phase_calls = 0;
    static unsigned long long recb_phase_zero_candidates = 0;
    static unsigned long long recb_phase_positive_candidates = 0;
    static double recb_phase_host_boundary_ms = 0.0;

    enum RecbRunMetric {
      RUN_LAUNCHES = 0, RUN_SECTOR_PAIRS, RUN_ATTEMPTS, RUN_HOPS,
      RUN_MAX_HOPS_ONE_WALKER, RUN_MULTI_HOP, RUN_CAPTURED,
      RUN_CAPTURED_NO_MOVE, RUN_EXITED, RUN_CAPPED, RUN_STALE,
      RUN_UNCLASSIFIED, RUN_TARGET_MO, RUN_TARGET_RE, RUN_TARGET_OTHER,
      RUN_LOCK_FAIL, RUN_SRC_CHANGED, RUN_DST_MO_CHANGED,
      RUN_DST_RE_CHANGED, RUN_DST_OTHER_CHANGED, RUN_TX_FAIL,
      RUN_ROSTER_TYPE_MISMATCH, RUN_STALE_AT_START, RUN_STALE_SRC_CHANGED,
      RUN_CLASSIFICATION_BAD_LAUNCHES, RUN_ROSTER_HASH_BAD_LAUNCHES,
      RUN_MAX_ROSTER_HASH_ABS_DIFF, RUN_METRIC_COUNT
    };

    struct RecbRunSummary {
      unsigned long long metric[RUN_METRIC_COUNT]{};
      unsigned long long op_attempts[RECB_OP_STRATA]{};
      unsigned long long op_successes[RECB_OP_STRATA]{};
      unsigned long long op_failures[RECB_OP_STRATA]{};
      unsigned long long direction_counts[RECB_DIRECTION_STRATA]{};
      unsigned long long strata[RECB_STRATUM_COUNT][RECB_STRATUM_OUTCOME_COUNT]{};
    };

    static RecbRunSummary recb_run_summary{};

    static void accumulateRecbRunSummary(const RecbVerifyStats& s) {
      RecbRunSummary& r = recb_run_summary;
      ++r.metric[RUN_LAUNCHES];
      r.metric[RUN_SECTOR_PAIRS] += s.sector_pairs;
      r.metric[RUN_ATTEMPTS] += s.attempts;
      r.metric[RUN_HOPS] += s.successful_hops;
      if (s.max_successful_hops > r.metric[RUN_MAX_HOPS_ONE_WALKER])
        r.metric[RUN_MAX_HOPS_ONE_WALKER] = s.max_successful_hops;
      r.metric[RUN_MULTI_HOP] += s.multi_hop_pairs;
      r.metric[RUN_CAPTURED] += s.captured_walks;
      r.metric[RUN_CAPTURED_NO_MOVE] += s.captured_no_move;
      r.metric[RUN_EXITED] += s.exited_walks;
      r.metric[RUN_CAPPED] += s.capped_walks;
      r.metric[RUN_STALE] += s.stale_source_walkers;
      r.metric[RUN_UNCLASSIFIED] += s.unclassified_walkers;
      r.metric[RUN_TARGET_MO] += s.target_mo;
      r.metric[RUN_TARGET_RE] += s.target_re;
      r.metric[RUN_TARGET_OTHER] += s.target_other;
      r.metric[RUN_LOCK_FAIL] += s.lock_failures;
      r.metric[RUN_SRC_CHANGED] += s.src_check_failures;
      r.metric[RUN_DST_MO_CHANGED] += s.dst_mo_check_failures;
      r.metric[RUN_DST_RE_CHANGED] += s.dst_re_check_failures;
      r.metric[RUN_DST_OTHER_CHANGED] += s.dst_other_check_failures;
      r.metric[RUN_TX_FAIL] += s.transaction_failures;
      r.metric[RUN_ROSTER_TYPE_MISMATCH] += s.roster_type_mismatches;
      r.metric[RUN_STALE_AT_START] += s.stale_at_start;
      r.metric[RUN_STALE_SRC_CHANGED] += s.stale_src_changed;
      const unsigned long long classified =
          static_cast<unsigned long long>(s.captured_walks) + s.exited_walks +
          s.capped_walks + s.stale_source_walkers;
      if (s.unclassified_walkers != 0 ||
          classified + s.unclassified_walkers != s.sector_pairs)
        ++r.metric[RUN_CLASSIFICATION_BAD_LAUNCHES];
      if (s.roster_live_count != s.hash_pair_count) {
        ++r.metric[RUN_ROSTER_HASH_BAD_LAUNCHES];
        const unsigned long long diff = s.roster_live_count > s.hash_pair_count
            ? static_cast<unsigned long long>(s.roster_live_count - s.hash_pair_count)
            : static_cast<unsigned long long>(s.hash_pair_count - s.roster_live_count);
        if (diff > r.metric[RUN_MAX_ROSTER_HASH_ABS_DIFF])
          r.metric[RUN_MAX_ROSTER_HASH_ABS_DIFF] = diff;
      }
      for (int i = 0; i < RECB_OP_STRATA; ++i) {
        r.op_attempts[i] += s.op_attempts[i];
        r.op_successes[i] += s.op_successes[i];
        r.op_failures[i] += s.op_failures[i];
      }
      for (int i = 0; i < RECB_DIRECTION_STRATA; ++i)
        r.direction_counts[i] += s.direction_counts[i];
      for (int i = 0; i < RECB_STRATUM_COUNT; ++i)
        for (int j = 0; j < RECB_STRATUM_OUTCOME_COUNT; ++j)
          r.strata[i][j] += s.strata[i][j];
    }

    static void mergeRecbRunSummary(RecbRunSummary& dst,
                                    const RecbRunSummary& src) {
      for (int i = 0; i < RUN_METRIC_COUNT; ++i) {
        if (i == RUN_MAX_HOPS_ONE_WALKER || i == RUN_MAX_ROSTER_HASH_ABS_DIFF) {
          if (src.metric[i] > dst.metric[i]) dst.metric[i] = src.metric[i];
        } else {
          dst.metric[i] += src.metric[i];
        }
      }
      for (int i = 0; i < RECB_OP_STRATA; ++i) {
        dst.op_attempts[i] += src.op_attempts[i];
        dst.op_successes[i] += src.op_successes[i];
        dst.op_failures[i] += src.op_failures[i];
      }
      for (int i = 0; i < RECB_DIRECTION_STRATA; ++i)
        dst.direction_counts[i] += src.direction_counts[i];
      for (int i = 0; i < RECB_STRATUM_COUNT; ++i)
        for (int j = 0; j < RECB_STRATUM_OUTCOME_COUNT; ++j)
          dst.strata[i][j] += src.strata[i][j];
    }

    static void logRecbRunSummary() {
      if (!recbCorrectnessDebugEnabled()) return;
      int rank = 0, ranks = 1;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &ranks);
      std::vector<RecbRunSummary> gathered(rank == 0 ? ranks : 0);
      MPI_Gather(&recb_run_summary, static_cast<int>(sizeof(RecbRunSummary)), MPI_BYTE,
                 rank == 0 ? gathered.data() : nullptr,
                 static_cast<int>(sizeof(RecbRunSummary)), MPI_BYTE, 0, MPI_COMM_WORLD);
      if (rank != 0) return;

      RecbRunSummary all{};
      for (int r = 0; r < ranks; ++r) {
        const RecbRunSummary& x = gathered[r];
        kiwi::logs::v(" ",
            " [recb-rank-summary] rank={} launches={} sector_pairs={} attempts={} hops={} "
            "lock_fail={} src_changed={} dst_changed={} tx_fail={} "
            "classification_bad_launches={} roster_hash_bad_launches={}.\n",
            r, x.metric[RUN_LAUNCHES], x.metric[RUN_SECTOR_PAIRS],
            x.metric[RUN_ATTEMPTS], x.metric[RUN_HOPS], x.metric[RUN_LOCK_FAIL],
            x.metric[RUN_SRC_CHANGED],
            x.metric[RUN_DST_MO_CHANGED] + x.metric[RUN_DST_RE_CHANGED] +
                x.metric[RUN_DST_OTHER_CHANGED],
            x.metric[RUN_TX_FAIL], x.metric[RUN_CLASSIFICATION_BAD_LAUNCHES],
            x.metric[RUN_ROSTER_HASH_BAD_LAUNCHES]);
        mergeRecbRunSummary(all, x);
      }
      kiwi::logs::v(" ",
          " [recb-run-summary] ranks={} launches={} sector_pairs={} attempts={} "
          "successful_hops={} max_hops_one_walker={} multi_hop_walkers={} "
          "captured={} captured_no_move={} exited={} capped={} stale={} unclassified={}.\n",
          ranks, all.metric[RUN_LAUNCHES], all.metric[RUN_SECTOR_PAIRS],
          all.metric[RUN_ATTEMPTS], all.metric[RUN_HOPS],
          all.metric[RUN_MAX_HOPS_ONE_WALKER], all.metric[RUN_MULTI_HOP],
          all.metric[RUN_CAPTURED], all.metric[RUN_CAPTURED_NO_MOVE],
          all.metric[RUN_EXITED], all.metric[RUN_CAPPED], all.metric[RUN_STALE],
          all.metric[RUN_UNCLASSIFIED]);
      kiwi::logs::v(" ",
          " [recb-run-integrity] classification_bad_launches={} "
          "roster_hash_bad_launches={} max_roster_hash_abs_diff={} "
          "roster_type_mismatches={} stale_at_start={} stale_src_changed={}.\n",
          all.metric[RUN_CLASSIFICATION_BAD_LAUNCHES],
          all.metric[RUN_ROSTER_HASH_BAD_LAUNCHES],
          all.metric[RUN_MAX_ROSTER_HASH_ABS_DIFF],
          all.metric[RUN_ROSTER_TYPE_MISMATCH], all.metric[RUN_STALE_AT_START],
          all.metric[RUN_STALE_SRC_CHANGED]);
      kiwi::logs::v(" ",
          " [recb-run-conflicts] lock_fail={} src_changed={} dst_changed(mo/re/other)={}/{}/{} "
          "tx_fail={} targets(mo/re/other)={}/{}/{}.\n",
          all.metric[RUN_LOCK_FAIL], all.metric[RUN_SRC_CHANGED],
          all.metric[RUN_DST_MO_CHANGED], all.metric[RUN_DST_RE_CHANGED],
          all.metric[RUN_DST_OTHER_CHANGED], all.metric[RUN_TX_FAIL],
          all.metric[RUN_TARGET_MO], all.metric[RUN_TARGET_RE],
          all.metric[RUN_TARGET_OTHER]);
      kiwi::logs::v(" ",
          " [recb-run-strata-schema] eligible=hardware-conflict candidate before locking; "
          "each eligible count must end in exactly one of lock_fail/src_changed/dst_changed/tx_fail/success.\n");
      const char* group_name[4] = {"direction", "op", "num_re", "num_sia"};
      const int group_begin[4] = {0, RECB_DIRECTION_STRATA,
          RECB_DIRECTION_STRATA + RECB_OP_STRATA,
          RECB_DIRECTION_STRATA + RECB_OP_STRATA + RECB_NUM_RE_STRATA};
      const int group_size[4] = {RECB_DIRECTION_STRATA, RECB_OP_STRATA,
                                RECB_NUM_RE_STRATA, RECB_NUM_SIA_STRATA};
      unsigned int bad_partition_bins = 0;
      for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < group_size[g]; ++i) {
          const unsigned long long* v = all.strata[group_begin[g] + i];
          const unsigned long long terminal = v[RECB_LOCK_FAIL] +
              v[RECB_SRC_CHANGED] + v[RECB_DST_CHANGED] +
              v[RECB_TX_FAIL] + v[RECB_SUCCESS];
          if (terminal != v[RECB_ELIGIBLE]) ++bad_partition_bins;
          kiwi::logs::v(" ",
              " [recb-run-stratum] group={} bin={} eligible={} lock_fail={} "
              "src_changed={} dst_changed={} tx_fail={} success={}.\n",
              group_name[g], (g == 1 ? i + 1 : i), v[RECB_ELIGIBLE],
              v[RECB_LOCK_FAIL], v[RECB_SRC_CHANGED], v[RECB_DST_CHANGED],
              v[RECB_TX_FAIL], v[RECB_SUCCESS]);
        }
      }
      kiwi::logs::v(" ",
          " [recb-run-strata-check] bad_partition_bins={} (expected 0).\n",
          bad_partition_bins);
    }

    static void initRecbProfileEvents() {
      if (recb_phase_events_inited) return;
      for (int i = 0; i < RECB_PHASE_EVENT_COUNT; ++i) {
        HANDLE_HIP(hipEventCreate(&recb_phase_events[i]));
      }
      recb_phase_events_inited = true;
    }

    static void destroyRecbProfileEvents() {
      if (!recb_phase_events_inited) return;
      for (int i = 0; i < RECB_PHASE_EVENT_COUNT; ++i) {
        hipEventDestroy(recb_phase_events[i]);
      }
      recb_phase_events_inited = false;
    }

    static void accumulateRecbPhaseTimings(int candidate_count) {
      HANDLE_HIP(hipEventSynchronize(recb_phase_events[RECB_PHASE_AFTER_CLEAR]));
      for (int i = 0; i + 1 < RECB_PHASE_EVENT_COUNT; ++i) {
        float elapsed_ms = 0.0f;
        HANDLE_HIP(hipEventElapsedTime(&elapsed_ms, recb_phase_events[i],
                                      recb_phase_events[i + 1]));
        recb_phase_ms[i] += static_cast<double>(elapsed_ms);
      }
      ++recb_phase_calls;
      if (candidate_count == 0) ++recb_phase_zero_candidates;
      else ++recb_phase_positive_candidates;
    }

    static void logRecbPhaseTimings() {
      if (!recbPhaseTimingEnabled() || recb_phase_calls == 0) return;
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      const double inv_calls = 1.0 / static_cast<double>(recb_phase_calls);
      kiwi::logs::v(" ",
          " [recb-phase-total] rank={} calls={} zero_candidates={} positive_candidates={} "
          "scan_size={} capacity={} ms(filter/count_d2h/walker/boundary_compact/refresh/boundary_type/copy/clear/host_boundary)="
          "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}.\n",
          rank, recb_phase_calls, recb_phase_zero_candidates,
          recb_phase_positive_candidates, pair_roster_scan_size, pair_atoms_size,
          recb_phase_ms[0], recb_phase_ms[1], recb_phase_ms[2], recb_phase_ms[3],
          recb_phase_ms[4], recb_phase_ms[5], recb_phase_ms[6], recb_phase_ms[7],
          recb_phase_host_boundary_ms);
      kiwi::logs::v(" ",
          " [recb-phase-avg] rank={} us_per_call(filter/count_d2h/walker/boundary_compact/refresh/boundary_type/copy/clear/host_boundary)="
          "{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}.\n",
          rank, recb_phase_ms[0] * 1000.0 * inv_calls,
          recb_phase_ms[1] * 1000.0 * inv_calls,
          recb_phase_ms[2] * 1000.0 * inv_calls,
          recb_phase_ms[3] * 1000.0 * inv_calls,
          recb_phase_ms[4] * 1000.0 * inv_calls,
          recb_phase_ms[5] * 1000.0 * inv_calls,
          recb_phase_ms[6] * 1000.0 * inv_calls,
          recb_phase_ms[7] * 1000.0 * inv_calls,
          recb_phase_host_boundary_ms * 1000.0 * inv_calls);
    }
}

bool recb_phase_timing_enabled() {
  return gpuSimulate::recbPhaseTimingEnabled();
}

void gpuRecbPhaseAddHostBoundary(double seconds) {
  if (seconds > 0.0) gpuSimulate::recb_phase_host_boundary_ms += seconds * 1000.0;
}

void resetGpuCommMaintenanceProfiler() {
    gpuSimulate::comm_gpu_maintenance_profiler = GpuCommMaintenanceProfiler{};
}

GpuCommMaintenanceProfiler getGpuCommMaintenanceProfiler() {
    return gpuSimulate::comm_gpu_maintenance_profiler;
}

void gpuCommProfileAddPack(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.pack_time += seconds;
}

void gpuCommProfileAddNccl(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.nccl_time += seconds;
}

void gpuCommProfileAddHashOp(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.hash_op_time += seconds;
}

void gpuCommProfileAddCompactSample(double seconds, bool first_of_dim) {
    if (first_of_dim) {
        gpuSimulate::comm_gpu_maintenance_profiler.compact_first_sum += seconds;
        gpuSimulate::comm_gpu_maintenance_profiler.compact_first_n += 1.0;
    } else {
        gpuSimulate::comm_gpu_maintenance_profiler.compact_rest_sum += seconds;
        gpuSimulate::comm_gpu_maintenance_profiler.compact_rest_n += 1.0;
    }
}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z);
__device__ int isGhostLat(_type_lattice_id lid);
__device__ int isSurfaceLat(_type_lattice_id lid);
static __device__ __forceinline__ GPUHashSet* commBucketPtr(int idx);

static int parseLocalRankEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return -1;
  char* end = nullptr;
  const long rank = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || rank < 0) return -1;
  return static_cast<int>(rank);
}

// Node-local rank for GPU bind. MPI_COMM_TYPE_SHARED is used first so
// sbatch+mpirun cannot inherit a stale SLURM_LOCALID=0 onto remote nodes.
static int getLocalProcessRank(int fallback_rank) {
  int mpi_inited = 0;
  MPI_Initialized(&mpi_inited);
  if (mpi_inited) {
    MPI_Comm node_comm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, fallback_rank,
                            MPI_INFO_NULL, &node_comm) == MPI_SUCCESS &&
        node_comm != MPI_COMM_NULL) {
      int local_rank = 0;
      MPI_Comm_rank(node_comm, &local_rank);
      MPI_Comm_free(&node_comm);
      return local_rank;
    }
  }

  const char* env_names[] = {
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "MV2_COMM_WORLD_LOCAL_RANK",
      "MPI_LOCALRANKID",
      "PMI_LOCAL_RANK",
      "SLURM_LOCALID",
  };
  for (const char* name : env_names) {
    const int rank = parseLocalRankEnv(name);
    if (rank >= 0) return rank;
  }
  return fallback_rank;
}

static int getEnvIntOrDefault(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') return default_value;
  return static_cast<int>(parsed);
}

static bool shouldPrintGpuBindDebug(int process_rank, int local_rank) {
  if (std::getenv("KMC_GPU_BIND_DEBUG") == nullptr) return false;
  const int max_global = getEnvIntOrDefault("KMC_GPU_BIND_DEBUG_MAX_GLOBAL", 16);
  const int local_filter = getEnvIntOrDefault("KMC_GPU_BIND_DEBUG_LOCAL_RANK", -1);
  if (local_filter >= 0) return local_rank == local_filter;
  return process_rank < max_global;
}
// =================================================================================
//[防线 L1] 组内冲突拦截器 (Intra-Tile Resolvers)
// =================================================================================
// 针对双目标 (calcExchangePairs 用)
template <typename TileType>
__device__ void resolve_intra_tile_conflicts(AtomTx& tx, TileType tile) {
    int my_rank = tile.thread_rank();
    bool conflict = false;
    const bool me_can_conflict = tx.active && tx.op_type != 0;

    for (int i = 0; i < tile.size(); ++i) {
        long int other_src = (long int)tile.shfl((unsigned long long int)tx.src_id, i);
        long int other_dst = (long int)tile.shfl((unsigned long long int)tx.dst_id, i);
        bool other_active  = tile.shfl(tx.active, i);
        int other_op       = tile.shfl(tx.op_type, i);

        if (me_can_conflict && other_active && other_op != 0 && i != my_rank) {
             bool overlap = (tx.dst_id == other_dst) ||
                           (tx.dst_id == other_src) ||
                           (tx.src_id == other_dst) ||
                           (tx.src_id == other_src);
            if (overlap && my_rank > i) {
                conflict = true;
            }
        }
    }
    if (conflict) tx.op_type = 0;
}

// 针对单目标 (update_hash_set 用)
template <typename TileType>
__device__ void resolve_intra_tile_conflicts_single(long int id, bool& active, TileType tile) {
    if (!active) return;
    int my_rank = tile.thread_rank();
    bool conflict = false;

    for (int i = 0; i < tile.size(); ++i) {
        long int other_id = (long int)tile.shfl((unsigned long long int)id, i);
        bool other_active = tile.shfl(active, i);
        if (other_active && id == other_id && i != my_rank && my_rank > i) {
            conflict = true;
        }
    }
    if (conflict) active = false; 
}

// =================================================================================
// [防线 L2] 64 位全局位图锁 (Wavefront-Aligned Bitmap Locks)
// =================================================================================
// 针对双目标 (calcExchangePairs 用)
__device__ inline long int get_safe_lock_id(long int id) {
    long int total_size = g_meta->size_x * g_meta->size_y * g_meta->size_z;
    return (id % total_size + total_size) % total_size;
}
__device__ inline bool try_acquire_bitmap_locks(long int src_id, long int dst_id) {
    if (src_id == dst_id) return false;
    long int safe_src = get_safe_lock_id(src_id);
    long int safe_dst = get_safe_lock_id(dst_id);

    long int min_id = min(safe_src, safe_dst);
    long int max_id = max(safe_src, safe_dst);

    unsigned long long int idx_min = min_id / 64;
    unsigned long long int mask_min = 1ULL << (min_id % 64); // 必须用 1ULL!
    unsigned long long int idx_max = max_id / 64;
    unsigned long long int mask_max = 1ULL << (max_id % 64);

    if ((atomicOr(&g_Global_Bitmap_Lock[idx_min], mask_min) & mask_min) == 0) {
        if ((atomicOr(&g_Global_Bitmap_Lock[idx_max], mask_max) & mask_max) == 0) {
            return true; // 双锁达成
        } else {
            atomicAnd(&g_Global_Bitmap_Lock[idx_min], ~mask_min); // 回退第一把锁
            return false;
        }
    }
    return false;
}

__device__ inline void release_bitmap_locks(long int src_id, long int dst_id) {
    long int safe_src = get_safe_lock_id(src_id);
    long int safe_dst = get_safe_lock_id(dst_id);
    atomicAnd(&g_Global_Bitmap_Lock[safe_src / 64], ~(1ULL << (safe_src % 64)));
    atomicAnd(&g_Global_Bitmap_Lock[safe_dst / 64], ~(1ULL << (safe_dst % 64)));
}

// 针对单目标 (update_hash_set 用)
__device__ inline bool try_acquire_bitmap_lock_single(long int id) {
    unsigned long long int mask = 1ULL << (id % 64);
    return (atomicOr(&g_Global_Bitmap_Lock[id / 64], mask) & mask) == 0;
}

__device__ inline void release_bitmap_lock_single(long int id) {
    unsigned long long int mask = 1ULL << (id % 64);
    atomicAnd(&g_Global_Bitmap_Lock[id / 64], ~mask);
}
// =========================================================
// 单线程查找操作 (Single-Thread Contains)
// 完美兼容 Tile 协作组的桶状哈希表结构 (Bucketed Hash Map)
// 适用于未声明协作组的普通线程调用
// 返回值: true (找到), false (未找到)
// =========================================================
__device__ __forceinline__ int recb_hash_table_index(const GPUHashSet* set) {
    if (set == g_MoRe_Hash) return 0;
    if (set == g_MoMo_Hash) return 1;
    if (set == g_Re_Hash) return 2;
    if (set == g_V_Hash) return 3;
    if (set == g_Busy_Set) return 4;
    if (set == g_ghost_Hash) return 5;
    if (set == g_surface_Hash) return 6;
    return -1;
}

__device__ __forceinline__ void record_hash_query(GPUHashSet* set) {
    if (!g_hash_stats_on) return;
    const int idx = recb_hash_table_index(set);
    if (idx >= 0) atomicAdd(&g_hash_query[idx], 1ULL);
}

__device__ __forceinline__ void record_hash_update(GPUHashSet* set) {
    if (!g_hash_stats_on) return;
    const int idx = recb_hash_table_index(set);
    if (idx >= 0) atomicAdd(&g_hash_update[idx], 1ULL);
}

__device__ __forceinline__ void record_hash_probe(GPUHashSet* set,
                                                   unsigned int probe_len) {
    if (!g_hash_stats_on) return;
    const int idx = recb_hash_table_index(set);
    if (idx >= 0 && probe_len > g_probe_max[idx]) {
      atomicMax(&g_probe_max[idx], probe_len);
    }
    if (idx == 2) {
      int threshold = -1;
      if (probe_len == 1) threshold = 0;
      else if (probe_len == 2) threshold = 1;
      else if (probe_len == 5) threshold = 2;
      else if (probe_len == 9) threshold = 3;
      else if (probe_len == 17) threshold = 4;
      else if (probe_len == 33) threshold = 5;
      else if (probe_len == 65) threshold = 6;
      if (threshold >= 0) atomicAdd(&g_re_probe_reach[threshold], 1ULL);
    }
}

__device__ __forceinline__ void record_more_insert(GPUHashSet* set,
                                                     long int replaced_value) {
    if (set != g_MoRe_Hash) return;
    (void)replaced_value;
    if (g_more_dirty == 0u) atomicCAS(&g_more_dirty, 0u, 1u);
}

__device__ __forceinline__ void record_more_remove(GPUHashSet* set) {
    if (set != g_MoRe_Hash) return;
    if (g_more_dirty == 0u) atomicCAS(&g_more_dirty, 0u, 1u);
}

__device__ bool hash_set_contains(GPUHashSet* set, long int key) {
    if (set == nullptr || key <= 0) return false;
    record_hash_query(set);
    unsigned int loop_count = 0;
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    while (loop_count <= max_probe ) {
        const unsigned int probe_len = loop_count + 1;
        record_hash_probe(set, probe_len);
        int base_index = buck_num * BUCKET_SIZE;
        bool has_empty = false;

        // 禁止展开：该函数在主循环中被大量内联，64 槽全展开会让调用者函数体爆炸，
        // 直接导致后端寄存器分配失败。
        #pragma unroll 1
        for (int i = 0; i < BUCKET_SIZE; ++i) {
            long int val = atomic_load_relaxed(&set->keys[base_index + i]);
            if (val == key) return true;
            if (val == set->empty_flag) has_empty = true;
        }

        // 必须先扫完整个 bucket 再根据 empty 停止。协作插入可能让同一 bucket
        // 出现非严格的 lane 顺序，不能在遇到第一个 empty 时提前漏掉后续 key。
        if (has_empty) return false;

        loop_count++;
        if (loop_count <= max_probe) {
          buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
        }
    }
    return false;
}
//单线程插入
// 返回值: 1 (插入成功), 0 (已存在，未重复插入), -1 (表满失败)
__device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
    if (set == nullptr || key <= 0) return -1;
    record_hash_update(set);
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    // CAS 被同 bucket 的其他 key 抢占时从头重查，避免在并发下制造重复 key。
    for (unsigned int retry = 0; retry < 64; ++retry) {
        long int* first_tombstone = nullptr;
        unsigned int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);

        while (loop_count <= max_probe) {
            const unsigned int probe_len = loop_count + 1;
            record_hash_probe(set, probe_len);
            int base_index = buck_num * BUCKET_SIZE;
            long int* first_empty = nullptr;

            // 禁止展开：同 hash_set_contains，控制调用者函数体规模。
            #pragma unroll 1
            for (int i = 0; i < BUCKET_SIZE; ++i) {
                long int* target_addr = &set->keys[base_index + i];
                const long int val = atomic_load_relaxed(target_addr);
                if (val == key) return 0;
                if (val == set->tombstone_flag && first_tombstone == nullptr) {
                    first_tombstone = target_addr;
                } else if (val == set->empty_flag && first_empty == nullptr) {
                    first_empty = target_addr;
                }
            }

            // empty 表示 probe chain 到此结束；优先复用更早遇到的 tombstone。
            if (first_empty != nullptr) {
                long int* target_addr = first_tombstone != nullptr ? first_tombstone : first_empty;
                const long int expected = first_tombstone != nullptr
                                              ? set->tombstone_flag
                                              : set->empty_flag;
                const unsigned long long int old_val = atomicCAS(
                    reinterpret_cast<unsigned long long int*>(target_addr),
                    static_cast<unsigned long long int>(expected),
                    static_cast<unsigned long long int>(key));
                if (old_val == static_cast<unsigned long long int>(expected)) {
                    record_more_insert(set, expected);
                    return 1;
                }
                break; // candidate 被抢占，从 home bucket 重新验证 key 是否已存在
            }

            loop_count++;
            if (loop_count <= max_probe) {
                buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
            }
        }

        // probe 范围内没有 empty，但可以安全复用已记录的 tombstone。
        if (loop_count > max_probe && first_tombstone != nullptr) {
            const unsigned long long int old_val = atomicCAS(
                reinterpret_cast<unsigned long long int*>(first_tombstone),
                static_cast<unsigned long long int>(set->tombstone_flag),
                static_cast<unsigned long long int>(key));
            if (old_val == static_cast<unsigned long long int>(set->tombstone_flag)) {
                record_more_insert(set, set->tombstone_flag);
                return 1;
            }
        } else if (loop_count > max_probe) {
            return -1;
        }
    }
    return -1;
}
//单线程删除
// 返回值: 1 (删除成功), 0 (未找到该 Key)
__device__ int hash_set_remove_atom_op(GPUHashSet* set, long int key) {
    if (set == nullptr || key <= 0) return 0;
    record_hash_update(set);
    unsigned int loop_count = 0;
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);

    while (loop_count <= max_probe) {
        const unsigned int probe_len = loop_count + 1;
        record_hash_probe(set, probe_len);
        int base_index = buck_num * BUCKET_SIZE;
        bool has_empty = false;

        // 禁止展开：同 hash_set_contains，控制调用者函数体规模。
        #pragma unroll 1
        for (int i = 0; i < BUCKET_SIZE; ++i) {
            long int* target_addr = &set->keys[base_index + i];
            long int val = atomic_load_relaxed(target_addr);

            // 情况 1: 找到了目标 Key
            if (val == key) {
                // 将其原子替换为墓碑标记
                unsigned long long int old_val = atomicCAS(
                    (unsigned long long int*)target_addr, 
                    (unsigned long long int)key, 
                    (unsigned long long int)set->tombstone_flag
                );

                // 抢占成功
                if (old_val == (unsigned long long int)key) {
                    record_more_remove(set);
                    return 1;
                }

                // 抢占失败（说明就在这一瞬间，这个 Key 刚好被别的逻辑删了或改了）
                i--; // 重新检查
                continue;
            }

            if (val == set->empty_flag) has_empty = true;
        }

        if (has_empty) return 0;

        loop_count++;
        if (loop_count <= max_probe) {
          buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
        }
    }
    return 0;
}


template <typename TileType>
__device__ __noinline__ int tiled_contains_op(GPUHashSet* set, long int key, TileType tile, bool active) {
    int final_result = 0;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int leader_key = tile.shfl(key, leader); 
        
        int lane = tile.thread_rank();
        if (lane == leader) record_hash_query(set);

        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        int searching = 1;

        // 【日志】：Leader 宣布任务开始
        // if (lane == leader) {
        //     printf("  -> [Contains] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }

        while(searching) {
            if (lane == leader) record_hash_probe(set, static_cast<unsigned int>(loop_count + 1));
            // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
            long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
            
            uint64_t match_mask = tile.ballot(val == leader_key);
            uint64_t empty_mask = tile.ballot(val == set->empty_flag);

            if (match_mask != 0) {
                leader_result = 1; 
                searching = 0; 
                // if (lane == leader) printf("    -> [HIT] Key %ld found at bucket %u by mask %llx\n", leader_key, buck_num, match_mask);
            } else if (empty_mask != 0) {
                leader_result = 0; 
                searching = 0; 
                // if (lane == leader) printf("    -> [MISS] Encountered empty slots at bucket %u. Key %ld does not exist.\n", buck_num, leader_key);
            } else {
                loop_count++;
                buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
                // if (lane == leader) printf("    -> [COLLISION] Bucket full, probing next bucket %u (Loop %d)\n", buck_num, loop_count);

                if (loop_count > set->num_buckets ) {  
                    leader_result = 0; 
                    searching = 0; 
                    // if (lane == leader) printf("    -> [FAIL] Hash table fully probed. Key %ld not found.\n", leader_key);
                }
            }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}

// ---------------------------------------------------------
// 2. Tiled Insert (插入操作)
// 返回值: 1 插入成功, -1 已经存在(跳过), -2 表满失败
// ---------------------------------------------------------
template <typename TileType>
__device__ int tiled_insert_op(GPUHashSet* set, long int key, TileType tile, bool active, int max_probes) {
    int final_result = 0; 
    const unsigned int probe_limit = hash_effective_max_probes(set->num_buckets);
    const unsigned int effective_max_probes =
        max_probes < static_cast<int>(probe_limit) ? static_cast<unsigned int>(max_probes) : probe_limit;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        // long int leader_key = tile.shfl(key, leader);
        long int leader_key = (long int)tile.shfl((unsigned long long int)key, leader);
        
        int lane = tile.thread_rank();
        if (lane == leader) record_hash_update(set);
        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        // if (lane == leader) {
        //     printf("  -> [Insert] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }
        int searching = 1;
        while(searching){
          if (lane == leader) record_hash_probe(set, static_cast<unsigned int>(loop_count + 1));
          // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
          long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
          if (tile.ballot(val == leader_key) != 0) {
              leader_result = 0;
              searching = 0;
          } else {

          // // 1. 插入之前检查当前key是否已经存在
          // if (tile.ballot(val == leader_key) != 0) {
          //     leader_result = -1; // 已存在
          //     searching = 0;
          // }else{
            // 2. 寻找空位或墓碑
            uint64_t candidate_mask = tile.ballot(val == set->empty_flag) |
                                      tile.ballot(val == set->tombstone_flag);
            
            if (candidate_mask != 0) {
                int target_lane = __ffsll((unsigned long long int)candidate_mask) - 1;
                 int cas_success = 0; 
                    
                if (lane == target_lane) {
                    // #if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                    // long int expected = val;
                    // cas_success = set->keys[buck_num * 64 + lane].compare_exchange_strong(
                    //         expected, leader_key, 
                    //         hip::memory_order_seq_cst, hip::memory_order_seq_cst);
                    unsigned long long int* addr = (unsigned long long int*)&set->keys[buck_num * BUCKET_SIZE + lane];
                    unsigned long long int old_val = atomicCAS(addr, (unsigned long long int)val, (unsigned long long int)leader_key);
                    cas_success = (old_val == (unsigned long long int)val) ? 1 : 0;
                    if (cas_success) record_more_insert(set, val);
                    // #endif
                    // cas_success = do_safe_atomic_cas((long int*)&set->keys[buck_num * 64 + lane], val, leader_key);
                }
                
                if (tile.ballot(cas_success!=0) != 0) {
                    leader_result = 1; 
                    searching = 0;
                }
                // 如果插入失败(被同一Wave内其他操作抢占)，继续在当前桶内重试
            } else {
                // 桶已满，线性探测到下一个桶
                loop_count++;
                buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
                if (loop_count > effective_max_probes) {
                    leader_result = -2; // 插入失败(表满)
                    searching = 0;
                }
            }
          }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}

// ---------------------------------------------------------
// 3. Tiled Remove (删除操作)
// 返回值: 1 删除成功, 0 未找到或已被别人删除
// ---------------------------------------------------------
template <typename TileType>
__device__ int tiled_remove_op(GPUHashSet* set, long int key, TileType tile, bool active, int max_probes) {
    int final_result = 0;
    const unsigned int probe_limit = hash_effective_max_probes(set->num_buckets);
    const unsigned int effective_max_probes =
        max_probes < static_cast<int>(probe_limit) ? static_cast<unsigned int>(max_probes) : probe_limit;
    uint64_t work_mask = tile.ballot(active); 
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        // long int leader_key = tile.shfl(key, leader);
        long int leader_key = (long int)tile.shfl((unsigned long long int)key, leader);
        
        int lane = tile.thread_rank();
        if (lane == leader) record_hash_update(set);
        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        // if (lane == leader) {
        //     printf("  -> [del] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }
        int searching = 1;
        while(searching){
            if (lane == leader) record_hash_probe(set, static_cast<unsigned int>(loop_count + 1));
            // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
            long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
            uint64_t match_mask = tile.ballot(val == leader_key);
            
            if (match_mask != 0) {
                int target_lane = __ffsll((unsigned long long int)match_mask) - 1;
                int cas_success = 0;
                
                if (lane == target_lane) {
                    // #if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                    // long int expected = leader_key;
                    // cas_success = set->keys[buck_num * 64 + lane].compare_exchange_strong(
                    //         expected, set->tombstone_flag, 
                    //         hip::memory_order_seq_cst, hip::memory_order_seq_cst);
                    unsigned long long int* addr = (unsigned long long int*)&set->keys[buck_num * BUCKET_SIZE + lane];
                    unsigned long long int old_val = atomicCAS(addr, (unsigned long long int)leader_key, (unsigned long long int)set->tombstone_flag);
                    cas_success = (old_val == (unsigned long long int)leader_key) ? 1 : 0;
                    if (cas_success) record_more_remove(set);
                    // #endif
                    // cas_success = do_safe_atomic_cas((long int*)&set->keys[buck_num * 64 + lane], leader_key, set->tombstone_flag);
                }
                
                // 用 ballot 判定是否至少一个线程成功删除了
                leader_result = (tile.ballot(cas_success!=0) != 0) ? 1 : 0;
                searching = 0;
            } else if (tile.ballot(val == set->empty_flag) != 0) {
                leader_result = 0; // 未找到
                searching = 0;
            }else{
              loop_count++;
              buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
              if (loop_count > effective_max_probes) {
                  leader_result = 0; // 未找到 (表满)
                  searching = 0;
              }
            }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}


__device__ void printHashTableKernel(GPUHashSet* table, int max_entries, bool verbose) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== GPU Hash Table Info ===\n");
        printf("Capacity: %d\n", table->capacity);
        printf("Empty Flag: %ld\n", table->empty_flag);
        printf("Tombstone Flag: %ld\n", table->tombstone_flag);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < display_count; i++) {
            long int key = table->keys[i];
            
            if (key == table->empty_flag) {
                if (verbose) printf("[%3d]: EMPTY\n", i);
            } else if (key == table->tombstone_flag) {
                if (verbose) printf("[%3d]: TOMBSTONE\n", i);
            } else {
                printf("[%3d]: %ld\n", i, key);
                valid_count++;
            }
        }
        
        if (table->capacity > display_count) {
            printf("... and %d more entries.\n", table->capacity - display_count);
        }
        
        float load_factor = (float)valid_count / table->capacity;
        printf("Total valid entries: %d\n", valid_count);
        printf("Load factor: %.3f\n", load_factor);
        printf("===========================\n");
    }
}
__global__ void print_pair_atoms(int arr_size){
  printf("===========start print pair==========");
  for(int i=0;i<arr_size * 2;i++){
    printf("%lld\t",g_Pair_Atoms[i]);
    if(i%10 == 0)
      printf("\n");
  }
  
  printf("===========end print pair============");
}
__device__ void printDevMetaKernel(const dev_meta* meta) {
    // 仅由第一个线程执行打印，避免输出重复
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== dev_meta Structure Information ===\n");
        
        // 打印晶格尺寸信息
        printf("Lattice Sizes:\n");
        printf("  size_x: %lld (simulation lattice size)\n", meta->size_x);
        printf("  size_y: %lld\n", meta->size_y);
        printf("  size_z: %lld\n", meta->size_z);
        
        // 打印当前进程的模拟盒尺寸
        printf("\nCurrent Process Box Sizes:\n");
        printf("  box_x: %lld (2x real size for BCC structure)\n", meta->box_x);
        printf("  box_y: %lld (same as real size)\n", meta->box_y);
        printf("  box_z: %lld (same as real size)\n", meta->box_z);
        
        // 打印全局模拟盒尺寸
        printf("\nGlobal Simulation Box Sizes:\n");
        printf("  g_box_x: %lld\n", meta->g_box_x);
        printf("  g_box_y: %lld\n", meta->g_box_y);
        printf("  g_box_z: %lld\n", meta->g_box_z);
        
        // 打印全局坐标基准
        printf("\nGlobal Coordinate Baselines:\n");
        printf("  g_base_x: %lld (for global lattice ID calculation)\n", meta->g_base_x);
        printf("  g_base_y: %lld\n", meta->g_base_y);
        printf("  g_base_z: %lld\n", meta->g_base_z);
        
        // 打印幽灵晶格尺寸
        printf("\nGhost Lattice Sizes:\n");
        printf("  ghost_x: %lld (2x real size)\n", meta->ghost_x);
        printf("  ghost_y: %lld\n", meta->ghost_y);
        printf("  ghost_z: %lld\n", meta->ghost_z);
        
        // 打印ID相关信息
        printf("\nID Information:\n");
        printf("  _max_id: %d (max local lattice ID)\n", meta->_max_id);
        printf("  local_base_id: %d (global ID = local ID + this value)\n", meta->local_base_id);
        
        // 打印regions数组（分维度显示）
        printf("\nRegions Array (48 elements):\n");
        const int regions_per_dim = 8;
        for (int dim = 0; dim < 3; ++dim) {
            printf("  Dimension %d Regions:\n", dim);
            for (int region = 0; region < regions_per_dim; ++region) {
                int index = dim * regions_per_dim + region;
                printf("    Region %d: %lld", region, meta->regions[index]);
                if (region % 2 == 1) {
                    printf(" (boundary: %lld to %lld)", meta->regions[index - 1], meta->regions[index]);
                }
                printf("\n");
            }
        }
        
        printf("=====================================\n");
    }
}
__device__ void printTopN(GPUHashSet* table, int max_entries){
   if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== TOP %d info===\n",max_entries);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < table->capacity; i++) {
            long int key = table->keys[i];
            if(valid_count >= max_entries)
              break;
            if (key == table->empty_flag) {
                continue;
            } else if (key == table->tombstone_flag) {
                continue;
            } else {
                printf("[%3d]: %ld\t", i, key);
                valid_count++;
            }
        }
        printf("\n");
      
        printf("===========================\n");
    }
}
__global__ void print_g_Pair_Atoms(int arr_size){
  printf("==========pair_atoms============\n");
  for(int i = 0;i<arr_size;i++){
    printf("%ld\t",g_Pair_Atoms[i]);
    if(i%5==0)
      printf("\n");
  }
  printf("\n");
  printf("================================\n");
}
void printDevMeta(const dev_meta& meta) {
    std::cout << "=== dev_meta Structure Information ===" << std::endl;
    
    // 打印晶格尺寸信息
    std::cout << "Lattice Sizes:" << std::endl;
    std::cout << "  size_x: " << meta.size_x << " (simulation lattice size)" << std::endl;
    std::cout << "  size_y: " << meta.size_y << std::endl;
    std::cout << "  size_z: " << meta.size_z << std::endl;
    
    // 打印当前进程的模拟盒尺寸（注意box_x的特殊说明）
    std::cout << "\nCurrent Process Box Sizes:" << std::endl;
    std::cout << "  box_x: " << meta.box_x << " (2x real size for BCC structure)" << std::endl;
    std::cout << "  box_y: " << meta.box_y << " (same as real size)" << std::endl;
    std::cout << "  box_z: " << meta.box_z << " (same as real size)" << std::endl;
    
    // 打印全局模拟盒尺寸
    std::cout << "\nGlobal Simulation Box Sizes:" << std::endl;
    std::cout << "  g_box_x: " << meta.g_box_x << std::endl;
    std::cout << "  g_box_y: " << meta.g_box_y << std::endl;
    std::cout << "  g_box_z: " << meta.g_box_z << std::endl;
    
    // 打印全局坐标基准
    std::cout << "\nGlobal Coordinate Baselines:" << std::endl;
    std::cout << "  g_base_x: " << meta.g_base_x << " (for global lattice ID calculation)" << std::endl;
    std::cout << "  g_base_y: " << meta.g_base_y << std::endl;
    std::cout << "  g_base_z: " << meta.g_base_z << std::endl;
    
    // 打印幽灵晶格尺寸
    std::cout << "\nGhost Lattice Sizes:" << std::endl;
    std::cout << "  ghost_x: " << meta.ghost_x << " (2x real size)" << std::endl;
    std::cout << "  ghost_y: " << meta.ghost_y << std::endl;
    std::cout << "  ghost_z: " << meta.ghost_z << std::endl;
    
    // 打印ID相关信息
    std::cout << "\nID Information:" << std::endl;
    std::cout << "  _max_id: " << meta._max_id << " (max local lattice ID)" << std::endl;
    std::cout << "  local_base_id: " << meta.local_base_id << " (global ID = local ID + this value)" << std::endl;
    
    // 打印regions数组（分维度显示）
    std::cout << "\nRegions Array (48 elements):" << std::endl;
    const int regions_per_dim = 8;  // 假设每个维度有8个区域
    for (int dim = 0; dim < 3; ++dim) {
        std::cout << "  Dimension " << dim << " Regions:" << std::endl;
        for (int region = 0; region < regions_per_dim; ++region) {
            int index = dim * regions_per_dim + region;
            std::cout << "    Region " << region << ": " << meta.regions[index];
            if (region % 2 == 1) {
                std::cout << " (boundary: " << meta.regions[index - 1] << " to " << meta.regions[index] << ")";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "=====================================" << std::endl;
}
__device__ void checkpointers() {
    // 检查g_MoRe_Hash指针
    if (g_MoRe_Hash == nullptr) {
        printf("ERROR: g_MoRe_Hash is NULL!\n");
    } else {
        printf("g_MoRe_Hash is valid, add:%p capacity: %d\n", g_MoRe_Hash,g_MoRe_Hash->capacity);
    }
    
    // 检查g_MoMo_Hash指针
    if (g_MoMo_Hash == nullptr) {
        printf("ERROR: g_MoMo_Hash is NULL!\n");
    } else {
        printf("g_MoMo_Hash is valid, , add:%p,capacity: %d\n", g_MoMo_Hash,g_MoMo_Hash->capacity);
    }
    
    // 检查g_Re_Hash指针
    if (g_Re_Hash == nullptr) {
        printf("ERROR: g_Re_Hash is NULL!\n");
    } else {
        printf("g_Re_Hash is valid, capacity: %d\n", g_Re_Hash->capacity);
    }
    
    // 检查g_V_Hash指针
    if (g_V_Hash == nullptr) {
        printf("ERROR: g_V_Hash is NULL!\n");
    } else {
        printf("g_V_Hash is valid, capacity: %d\n", g_V_Hash->capacity);
    }
    
    // 检查g_meta指针
    if (g_meta == nullptr) {
        printf("ERROR: g_meta is NULL!\n");
    } else {
        printf("g_meta is valid, size_x: %lld\n", g_meta->size_x);
    }
    
    // 检查g_Busy_Set指针
    if (g_Busy_Set == nullptr) {
        printf("ERROR: g_Busy_Set is NULL!\n");
    } else {
        printf("g_Busy_Set is valid, capacity: %d\n", g_Busy_Set->capacity);
    }
    
    // 检查g_surface_Hash指针
    if (g_surface_Hash == nullptr) {
        printf("ERROR: g_surface_Hash is NULL!\n");
    } else {
        printf("g_surface_Hash is valid, capacity: %d\n", g_surface_Hash->capacity);
    }
    
    // 检查g_ghost_Hash指针
    if (g_ghost_Hash == nullptr) {
        printf("ERROR: g_ghost_Hash is NULL!\n");
    } else {
        printf("g_ghost_Hash is valid, capacity: %d\n", g_ghost_Hash->capacity);
    }
    
    // 检查g_Pair_Atoms指针
    if (g_Pair_Atoms == nullptr) {
        printf("ERROR: g_Pair_Atoms is NULL!\n");
    } else {
        printf("g_Pair_Atoms is valid, address: 0x%p\n", (void*)g_Pair_Atoms);
    }
}

__global__ void print_surface_and_ghost(){
  int g=0,s=0;
  long int x=0,y=0,z=0;
 

  printf("++++++++++++s+++++++++++++++++++++\n");
  for(int i=0;i<g_surface_Hash->capacity;i++)
    if(g_surface_Hash->keys[i]!=-1 && g_surface_Hash->keys[i]!=-1){
      long int id = g_surface_Hash->keys[i];
      x = id%g_meta->size_x;
      y = (id / g_meta->size_x) % g_meta->size_y;
      z = id / (g_meta->size_x * g_meta->size_y);
      printf("s(%ld,%ld)-(%ld,%ld,%ld)\t",g_surface_Hash->keys[i],getType(g_surface_Hash->keys[i]),x,y,z);
      s++;
      if(s%5==0)
      printf("\n");
    }
  printf("\n");
  printf("sssssssssssssss\n");
  for(int i=0;i<g_ghost_Hash->capacity;i++)
    if(g_ghost_Hash->keys[i]!=-1 && g_ghost_Hash->keys[i]!=-1){
      long int id = g_ghost_Hash->keys[i];
      x = id%g_meta->size_x;
      y = (id / g_meta->size_x) % g_meta->size_y;
      z = id / (g_meta->size_x * g_meta->size_y);
      printf("g(%ld,%ld)-(%ld,%ld,%ld)\t",g_ghost_Hash->keys[i],getType(g_ghost_Hash->keys[i]),x,y,z);
      g++;
      if(g%5==0)
      printf("\n");
    }
    printf("\n");
}
__global__ void print_Hash_and_Pair(int arr_size,int flag){
  long int pair_cnt=0,momo=0,more=0;
  for(int i=0;i<arr_size * 2;i++)
    if(g_Pair_Atoms[i]!=-1)
      pair_cnt++;
  for(int i=0;i<g_MoMo_Hash->capacity;i++)
    if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
      momo++;
  for(int i=0;i<g_MoRe_Hash->capacity;i++)
    if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
      more++;

  printf("==current hash and pairs========\n");
  if(flag == 1)
    printf("after:");
  printf("arr_size:%ld,pair: %ld\t momo %ld\t more %ld\n",arr_size,pair_cnt,momo,more);
  printf("==============================\n");
}
__global__ void print_momo_and_more(){
   printf("----------------------after momo-----------------------\n");
  for(int i=0;i<g_MoMo_Hash->capacity;i++)
    if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
      printf("%ld\t",g_MoMo_Hash->keys[i]);
  printf("\n");
  printf("----------------------after more-----------------------\n");
  for(int i=0;i<g_MoRe_Hash->capacity;i++)
    if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
      printf("%ld\t",g_MoRe_Hash->keys[i]);
  printf("\n");
}
__device__ double _1nnMapToSwitch(bonds::PairBond::bond_type bondType){
    switch (bondType) {
      case bonds::PairBond::MoMo:
        return -1.991;
      case bonds::PairBond::VV:
        return 0.199;
      case bonds::PairBond::ReRe:
        return -2.201;
      case bonds::PairBond::NiNi:
        return -1.618;
      case bonds::PairBond::MnMn:
        return -0.271;
      case bonds::PairBond::SiSi:
        return -0.716;
      case bonds::PairBond::VMo:
        return -0.755318;
      case bonds::PairBond::MoRe:
        return -2.096;
      case bonds::PairBond::MoNi:
        return -1.8045;
      case bonds::PairBond::MoMn:
        return -0.446;
      case bonds::PairBond::MoSi:
        return -0.683;
      case bonds::PairBond::VRe:
        return -0.920318;
      case bonds::PairBond::VNi:
        return -0.708818;
      case bonds::PairBond::VMn:
        return -0.038;
      case bonds::PairBond::VSi:
        return -0.344;
      case bonds::PairBond::ReNi:
        return -1.9095;
      case bonds::PairBond::ReMn:
        return -0.365;
      case bonds::PairBond::ReSi:
        return -0.611;
      case bonds::PairBond::MnNi:
        return -0.366;
      case bonds::PairBond::MnSi:
        return -0.488;
      case bonds::PairBond::NiSi:
        return -0.723;
      default:
        // 处理默认情况
        return 0.0;
    }
}

__device__ double _2nnMapToSwitch(bonds::PairBond::bond_type bondType){
    switch (bondType) {
      case bonds::PairBond::MoMo:
        return -0.9955;
      case bonds::PairBond::VV:
        return -0.245;
      case bonds::PairBond::ReRe:
        return -1.1005;
      case bonds::PairBond::NiNi:
        return -0.809;
      case bonds::PairBond::MnMn:
        return -0.611;
      case bonds::PairBond::SiSi:
        return -0.389;
      case bonds::PairBond::VMo:
        return -0.377659;
      case bonds::PairBond::MoRe:
        return -1.048;
      case bonds::PairBond::MoNi:
        return -0.90225;
      case bonds::PairBond::MoMn:
        return -0.631;
      case bonds::PairBond::MoSi:
        return -0.469;
      case bonds::PairBond::VRe:
        return -0.460159;
      case bonds::PairBond::VNi:
        return -0.354409;
      case bonds::PairBond::VSi:
        return -0.248;
      case bonds::PairBond::VMn:
        return -0.203;
      case bonds::PairBond::ReNi:
        return -0.95475;
      case bonds::PairBond::ReMn:
        return -0.621;
      case bonds::PairBond::ReSi:
        return -0.566;
      case bonds::PairBond::MnNi:
        return -0.496;
      case bonds::PairBond::MnSi:
        return -0.316;
      case bonds::PairBond::NiSi:
        return -0.521;
      default:
        // 处理默认情况
        return 0.0;
    }
}
__device__ int hash_set_contains_debug(GPUHashSet* table, long int key) {
  table = g_MoMo_Hash;
  if(table == nullptr)
            printf("momo is null1");
  printf("table addr:%p,key:%ld",table,key);
  if (table == nullptr) {
      printf("Error: table is nullptr\n");
      return 0;
  }
  if (table->keys == nullptr) {
      printf("Error: table->keys is nullptr\n");
      return 0;
  }
  if (table->capacity <= 0) {
      printf("Error: Invalid capacity (%d)\n", table->capacity);
      return 0;
  }

  int index = key % table->capacity;
  if (index < 0 || index >= table->capacity) {
      printf("Error: Computed index out of range (%d)\n", index);
      return 0;
  }

  printf("Searching for key: %d, start index: %d, capacity: %d\n", key, index, table->capacity);

  const int original_index = index;
  int attempts = 0;

  do {
      int current = table->keys[index];

      printf("Checking index: %d, key at index: %d\n", index, current);

      if (current == key) {
          printf("Key %d found at index %d\n", key, index);
          return 1;
      }
      if (current == table->empty_flag) {
          printf("Empty slot encountered at index %d, terminating search\n", index);
          break;  // 遇到空槽提前终止
      }

      index = (index + 1) % table->capacity;
      attempts++;

      if (attempts > table->capacity) {
          printf("Error: Infinite loop detected, breaking out\n");
          break;
      }

  } while (index != original_index);  // 完整遍历整个表

  printf("Key %d not found\n", key);
  return 0;
}


// 原子交换操作 - 支持long类型

// __device__ int hash_set_remove_atom_op(GPUHashSet* table, long key) {
//     unsigned long long index = key % table->capacity;
//     unsigned long long original_index = index;

//     volatile long* keys = (volatile long*)table->keys;
//     do {
//         long current = keys[index];
        
//         if (current == table->empty_flag) {
//             return 0;
//         }
        
//         if (current == key) {
//             long old = atomicCASLong((long*)&table->keys[index], key, table->tombstone_flag);
//             if (old == key) return 1;
//         }
        
//         index = (index + 1) % table->capacity;
//     } while (index != original_index);
    
//     return 0;
// }

// __device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
//   unsigned long long pos = key % set->capacity;  // 保持修正后的哈希计算
//   unsigned long long start_pos = pos;

//   volatile long* keys = (volatile long*)set->keys;
//   do {
//       long current = keys[pos];
      
//       if (current == key) {
//           return 0; // 已存在相同key
//       }
      
//       if (current == set->empty_flag || current == set->tombstone_flag) {
//           long old = atomicCASLong((long*)&set->keys[pos], current, key);
//           if (old == current) return 1;
//           if (old == key) return 0;
//       }
      
//       pos = (pos + 1) % set->capacity;
//   } while (pos != start_pos);
  
//   return 0; // 哈希表已满
// }

__global__ void calcExchangeAtoms(dev_atom *atoms, const _type_lattice_count count){

    const _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;//用这个来标志dev_atom中的原子数
    if(atoms == nullptr) return;
    if(tid < count){
      if(atoms[tid].exchange == -1){
        atoms[tid].exchange = 0;
        return;
      }
      // atoms[tid].exchange = 1;
      for(int i = 0; i < LatticesList::MAX_1NN; i++){
        if (atoms[tid].atom1nn_type[i] == LatticeTypes::MoMo || atoms[tid].atom1nn_type[i] == LatticeTypes::MoRe) atoms[tid].numSIA++;
      }
      if(atoms[tid].numSIA >= 2) {
        // printf("catched by more");
          return;
      }

      if(atoms[tid].atom_type == LatticeTypes::MoRe){
        for(int i = 0; i< LatticesList::MAX_1NN;i++){
          if(atoms[tid].atom1nn_type[i] == LatticeTypes::Re) atoms[tid].numRe++;
        }
        if (atoms[tid].numRe >= 5) {
          // printf("catched by re");
          return;
        }
      }

      _type_lattice_count randomLattice_id = 0;
      LatticeTypes::lat_type randomLattice_type = LatticeTypes::V;

      // randomLattice_id = atoms[tid].atom12nn[atoms[tid].random_num1];
      // randomLattice_type = atoms[tid].atom12nn_type[atoms[tid].random_num1];
      // if(atoms[tid].random_num1 < LatticesList::MAX_1NN){
      //   randomLattice_id = atoms[tid].atom1nn[atoms[tid].random_num1];
      //   randomLattice_type = atoms[tid].atom1nn_type[atoms[tid].random_num1];
      // }
      // else{
      //   randomLattice_id = atoms[tid].atom2nn[atoms[tid].random_num1 -  LatticesList::MAX_1NN ];
      //   randomLattice_type = atoms[tid].atom2nn_type[atoms[tid].random_num1 -  LatticesList::MAX_1NN];
      // }
      // printf("tid: %d, atomtype: %d ,randomtype: %d\n",tid,atoms[tid].atom_type,randomLattice_type);
      //获取随机数位置的粒子的信息
      // kiwi::logs::v(" ","atom type is {},random type is:{}\n",atoms[tid].atom_type,randomLattice_type);

      if (atoms[tid].exchange_type == LatticeTypes::Mo || atoms[tid].exchange_type == LatticeTypes::Re){
        if(atoms[tid].atom_type == LatticeTypes::MoMo){
          if(atoms[tid].exchange_type == LatticeTypes::Mo){   
            atoms[tid].exchange = 1;
          }
          else if(atoms[tid].exchange_type == LatticeTypes::Re){
            atoms[tid].exchange = 1;
          }
        }else if(atoms[tid].atom_type == LatticeTypes::MoRe){
          if(atoms[tid].numRe == 0){
            if(atoms[tid].exchange_type == LatticeTypes::Mo){
              atoms[tid].exchange = 1;
            }
            else{
              atoms[tid].exchange = 1;
            }
          }
          else if(atoms[tid].numRe > 0 && atoms[tid].numRe < LatticesList::MAX_1NN){
            if(atoms[tid].ranmov > (static_cast<double>(atoms[tid].numRe) / LatticesList::MAX_1NN)) {
                if (atoms[tid].exchange_type == LatticeTypes::Mo) { 
                  atoms[tid].exchange = 1;
                }
                else if(atoms[tid].exchange_type == LatticeTypes::Re){
                  atoms[tid].exchange = 1;
                }
            }
          }else return;
        }
      }
      // printf("tid: %d, atoms[tid].exchange:%d, id is %d\n",tid,atoms[tid].exchange,atoms[tid].exchange_id);
    }


}
__global__ void calcRatesGPU(dev_Vacancy *deviceVacancyInfo, dev_nnLattice *deviceNeighbourInfo,
                             double* d_sum_rates, double attempt_freq, double temperature, _type_lattice_count vac_total) {
  // 获取线程的全局ID
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 洗牌函数、寄存器和共享内存优化原子加
  if (tid == 0) *d_sum_rates = 0.0;
  __shared__ double tmp_sh;
  if (threadIdx.x == 0) tmp_sh = 0.0;
  __syncthreads();

  // 判断当前tid是否可用
  if(tid < vac_total) {
    double totalRatesOfCurVac = 0.0;
    
    deviceVacancyInfo[tid].rates[0] = 0;
    deviceVacancyInfo[tid].rates[1] = 0;
    deviceVacancyInfo[tid].rates[2] = 0;
    deviceVacancyInfo[tid].rates[3] = 0;
    deviceVacancyInfo[tid].rates[4] = 0;
    deviceVacancyInfo[tid].rates[5] = 0;
    deviceVacancyInfo[tid].rates[6] = 0;
    deviceVacancyInfo[tid].rates[7] = 0;

    double vac_energy = 0;
    int first_ir = 1;
    unsigned char atom_nei_status = 0;
    for (int b = 0; b < 8; b++) {
      if (deviceNeighbourInfo[tid * NN_TOTAL + b].type._type <= LatticeTypes::Si && deviceNeighbourInfo[tid * NN_TOTAL + b].type._type != LatticeTypes::V) { // can trans
        atom_nei_status |= 1 << b;
      }
    }

    for (unsigned char b = 0; b < 8; b++) {
      if ((atom_nei_status >> b) & 1) { // the neighbour lattice is available
        double active_energy;
        switch (deviceNeighbourInfo[tid * NN_TOTAL + b].type._type) {
          case LatticeTypes::Mo:
            active_energy = 0.65;
            break;
          case LatticeTypes::Re:
            active_energy = 0.54;
            break;
          case LatticeTypes::Mn:
            active_energy = 1.03;
            break;
          case LatticeTypes::Ni:
            active_energy = 0.68;
            break;
          case LatticeTypes::Si:
            active_energy = 0.42;
            break;
          default:
            active_energy = 0;
            break;
        }
        // calculate system energy before transition.
        // const LatticeTypes::lat_type ghost_atom = d_lat_list[tid * 8 + b].type._type;
        double e_before = 0;
        {
          // bonds energy of src lattice contributed by its 1nn/2nn neighbour lattice.
          double e_src, e_des;
          unsigned char _1nn_status = 0xFF;
          unsigned char _2nn_status = 0x2F;
          double energy = 0.0;
          if(first_ir == 1) {
            // Traver all 1nn neighbour lattices, and calculate bond energy contribution.
            for (int b2 = 0; b2 < 8; b2++) {
              if ((_1nn_status >> b2) & 0x1) {
                // we assume that src_atom_type is single atom or vacancy.
                if (deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type <= LatticeTypes::Si) {
                  // it is vacancy or single atom.
                  LatticeTypes::lat_type temp_1nn = deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type;
                  LatticeTypes::lat_type temp_src = LatticeTypes::V;
                  if (temp_1nn > temp_src) {
                    LatticeTypes::lat_type temp = temp_1nn;
                    temp_1nn = temp_src;
                    temp_src = temp;
                  }
                  // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                  const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                  vac_energy += _1nnMapToSwitch(bond);
                }
              }
            }
        
            // Traver all 2nn neighbour lattices, and calculate bond energy contribution.
            for (int b2 = 0; b2 < 6; b2++) {
              if ((_2nn_status >> b2) & 0x1) {
                // we assume that src_atom_type is single atom or vacancy.

                if (deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type <= LatticeTypes::Si) {
                  // it is vacancy or single atom.
                  LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type;
                  LatticeTypes::lat_type temp_src = LatticeTypes::V;
                  if (temp_2nn > temp_src) {
                    LatticeTypes::lat_type temp = temp_2nn;
                    temp_2nn = temp_src;
                    temp_src = temp;
                  }
                  // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                  const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                  vac_energy += _2nnMapToSwitch(bond);
                }
              }
            }
            first_ir = 0;
          }
          e_src = vac_energy;
//_______________________________________
          energy = 0.0;

          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }

          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          // if(is_first) *energy_temp2 = energy;
          // is_first = 0;
          e_des = energy;

          // double e_des = bonds::BondsCounter::count(&lattice_list, target_id, target_lattice.type);
          e_before = e_src + e_des;
          // energy_temp2[b] = energy;
        }

// ______________________________________________________________________________________________________________________________________



        // calculate system energy after transition.

        double e_after = 0;
        {
          // bonds energy of src lattice contributed by its 1nn/2nn neighbour lattice.
          double e_src, e_des;
          unsigned char _1nn_status = 0xFF;
              
          double energy = 0.0;

          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              LatticeTypes::lat_type temp_type = deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type;
              if (b2 == b){
                temp_type = LatticeTypes::V;
              }
              if (temp_type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = temp_type; 
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }

          // Traver all 2nn neighbour lattices, and calculate bond energy contribution.
          unsigned char _2nn_status = 0x2F;
          
          // lat_list->get2nn(source_id, _2nn_neighbour);
          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          e_src = energy;
//_______________________________________
          energy = 0.0;
          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              LatticeTypes::lat_type temp_type = deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type;
              if (b2 + b == 7) {
                temp_type = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
              }
              // if (d_target_lattice_1neighbour[tid * 64 + b * 8 + b2].id == d_lattices[tid].id){
              //   temp_type = d_lat_list[tid * 8 + b].type._type;
              // }
              if (temp_type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = temp_type;
                LatticeTypes::lat_type temp_src = LatticeTypes::V;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }
          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type;;
                LatticeTypes::lat_type temp_src = LatticeTypes::V;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          e_des = energy;
          e_after = e_src + e_des;
        }
        active_energy += (e_after - e_before) / 2;
        double rate = attempt_freq * __expf(-active_energy / (8.617330350E-5 * temperature));
        deviceVacancyInfo[tid].rates[b] = rate;
        totalRatesOfCurVac += rate;
      }
    }
    int delta = 1;
    for (int j = 0; j < 6; j++) {
      totalRatesOfCurVac += __shfl_down(totalRatesOfCurVac, delta, 64);
      delta += delta;
    }
    if (threadIdx.x % 64 == 0) atomicAdd(&tmp_sh, totalRatesOfCurVac);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(d_sum_rates, tmp_sh);
  }
}

__global__ void selectEventGPU(dev_Vacancy *deviceVacancyInfo, dev_nnLattice *deviceNeighbourInfo, 
                               dev_event *d_event, _type_lattice_count vac_total, double excepted_rate) {

  // 获取线程的全局ID
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0) {
    double rate_accumulator = 0;
    int tag = 1;
    for(_type_lattice_count i = 0; i < vac_total; i++) {
      if(tag == 1) {
        for (int rate_index = 0; rate_index < 8; rate_index++) {
          rate_accumulator += deviceVacancyInfo[i].rates[rate_index];
          if(rate_accumulator > excepted_rate) {
            if(tag == 1) {
              d_event->from_id = deviceVacancyInfo[i].id;
              d_event->to_id = deviceNeighbourInfo[i * NN_TOTAL + rate_index].id;
              d_event->to_type._type = deviceNeighbourInfo[i * NN_TOTAL + rate_index].type._type;
              g_recb_event_from = d_event->from_id;
              g_recb_event_to = d_event->to_id;
              __threadfence();
              g_recb_event_valid = 1u;
              tag = 0;
              break;
            }
          }
        } 
      } else break;
    }
  }
}

__global__ void apply_selected_event_kernel(const dev_event* ev, int sector_id, long int pair_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  dev_event e = ev[0];
  long int from_id = e.from_id;
  long int to_id = e.to_id;
  LatticeTypes::lat_type to_type = static_cast<LatticeTypes::lat_type>(e.to_type._type);

  hash_set_remove_atom_op(g_V_Hash, from_id);
  hash_set_insert_atom_op(g_V_Hash, to_id);

  switch (to_type) {
    case LatticeTypes::MoRe:
      hash_set_remove_atom_op(g_MoRe_Hash, to_id);
      hash_set_insert_atom_op(g_MoRe_Hash, from_id);
      break;
    case LatticeTypes::MoMo:
      hash_set_remove_atom_op(g_MoMo_Hash, to_id);
      hash_set_insert_atom_op(g_MoMo_Hash, from_id);
      break;
    case LatticeTypes::Re:
      hash_set_remove_atom_op(g_Re_Hash, to_id);
      hash_set_insert_atom_op(g_Re_Hash, from_id);
      break;
    case LatticeTypes::V:
      break;
    case LatticeTypes::Mo:
      break;
    default:
      break;
  }

  // 修复 roster/hash 不守恒：空位事件把 MoMo/MoRe 间隙对从 to_id 挪到 from_id 时，
  // 哈希表已更新，但持久花名册 g_Pair_Atoms 中该间隙对的 slot 仍指向 to_id（现已是空位 V），
  // 下次 recb 开头 getType(to_id)=V 会误判为 stale。这里同步：把指向 to_id 的 slot 改指 from_id。
  if (to_type == LatticeTypes::MoMo || to_type == LatticeTypes::MoRe) {
    if (g_Pair_Atoms && pair_count > 0) {
      for (long int i = 0; i < pair_count; ++i) {
        if (g_Pair_Atoms[i] == to_id) {
          g_Pair_Atoms[i] = from_id;
          break; // 一个间隙对只对应一个 slot，找到即可
        }
      }
    }
  }

  if (isGhostLat(to_id) && g_meta && g_comm_ghost_regions) {
    _type_lattice_id x = to_id % g_meta->size_x;
    _type_lattice_id y = (to_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = to_id / (g_meta->size_x * g_meta->size_y);
    int base = sector_id * 7;
    for (int k = 0; k < 7; k++) {
      const auto* r = &g_comm_ghost_regions[base + k];
      if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
        GPUHashSet* b = commBucketPtr(k);
        if (b) hash_set_insert_atom_op(b, to_id);
        break;
      }
    }
  }

  if (isSurfaceLat(from_id)) {
    _type_lattice_id x = from_id % g_meta->size_x;
    _type_lattice_id y = (from_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = from_id / (g_meta->size_x * g_meta->size_y);
    if (g_comm_surface_offsets && g_comm_surface_boxes && g_comm_surface_buckets_ptr) {
      for (int b = 0; b < 24; b++) {
        int start = g_comm_surface_offsets[b];
        int end = g_comm_surface_offsets[b + 1];
        for (int j = start; j < end; j++) {
          CommBox box = g_comm_surface_boxes[j];
          if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
            GPUHashSet* set = g_comm_surface_buckets_ptr[b];
            if (set) hash_set_insert_atom_op(set, from_id);
            break;
          }
        }
      }
    }
  }

  if (isSurfaceLat(to_id)) {
    _type_lattice_id x = to_id % g_meta->size_x;
    _type_lattice_id y = (to_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = to_id / (g_meta->size_x * g_meta->size_y);
    if (g_comm_surface_offsets && g_comm_surface_boxes && g_comm_surface_buckets_ptr) {
      for (int b = 0; b < 24; b++) {
        int start = g_comm_surface_offsets[b];
        int end = g_comm_surface_offsets[b + 1];
        for (int j = start; j < end; j++) {
          CommBox box = g_comm_surface_boxes[j];
          if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
            GPUHashSet* set = g_comm_surface_buckets_ptr[b];
            if (set) hash_set_insert_atom_op(set, to_id);
            break;
          }
        }
      }
    }
  }
}

void selectAndPerformEventGPU(double excepted_rand, int rank, _type_lattice_count step, int sect,
                              std::array<std::unordered_set<_type_lattice_id>, 7>& exchange_ghost, const unsigned int sector_id,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z,
                              std::array<comm::Region<comm::_type_lattice_size>, 56>& ghost_region){
  using namespace gpuSimulate;
  commSetGhostRegions(&ghost_region[0], 56);

  blockSize = 1;
  numBlocks = 1;
  hipLaunchKernelGGL(selectEventGPU, numBlocks, blockSize, 0, 0, 
                    deviceVacancyInfo,
                    deviceNeighbourInfo,
                    d_event,
                    currentVacCount,
                     *sum_rates * excepted_rand);

  CHECK_FOR_ERROR();
  if (gpu_comm_enabled()) {
    hipLaunchKernelGGL(apply_selected_event_kernel, 1, 1, 0, 0, d_event,
                       static_cast<int>(sector_id), pair_roster_scan_size);
    CHECK_FOR_ERROR();
    return;
  }

  HANDLE_HIP(hipMemcpy(h_event, d_event, sizeof(dev_event), hipMemcpyDeviceToHost));
  to_x = h_event->to_id % lattice_list->meta.size_x;
  to_y = (h_event->to_id / lattice_list->meta.size_x) % lattice_list->meta.size_y;
  to_z = h_event->to_id / (lattice_list->meta.size_x * lattice_list->meta.size_y);

  exchange_lattices(h_event);
  if(lattice_list->meta.isGhostLat(h_event->to_id)) {
    for(int i = 0; i < 7; i++) {
      if(ghost_region[sector_id * 7 + i].x_low <= to_x && to_x < ghost_region[sector_id * 7 + i].x_high && 
         ghost_region[sector_id * 7 + i].y_low <= to_y && to_y < ghost_region[sector_id * 7 + i].y_high && 
         ghost_region[sector_id * 7 + i].z_low <= to_z && to_z < ghost_region[sector_id * 7 + i].z_high) {
          if(!exchange_ghost[i].count(h_event->to_id)) exchange_ghost[i].emplace(h_event->to_id);
          break;
      }
    }
  }
  if (lattice_list->meta.isSurfaceLat(h_event->from_id)) add_surface(h_event->from_id, exchange_surface_x, exchange_surface_y, exchange_surface_z);
  if (lattice_list->meta.isSurfaceLat(h_event->to_id)) add_surface(h_event->to_id, exchange_surface_x, exchange_surface_y, exchange_surface_z);
}
__global__ void setGlobalPointer(dev_meta *meta,
                            GPUHashSet *MoRe_Hash,GPUHashSet *MoMo_Hash,
                            GPUHashSet *Re_Hash,GPUHashSet *V_Hash,GPUHashSet *Busy_Set_Hash,
                            GPUHashSet *surface_Hash,GPUHashSet *ghost_Hash,
                            long int *Pair_Atoms,
                            unsigned long long int* d_bitmap_lock,
                            hiprandStatePhilox4_32_10_t *rng_states
                            ){
  if (threadIdx.x == 0 && blockIdx.x == 0) {
  g_meta = meta;
  g_MoRe_Hash = MoRe_Hash;
  g_MoMo_Hash = MoMo_Hash;
  // printf("MoMo:%p\n",MoMo_Hash);
  // printf("g_MoMo:%p\n",g_MoMo_Hash);
  g_Re_Hash = Re_Hash;
  g_V_Hash = V_Hash;
  g_Busy_Set = Busy_Set_Hash;
  g_surface_Hash = surface_Hash;
  g_ghost_Hash = ghost_Hash;
  g_Pair_Atoms = Pair_Atoms;
  g_Global_Bitmap_Lock = d_bitmap_lock;
  g_rng_states = rng_states;

    // printHashTableKernel(g_MoMo_Hash,200,false);
    // printDevMetaKernel(g_meta);


  }
}

// 每个 walker 线程初始化一条独立的 Philox 随机数流：
// 同一 seed + 子序列号 tid => 各流互相独立且整跑可复现；整个模拟只调用一次。
__global__ void initRngStates(long int n, unsigned long long seed) {
  long int tid = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  if (tid < n) {
    hiprand_init(seed, (unsigned long long)tid, 0, &g_rng_states[tid]);
  }
}

__global__ void setCommGhostPointers(GPUHashSet *b0, GPUHashSet *b1, GPUHashSet *b2, GPUHashSet *b3, GPUHashSet *b4, GPUHashSet *b5, GPUHashSet *b6,
                                     comm::Region<comm::_type_lattice_size> *ghost_regions) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_comm_ghost_bucket0 = b0;
    g_comm_ghost_bucket1 = b1;
    g_comm_ghost_bucket2 = b2;
    g_comm_ghost_bucket3 = b3;
    g_comm_ghost_bucket4 = b4;
    g_comm_ghost_bucket5 = b5;
    g_comm_ghost_bucket6 = b6;
    if (ghost_regions) g_comm_ghost_regions = ghost_regions;
  }
}

static __device__ __forceinline__ GPUHashSet* commBucketPtr(int idx) {
  switch (idx) {
    case 0: return g_comm_ghost_bucket0;
    case 1: return g_comm_ghost_bucket1;
    case 2: return g_comm_ghost_bucket2;
    case 3: return g_comm_ghost_bucket3;
    case 4: return g_comm_ghost_bucket4;
    case 5: return g_comm_ghost_bucket5;
    case 6: return g_comm_ghost_bucket6;
    default: return nullptr;
  }
}

__global__ void bucketizeGhostIdsKernel(const long int *ids, int n, int sector_id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!g_meta || !g_comm_ghost_regions) return;
  long int id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  int base = sector_id * 7;
  for (int k = 0; k < 7; k++) {
    const auto* r = &g_comm_ghost_regions[base + k];
    if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
      GPUHashSet* b = commBucketPtr(k);
      if (b) hash_set_insert_atom_op(b, id);
      break;
    }
  }
}

__global__ void insertGhostIdKernel(long int id, int sector_id) {
  if (!g_meta || !g_comm_ghost_regions) return;
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  int base = sector_id * 7;
  for (int k = 0; k < 7; k++) {
    const auto* r = &g_comm_ghost_regions[base + k];
    if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
      GPUHashSet* b = commBucketPtr(k);
      if (b) hash_set_insert_atom_op(b, id);
      break;
    }
  }
}

__global__ void compactHashSetKernel(GPUHashSet *set, long int *out, int *out_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!set) return;
  if (i >= set->capacity) return;
  long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  int pos = atomicAdd(out_count, 1);
  out[pos] = key;
}

__global__ void compactHashSetBoundedKernel(GPUHashSet *set, long int *out, int *out_count, int max_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!set || !out || !out_count) return;
  if (i >= set->capacity) return;
  long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  int pos = atomicAdd(out_count, 1);
  if (pos < 0 || pos >= max_out) return;
  out[pos] = key;
}

void commSetGhostRegions(const comm::Region<comm::_type_lattice_size> *ghost_region, int n) {
  if (!ghost_region) return;
  if (n <= 0) return;
  if (gpuSimulate::d_comm_ghost_regions == nullptr) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_ghost_regions), static_cast<size_t>(n) * sizeof(comm::Region<comm::_type_lattice_size>)));
  }
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_ghost_regions, ghost_region, static_cast<size_t>(n) * sizeof(comm::Region<comm::_type_lattice_size>), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommGhostPointers, 1, 1, 0, 0,
                     gpuSimulate::comm_ghost_bucket0 ? gpuSimulate::comm_ghost_bucket0->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket1 ? gpuSimulate::comm_ghost_bucket1->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket2 ? gpuSimulate::comm_ghost_bucket2->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket3 ? gpuSimulate::comm_ghost_bucket3->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket4 ? gpuSimulate::comm_ghost_bucket4->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket5 ? gpuSimulate::comm_ghost_bucket5->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket6 ? gpuSimulate::comm_ghost_bucket6->device_ptr() : nullptr,
                     gpuSimulate::d_comm_ghost_regions);
  hipDeviceSynchronize();
}

void commInitGhostBuckets(int capacity) {
  if (capacity <= 0) return;
  gpuSimulate::comm_ghost_capacity = capacity * 2;
  if (!gpuSimulate::comm_ghost_bucket0) gpuSimulate::comm_ghost_bucket0 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket1) gpuSimulate::comm_ghost_bucket1 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket2) gpuSimulate::comm_ghost_bucket2 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket3) gpuSimulate::comm_ghost_bucket3 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket4) gpuSimulate::comm_ghost_bucket4 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket5) gpuSimulate::comm_ghost_bucket5 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket6) gpuSimulate::comm_ghost_bucket6 = new HIPHashSet(capacity);

  gpuSimulate::comm_ghost_capacity = gpuSimulate::comm_ghost_bucket0->get_capacity();
  hipLaunchKernelGGL(setCommGhostPointers, 1, 1, 0, 0,
                     gpuSimulate::comm_ghost_bucket0->device_ptr(),
                     gpuSimulate::comm_ghost_bucket1->device_ptr(),
                     gpuSimulate::comm_ghost_bucket2->device_ptr(),
                     gpuSimulate::comm_ghost_bucket3->device_ptr(),
                     gpuSimulate::comm_ghost_bucket4->device_ptr(),
                     gpuSimulate::comm_ghost_bucket5->device_ptr(),
                     gpuSimulate::comm_ghost_bucket6->device_ptr(),
                     gpuSimulate::d_comm_ghost_regions);
  hipDeviceSynchronize();
}

void commClearGhostBuckets() {
  if (gpuSimulate::comm_ghost_bucket0) gpuSimulate::comm_ghost_bucket0->clear();
  if (gpuSimulate::comm_ghost_bucket1) gpuSimulate::comm_ghost_bucket1->clear();
  if (gpuSimulate::comm_ghost_bucket2) gpuSimulate::comm_ghost_bucket2->clear();
  if (gpuSimulate::comm_ghost_bucket3) gpuSimulate::comm_ghost_bucket3->clear();
  if (gpuSimulate::comm_ghost_bucket4) gpuSimulate::comm_ghost_bucket4->clear();
  if (gpuSimulate::comm_ghost_bucket5) gpuSimulate::comm_ghost_bucket5->clear();
  if (gpuSimulate::comm_ghost_bucket6) gpuSimulate::comm_ghost_bucket6->clear();
}

void commBuildGhostBucketsFromList(int sector_id, const long int *d_ids, int n) {
  if (!d_ids) return;
  if (n <= 0) return;
  if (gpuSimulate::d_comm_ghost_regions == nullptr) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(bucketizeGhostIdsKernel, blockNum, blockSize, 0, 0, d_ids, n, sector_id);
  hipDeviceSynchronize();
}

void commInsertGhostId(int sector_id, long int id) {
  if (gpuSimulate::d_comm_ghost_regions == nullptr) return;
  hipLaunchKernelGGL(insertGhostIdKernel, 1, 1, 0, 0, id, sector_id);
}

void commCompactGhostBucket(int bucket, long int *d_out, int *d_count) {
  if (!d_out || !d_count) return;
  HANDLE_HIP(hipMemset(d_count, 0, sizeof(int)));
  HIPHashSet *b = nullptr;
  switch (bucket) {
    case 0: b = gpuSimulate::comm_ghost_bucket0; break;
    case 1: b = gpuSimulate::comm_ghost_bucket1; break;
    case 2: b = gpuSimulate::comm_ghost_bucket2; break;
    case 3: b = gpuSimulate::comm_ghost_bucket3; break;
    case 4: b = gpuSimulate::comm_ghost_bucket4; break;
    case 5: b = gpuSimulate::comm_ghost_bucket5; break;
    case 6: b = gpuSimulate::comm_ghost_bucket6; break;
    default: b = nullptr; break;
  }
  if (!b) return;
  int cap = gpuSimulate::comm_ghost_capacity;
  int blockSize = 256;
  int blockNum = (cap + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(compactHashSetKernel, blockNum, blockSize, 0, 0, b->device_ptr(), d_out, d_count);
  hipDeviceSynchronize();
}

int commGhostCapacity() {
  return gpuSimulate::comm_ghost_capacity;
}

__global__ void setCommSurfaceMeta(GPUHashSet **bucket_ptrs, CommBox *boxes, int *offsets) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_comm_surface_buckets_ptr = bucket_ptrs;
    g_comm_surface_boxes = boxes;
    g_comm_surface_offsets = offsets;
  }
}

__global__ void bucketizeSurfaceIdsKernel(const long int *ids, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long int id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
  for (int b = 0; b < 24; b++) {
    int start = g_comm_surface_offsets[b];
    int end = g_comm_surface_offsets[b + 1];
    for (int j = start; j < end; j++) {
      CommBox box = g_comm_surface_boxes[j];
      if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
        GPUHashSet* set = g_comm_surface_buckets_ptr[b];
        if (set) hash_set_insert_atom_op(set, id);
        break;
      }
    }
  }
}

__global__ void insertSurfaceIdKernel(long int id) {
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
  for (int b = 0; b < 24; b++) {
    int start = g_comm_surface_offsets[b];
    int end = g_comm_surface_offsets[b + 1];
    for (int j = start; j < end; j++) {
      CommBox box = g_comm_surface_boxes[j];
      if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
        GPUHashSet* set = g_comm_surface_buckets_ptr[b];
        if (set) hash_set_insert_atom_op(set, id);
        break;
      }
    }
  }
}

void commSetSurfaceBoxes(const CommBox *boxes, int boxes_count, const int *offsets, int offsets_count) {
  if (boxes_count < 0 || offsets_count <= 0) return;
  if (boxes_count > 0) {
    if (gpuSimulate::d_comm_surface_boxes && gpuSimulate::comm_surface_boxes_cap < boxes_count) {
      HANDLE_HIP(hipFree(gpuSimulate::d_comm_surface_boxes));
      gpuSimulate::d_comm_surface_boxes = nullptr;
      gpuSimulate::comm_surface_boxes_cap = 0;
    }
    if (!gpuSimulate::d_comm_surface_boxes) {
      HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_boxes), static_cast<size_t>(boxes_count) * sizeof(CommBox)));
      gpuSimulate::comm_surface_boxes_cap = boxes_count;
    }
    HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_boxes, boxes, static_cast<size_t>(boxes_count) * sizeof(CommBox), hipMemcpyHostToDevice));
  }
  if (gpuSimulate::d_comm_surface_offsets && gpuSimulate::comm_surface_offsets_cap < offsets_count) {
    HANDLE_HIP(hipFree(gpuSimulate::d_comm_surface_offsets));
    gpuSimulate::d_comm_surface_offsets = nullptr;
    gpuSimulate::comm_surface_offsets_cap = 0;
  }
  if (!gpuSimulate::d_comm_surface_offsets) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_offsets), static_cast<size_t>(offsets_count) * sizeof(int)));
    gpuSimulate::comm_surface_offsets_cap = offsets_count;
  }
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_offsets, offsets, static_cast<size_t>(offsets_count) * sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommSurfaceMeta, 1, 1, 0, 0, gpuSimulate::d_comm_surface_bucket_ptrs, gpuSimulate::d_comm_surface_boxes, gpuSimulate::d_comm_surface_offsets);
  hipDeviceSynchronize();
}

void commInitSurfaceBuckets(int capacity) {
  if (capacity <= 0) return;
  gpuSimulate::comm_surface_capacity = capacity * 2;
  bool flag = false;
  for (int i = 0; i < 24; i++) {
    if (!gpuSimulate::comm_surface_buckets[i]){
      gpuSimulate::comm_surface_buckets[i] = new HIPHashSet(capacity);
      if(!flag){
        gpuSimulate::comm_surface_capacity = gpuSimulate::comm_surface_buckets[i]->get_capacity();
        flag=true;
      }
    } 
  }
  

  if (!gpuSimulate::d_comm_surface_bucket_ptrs) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_bucket_ptrs), 24 * sizeof(GPUHashSet*)));
  }
  GPUHashSet* h_ptrs[24];
  for (int i = 0; i < 24; i++) h_ptrs[i] = gpuSimulate::comm_surface_buckets[i]->device_ptr();
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_bucket_ptrs, h_ptrs, 24 * sizeof(GPUHashSet*), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommSurfaceMeta, 1, 1, 0, 0, gpuSimulate::d_comm_surface_bucket_ptrs, gpuSimulate::d_comm_surface_boxes, gpuSimulate::d_comm_surface_offsets);
  hipDeviceSynchronize();
}

void commClearSurfaceBuckets() {
  for (int i = 0; i < 24; i++) {
    if (gpuSimulate::comm_surface_buckets[i]) gpuSimulate::comm_surface_buckets[i]->clear();
  }
}

void commClearSurfaceSector(int sect) {
  if (sect < 0 || sect >= 8) return;
  int base = sect * 3;
  for (int i = 0; i < 3; i++) {
    int idx = base + i;
    if (gpuSimulate::comm_surface_buckets[idx]) gpuSimulate::comm_surface_buckets[idx]->clear();
  }
}

void commBuildSurfaceBucketsFromList(const long int *d_ids, int n) {
  if (!d_ids) return;
  if (n <= 0) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(bucketizeSurfaceIdsKernel, blockNum, blockSize, 0, 0, d_ids, n);
  hipDeviceSynchronize();
}

void commInsertSurfaceId(long int id) {
  hipLaunchKernelGGL(insertSurfaceIdKernel, 1, 1, 0, 0, id);
}

void commCompactSurfaceBucket(int bucket, long int *d_out, int *d_count) {
  if (!d_out || !d_count) return;
  if (bucket < 0 || bucket >= 24) return;
  HANDLE_HIP(hipMemset(d_count, 0, sizeof(int)));
  HIPHashSet *b = gpuSimulate::comm_surface_buckets[bucket];
  if (!b) return;
  int cap = gpuSimulate::comm_surface_capacity;
  int blockSize = 256;
  int blockNum = (cap + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(compactHashSetKernel, blockNum, blockSize, 0, 0, b->device_ptr(), d_out, d_count);
  hipDeviceSynchronize();
}

int commSurfaceCapacity() {
  return gpuSimulate::comm_surface_capacity;
}

__device__ int isGhostLat(_type_lattice_id lid);
__device__ int isSurfaceLat(_type_lattice_id lid);

__global__ void forwardFromRecvPayloadKernel(const ChangeLattice_GPU *payload, int n, int sector_id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ChangeLattice_GPU c = payload[i];
  long int id = getId(c.x, c.y, c.z);
  if (isGhostLat(id) && g_comm_ghost_regions) {
    int base = sector_id * 7;
    for (int k = 0; k < 7; k++) {
      const auto* r = &g_comm_ghost_regions[base + k];
      if (r->x_low <= c.x && c.x < r->x_high && r->y_low <= c.y && c.y < r->y_high && r->z_low <= c.z && c.z < r->z_high) {
        GPUHashSet* b = commBucketPtr(k);
        if (b) hash_set_insert_atom_op(b, id);
        break;
      }
    }
  }
  if (isSurfaceLat(id)) {
    if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
    for (int b = 0; b < 24; b++) {
      int start = g_comm_surface_offsets[b];
      int end = g_comm_surface_offsets[b + 1];
      for (int j = start; j < end; j++) {
        CommBox box = g_comm_surface_boxes[j];
        if (box.x_low <= c.x && c.x < box.x_high && box.y_low <= c.y && c.y < box.y_high && box.z_low <= c.z && c.z < box.z_high) {
          GPUHashSet* set = g_comm_surface_buckets_ptr[b];
          if (set) hash_set_insert_atom_op(set, id);
          break;
        }
      }
    }
  }
}

void commForwardFromRecvPayload(const ChangeLattice_GPU *d_payload, int n, int sector_id) {
  if (!d_payload) return;
  if (n <= 0) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(forwardFromRecvPayloadKernel, blockNum, blockSize, 0, 0, d_payload, n, sector_id);
}
__global__ void printpointer(){
  if(g_MoMo_Hash == nullptr)
    printf("is a null\n");
  else
    printf("not null\n");
}
// noinline：本函数内联 4 次全桶扫描的 hash_set_contains，若在 walker 主循环里
// 全部内联（目标选择 1 次 + 锁内 DCL 2 次），函数体规模会把后端寄存器分配打爆。
// 正确性优先；后续性能优化再评估是否放开。
__device__ __noinline__ LatticeTypes::lat_type getType(const long int& id){
      // printf("START GET TYPE");
      if(hash_set_contains(g_MoRe_Hash,id))       return LatticeTypes::MoRe;
      else if(hash_set_contains(g_MoMo_Hash,id))  return LatticeTypes::MoMo;
      else if(hash_set_contains(g_Re_Hash,id))    return LatticeTypes::Re;
      else if(hash_set_contains(g_V_Hash,id))     return LatticeTypes::V;
      else return LatticeTypes::Mo;

}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z){
  _type_lattice_id res = x + g_meta->size_x * (y + z * g_meta->size_y);
  return (res<0 ? 0 : res);
  // return x + g_meta->size_x * (y + z * g_meta->size_y);
}
__device__ void getnnlattice(_type_lattice_id latti_id, dev_Lattice *_1nn_list_lattice){
  _1nn_list_lattice->id = latti_id;
  if(hash_set_contains(g_MoRe_Hash,latti_id))        _1nn_list_lattice->type = LatticeTypes::MoRe;
  else if(hash_set_contains(g_MoMo_Hash,latti_id))   _1nn_list_lattice->type = LatticeTypes::MoMo;
  else if(hash_set_contains(g_Re_Hash,latti_id))     _1nn_list_lattice->type = LatticeTypes::Re;
  else if(hash_set_contains(g_V_Hash,latti_id))      _1nn_list_lattice->type = LatticeTypes::V;
  else _1nn_list_lattice->type = LatticeTypes::Mo;

}
#if 0
// 旧单表校验实现已停用：它无法表达“Mo 是所有显式哈希表均未命中”。
// recb 当前的锁内校验直接比较 getType(id) 与期望类型。
__device__ bool single_chk_type(long int id, LatticeTypes::lat_type dcl_type) {
    GPUHashSet* target_hash = nullptr;

    // 1. 根据目标类型选择哈希表
    if (dcl_type == LatticeTypes::MoRe) {
        target_hash = g_MoRe_Hash;
    } else if (dcl_type == LatticeTypes::MoMo) {
        target_hash = g_MoMo_Hash;
    } else if (dcl_type == LatticeTypes::Re) {
        target_hash = g_Re_Hash;
    } else if (dcl_type == LatticeTypes::V) {
        target_hash = g_V_Hash;
    }

    if (target_hash == nullptr) {
        return false;
    }

    return hash_set_contains(target_hash, id);
}
#endif
// 实验性的 tile 协作校验；当前 recb 正确性基线不调用它。
template <typename TileType>
__device__ bool tiled_chk_type(long int id, LatticeTypes::lat_type dcl_type, TileType tile, bool active) {
    // LatticeTypes::lat_type final_result = LatticeTypes::Mo; 
    bool final_found = false;
    uint64_t work_mask = tile.ballot(active);
    
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int target_id = (long int)tile.shfl((unsigned long long int)id, leader);
        // LatticeTypes::lat_type target_type = (LatticeTypes::lat_type)tile.shfl((int)dcl_type, leader);
        LatticeTypes::lat_type target_type = static_cast<LatticeTypes::lat_type>(tile.shfl(static_cast<int>(dcl_type), leader));
        GPUHashSet* target_hash = nullptr;
        if (target_type == LatticeTypes::MoRe) {
            target_hash = g_MoRe_Hash;
        } else if (target_type == LatticeTypes::MoMo) {
            target_hash = g_MoMo_Hash;
        } else if (target_type == LatticeTypes::Re) {
            target_hash = g_Re_Hash;
        } else if (target_type == LatticeTypes::V) {
            target_hash = g_V_Hash;
        }

        bool current_found = false;
        if (target_type == LatticeTypes::Mo) {
          // Mo 没有独立哈希表：与 getType() 保持一致，只有当目标不属于
          // 任何显式存储的非 Mo 类型时，才能确认它仍然是 Mo。
          const bool is_more = collective_bucket_contains(g_MoRe_Hash, target_id, tile);
          const bool is_momo = collective_bucket_contains(g_MoMo_Hash, target_id, tile);
          const bool is_re = collective_bucket_contains(g_Re_Hash, target_id, tile);
          const bool is_v = collective_bucket_contains(g_V_Hash, target_id, tile);
          current_found = !(is_more || is_momo || is_re || is_v);
        } else if (target_hash != nullptr) {
          current_found = collective_bucket_contains(target_hash, target_id, tile);
        }

        if (tile.thread_rank() == leader) {
            final_found = current_found;
        }
        
        work_mask &= ~(1ULL << leader);
    }
    return final_found; // 只有原本 active 的线程会拿到有效值
}


template <typename TileType>
__device__ LatticeTypes::lat_type tiled_get_type(long int id, TileType tile, bool active) {
    LatticeTypes::lat_type final_result = LatticeTypes::Mo; 
    uint64_t work_mask = tile.ballot(active);
    
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int target_id = (long int)tile.shfl((unsigned long long int)id, leader); 
        LatticeTypes::lat_type leader_res = LatticeTypes::Mo;

        if (collective_bucket_contains(g_MoRe_Hash, target_id, tile)) {
            leader_res = LatticeTypes::MoRe;
        } else if (collective_bucket_contains(g_MoMo_Hash, target_id, tile)) {
            leader_res = LatticeTypes::MoMo;
        } else if (collective_bucket_contains(g_Re_Hash, target_id, tile)) {
            leader_res = LatticeTypes::Re;
        } else if (collective_bucket_contains(g_V_Hash, target_id, tile)) {
            leader_res = LatticeTypes::V;
        }

        if (tile.thread_rank() == leader) {
            final_result = leader_res;
        }
        
        work_mask &= ~(1ULL << leader);
    }
    return final_result; // 只有原本 active 的线程会拿到有效值
}
//只计算邻居，不进行哈希查找 getId返回值是-1的话没考虑啊？？
__device__ void get1nn_ids(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z, _type_lattice_id* nn_list) {
  if(x % 2 == 0) {
    nn_list[0] = getId(x - 1, y - 1, z - 1);
    nn_list[1] = getId(x - 1, y - 1, z);
    nn_list[2] = getId(x - 1, y, z - 1);
    nn_list[3] = getId(x - 1, y, z);
    nn_list[4] = getId(x + 1, y - 1, z - 1);
    nn_list[5] = getId(x + 1, y - 1, z);
    nn_list[6] = getId(x + 1, y, z - 1);
    nn_list[7] = getId(x + 1, y, z);
  } else {
    nn_list[0] = getId(x - 1, y, z);
    nn_list[1] = getId(x - 1, y, z + 1);
    nn_list[2] = getId(x - 1, y + 1, z);
    nn_list[3] = getId(x - 1, y + 1, z + 1);
    nn_list[4] = getId(x + 1, y, z);
    nn_list[5] = getId(x + 1, y, z + 1);
    nn_list[6] = getId(x + 1, y + 1, z);
    nn_list[7] = getId(x + 1, y + 1, z + 1);
  }
}
__device__ void get1nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                  dev_Lattice *_1nn_list){
  _type_lattice_id latti_id;
  if(x % 2 == 0){
    latti_id = getId(x - 1, y - 1, z - 1);
    getnnlattice(latti_id, &_1nn_list[0]);
    latti_id = getId(x - 1, y - 1, z);
    getnnlattice(latti_id, &_1nn_list[1]);
    latti_id = getId(x - 1, y, z - 1);
    getnnlattice(latti_id, &_1nn_list[2]);
    latti_id = getId(x - 1, y, z);
    getnnlattice(latti_id, &_1nn_list[3]);
    latti_id = getId(x + 1, y - 1, z - 1);
    getnnlattice(latti_id, &_1nn_list[4]);
    latti_id = getId(x + 1, y - 1, z);
    getnnlattice(latti_id, &_1nn_list[5]);
    latti_id = getId(x + 1, y, z - 1);
    getnnlattice(latti_id, &_1nn_list[6]);
    latti_id = getId(x + 1, y, z);
    getnnlattice(latti_id, &_1nn_list[7]);
  }else {
    latti_id = getId(x - 1, y, z);
    getnnlattice(latti_id, &_1nn_list[0]);
    latti_id = getId(x - 1, y, z + 1);
    getnnlattice(latti_id, &_1nn_list[1]);
    latti_id = getId(x - 1, y + 1, z);
    getnnlattice(latti_id, &_1nn_list[2]);
    latti_id = getId(x - 1, y + 1, z + 1);
    getnnlattice(latti_id, &_1nn_list[3]);
    latti_id = getId(x + 1, y, z);
    getnnlattice(latti_id, &_1nn_list[4]);
    latti_id = getId(x + 1, y, z + 1);
    getnnlattice(latti_id, &_1nn_list[5]);
    latti_id = getId(x + 1, y + 1, z);
    getnnlattice(latti_id, &_1nn_list[6]);
    latti_id = getId(x + 1, y + 1, z + 1);
    getnnlattice(latti_id, &_1nn_list[7]);
  }
}
__device__ void get2nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                              dev_Lattice *_2nn_list){
  _type_lattice_id latti_id;
  latti_id = getId(x - 2, y, z);
  getnnlattice(latti_id, &_2nn_list[0]);
  // _2nn_list[0] = &_lattices[z + 0][y + 0][x - 2];
  latti_id = getId(x, y - 1, z);
  getnnlattice(latti_id, &_2nn_list[1]);
  // _2nn_list[1] = &_lattices[z + 0][y - 1][x + 0];
  latti_id = getId(x, y, z - 1);
  getnnlattice(latti_id, &_2nn_list[2]);
  // _2nn_list[2] = &_lattices[z - 1][y + 0][x + 0];
  latti_id = getId(x, y, z + 1);
  getnnlattice(latti_id, &_2nn_list[3]);
  // _2nn_list[3] = &_lattices[z + 1][y + 0][x + 0];
  latti_id = getId(x, y + 1, z);
  getnnlattice(latti_id, &_2nn_list[4]);
  // _2nn_list[4] = &_lattices[z + 0][y + 1][x + 0];
  latti_id = getId(x + 2, y, z);
  getnnlattice(latti_id, &_2nn_list[5]);
  // _2nn_list[5] = &_lattices[z + 0][y + 0][x + 2];
}

__device__ int randomInt(unsigned long long seed, int min, int max) {
    // // 获取线程的唯一标识符
    // int threadSeed = threadIdx.x + blockIdx.x * blockDim.x; 

    // // 初始化每个线程的hiprandState
    // hiprandState state;
    // hiprand_init(seed + threadSeed, threadIdx.x, 0, &state); 

    // // 生成一个随机数，范围在[min, max]之间
    // int randNum = hiprand(&state) % (max - min + 1) + min;

    // return randNum;
    return 1;
}

__device__ void getRandomLattice(const _type_lattice_coord& x, const _type_lattice_coord& y, const _type_lattice_coord& z,
                                _type_lattice_coord& temp_x, _type_lattice_coord& temp_y, _type_lattice_coord& temp_z, const int& randomValue){
  // if(randomValue < 0 || randomValue >= 14){
  //   printf("wrong random value:%d\n",randomValue);
  // }
  switch (randomValue)
  {
    case 0:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y - 1;
        temp_z = z - 1;
      } else {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z;
      }
      break;
    case 1:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y - 1;
        temp_z = z;
      } else {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z + 1;
      }
      break;
    case 2:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z - 1;
      } else {
        temp_x = x - 1;
        temp_y = y + 1;
        temp_z = z;
      }
      break;
    case 3:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z;
      } else {
        temp_x = x - 1;
        temp_y = y + 1;
        temp_z = z + 1;
      }
      break;
    case 4:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y - 1;
        temp_z = z - 1;
      } else {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z;
      }
      break;
    case 5:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y - 1;
        temp_z = z;
      } else {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z + 1;
      }
      break;
    case 6:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z - 1;
      } else {
        temp_x = x + 1;
        temp_y = y + 1;
        temp_z = z;
      }
      break;
    case 7:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z;
      } else {
        temp_x = x + 1;
        temp_y = y + 1;
        temp_z = z + 1;
      }
      break;
    case 8:
      temp_x = x - 2;
      temp_y = y;
      temp_z = z;
      break;
    case 9:
      temp_x = x;
      temp_y = y - 1;
      temp_z = z;
      break;
    case 10:
      temp_x = x;
      temp_y = y;
      temp_z = z - 1;
      break;
    case 11:
      temp_x = x;
      temp_y = y;
      temp_z = z + 1;
      break;
    case 12:
      temp_x = x;
      temp_y = y + 1;
      temp_z = z;
      break;
    case 13:
      temp_x = x + 2;
      temp_y = y;
      temp_z = z;
      break;
    default:
      // printf("wrong random period,%d\n",randomValue);
      // abort();
      break;
  }
}

__device__ float randomFloat(unsigned long long seed) {
    // 获取线程的唯一标识符
    unsigned long long seed1 = clock();
    
    // 定义并初始化hiprandState
    hiprandState state;
    hiprand_init(seed + seed1, threadIdx.x, 0, &state);  // 初始化状态
    
    // 生成 0 到 1 之间的浮点数
    float randNum = hiprand_uniform(&state);  // 使用 hiprand_uniform 生成浮点数
    return randNum;
    // return 0;
}


__device__ void getCoordByLId(_type_lattice_id lid, _type_lattice_coord *x, _type_lattice_coord *y,
                            _type_lattice_coord *z){
   _type_lattice_id lid2 = lid;
    *x = lid % g_meta->size_x;
    lid = lid /  g_meta->size_x;
    *y = lid %  g_meta->size_y;
    *z = lid /  g_meta->size_y;
}
__device__ int isGhostLat(_type_lattice_id lid){
  _type_lattice_coord x = 0, y = 0, z = 0;
  getCoordByLId(lid, &x, &y, &z);
  x -= g_meta->ghost_x;
  y -= g_meta->ghost_y;
  z -= g_meta->ghost_z;

  return x < 0 || y < 0 || z < 0 || x >= g_meta->box_x || y >= g_meta->box_y || z >= g_meta->box_z;

}
// __device__ int isGhostLat(_type_lattice_id lid){
//   _type_lattice_size x = lid%g_meta->size_x;
//   _type_lattice_size y = (lid / g_meta->size_x) % g_meta->size_y;
//   _type_lattice_size z = lid / (g_meta->size_x * g_meta->size_y);
//   if(x<g_meta->ghost_x)
//     return 1;
//   if(x>=g_meta->ghost_x+g_meta->box_x*2)
//     return 1;

//   if(y<g_meta->ghost_y)
//     return 1;
//   if(x>=g_meta->ghost_y+g_meta->box_y)
//     return 1;

//   if(y<g_meta->ghost_z)
//     return 1;
//   if(x>=g_meta->ghost_z+g_meta->box_z)
//     return 1;

//   return 0;
// }

__device__ int isSurfaceLat(_type_lattice_id lid){
   _type_lattice_coord x = 0, y = 0, z = 0;
    getCoordByLId(lid, &x, &y, &z);
    x -= g_meta->ghost_x;
    y -= g_meta->ghost_y;
    z -= g_meta->ghost_z;
    return (0 <= x && x < g_meta->ghost_x) ||(0 <= y && y < g_meta->ghost_y) || (0 <= z && z < g_meta->ghost_z) || (x >= (g_meta->box_x - g_meta->ghost_x) && x < g_meta->box_x) || (y >= (g_meta->box_y - g_meta->ghost_y) && y < g_meta->box_y) || (z >= (g_meta->box_z - g_meta->ghost_z) && z < g_meta->box_z);
}

__global__ void calcExchangePairs_test(int sector_id,int arr_size){
    _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= arr_size)
      return;    

    int flag = false;
    _type_lattice_id id = g_Pair_Atoms[tid];
      // 1. 插入测试
      int test_key = id+1; // 用线程ID作为测试的键
      int insert_result = hash_set_insert_atom_op(g_MoMo_Hash, test_key);
      if (insert_result == 1) {
          printf("Thread %d: Successfully inserted key %d into g_Busy_Set.\n", tid, test_key);
          flag++;
      } else {
          printf("Thread %d: Failed to insert key %d into g_Busy_Set.\n", tid, test_key);
          flag=0;
      }

      // 2. 查找测试
      int contains_result = hash_set_contains(g_MoMo_Hash, test_key);
      if (contains_result == 1) {
          flag++;
          printf("Thread %d: Key %d found in g_Busy_Set.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Key %d NOT found in g_Busy_Set.\n", tid, test_key);
      }

      // 3. 删除测试
      int remove_result = hash_set_remove_atom_op(g_MoMo_Hash, test_key);
      if (remove_result == 1) {
          flag++;
          printf("Thread %d: Successfully removed key %d from g_Busy_Set.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Failed to remove key %d from g_Busy_Set.\n", tid, test_key);
      }

      // 验证删除后是否仍然存在
      contains_result = hash_set_contains(g_MoMo_Hash, test_key);
      if (contains_result == 1) {
          flag++;
          printf("Thread %d: Key %d still found in g_Busy_Set after removal.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Key %d NOT found in g_Busy_Set after removal.\n", tid, test_key);
      }
  }

// __global__ void func_check(int sector_id,int arr_size){
//   _type_lattice_id id = g_Pair_Atoms[0];
//   getType(id);
//   geti
// }

__global__ void update_Pair_Atoms(int arr_size){
    //这个函数用于更新g_pair_atoms
    /*
      一共有两种情况需要更新pair_Atoms
      1,每次calcExchangePairs计算结束后如果发生了交换，则交换的位置要被更新到此函数中
      2,通信过程中需要更新
      这个函数应用于第二种更新，第一种更新直接放到函数中执行就行
    */
    //todo 更新函数有问题
    for(int i=0;i<arr_size;i++)
      g_Pair_Atoms[i] = -1; 
    int index = 0;
    long int momo=0,more=0;
    // printf("-------------MOMO_DATA-------------\n");
    for(int i=0;i<g_MoMo_Hash->capacity;i++){
      if(g_MoMo_Hash->keys[i]!=g_MoMo_Hash->empty_flag && g_MoMo_Hash->keys[i]!=g_MoMo_Hash->tombstone_flag){
        g_Pair_Atoms[index++]=g_MoMo_Hash->keys[i];
        if(!isGhostLat(g_MoMo_Hash->keys[i])){
          // printf("%ld\t",g_MoMo_Hash->keys[i]);
          momo++;
        }
          
      }
    }
    // printf("\n");
    //  printf("-------------MORe_DATA-------------\n");
    for(int i=0;i<g_MoRe_Hash->capacity;i++){
      if(g_MoRe_Hash->keys[i]!=g_MoRe_Hash->empty_flag && g_MoRe_Hash->keys[i]!=g_MoRe_Hash->tombstone_flag){
        g_Pair_Atoms[index++]=g_MoRe_Hash->keys[i];
        if(!isGhostLat(g_MoRe_Hash->keys[i])){
          // printf("%ld\t",g_MoRe_Hash->keys[i]);
          more++;
        }
      }
    }
    // printf("\n");
    // printf("momo num is :%ld\t",momo);
    // printf("more num is:%ld\n",more);
    // printf("AFTER UPDATE atom_num is:%ld/%ld\n",index,arr_size);

}
// Insert-Then-Remove 
template <typename TileType>
__device__ __noinline__ void perform_batch_update_tiled(AtomTx& tx, TileType tile) {
  tx.success = false; // 默认失败
  if (tx.op_type != 0 && tx.src_id == tx.dst_id) {
  
      // if (tile.thread_rank() == 0) {
      //     printf("⚠️ [SELF COLLISION] ... \n");
      // }
      tx.op_type = 0; 
  }
    bool is_type1 = (tx.op_type == 1);
  if (tile.any(is_type1)) {
      // Step 1: 尝试插入 Dst (必须是空位)
      int ins_res = tiled_insert_op(g_MoMo_Hash, tx.dst_id, tile, is_type1,g_MoMo_Hash->num_buckets);
      bool dst_inserted = (is_type1 && ins_res == 1);

      // Step 2: 只有插入成功，才移除 Src
      int rem_res = 0;
      if (tile.any(dst_inserted)) {
          rem_res = tiled_remove_op(g_MoMo_Hash, tx.src_id, tile, dst_inserted,g_MoMo_Hash->num_buckets);
      }
      bool src_removed = (dst_inserted && rem_res == 1);

      // Step 3: 异常处理 (Insert 成功但 Remove 失败 -> 双重存在)
      // 必须回滚 Dst (删除刚刚插入的)
      bool need_rollback = (dst_inserted && !src_removed);
      if (tile.any(need_rollback)) {
        tiled_remove_op(g_MoMo_Hash, tx.dst_id, tile, need_rollback,g_MoMo_Hash->num_buckets);
      }

      // 只有两步都成功才算成功
       if (src_removed){
        tx.success = true;
        // if (tile.thread_rank() == 0) {
        //   printf("[TYPE1_OK] MoMo_src:%ld + dstid:%ld\n", tx.src_id, tx.dst_id);
        // }
      }
  }

  // 2. 处理类型 2：MoMo + Re -> Mo + MoRe (Reaction)
  bool is_type2 = (tx.op_type == 2);
  if (tile.any(is_type2)) {
      // Step 1: 尝试插入 MoRe (Dst) - 这是最关键的一步，如果满/冲突则直接放弃
      int ins_more_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type2,g_MoRe_Hash->num_buckets);
      bool more_inserted = (is_type2 && ins_more_res == 1);

      // Step 2: 只有插入成功，才去销毁旧原料 (Re 和 MoMo)
      int rem_re_res = 0;
      int rem_momo_res = 0;
      if (tile.any(more_inserted)) {
          rem_re_res = tiled_remove_op(g_Re_Hash, tx.dst_id, tile, more_inserted,g_Re_Hash->num_buckets);
          rem_momo_res = tiled_remove_op(g_MoMo_Hash, tx.src_id, tile, more_inserted,g_MoMo_Hash->num_buckets);
      }
      bool re_removed = (more_inserted && rem_re_res == 1);
      bool momo_removed = (more_inserted && rem_momo_res == 1);

      // Step 3: 提交或回滚。所有 lane 必须共同执行下面的 tile collective；
      // 不能把 tile.any 放进仅失败 lane 才进入的分歧分支。
      const bool type2_committed = re_removed && momo_removed;
      if (type2_committed) {
        tx.success = true;
      }

      const bool need_revert_more = more_inserted && !type2_committed;
      if (tile.any(need_revert_more)) {
        tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, need_revert_more,
                        g_MoRe_Hash->num_buckets);
      }

      const bool need_restore_re = more_inserted && re_removed && !momo_removed;
      if (tile.any(need_restore_re)) {
        tiled_insert_op(g_Re_Hash, tx.dst_id, tile, need_restore_re,
                        g_Re_Hash->num_buckets);
      }

      const bool need_restore_momo = more_inserted && !re_removed && momo_removed;
      if (tile.any(need_restore_momo)) {
        tiled_insert_op(g_MoMo_Hash, tx.src_id, tile, need_restore_momo,
                        g_MoMo_Hash->num_buckets);
      }
  }

  // Type 3: MoRe 搬家 (Move to Empty) 
  bool is_type3 = (tx.op_type == 3);
  if (tile.any(is_type3)) {
      // Step 1: Insert Dst
      int ins_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type3,g_MoRe_Hash->num_buckets);
      bool dst_inserted = (is_type3 && ins_res == 1);
      
      // Step 2: Remove Src
      int rem_res = 0;
      if (tile.any(dst_inserted)) {
          rem_res = tiled_remove_op(g_MoRe_Hash, tx.src_id, tile, dst_inserted,g_MoRe_Hash->num_buckets);
      }
      bool src_removed = (dst_inserted && rem_res == 1);

      // Step 3: Rollback Dst if needed
      bool need_rollback = (dst_inserted && !src_removed);
      if (tile.any(need_rollback)) {
          tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, need_rollback,g_MoRe_Hash->num_buckets);
      }

      // if (src_removed) tx.success = true;
    if (src_removed){
      tx.success = true;
      // if (tile.thread_rank() == 0) {
      //   printf("[TYPE1_OK] MoMo_src:%ld + dstid:%ld\n", tx.src_id, tx.dst_id);
      // }
    }
  }
  
  // Type 4: MoRe + Re -> Re + MoRe (MoRe 换位)
  bool is_type4 = (tx.op_type == 4);
  if (tile.any(is_type4)) {
      // Step 1: 尝试插入所有新身份 (Re at Src, MoRe at Dst)
      int ins_re_res = tiled_insert_op(g_Re_Hash, tx.src_id, tile, is_type4,g_Re_Hash->num_buckets);
      int ins_more_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type4,g_MoRe_Hash->num_buckets);
      
      bool re_inserted = (is_type4 && ins_re_res == 1);
      bool more_inserted = (is_type4 && ins_more_res == 1);
      bool both_inserted = (re_inserted && more_inserted);

      // Step 2: 只有都插入成功，才删除旧身份
      int rem_more_res = 0;
      int rem_re_res = 0;
      if (tile.any(both_inserted)) {
          rem_more_res = tiled_remove_op(g_MoRe_Hash, tx.src_id, tile, both_inserted,g_MoRe_Hash->num_buckets);
          rem_re_res = tiled_remove_op(g_Re_Hash, tx.dst_id, tile, both_inserted,g_Re_Hash->num_buckets);
      }
      bool more_removed = (both_inserted && rem_more_res == 1);
      bool re_removed = (both_inserted && rem_re_res == 1);

      // Step 3: 提交或回滚；与 type 2 相同，collective 保持整 tile 收敛。
      const bool type4_committed = more_removed && re_removed;
      if (type4_committed) {
        tx.success = true;
      }

      const bool fail_ins_re = re_inserted && !more_inserted;
      if (tile.any(fail_ins_re)) {
        tiled_remove_op(g_Re_Hash, tx.src_id, tile, fail_ins_re,
                        g_Re_Hash->num_buckets);
      }
      const bool fail_ins_more = !re_inserted && more_inserted;
      if (tile.any(fail_ins_more)) {
        tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, fail_ins_more,
                        g_MoRe_Hash->num_buckets);
      }

      const bool fail_rem_both = both_inserted && !type4_committed;
      if (tile.any(fail_rem_both)) {
        tiled_remove_op(g_Re_Hash, tx.src_id, tile, fail_rem_both,
                        g_Re_Hash->num_buckets);
        tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, fail_rem_both,
                        g_MoRe_Hash->num_buckets);
      }

      const bool need_restore_more = both_inserted && more_removed && !re_removed;
      if (tile.any(need_restore_more)) {
        tiled_insert_op(g_MoRe_Hash, tx.src_id, tile, need_restore_more,
                        g_MoRe_Hash->num_buckets);
      }

      const bool need_restore_re4 = both_inserted && !more_removed && re_removed;
      if (tile.any(need_restore_re4)) {
        tiled_insert_op(g_Re_Hash, tx.dst_id, tile, need_restore_re4,
                        g_Re_Hash->num_buckets);
      }
  }
  // 按类别调用独立的函数
  // if (tile.any(tx.op_type == 1)) process_type1(tx, tile);
  // if (tile.any(tx.op_type == 2)) process_type2(tx, tile);
  // if (tile.any(tx.op_type == 3)) process_type3(tx, tile);
  // if (tile.any(tx.op_type == 4)) process_type4(tx, tile);
  // 3. 处理 Ghost 和 Surface 更新。每个 collective 仍由整 tile 调用，
  // tx.success 只决定当前 lane 是否贡献一个实际请求。
  const bool update_ghost = tx.success && isGhostLat(tx.dst_id);
  if (tile.any(update_ghost)) {
      tiled_insert_op(g_ghost_Hash, tx.dst_id, tile, update_ghost,g_ghost_Hash->num_buckets);
  }
  const bool update_surface = tx.success && isSurfaceLat(tx.dst_id);
  if (tile.any(update_surface)) {
      tiled_insert_op(g_surface_Hash, tx.dst_id, tile, update_surface,g_surface_Hash->num_buckets);
  }
  const bool update_surface_src = tx.success && isSurfaceLat(tx.src_id);
  if (tile.any(update_surface_src)) {
      tiled_insert_op(g_surface_Hash, tx.src_id, tile, update_surface_src,g_surface_Hash->num_buckets);
  }
  return;
}

// GPU correctness baseline：一个 walker 线程独立完成 Remove-Then-Insert，
// 顺序与 original_kmc 的 CPU recb_solver 一致。
// 跨线程的同位置互斥由外层 src/dst 位图锁保证，哈希槽竞争由 atomicCAS 保证。
__device__ __noinline__ void perform_batch_update_single(AtomTx& tx) {
  tx.success = false;
  if (tx.op_type == 0) return;
  if (tx.src_id == tx.dst_id) {
    tx.op_type = 0;
    return;
  }

  if (tx.op_type == 1) {
    if (hash_set_remove_atom_op(g_MoMo_Hash, tx.src_id) != 1) return;
    if (hash_set_insert_atom_op(g_MoMo_Hash, tx.dst_id) == 1) {
      tx.success = true;
    } else {
      hash_set_insert_atom_op(g_MoMo_Hash, tx.src_id);
    }
  } else if (tx.op_type == 2) {
    if (hash_set_remove_atom_op(g_Re_Hash, tx.dst_id) != 1) return;
    if (hash_set_remove_atom_op(g_MoMo_Hash, tx.src_id) != 1) {
      hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
      return;
    }
    if (hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id) == 1) {
      tx.success = true;
    } else {
      hash_set_insert_atom_op(g_MoMo_Hash, tx.src_id);
      hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
    }
  } else if (tx.op_type == 3) {
    if (hash_set_remove_atom_op(g_MoRe_Hash, tx.src_id) != 1) return;
    if (hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id) == 1) {
      tx.success = true;
    } else {
      hash_set_insert_atom_op(g_MoRe_Hash, tx.src_id);
    }
  } else if (tx.op_type == 4) {
    if (hash_set_remove_atom_op(g_Re_Hash, tx.dst_id) != 1) return;
    if (hash_set_remove_atom_op(g_MoRe_Hash, tx.src_id) != 1) {
      hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
      return;
    }
    if (hash_set_insert_atom_op(g_Re_Hash, tx.src_id) != 1) {
      hash_set_insert_atom_op(g_MoRe_Hash, tx.src_id);
      hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
      return;
    }
    if (hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id) == 1) {
      tx.success = true;
    } else {
      hash_set_remove_atom_op(g_Re_Hash, tx.src_id);
      hash_set_insert_atom_op(g_MoRe_Hash, tx.src_id);
      hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
    }
  }

  if (tx.success) {
    if (isGhostLat(tx.dst_id)) {
      hash_set_insert_atom_op(g_ghost_Hash, tx.dst_id);
    }
    if (isSurfaceLat(tx.dst_id)) {
      hash_set_insert_atom_op(g_surface_Hash, tx.dst_id);
    }
    if (isSurfaceLat(tx.src_id)) {
      hash_set_insert_atom_op(g_surface_Hash, tx.src_id);
    }
  }
}

__device__ int decide_op_type(LatticeTypes::lat_type cur_type,LatticeTypes::lat_type des_type,bool flag) {
    if (cur_type == LatticeTypes::MoMo) {
      if (des_type == LatticeTypes::Mo) return 1; // 交换位置 
      if (des_type == LatticeTypes::Re) return 2; // (momo 变成 mo，re 变成 more)
    }
    if (cur_type == LatticeTypes::MoRe) {
      if (flag && des_type == LatticeTypes::Mo) return 3; // MoRe 搬家
      if (flag && des_type == LatticeTypes::Re) return 4; // MoRe 换位
    }
    return 0; // 不满足条件
}

// template <typename TileType>
// __device__ bool collective_bucket_contains(GPUHashSet* set, long int target_key, TileType tile) {
//     unsigned int buck_num = set->hasher(target_key) % set->num_buckets;
//     int lane = tile.thread_rank();
//     int loop_count = 0;

//     // 必须加上线性探测的循环，处理桶溢出的情况
//     while (loop_count < set->num_buckets) {
//         // 全组 64 线程同时加载整个桶
//         long int val = atomic_load_relaxed(&set->keys[buck_num * 64 + lane]);

//         if (tile.any(val == target_key)) {
//             return true; 
//         }
//         if (tile.any(val == set->empty_flag)) {
//             return false;
//         }
//         buck_num = (buck_num + 1) % set->num_buckets;
//         loop_count++;
//     }
//     return false;
// }

template <typename TileType>
__device__ bool collective_bucket_contains(GPUHashSet* set, long int target_key, TileType tile) {
    // 即使 target_key <= 0，也不能提前 return，必须陪全组参与 tile.any 防死锁
    if (set == nullptr) return false;

    const bool valid_key = (target_key > 0);
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int loop_count = 0;
    unsigned int buck_num = valid_key ? hash_probe_bucket(set->num_buckets, target_key, loop_count) : 0;
    const int lane = tile.thread_rank();
    if (lane == 0 && valid_key) record_hash_query(set);
    bool done = !valid_key;
    bool result = false;
    // const int MAX_PROBE_STEPS = 3;

    while (loop_count <= max_probe) {
        if (lane == 0 && !done) record_hash_probe(set, loop_count + 1);
        long int val = GPU_HASH_EMPTY_KEY;
        if (!done) {
            val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
        }

        const uint64_t active_mask = tile.ballot(!done);
        if (active_mask == 0) break;

        const uint64_t match_mask = tile.ballot(!done && val == target_key);
        const uint64_t empty_mask = tile.ballot(!done && val == GPU_HASH_EMPTY_KEY);

        if (match_mask != 0) {
            result = true;
            done = true;
        } else if (empty_mask != 0) {
            result = false;
            done = true;
        } else if (!done) {
            loop_count++;
            if (loop_count > max_probe) {
                result = false;
                done = true;
            } else {
                buck_num = hash_probe_bucket(set->num_buckets, target_key, loop_count);
            }
        }
    }
    return result;
}



template <typename TileType>
__device__ void tiled_check_neighbors(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, 
                                      TileType tile, bool active, int& out_numSIA, int& out_numRe) 
{
    out_numSIA = 0;
    out_numRe = 0;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        
        _type_lattice_size lx = tile.shfl(x, leader);
        _type_lattice_size ly = tile.shfl(y, leader);
        _type_lattice_size lz = tile.shfl(z, leader);
        
        _type_lattice_id nn_ids[8];
        get1nn_ids(lx, ly, lz, nn_ids); 

        int leader_SIA = 0;
        int leader_Re = 0;
        #pragma unroll 1 
        for (int i = 0; i < 8; i++) {
            _type_lattice_id nid = nn_ids[i];
           
            if (collective_bucket_contains(g_MoRe_Hash, nid, tile) || collective_bucket_contains(g_MoMo_Hash, nid, tile)) {
                leader_SIA++;
            } 
            else if (collective_bucket_contains(g_Re_Hash, nid, tile)) {
                leader_Re++;
            }
        }

        if (tile.thread_rank() == leader) {
            out_numSIA = leader_SIA;
            out_numRe = leader_Re;
        }
        work_mask &= ~(1ULL << leader);
    }
}
template <typename TileType>
__device__ void smart_tiled_check_neighbors(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, 
                                            TileType tile, bool active, int& out_numSIA, int& out_numRe) 
{
    out_numSIA = 0;
    out_numRe = 0;
    
    uint64_t work_mask = tile.ballot(active);
    
    // 统计当前 64 个人里，到底有几个 active 的？(__popcll 是极速硬件指令)
    int active_count = __popcll((unsigned long long int)work_mask);

    // 设置临界值 THRESHOLD// 通常在 3 到 6 之间
    const int THRESHOLD = 4; 

    if (active_count > 0 && active_count <= THRESHOLD) {
        while (work_mask != 0) {
            int leader = __ffsll((unsigned long long int)work_mask) - 1;
            work_mask &= ~(1ULL << leader);
        }
    } 
    else if (active_count > THRESHOLD) {
        if (active) {
            _type_lattice_id nn_ids[8];
            get1nn_ids(x, y, z, nn_ids); 
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                _type_lattice_id nid = nn_ids[i];
                if (nid < 0) continue;
                if (hash_set_contains(g_MoRe_Hash, nid) || hash_set_contains(g_MoMo_Hash, nid)) out_numSIA++;
                else if (hash_set_contains(g_Re_Hash, nid)) out_numRe++;
            }
        }
    }
}
__device__ void tiled_check_neighbors_single(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, int& out_numSIA, int& out_numRe)
{
    out_numSIA = 0;
    out_numRe = 0;

    _type_lattice_id nn_ids[8];
    get1nn_ids(x, y, z, nn_ids);

    // 禁止 unroll：每次迭代内联 2~3 个全桶扫描的 hash_set_contains，
    // 全部展开会把调用者的寄存器需求推高数倍，后端直接分配失败。
    #pragma unroll 1
    for (int i = 0; i < 8; i++) {
        _type_lattice_id nid = nn_ids[i];
        if (nid < 0) continue;

        if (hash_set_contains(g_MoRe_Hash, nid) || hash_set_contains(g_MoMo_Hash, nid)) {
            out_numSIA++;
        } else if (hash_set_contains(g_Re_Hash, nid)) {
            out_numRe++;
        }
    }
}

__device__ __forceinline__ int recb_walker_attempt_bin(unsigned int attempts) {
  if (attempts == 0u) return 0;
  if (attempts <= 7u) return 1;
  if (attempts <= 31u) return 2;
  if (attempts <= 63u) return 3;
  if (attempts <= 127u) return 4;
  if (attempts <= 255u) return 5;
  if (attempts <= 511u) return 6;
  if (attempts <= 1023u) return 7;
  if (attempts <= 2047u) return 8;
  if (attempts <= 4095u) return 9;
  return 10;
}
// __launch_bounds__(128) 与实际启动 blockSize=128 一致：缺省时编译器按
// 1024 线程/块的最坏情况压缩寄存器预算，本 kernel 的统计计数器会直接把
// 后端寄存器分配打爆（no registers from class available）。
__global__ void __launch_bounds__(128) calcExchangePairs(int sector_id,int arr_size, RecbVerifyStats *stats,
                                  const long int* __restrict__ sector_slots,
                                  bool verify_detail,
                                  bool collect_hash_stats,
                                  unsigned long long* __restrict__ walker_lengths
){
  if (verify_detail && blockIdx.x == 0 && threadIdx.x == 0) {
    printf("begin calcExchangePairs\n");
  }
  __shared__ unsigned int block_direction_counts[14];
  if (verify_detail) {
    if (threadIdx.x < 14) block_direction_counts[threadIdx.x] = 0;
    __syncthreads();
  }
  // 初始化状态机
  AtomTx tx = {0, 0, true, false, 0};

  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  // slot 是 walker 在花名册 g_Pair_Atoms 中的槽位号（身份），tid 只是本次启动的线程号（临时工牌）。
  // sector_slots 为 recb 前筛出的本扇区槽位列表；传 nullptr 时退回旧行为（slot = tid）。
  long int slot = tid;
  if (sector_slots != nullptr && tid < arr_size) {
    slot = sector_slots[tid];
  }
  _type_lattice_id id = 0;

  bool valid_thread = (g_Pair_Atoms != nullptr && tid < arr_size);
  if (valid_thread) {
    id = g_Pair_Atoms[slot];
  }

  if(!valid_thread || id<=0){
    tx.active = false;
    id = 0;
  }

  _type_lattice_size x = 0;
  _type_lattice_size y = 0;
  _type_lattice_size z = 0;
  const long int x_min = 2*g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2*g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];

  int my_hops = 0;
  // per-thread 计数用 32 位即可（单 walker 尝试次数 ≤ KMC_RECB_MAX_ITERATIONS），
  // 聚合到全局 64 位计数器时再 widen；64 位局部变量会翻倍占用 VGPR。
  unsigned int my_attempts = 0;
  unsigned int my_target_mo = 0;
  unsigned int my_target_re = 0;
  unsigned int my_target_other = 0;
  unsigned int my_lock_failures = 0;
  unsigned int my_src_check_failures = 0;
  unsigned int my_dst_mo_check_failures = 0;
  unsigned int my_dst_re_check_failures = 0;
  unsigned int my_dst_other_check_failures = 0;
  unsigned int my_transaction_failures = 0;
  unsigned int my_extra_rng_draws = 0;
  unsigned int my_captured_walks = 0;
  unsigned int my_exited_walks = 0;
  unsigned int my_capped_walks = 0;
  unsigned int my_stale_source_walkers = 0;
  int started_in_sector = 0;
  int captured_no_move = 0;

  if (id > 0) {
    x = id % g_meta->size_x;
    y = (id / g_meta->size_x) % g_meta->size_y;
    z = id / (g_meta->size_x * g_meta->size_y);
  }
  const bool in0 = (id > 0) &&
                   (x_min <= x && x < x_max &&
                    y_min <= y && y < y_max &&
                    z_min <= z && z < z_max);
  if (in0) {
    started_in_sector = 1;
    if (stats) atomicAdd(&stats->sector_pairs, 1u);
  }

  LatticeTypes::lat_type cur_type = tx.active ? getType(id) : LatticeTypes::Mo;
  // original_kmc 在每次 walker 循环开头都会断言当前位置仍是该 pair。
  // GPU roster 若已陈旧，不能让它带着 Mo/Re 身份进入随机游走。
  if (tx.active && cur_type != LatticeTypes::MoMo && cur_type != LatticeTypes::MoRe) {
    if (stats) {
      atomicAdd(&stats->roster_type_mismatches, 1u);
      atomicAdd(&stats->stale_at_start, 1u);
      if (isGhostLat(id)) atomicAdd(&stats->stale_start_ghost, 1u);
      if (cur_type == LatticeTypes::Re) atomicAdd(&stats->stale_start_is_re, 1u);
      if (cur_type == LatticeTypes::V) atomicAdd(&stats->stale_start_is_v, 1u);
    }
    // 排查日志：定位 stale 的具体来源。打印前几个 slot/id/坐标/实际类型/是否 ghost，
    // 用来区分"游走竞态"还是"通信移交后 slot 没同步"。verify_detail 打开时才会打印。
    if (verify_detail && stats) {
      unsigned int seq = atomicAdd(&stats->stale_debug_seq, 1u);
      if (seq < 8) {
        printf("[recb-stale-debug] sector=%d slot=%ld id=%ld type=%d x=%ld y=%ld z=%ld ghost=%d\n",
               sector_id, (long)slot, (long)id, (int)cur_type,
               (long)x, (long)y, (long)z, isGhostLat(id));
      }
    }
    g_Pair_Atoms[slot] = -1;
    if (started_in_sector) my_stale_source_walkers = 1;
    tx.active = false;
  }
  if (tx.active) tx.src_id = id;

  int loop_iterations = 0;

  // GPU scalar correctness baseline：每个 walker 独立推进到明确终止状态。
  // 此路径不依赖 cooperative-groups ballot/shuffle，避免遗漏 wave64 的高 lane。
  while (tx.active) {
        // 保险丝限制循环尝试次数；成功跳跃次数由 my_hops 单独统计。
        // 触发仅意味着 walker 本次事件提前停下，g_Pair_Atoms 保留，下一事件照常重走。
        if (loop_iterations >= KMC_RECB_MAX_ITERATIONS) {
          tx.active = false;
          my_capped_walks = 1;
          break;
        }

        const bool in_bounds = (x_min <= x && x < x_max &&
                                y_min <= y && y < y_max &&
                                z_min <= z && z < z_max);
        if (!in_bounds) {
          // 本扇区结束，但 pair 仍在本地其他扇区时必须保留花名册槽位。
          my_exited_walks = 1;
          tx.active = false;
          break;
        }

        // --- 当前 walker 独立查邻居 ---
        int numSIA = 0, numRe = 0;
        tiled_check_neighbors_single(x, y, z, numSIA, numRe);
        if (numSIA >= 2 || (cur_type == LatticeTypes::MoRe && numRe >= 5)) {
          if (my_hops == 0) captured_no_move = 1;
          my_captured_walks = 1;
          tx.active = false;
          break;
        }
       
        // --- 计算随机跳跃目标 ---
        _type_lattice_size temp_x = 0, temp_y = 0, temp_z = 0;
        _type_lattice_id random_id = 0;
        if (tx.active) {
            loop_iterations++;
            my_attempts++;
            // 每线程持久 Philox 流：方向与 walker 处境解耦，每步都是新骰子
            int randomNumber = (int)(hiprand_uniform(&g_rng_states[slot]) * 14.0f);
            if (randomNumber > 13) randomNumber = 13; // hiprand_uniform 返回 (0,1]，防边界
            if (verify_detail) atomicAdd(&block_direction_counts[randomNumber], 1u);
            getRandomLattice(x, y, z, temp_x, temp_y, temp_z, randomNumber);
            random_id = getId(temp_x, temp_y, temp_z);
        }
        LatticeTypes::lat_type rand_type = tx.active? getType(random_id):(LatticeTypes::Mo);

        if (tx.active) {
            if (verify_detail) {
              if (rand_type == LatticeTypes::Mo) my_target_mo++;
              else if (rand_type == LatticeTypes::Re) my_target_re++;
              else my_target_other++;
            }
            if (rand_type == LatticeTypes::Mo || rand_type == LatticeTypes::Re) {
                bool flag1 = (numRe == 0);
                bool flag2 = false;
                if (numRe > 0 && numRe < LatticesList::MAX_1NN) {
                    if (collect_hash_stats) ++my_extra_rng_draws;
                    double ranmov = (double)hiprand_uniform(&g_rng_states[slot]);
                    flag2 = (ranmov > (static_cast<double>(numRe) / 8.0));
                }
                tx.op_type = decide_op_type(cur_type, rand_type, flag1 || flag2);
                tx.dst_id = random_id;
            } else {
                tx.op_type = 0;
            }
        } else {
            tx.op_type = 0;
        }
        //----加锁----
        bool locks_acquired = false;
        if (tx.active && tx.op_type != 0) {
            locks_acquired = try_acquire_bitmap_locks(tx.src_id, tx.dst_id);
            if (locks_acquired) {
              // 获取语义：拿到锁之前的哈希表读取不能重排到拿锁之后。
              // GPU 弱内存序下 atomicOr 只保证原子性，不保证顺序，需要显式 fence。
              __threadfence();
            } else {
              my_lock_failures++;
              tx.op_type = 0;
            }
        }
        // DCL 状态二次校验：只有拿到 src/dst 位置锁的线程才读取最新类型。
        // 每个 walker 独立走 getType；当前 correctness baseline 不依赖 tile 查询。
        bool need_verify = locks_acquired && (tx.op_type != 0);

        const LatticeTypes::lat_type safe_src_type =
            need_verify ? getType(tx.src_id) : cur_type;
        const LatticeTypes::lat_type safe_dst_type =
            need_verify ? getType(tx.dst_id) : rand_type;
        const bool src_chk = !need_verify || (safe_src_type == cur_type);
        const bool dst_chk = !need_verify || (safe_dst_type == rand_type);

        if (need_verify) {
          if (!src_chk) my_src_check_failures++;
          if (!dst_chk) {
            if (rand_type == LatticeTypes::Mo) my_dst_mo_check_failures++;
            else if (rand_type == LatticeTypes::Re) my_dst_re_check_failures++;
            else my_dst_other_check_failures++;
          }
          if (!src_chk) {
            // original_kmc 对源类型不变使用 assert；这里同样 fail closed。
            // 源一旦与 walker 身份不符，就不能刷新后继续（可能是重复 roster）。
            if (g_Pair_Atoms[slot] == tx.src_id) {
              g_Pair_Atoms[slot] = -1;
            }
            tx.active = false;
            my_stale_source_walkers = 1;
            if (stats) atomicAdd(&stats->stale_src_changed, 1u);
            tx.op_type = 0;
          } else if (!dst_chk) {
            // 等锁期间目的地被别的 walker 改变属于正常竞争，本次放弃并重选。
            tx.op_type = 0;
          }
        }
        const bool transaction_attempted = need_verify && src_chk && dst_chk;
        const int attempted_op = transaction_attempted ? tx.op_type : 0;
        if (verify_detail && attempted_op >= 1 && attempted_op <= 4) {
          atomicAdd(&stats->op_attempts[attempted_op - 1], 1ULL);
        }

        //---- GPU scalar correctness baseline：owner 线程独立完成事务 ---
        if (tx.op_type != 0) {
          perform_batch_update_single(tx);
        }
        if (transaction_attempted && !tx.success) {
          my_transaction_failures++;
        }
        if (verify_detail && attempted_op >= 1 && attempted_op <= 4) {
          if (tx.success) atomicAdd(&stats->op_successes[attempted_op - 1], 1ULL);
          else atomicAdd(&stats->op_failures[attempted_op - 1], 1ULL);
        }
        //----解锁----
        if (locks_acquired) {
            // 先发布该 owner 对多个哈希表的写入，再释放逻辑位置锁。
            __threadfence();
            release_bitmap_locks(tx.src_id, tx.dst_id);
        }
        //----更新状态----
        // 对齐 original recb_solver：成功换格后继续在本 while 里跳，直到抓住或出扇区。
        if (tx.active && tx.op_type != 0 && tx.success) {
            x = temp_x;
            y = temp_y;
            z = temp_z;
            tx.src_id = id = random_id;
            if(tx.op_type == 2 || tx.op_type == 4) cur_type = LatticeTypes::MoRe;
            my_hops++;
            const bool dest_in_sector = (x_min <= x && x < x_max &&
                                         y_min <= y && y < y_max &&
                                         z_min <= z && z < z_max);
            // 槽位统一保留新位置 id（含 ghost）：花名册与 MoMo/MoRe 全量哈希同口径，
            // 不变量 roster_live == hash_pairs 才成立。ghost 位置不属于任何本地扇区，
            // filter 不会再调度它，等下一次 rebuild 重排即可。
            g_Pair_Atoms[slot] = random_id;
            if (isGhostLat(random_id)) {
              my_exited_walks = 1;
              tx.active = false;
            } else if (!dest_in_sector) {
              // 本扇区结束，但 pair 仍在本地其他扇区，槽位保留供后续扇区调度。
              my_exited_walks = 1;
              tx.active = false;
            }
        }
        tx.op_type = 0;
    }
    if (stats && started_in_sector) {
      if (my_attempts > 0) atomicAdd(&stats->attempts, (unsigned long long)my_attempts);
      if (my_hops > 0) atomicAdd(&stats->successful_hops, (unsigned long long)my_hops);
      atomicMax(&stats->max_successful_hops, (unsigned int)my_hops);
      if (my_hops >= 2) atomicAdd(&stats->multi_hop_pairs, 1u);
      if (captured_no_move) atomicAdd(&stats->captured_no_move, 1u);
      if (my_captured_walks) atomicAdd(&stats->captured_walks, 1u);
      if (my_exited_walks) atomicAdd(&stats->exited_walks, 1u);
      if (my_capped_walks) atomicAdd(&stats->capped_walks, 1u);
      if (my_stale_source_walkers) atomicAdd(&stats->stale_source_walkers, 1u);
      if (!my_captured_walks && !my_exited_walks && !my_capped_walks &&
          !my_stale_source_walkers) {
        atomicAdd(&stats->unclassified_walkers, 1u);
      }
      if (my_lock_failures) atomicAdd(&stats->lock_failures, (unsigned long long)my_lock_failures);
      if (my_src_check_failures) atomicAdd(&stats->src_check_failures, (unsigned long long)my_src_check_failures);
      if (my_dst_mo_check_failures) atomicAdd(&stats->dst_mo_check_failures, (unsigned long long)my_dst_mo_check_failures);
      if (my_dst_re_check_failures) atomicAdd(&stats->dst_re_check_failures, (unsigned long long)my_dst_re_check_failures);
      if (my_dst_other_check_failures) atomicAdd(&stats->dst_other_check_failures, (unsigned long long)my_dst_other_check_failures);
      if (my_transaction_failures) atomicAdd(&stats->transaction_failures, (unsigned long long)my_transaction_failures);
      if (verify_detail) {
        if (my_target_mo) atomicAdd(&stats->target_mo, (unsigned long long)my_target_mo);
        if (my_target_re) atomicAdd(&stats->target_re, (unsigned long long)my_target_re);
        if (my_target_other) atomicAdd(&stats->target_other, (unsigned long long)my_target_other);
      }
    }
    if (collect_hash_stats && walker_lengths && tid < arr_size) {
      walker_lengths[tid] =
          (static_cast<unsigned long long>(my_extra_rng_draws) << 32) |
          static_cast<unsigned int>(my_attempts);
    }
    if (verify_detail) {
      __syncthreads();
      if (threadIdx.x < 14) {
        atomicAdd(&stats->direction_counts[threadIdx.x],
                  (unsigned long long)block_direction_counts[threadIdx.x]);
      }
    }
}

// 单独汇总完整 walker 长度，避免诊断归约增加主随机游走 kernel 的寄存器压力。
template <typename Tile>
__device__ __forceinline__ void check_neighbors_tile8(
    _type_lattice_size x, _type_lattice_size y, _type_lattice_size z,
    Tile tile, int& num_sia, int& num_re) {
  const int lane = tile.thread_rank();
  const int q = lane & 3;
  const int dx = lane < 4 ? -1 : 1;
  const bool even_x = (x & 1) == 0;
  const int dy = even_x ? (q < 2 ? -1 : 0) : (q < 2 ? 0 : 1);
  const int dz = even_x ? ((q & 1) ? 0 : -1) : ((q & 1) ? 1 : 0);
  const _type_lattice_id nid = getId(x + dx, y + dy, z + dz);

  int sia = 0;
  int re = 0;
  if (nid > 0) {
    sia = hash_set_contains(g_MoRe_Hash, nid) ||
          hash_set_contains(g_MoMo_Hash, nid);
    re = !sia && hash_set_contains(g_Re_Hash, nid);
  }
  #pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    sia += tile.shfl_down(sia, offset);
    re += tile.shfl_down(re, offset);
  }
  num_sia = tile.shfl(sia, 0);
  num_re = tile.shfl(re, 0);
}

__device__ __forceinline__ void record_recb_stratum_outcome(
    unsigned long long* block_strata, int direction, int op_type, int num_re,
    int num_sia, RecbStratumOutcome outcome) {
  if (!block_strata || op_type < 1 || op_type > RECB_OP_STRATA) return;
  if (direction < 0) direction = 0;
  if (direction >= RECB_DIRECTION_STRATA) direction = RECB_DIRECTION_STRATA - 1;
  if (num_re < 0) num_re = 0;
  if (num_re >= RECB_NUM_RE_STRATA) num_re = RECB_NUM_RE_STRATA - 1;
  int sia_bin = num_sia < 0 ? 0 : (num_sia >= 2 ? 2 : num_sia);
  const int op_base = RECB_DIRECTION_STRATA;
  const int re_base = op_base + RECB_OP_STRATA;
  const int sia_base = re_base + RECB_NUM_RE_STRATA;
  atomicAdd(&block_strata[direction * RECB_STRATUM_OUTCOME_COUNT + outcome], 1ULL);
  atomicAdd(&block_strata[(op_base + op_type - 1) * RECB_STRATUM_OUTCOME_COUNT + outcome], 1ULL);
  atomicAdd(&block_strata[(re_base + num_re) * RECB_STRATUM_OUTCOME_COUNT + outcome], 1ULL);
  atomicAdd(&block_strata[(sia_base + sia_bin) * RECB_STRATUM_OUTCOME_COUNT + outcome], 1ULL);
}

template <typename Tile>
__device__ void run_recb_walker_tile8(
    int sector_id, int work_index, RecbVerifyStats* stats,
    const long int* __restrict__ sector_slots, bool verify_detail,
    bool collect_hash_stats,
    unsigned long long* __restrict__ walker_lengths,
    unsigned long long* block_strata, Tile tile) {
  const bool leader = tile.thread_rank() == 0;
  long int slot = leader ? sector_slots[work_index] : 0;
  slot = tile.shfl(slot, 0);
  _type_lattice_id id = leader ? g_Pair_Atoms[slot] : 0;
  id = tile.shfl(id, 0);

  AtomTx tx = {0, 0, id > 0, false, 0};
  _type_lattice_size x = id > 0 ? id % g_meta->size_x : 0;
  _type_lattice_size y = id > 0 ? (id / g_meta->size_x) % g_meta->size_y : 0;
  _type_lattice_size z = id > 0 ? id / (g_meta->size_x * g_meta->size_y) : 0;
  LatticeTypes::lat_type cur_type = LatticeTypes::Mo;
  if (leader && tx.active) cur_type = getType(id);
  cur_type = static_cast<LatticeTypes::lat_type>(tile.shfl(static_cast<int>(cur_type), 0));

  const long int x_min = 2 * g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2 * g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];

  unsigned int attempts = 0, hops = 0, extra_rng = 0;
  unsigned int target_mo = 0, target_re = 0, target_other = 0;
  unsigned int lock_failures = 0, src_failures = 0;
  unsigned int dst_mo_failures = 0, dst_re_failures = 0, dst_other_failures = 0;
  unsigned int tx_failures = 0, captured = 0, exited = 0, capped = 0, stale = 0;
  unsigned int captured_no_move = 0;

  if (leader && tx.active && stats) atomicAdd(&stats->sector_pairs, 1u);
  if (leader && tx.active && cur_type != LatticeTypes::MoMo && cur_type != LatticeTypes::MoRe) {
    if (stats) {
      atomicAdd(&stats->roster_type_mismatches, 1u);
      atomicAdd(&stats->stale_at_start, 1u);
      if (isGhostLat(id)) atomicAdd(&stats->stale_start_ghost, 1u);
      if (cur_type == LatticeTypes::Re) atomicAdd(&stats->stale_start_is_re, 1u);
      if (cur_type == LatticeTypes::V) atomicAdd(&stats->stale_start_is_v, 1u);
    }
    g_Pair_Atoms[slot] = -1;
    stale = 1;
    tx.active = false;
  }
  if (leader && tx.active) {
    tx.src_id = id;
  }

  while (tile.shfl(leader && tx.active ? 1 : 0, 0)) {
    if (leader) {
      if (attempts >= KMC_RECB_MAX_ITERATIONS) {
        capped = 1;
        tx.active = false;
        atomicExch(&g_recb_pairs_changed, 1u);
      } else if (!(x_min <= x && x < x_max && y_min <= y && y < y_max &&
                   z_min <= z && z < z_max)) {
        exited = 1;
        tx.active = false;
      }
    }
    if (!tile.shfl(leader && tx.active ? 1 : 0, 0)) break;

    x = tile.shfl(x, 0);
    y = tile.shfl(y, 0);
    z = tile.shfl(z, 0);
    int num_sia = 0, num_re = 0;
    check_neighbors_tile8(x, y, z, tile, num_sia, num_re);

    if (leader) {
      if (num_sia >= 2 || (cur_type == LatticeTypes::MoRe && num_re >= 5)) {
        captured_no_move = hops == 0;
        captured = 1;
        tx.active = false;
      } else {
        ++attempts;
        int direction = static_cast<int>(hiprand_uniform(&g_rng_states[slot]) * 14.0f);
        if (direction > 13) direction = 13;
        if (verify_detail && stats) atomicAdd(&stats->direction_counts[direction], 1ULL);

        _type_lattice_size nx = 0, ny = 0, nz = 0;
        getRandomLattice(x, y, z, nx, ny, nz, direction);
        const _type_lattice_id dst_id = getId(nx, ny, nz);
        const LatticeTypes::lat_type dst_type = getType(dst_id);
        if (verify_detail) {
          if (dst_type == LatticeTypes::Mo) ++target_mo;
          else if (dst_type == LatticeTypes::Re) ++target_re;
          else ++target_other;
        }

        tx.op_type = 0;
        if (dst_type == LatticeTypes::Mo || dst_type == LatticeTypes::Re) {
          bool accept = num_re == 0;
          if (num_re > 0 && num_re < LatticesList::MAX_1NN) {
            if (collect_hash_stats) ++extra_rng;
            accept = hiprand_uniform(&g_rng_states[slot]) >
                     static_cast<float>(num_re) / LatticesList::MAX_1NN;
          }
          tx.op_type = decide_op_type(cur_type, dst_type, accept);
          tx.dst_id = dst_id;
        }
        const int candidate_op = tx.op_type;
        if (candidate_op != 0) {
          record_recb_stratum_outcome(block_strata, direction, candidate_op,
                                      num_re, num_sia, RECB_ELIGIBLE);
        }

        bool locks_acquired = false;
        if (tx.op_type != 0) {
          locks_acquired = try_acquire_bitmap_locks(tx.src_id, tx.dst_id);
          if (locks_acquired) __threadfence();
          else {
            ++lock_failures;
            record_recb_stratum_outcome(block_strata, direction, candidate_op,
                                        num_re, num_sia, RECB_LOCK_FAIL);
            tx.op_type = 0;
          }
        }

        const bool need_verify = locks_acquired && tx.op_type != 0;
        const LatticeTypes::lat_type safe_src = need_verify ? getType(tx.src_id) : cur_type;
        const LatticeTypes::lat_type safe_dst = need_verify ? getType(tx.dst_id) : dst_type;
        const bool src_ok = !need_verify || safe_src == cur_type;
        const bool dst_ok = !need_verify || safe_dst == dst_type;
        if (need_verify) {
          if (!src_ok) ++src_failures;
          if (!dst_ok) {
            if (dst_type == LatticeTypes::Mo) ++dst_mo_failures;
            else if (dst_type == LatticeTypes::Re) ++dst_re_failures;
            else ++dst_other_failures;
          }
          if (!src_ok) {
            record_recb_stratum_outcome(block_strata, direction, candidate_op,
                                        num_re, num_sia, RECB_SRC_CHANGED);
            if (g_Pair_Atoms[slot] == tx.src_id) g_Pair_Atoms[slot] = -1;
            stale = 1;
            if (stats) atomicAdd(&stats->stale_src_changed, 1u);
            tx.active = false;
            tx.op_type = 0;
          } else if (!dst_ok) {
            record_recb_stratum_outcome(block_strata, direction, candidate_op,
                                        num_re, num_sia, RECB_DST_CHANGED);
            tx.op_type = 0;
          }
        }

        const bool transaction_attempted = need_verify && src_ok && dst_ok;
        const int attempted_op = transaction_attempted ? tx.op_type : 0;
        if (verify_detail && stats && attempted_op >= 1 && attempted_op <= 4)
          atomicAdd(&stats->op_attempts[attempted_op - 1], 1ULL);
        if (tx.op_type != 0) perform_batch_update_single(tx);
        if (transaction_attempted && !tx.success) ++tx_failures;
        if (transaction_attempted) {
          record_recb_stratum_outcome(block_strata, direction, candidate_op,
                                      num_re, num_sia,
                                      tx.success ? RECB_SUCCESS : RECB_TX_FAIL);
        }
        if (verify_detail && stats && attempted_op >= 1 && attempted_op <= 4) {
          if (tx.success) atomicAdd(&stats->op_successes[attempted_op - 1], 1ULL);
          else atomicAdd(&stats->op_failures[attempted_op - 1], 1ULL);
        }
        if (locks_acquired) {
          __threadfence();
          release_bitmap_locks(tx.src_id, tx.dst_id);
        }

        if (tx.active && tx.op_type != 0 && tx.success) {
          x = nx; y = ny; z = nz;
          tx.src_id = id = dst_id;
          if (tx.op_type == 2 || tx.op_type == 4) cur_type = LatticeTypes::MoRe;
          ++hops;
          atomicExch(&g_recb_pairs_changed, 1u);
          g_Pair_Atoms[slot] = dst_id;
          const bool in_sector = x_min <= x && x < x_max && y_min <= y && y < y_max &&
                                 z_min <= z && z < z_max;
          if (isGhostLat(dst_id) || !in_sector) {
            exited = 1;
            tx.active = false;
          }
        }
        tx.op_type = 0;
      }
    }
  }

  if (!leader) return;
  if (stats) {
    if (attempts) atomicAdd(&stats->attempts, static_cast<unsigned long long>(attempts));
    if (hops) atomicAdd(&stats->successful_hops, static_cast<unsigned long long>(hops));
    atomicMax(&stats->max_successful_hops, hops);
    if (hops >= 2) atomicAdd(&stats->multi_hop_pairs, 1u);
    if (captured_no_move) atomicAdd(&stats->captured_no_move, 1u);
    if (captured) atomicAdd(&stats->captured_walks, 1u);
    if (exited) atomicAdd(&stats->exited_walks, 1u);
    if (capped) atomicAdd(&stats->capped_walks, 1u);
    if (stale) atomicAdd(&stats->stale_source_walkers, 1u);
    if (!captured && !exited && !capped && !stale) atomicAdd(&stats->unclassified_walkers, 1u);
    if (lock_failures) atomicAdd(&stats->lock_failures, static_cast<unsigned long long>(lock_failures));
    if (src_failures) atomicAdd(&stats->src_check_failures, static_cast<unsigned long long>(src_failures));
    if (dst_mo_failures) atomicAdd(&stats->dst_mo_check_failures, static_cast<unsigned long long>(dst_mo_failures));
    if (dst_re_failures) atomicAdd(&stats->dst_re_check_failures, static_cast<unsigned long long>(dst_re_failures));
    if (dst_other_failures) atomicAdd(&stats->dst_other_check_failures, static_cast<unsigned long long>(dst_other_failures));
    if (tx_failures) atomicAdd(&stats->transaction_failures, static_cast<unsigned long long>(tx_failures));
    if (verify_detail) {
      if (target_mo) atomicAdd(&stats->target_mo, static_cast<unsigned long long>(target_mo));
      if (target_re) atomicAdd(&stats->target_re, static_cast<unsigned long long>(target_re));
      if (target_other) atomicAdd(&stats->target_other, static_cast<unsigned long long>(target_other));
    }
  }
  if (collect_hash_stats && walker_lengths) {
    walker_lengths[work_index] =
        (static_cast<unsigned long long>(extra_rng) << 32) | attempts;
  }
}

__global__ void __launch_bounds__(128) calcExchangePairsTile8(
    int sector_id, int work_count, RecbVerifyStats* stats,
    const long int* __restrict__ sector_slots, bool verify_detail,
    bool collect_hash_stats, unsigned long long* __restrict__ walker_lengths,
    unsigned int* __restrict__ work_head) {
  __shared__ unsigned long long
      block_strata[RECB_STRATUM_COUNT * RECB_STRATUM_OUTCOME_COUNT];
  if (verify_detail) {
    for (int i = threadIdx.x;
         i < RECB_STRATUM_COUNT * RECB_STRATUM_OUTCOME_COUNT;
         i += blockDim.x) block_strata[i] = 0ULL;
    __syncthreads();
  }
  cg::thread_block block = cg::this_thread_block();
  auto tile = cg::tiled_partition<8>(block);
  while (true) {
    unsigned int work_index = 0;
    if (tile.thread_rank() == 0) work_index = atomicAdd(work_head, 1u);
    work_index = tile.shfl(work_index, 0);
    if (work_index >= static_cast<unsigned int>(work_count)) break;
    run_recb_walker_tile8(sector_id, static_cast<int>(work_index), stats,
                          sector_slots, verify_detail, collect_hash_stats,
                          walker_lengths, verify_detail ? block_strata : nullptr,
                          tile);
  }
  if (verify_detail) {
    __syncthreads();
    for (int i = threadIdx.x;
         i < RECB_STRATUM_COUNT * RECB_STRATUM_OUTCOME_COUNT;
         i += blockDim.x) {
      const unsigned long long value = block_strata[i];
      if (value) atomicAdd(&stats->strata[0][0] + i, value);
    }
  }
}

__global__ void collect_recb_walker_stats(
    const unsigned long long* __restrict__ walker_lengths, int count) {
  __shared__ unsigned int block_attempts[128];
  __shared__ unsigned int block_extra_rng[128];
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = tid < static_cast<unsigned int>(count);
  const unsigned long long packed = valid ? walker_lengths[tid] : 0ULL;
  const unsigned int attempts = static_cast<unsigned int>(packed);
  const unsigned int extra_rng = static_cast<unsigned int>(packed >> 32);
  block_attempts[threadIdx.x] = attempts;
  block_extra_rng[threadIdx.x] = extra_rng;
  __syncthreads();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    g_recb_wave_size = static_cast<unsigned int>(warpSize);
  }
  if (valid) {
    atomicAdd(&g_recb_walker_attempt_hist[recb_walker_attempt_bin(attempts)], 1ULL);
    atomicMax(&g_recb_walker_attempt_max, attempts);
  }

  const unsigned int lane = threadIdx.x % warpSize;
  if (lane != 0u) return;
  const unsigned int base = threadIdx.x;
  unsigned int active_lanes = 0;
  unsigned int wave_max = 0;
  unsigned long long wave_sum = 0;
  unsigned long long extra_sum = 0;
  #pragma unroll 1
  for (unsigned int i = 0; i < warpSize && base + i < blockDim.x; ++i) {
    if (blockIdx.x * blockDim.x + base + i >= static_cast<unsigned int>(count)) break;
    const unsigned int value = block_attempts[base + i];
    ++active_lanes;
    wave_sum += value;
    extra_sum += block_extra_rng[base + i];
    if (value > wave_max) wave_max = value;
  }
  if (active_lanes > 0u) {
    atomicAdd(&g_recb_wave_tail[0], wave_sum);
    atomicAdd(&g_recb_wave_tail[1],
              static_cast<unsigned long long>(wave_max) * active_lanes);
    atomicAdd(&g_recb_wave_tail[2],
              static_cast<unsigned long long>(wave_max) * warpSize);
    atomicAdd(&g_recb_wave_tail[3], 1ULL);
    if (extra_sum > 0ULL) atomicAdd(&g_recb_extra_rng_draws, extra_sum);
  }
}

__global__ void calcExchangePairsTemp(int sector_id,int arr_size
){
  // print_g_Pair_Atoms(arr_size);
  // checkpointers();
  if(g_Pair_Atoms == nullptr)
    return;
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= arr_size)
    return;
  // if(tid >= 1)
  //   return;
  
  _type_lattice_id id = g_Pair_Atoms[tid];
  if(id <= 0)
    return;
  LatticeTypes::lat_type cur_type = getType(id);
  
  _type_lattice_size x = id%g_meta->size_x;
  _type_lattice_size y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_size z = id / (g_meta->size_x * g_meta->size_y);
  const long int x_min = 2*g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2*g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];
  // printf("id %ld, cur_type is %ld,(%ld,%ld,%ld) -----(%ld-%ld)(%ld-%ld)(%ld-%ld)\n",id,cur_type,x,y,z,x_min,x_max,y_min,y_max,z_min,z_max);

  //if not in this domain then skip

   // 定义区域边界
   

    // 检查每个维度的边界条件
    bool x_ok = (x_min <= x && x < x_max);
    bool y_ok = (y_min <= y && y < y_max);
    bool z_ok = (z_min <= z && z < z_max);

    // // 输出调试信息
  
  // if(!(2 * g_meta->regions[sector_id * 6 + 0] <= x && x < 2*g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5])){
  // //   // // // 输出调试信息
  //   // printf("id %ld: x_ok=%d, y_ok=%d, z_ok=%d\n", 
  //   //       id, x_ok, y_ok, z_ok);
  //   return;
  // }
    // printf("cal atom msg:\n");s
    // printf("Sector %d: x=[%ld,%ld), y=[%ld,%ld), z=[%ld,%ld)\t", 
    //       sector_id, x_min, x_max, y_min, y_max, z_min, z_max);
    // printf("Point %ld:(%ld,%ld,%ld): %d \n", id,
    //       x, y, z,getType(id));


  
  int numSIA=0;
  int numRe=0;
  int ret = 0;

  int num_more = 0;
  int num_momo = 0;
  int busy = 0;

  for(int i =0;i<g_MoRe_Hash->capacity;i++){
    if(g_MoRe_Hash->keys[i]!=-1&&g_MoRe_Hash->keys[i]!=-2)
      num_more++;
  }

  for(int i =0;i<g_MoMo_Hash->capacity;i++){
    if(g_MoMo_Hash->keys[i]!=-1&&g_MoMo_Hash->keys[i]!=-2)
      num_momo++;
  }

  // printf("begin %d + %d = %d \n",num_more,num_momo,num_momo+num_more);
  dev_Lattice nn_list[LatticesList::MAX_1NN];
  _type_lattice_size temp_x, temp_y, temp_z;
  dev_Lattice randomLattice;
  int loop_index = 0;
  while (2 * g_meta->regions[sector_id * 6 + 0] <= x && x < 2*g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5]) 
  {

    // printf("id:%ld enter loop:%d\n",id,loop_index);
    loop_index++;
    if(loop_index >=10)
      break;

    id = getId(x,y,z);

    if(cur_type != getType(id)){
      // printf("%d:%d id has wrong type %d - %d\n",tid,id,cur_type,getType(id));
    }

 
    if(hash_set_contains(g_Busy_Set,id)){
      continue;
    }else{
      ret = hash_set_insert_atom_op(g_Busy_Set,id);
      busy++;
    }
    numSIA = 0;
    get1nn(x,y,z,nn_list);
    for(int i=0;i < LatticesList::MAX_1NN;i++){
      if(nn_list[i].type == LatticeTypes::MoMo || nn_list[i].type == LatticeTypes::MoRe)
        numSIA++;
    }

    if(numSIA >= 2) {
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
      busy--;
      return;
    }
    numRe = 0;
    
    if (cur_type == LatticeTypes::MoRe) {
        for(int i=0;i < LatticesList::MAX_1NN;i++) {
          if (nn_list[i].type == LatticeTypes::Re) numRe++;
        }
        if (numRe >= 5) {
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1)
          busy--;
          return;
        }
    }

    int sum = tid + id + loop_index;
    int randomNumber = (sum % 14 + 14) % 14;  // 确保结果在0-13之间

    if(randomNumber < 0 || randomNumber >= 14){
        printf("%ld : random num generate error %d\n",(long)tid,randomNumber);
      }
    getRandomLattice(x,y,z,temp_x,temp_y,temp_z,randomNumber);
    randomLattice.id = getId(temp_x,temp_y,temp_z);

    
    if(hash_set_contains(g_Busy_Set,randomLattice.id)){
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
      busy--;
      continue;
    }else{
      ret = hash_set_insert_atom_op(g_Busy_Set,randomLattice.id);
      busy++;
    }
    // printf("%d:is finish select random number\n",tid);
    randomLattice.type = getType(randomLattice.id);
    LatticeTypes::lat_type rantype = randomLattice.type;

    // printf("%ld --- %ld(%ld,%ld,%ld)->%ld(%ld,%ld,%ld),\t%d->%d\n",tid,id,x,y,z,randomLattice.id,temp_x,temp_y,temp_z,cur_type,randomLattice.type);


    if(randomLattice.type == LatticeTypes::Mo || randomLattice.type == LatticeTypes::Re){
      
      if(cur_type == LatticeTypes::MoMo){
        bool exchanged = false;
        if (randomLattice.type == LatticeTypes::Mo) {
          int rm = hash_set_remove_atom_op(g_MoMo_Hash,id);
          if (rm != 1) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int ins = hash_set_insert_atom_op(g_MoMo_Hash,randomLattice.id);
          if (ins != 1) {
            hash_set_insert_atom_op(g_MoMo_Hash,id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          exchanged = true;
        } else if(randomLattice.type == LatticeTypes::Re) {
          int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
          if (rm_re != 1) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int rm_momo = hash_set_remove_atom_op(g_MoMo_Hash,id);
          if (rm_momo != 1) {
            hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
          if (ins_more != 1) {
            hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
            hash_set_insert_atom_op(g_MoMo_Hash,id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          cur_type = LatticeTypes::MoRe;
          exchanged = true;
        }

        if (!exchanged) {
          ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
          if(ret == 1) busy--;
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1) busy--;
          continue;
        }

        x = temp_x;
        y = temp_y;
        z = temp_z;
        
        if (isGhostLat(randomLattice.id)) {
          ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
        }
        if (isSurfaceLat(randomLattice.id)){
          ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
        }
          
        if (isSurfaceLat(id)){
          ret = hash_set_insert_atom_op(g_surface_Hash,id);
        }
      }
      else if(cur_type == LatticeTypes::MoRe){
        if(numRe == 0){
          bool exchanged = false;
          if (randomLattice.type == LatticeTypes::Mo) {
            int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
            if (rm_more != 1) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
            if (ins_more != 1) {
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            exchanged = true;
          } else {
            int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
            if (rm_re != 1) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
            if (rm_more != 1) {
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_re = hash_set_insert_atom_op(g_Re_Hash,id);
            if (ins_re != 1) {
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
            if (ins_more != 1) {
              hash_set_remove_atom_op(g_Re_Hash,id);
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            exchanged = true;
          }

          if (!exchanged) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          x = temp_x;
          y = temp_y;
          z = temp_z;
          // printf("2 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));

          //TODO:
          if (isGhostLat(randomLattice.id)) {
            ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
          }
          if (isSurfaceLat(randomLattice.id))
            ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
          if (isSurfaceLat(id)) 
            ret = hash_set_insert_atom_op(g_surface_Hash,id);
        }
        else if(numRe > 0 && numRe < LatticesList::MAX_1NN){
          double ranmov = randomFloat(tid + id + loop_index);
          if(ranmov > (static_cast<double>(numRe) / LatticesList::MAX_1NN)) {
            bool exchanged = false;
            if (randomLattice.type == LatticeTypes::Mo) {
              int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
              if (rm_more != 1) {
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
              if (ins_more != 1) {
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              exchanged = true;
            } else if(randomLattice.type == LatticeTypes::Re) {
              int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
              if (rm_re != 1) {
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
              if (rm_more != 1) {
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_re = hash_set_insert_atom_op(g_Re_Hash,id);
              if (ins_re != 1) {
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
              if (ins_more != 1) {
                hash_set_remove_atom_op(g_Re_Hash,id);
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              exchanged = true;
            } else {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }

            if (!exchanged) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            x = temp_x;
            y = temp_y;
            z = temp_z;
            // printf("3 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));
            //TODO
            if (isGhostLat(randomLattice.id)) {
              ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
            }
            if (isSurfaceLat(randomLattice.id)) 
              ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
            if (isSurfaceLat(id)) 
              ret = hash_set_insert_atom_op(g_surface_Hash,id);
         }else continue;
      }
        else{
          ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
          if(ret == 1)
          busy--;
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1)
          busy--;
          // printf("%d : more : %d momo : %d \n",tid,num_more,num_momo);
          return;
        } 
    }
    else {
      ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
      if(ret == 1)
        busy--;
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
        busy--;
      continue;
      }
    }  
    ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
    if(ret == 1)
    busy--;
    ret = hash_set_remove_atom_op(g_Busy_Set,id);
    if(ret == 1)
    busy--;

    
  }

}

// 测试内核
__global__ void test_kernel(GPUHashSet* gpu_set) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < gpu_set->capacity) {
      int key = gpu_set->keys[tid];
      if (key != gpu_set->empty_flag) {
          printf("Key at index %d: %d\n", tid, key);
      }
  }
}

__global__ void get_surface_and_ghost_type(long int *surface_type,long int *ghost_type){
      _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
      if(tid == 0){
        // printf("tp capacity:%ld\n",g_surface_Hash->capacity);
        for(int i=0;i<g_surface_Hash->capacity;i++){
          if(g_surface_Hash->keys[i] != -1 && g_surface_Hash->keys[i] != -2){
            // printf("tp2 %ld\t", g_surface_Hash->keys[i]);
            surface_type[i] = getType(g_surface_Hash->keys[i]);
            // printf("tp %ld\t", getType(g_surface_Hash->keys[i]));
          }
            
        }
        //  printf("\n");
      }else if(tid == 1){
        // printf("tp capacity:sld\n",g_ghost_Hash->capacity);
        for(int i=0;i<g_ghost_Hash->capacity;i++){
          if(g_ghost_Hash->keys[i] != -1 && g_ghost_Hash->keys[i] != -2){
            // printf("tp3 %ld\t", g_ghost_Hash->keys[i]);
            ghost_type[i] = getType(g_ghost_Hash->keys[i]);
            // printf("tp1 %ld\t", getType(g_ghost_Hash->keys[i]));
          }
            
        }
        // printf("\n");
    }
}
__global__ void compute_types_for_keys(const long int* __restrict__ keys,
                                      long int* __restrict__ out_types,
                                      int capacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    long int k = keys[tid];
    if (k != -1 && k != -2) out_types[tid] = getType(k);
}
__global__ void compact_valid_keys(const long int* __restrict__ keys,
                                  int capacity,
                                  long int* __restrict__ out_keys,
                                  int* __restrict__ out_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    long int k = keys[tid];
    if (k == -1 || k == -2) return;
    int idx = atomicAdd(out_count, 1);
    out_keys[idx] = k;
}

__global__ void filter_ids_in_region(const long int* __restrict__ in_ids,
                                     int in_count,
                                     int x_low, int x_high,
                                     int y_low, int y_high,
                                     int z_low, int z_high,
                                     long int* __restrict__ out_ids,
                                     int* __restrict__ out_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_count) return;
  long int id = in_ids[tid];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (x_low <= x && x < x_high && y_low <= y && y < y_high && z_low <= z && z < z_high) {
    int idx = atomicAdd(out_count, 1);
    out_ids[idx] = id;
  }
}

// recb 前筛选：把当前扇区盒子内的 walker 槽位号（g_Pair_Atoms 下标）压缩到 out_slots。
// 存槽位号而非 id：calcExchangePairs 游走后会写回 g_Pair_Atoms[slot]（新位置或 -1），
// 用槽位号保证写回永远对准自己的槽位。边界口径与 calcExchangePairs 完全一致：
// x 为双倍坐标（2*regions），y/z 直接用 regions；id<=0 为无效槽（离扇区标记）跳过。
__device__ __forceinline__ bool recb_event_affects_pair(long int pair_id) {
  if (!g_recb_event_valid) return true;
  if (pair_id == g_recb_event_from || pair_id == g_recb_event_to) return true;
  const _type_lattice_id x = pair_id % g_meta->size_x;
  const _type_lattice_id y = (pair_id / g_meta->size_x) % g_meta->size_y;
  const _type_lattice_id z = pair_id / (g_meta->size_x * g_meta->size_y);
  _type_lattice_id nn[8];
  get1nn_ids(x, y, z, nn);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (nn[i] == g_recb_event_from || nn[i] == g_recb_event_to) return true;
  }
  return false;
}

__global__ void filter_pair_slots_for_sector(int sector_id, long int roster_size,
                                             long int* __restrict__ out_slots,
                                             int* __restrict__ out_count,
                                             bool event_local_enabled,
                                             bool audit_event_local) {
  long int slot = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  if (slot >= roster_size) return;
  long int id = g_Pair_Atoms[slot];
  if (id <= 0) return;
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  const long int x_min = 2 * g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2 * g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];
  if (x_min <= x && x < x_max && y_min <= y && y < y_max && z_min <= z && z < z_max) {
    const bool local_pass = event_local_enabled && g_recb_sector_stable[sector_id];
    if (local_pass && !recb_event_affects_pair(id)) {
      if (!audit_event_local) return;

      // Debug-only full audit. A pair excluded by the locality predicate must
      // still satisfy the exact capture predicate used by the tile8 walker.
      atomicAdd(&g_recb_local_audit[0], 1u);
      const LatticeTypes::lat_type type = getType(id);
      int num_sia = 0, num_re = 0;
      tiled_check_neighbors_single(x, y, z, num_sia, num_re);
      const bool captured = num_sia >= 2 ||
          (type == LatticeTypes::MoRe && num_re >= 5);
      if (captured) return;

      // Do not let a debug run silently continue with a missed active pair:
      // recover it into this launch and force the following pass to be full.
      atomicAdd(&g_recb_local_audit[1], 1u);
      atomicExch(&g_recb_pairs_changed, 1u);
    }
    int idx = atomicAdd(out_count, 1);
    out_slots[idx] = slot;
  }
}

__global__ void begin_recb_pass(int sector_id, bool event_local_enabled) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    g_recb_pairs_changed = 0u;
    g_recb_was_full_pass = !event_local_enabled ||
                           g_recb_sector_stable[sector_id] == 0u;
    g_recb_local_audit[0] = 0u;
    g_recb_local_audit[1] = 0u;
  }
}

__global__ void finish_recb_pass(int sector_id) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (g_recb_pairs_changed) {
    g_recb_sector_stable[sector_id] = 0u;
  } else if (g_recb_was_full_pass) {
    g_recb_sector_stable[sector_id] = 1u;
  }
}

__global__ void invalidate_recb_stability() {
  if (blockIdx.x == 0 && threadIdx.x < 8) g_recb_sector_stable[threadIdx.x] = 0u;
}

// 正确性校验：仍留在本地 roster 中的每个 slot，必须指向 MoMo 或 MoRe。
// 独立 kernel 在随机游走完成后执行，避免与仍在运行的 walker 发生读写竞争。
// 已确认不再是 pair 的陈旧槽直接注销，避免下一 sector 再次调度错误 walker。
__global__ void validate_pair_slots(const long int* __restrict__ slots,
                                    int slot_count,
                                    RecbVerifyStats* stats) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= slot_count) return;
  const long int slot = slots[tid];
  const long int id = g_Pair_Atoms[slot];
  if (id <= 0) return;
  const LatticeTypes::lat_type type = getType(id);
  if (type != LatticeTypes::MoMo && type != LatticeTypes::MoRe) {
    if (stats) atomicAdd(&stats->roster_type_mismatches, 1u);
    g_Pair_Atoms[slot] = -1;
  }
}

// rebuild 后全量校验：清除花名册里指向"已不在 g_MoMo_Hash/g_MoRe_Hash 的 id"的陈旧 slot。
// 间隙对游走到 y 通信边界后，通信 update_hash_set 会把它移交给邻居（从 g_MoMo_Hash 删除，
// 旧位置变 Mo），花名册 slot 若未同步就会 stale。rebuild 从哈希表重建本应覆盖，但跨扇区
// 通信后可能仍有残留（update_hash_set 与 rebuild 之间的边界格点），这里兜底清理，
// 保证花名册与哈希表一致。
__global__ void reset_validate_counters() {
  g_validate_counters[0] = 0;
  g_validate_counters[1] = 0;
  g_validate_counters[2] = 0;
  g_validate_counters[3] = 0;
  g_hash_occ[0] = 0; g_hash_occ[1] = 0; g_hash_occ[2] = 0;
  g_hash_occ[3] = 0; g_hash_occ[4] = 0; g_hash_occ[5] = 0;
  g_hash_occ[6] = 0; g_hash_occ[7] = 0; g_hash_occ[8] = 0;
}

// 清零每张表的哈希操作统计计数器。
__global__ void reset_hash_stats() {
  for (int i = 0; i < RECB_HASH_TABLE_COUNT; ++i) {
    g_hash_query[i] = 0;
    g_hash_update[i] = 0;
    g_probe_max[i] = 0;
  }
  for (int i = 0; i < 7; ++i) g_re_probe_reach[i] = 0;
  for (int i = 0; i < RECB_WALKER_HIST_BINS; ++i) {
    g_recb_walker_attempt_hist[i] = 0;
  }
  for (int i = 0; i < 4; ++i) g_recb_wave_tail[i] = 0;
  g_recb_walker_attempt_max = 0;
  g_recb_wave_size = 0;
  g_recb_extra_rng_draws = 0;
}

__global__ void reset_more_occupancy(unsigned int capacity) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    (void)capacity;
    g_more_dirty = 0;
  }
}

__global__ void reset_hash_occupancy_slot(int idx) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && idx >= 0 && idx < 3) {
    g_hash_occ[idx * 3 + 0] = 0;
    g_hash_occ[idx * 3 + 1] = 0;
    g_hash_occ[idx * 3 + 2] = 0;
  }
}

// 统计一个哈希表的有效 key 数，累加到 g_validate_counters[2]
__global__ void count_hash_keys_all(GPUHashSet* set) {
  if (!set) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= set->capacity) return;
  const long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  atomicAdd(&g_validate_counters[2], 1u);
}

// 统计一张哈希表的占用分布（empty/tombstone/live），写入 g_hash_occ[idx*3 + 0/1/2]。
// 用于 rebuild 前量化"离满载还有多少余量"：有效占用 = (live + tombstone) / capacity，
// 它决定了开放寻址探测链的长度。idx: 0=MoMo, 1=MoRe, 2=Re。
__global__ void count_hash_occupancy(GPUHashSet* set, int idx) {
  __shared__ unsigned int block_counts[3];
  if (threadIdx.x < 3) block_counts[threadIdx.x] = 0;
  __syncthreads();
  if (!set) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < set->capacity) {
    const long int key = set->keys[i];
    if (key == set->empty_flag) atomicAdd(&block_counts[0], 1u);
    else if (key == set->tombstone_flag) atomicAdd(&block_counts[1], 1u);
    else atomicAdd(&block_counts[2], 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&g_hash_occ[idx * 3 + 0], block_counts[0]);
    atomicAdd(&g_hash_occ[idx * 3 + 1], block_counts[1]);
    atomicAdd(&g_hash_occ[idx * 3 + 2], block_counts[2]);
  }
}

// 统计花名册有效 slot 数，累加到 g_validate_counters[3]
__global__ void count_roster_slots(long int roster_size) {
  long int slot = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  if (slot >= roster_size) return;
  if (g_Pair_Atoms[slot] > 0) atomicAdd(&g_validate_counters[3], 1u);
}

// 从花名册重新插入有效 key 到哈希表。用于 rebuild 时清理 tombstone：
// clear 哈希表（全部置 empty）后，把 compact 收集到的有效 key 重新 hash 插入，
// 从而消除长期 remove 累积的 tombstone（tombstone 会拉长开放寻址探测链，
// 占用度逼近 1 时查找/插入会退化，甚至把“存在”误判为“不存在”，
// 进而被 validate_roster_all 误清，造成 roster/hash 差值累积）。
__global__ void reinsert_from_roster(GPUHashSet* set, const long int* roster,
                                     long int start, long int end) {
  const long int i = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  const long int idx = start + i;
  if (!set || !roster || idx >= end) return;
  hash_set_insert_atom_op(set, roster[idx]);
}

// recb 专用 MoRe 全表核验/压缩。每个 block 先在 shared memory 内归约，
// 全局只提交 empty/tombstone/live 三个计数；LIVE scratch 也按 block 一次性
// 预留连续区间，避免每个槽争抢全局计数器。
__global__ void compact_more_for_refresh(GPUHashSet* set, long int* live_keys,
                                         unsigned int* counts,
                                         long int live_keys_capacity) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned int block_counts[3];
  __shared__ unsigned int block_live_write;
  __shared__ unsigned int block_live_base;
  if (threadIdx.x < 3) block_counts[threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    block_live_write = 0;
    block_live_base = 0;
  }
  __syncthreads();

  bool is_live = false;
  long int key = GPU_HASH_EMPTY_KEY;
  if (set && live_keys && counts && i < set->capacity) {
    key = set->keys[i];
    if (key == set->empty_flag) {
      atomicAdd(&block_counts[0], 1u);
    } else if (key == set->tombstone_flag) {
      atomicAdd(&block_counts[1], 1u);
    } else {
      is_live = true;
      atomicAdd(&block_counts[2], 1u);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && set && live_keys && counts) {
    atomicAdd(&counts[0], block_counts[0]);
    atomicAdd(&counts[1], block_counts[1]);
    block_live_base = atomicAdd(&counts[2], block_counts[2]);
  }
  __syncthreads();

  if (is_live) {
    const unsigned int local_pos = atomicAdd(&block_live_write, 1u);
    const unsigned int global_pos = block_live_base + local_pos;
    if (static_cast<long int>(global_pos) < live_keys_capacity) {
      live_keys[global_pos] = key;
    }
  }
}

__global__ void reinsert_more_after_refresh(GPUHashSet* set,
                                            const long int* live_keys,
                                            unsigned int live_count,
                                            unsigned int* counts) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!set || !live_keys || !counts || i >= live_count) return;
  if (hash_set_insert_atom_op(set, live_keys[i]) != 1) {
    atomicAdd(&counts[3], 1u);
  }
}

static void refreshMoReAfterRecb(unsigned int sector_id, int launch_n,
                                 bool correctness_debug, bool hash_stats) {
  using namespace gpuSimulate;
  if (!MoRe_Hash || !MoRe_Hash->device_ptr() || !d_more_refresh_keys ||
      !d_more_refresh_counts || pair_atoms_size <= 0) {
    return;
  }

  const int block = 256;
  const int capacity = MoRe_Hash->get_capacity();
  unsigned int dirty = 0;
  HANDLE_HIP(hipMemcpyFromSymbol(&dirty, HIP_SYMBOL(g_more_dirty), sizeof(dirty)));

  unsigned int h_counts[4] = {h_more_cached_counts[0], h_more_cached_counts[1],
                              h_more_cached_counts[2], 0};
  const bool used_full_scan =
      correctness_debug || dirty != 0 || !h_more_cache_initialized;

  if (used_full_scan) {
    HANDLE_HIP(hipMemset(d_more_refresh_counts, 0, 4 * sizeof(unsigned int)));
    const int grid = (capacity + block - 1) / block;
    hipLaunchKernelGGL(compact_more_for_refresh, grid, block, 0, 0,
                       MoRe_Hash->device_ptr(), d_more_refresh_keys,
                       d_more_refresh_counts, pair_atoms_size);
    CHECK_FOR_ERROR();
    HANDLE_HIP(hipMemcpy(h_counts, d_more_refresh_counts,
                         4 * sizeof(unsigned int), hipMemcpyDeviceToHost));
    h_more_cached_counts[0] = h_counts[0];
    h_more_cached_counts[1] = h_counts[1];
    h_more_cached_counts[2] = h_counts[2];
    h_more_cache_initialized = true;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const double effective_load =
      capacity > 0 ? (h_counts[1] + h_counts[2]) / static_cast<double>(capacity) : 0.0;
  if (correctness_debug || hash_stats) {
    kiwi::logs::v(" ", " [recb-more-occupancy] rank={} sect={} launch={} source={} cap={} empty={} tombstone={} live={} eff_load={:.3f} action={}.\n",
                  rank, sector_id, launch_n,
                  used_full_scan ? "full-scan" : "cached-clean", capacity,
                  h_counts[0], h_counts[1], h_counts[2], effective_load,
                  h_counts[1] == 0 ? "skip-clean" : "refresh");
  }

  if (used_full_scan) {
    const unsigned long long classified_slots =
        static_cast<unsigned long long>(h_counts[0]) + h_counts[1] + h_counts[2];
    if (classified_slots != static_cast<unsigned long long>(capacity)) {
      kiwi::logs::v(" ", " [recb-more-refresh-fatal] rank={} sect={} launch={} occupancy_sum={} cap={}（占用统计不完整，禁止 clear）.\n",
                    rank, sector_id, launch_n, classified_slots, capacity);
      MPI_Abort(MPI_COMM_WORLD, 90);
      return;
    }

    if (h_counts[2] > static_cast<unsigned int>(pair_atoms_size)) {
      kiwi::logs::v(" ", " [recb-more-refresh-fatal] rank={} sect={} launch={} live={} scratch_capacity={}（LIVE key 尚未全部保存，禁止 clear 以免破坏哈希表）.\n",
                    rank, sector_id, launch_n, h_counts[2], pair_atoms_size);
      MPI_Abort(MPI_COMM_WORLD, 91);
      return;
    }
  }

  // 墓碑为零时无需 clear/reinsert；debug 模式虽然完成了全表核验，也直接返回。
  if (h_counts[1] == 0) {
    if (dirty != 0) {
      const unsigned int clean = 0;
      HANDLE_HIP(hipMemcpyToSymbol(HIP_SYMBOL(g_more_dirty), &clean, sizeof(clean)));
    }
    return;
  }

  // 此时所有 LIVE key 已完整保存。整表 clear 后再插回，墓碑转为 EMPTY，
  // 不会产生“在原探测链中间直接清墓碑”导致的 contains 提前终止。
  MoRe_Hash->clear();
  hipLaunchKernelGGL(reset_more_occupancy, 1, 1, 0, 0,
                     static_cast<unsigned int>(capacity));
  CHECK_FOR_ERROR();
  if (h_counts[2] > 0) {
    const int reinsert_grid = (static_cast<int>(h_counts[2]) + block - 1) / block;
    hipLaunchKernelGGL(reinsert_more_after_refresh, reinsert_grid, block, 0, 0,
                       MoRe_Hash->device_ptr(), d_more_refresh_keys,
                       h_counts[2], d_more_refresh_counts);
    CHECK_FOR_ERROR();
    HANDLE_HIP(hipDeviceSynchronize());
    HANDLE_HIP(hipMemcpy(&h_counts[3], d_more_refresh_counts + 3,
                         sizeof(unsigned int), hipMemcpyDeviceToHost));
  }
  if (h_counts[3] != 0) {
    kiwi::logs::v(" ", " [recb-more-refresh-fatal] rank={} sect={} launch={} reinsert_failures={} live={} cap={}.\n",
                  rank, sector_id, launch_n, h_counts[3], h_counts[2], capacity);
    MPI_Abort(MPI_COMM_WORLD, 92);
    return;
  }
  h_more_cached_counts[0] = static_cast<unsigned int>(capacity) - h_counts[2];
  h_more_cached_counts[1] = 0;
  h_more_cached_counts[2] = h_counts[2];
  const unsigned int clean = 0;
  HANDLE_HIP(hipMemcpyToSymbol(HIP_SYMBOL(g_more_dirty), &clean, sizeof(clean)));
}

static void initializeMoreOccupancyCounter() {
  using namespace gpuSimulate;
  if (!MoRe_Hash || !MoRe_Hash->device_ptr() || !d_more_refresh_keys ||
      !d_more_refresh_counts || pair_atoms_size <= 0) return;

  const int block = 256;
  const int capacity = MoRe_Hash->get_capacity();
  const int grid = (capacity + block - 1) / block;
  HANDLE_HIP(hipMemset(d_more_refresh_counts, 0, 4 * sizeof(unsigned int)));
  hipLaunchKernelGGL(compact_more_for_refresh, grid, block, 0, 0,
                     MoRe_Hash->device_ptr(), d_more_refresh_keys,
                     d_more_refresh_counts, pair_atoms_size);
  CHECK_FOR_ERROR();
  unsigned int h_counts[4] = {};
  HANDLE_HIP(hipMemcpy(h_counts, d_more_refresh_counts,
                       4 * sizeof(unsigned int), hipMemcpyDeviceToHost));
  const unsigned long long sum = static_cast<unsigned long long>(h_counts[0]) +
                                 h_counts[1] + h_counts[2];
  if (sum != static_cast<unsigned long long>(capacity) ||
      h_counts[2] > static_cast<unsigned int>(pair_atoms_size)) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    kiwi::logs::v(" ", " [recb-more-counter-init-fatal] rank={} cap={} empty={} tombstone={} live={} scratch_capacity={}.\n",
                  rank, capacity, h_counts[0], h_counts[1], h_counts[2],
                  pair_atoms_size);
    MPI_Abort(MPI_COMM_WORLD, 93);
    return;
  }
  h_more_cached_counts[0] = h_counts[0];
  h_more_cached_counts[1] = h_counts[1];
  h_more_cached_counts[2] = h_counts[2];
  h_more_cache_initialized = true;
  const unsigned int clean = 0;
  HANDLE_HIP(hipMemcpyToSymbol(HIP_SYMBOL(g_more_dirty), &clean, sizeof(clean)));
}

static void logReHashOccupancy(unsigned int sector_id, int launch_n, int rank) {
  using namespace gpuSimulate;
  if (!Re_Hash || !Re_Hash->device_ptr()) return;
  const int capacity = Re_Hash->get_capacity();
  if (capacity <= 0) return;
  const int block = 256;
  const int grid = (capacity + block - 1) / block;
  hipLaunchKernelGGL(reset_hash_occupancy_slot, 1, 1, 0, 0, 2);
  hipLaunchKernelGGL(count_hash_occupancy, grid, block, 0, 0,
                     Re_Hash->device_ptr(), 2);
  CHECK_FOR_ERROR();
  unsigned int h_occ[9] = {};
  HANDLE_HIP(hipMemcpyFromSymbol(h_occ, HIP_SYMBOL(g_hash_occ), sizeof(h_occ)));
  if (rank == 0) {
    const double effective_load =
        (h_occ[7] + h_occ[8]) / static_cast<double>(capacity);
    kiwi::logs::v(" ", " [recb-re-occupancy] sect={} launch={} cap={} empty={} tombstone={} live={} eff_load={:.3f}.\n",
                  sector_id, launch_n, capacity, h_occ[6], h_occ[7], h_occ[8],
                  effective_load);
  }
}

__global__ void validate_roster_all(long int roster_size, bool verify_detail) {
  long int slot = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  if (slot >= roster_size) return;
  const long int id = g_Pair_Atoms[slot];
  if (id <= 0) return;
  const LatticeTypes::lat_type type = getType(id);
  if (type != LatticeTypes::MoMo && type != LatticeTypes::MoRe) {
    // 兜底清理是正确性修复逻辑，始终执行；计数与 printf 仅为排查，收到
    // KMC_RECB_CORRECTNESS_DEBUG 开关里，关闭时零开销。
    g_Pair_Atoms[slot] = -1;
    if (verify_detail) {
      unsigned int seq = atomicAdd(&g_validate_counters[0], 1u);
      if (isGhostLat(id)) atomicAdd(&g_validate_counters[1], 1u);
      // 打印前 8 个被清除的 slot，用于判断：id 是否在通信边界(ghost)、实际类型是什么，
      // 从而区分"validate 误清(探测失败)"还是"真的移交后正确清除"。
      if (seq < 8) {
        _type_lattice_size xx = id % g_meta->size_x;
        _type_lattice_size yy = (id / g_meta->size_x) % g_meta->size_y;
        _type_lattice_size zz = id / (g_meta->size_x * g_meta->size_y);
        printf("[validate-debug] slot=%ld id=%ld type=%d x=%ld y=%ld z=%ld ghost=%d\n",
               (long)slot, (long)id, (int)type, (long)xx, (long)yy, (long)zz, isGhostLat(id));
      }
    }
  }
}

__global__ void count_roster_live_kernel(const long int* roster, long int n, RecbVerifyStats* stats) {
  long int i = blockIdx.x * (long int)blockDim.x + threadIdx.x;
  if (!roster || !stats || i >= n) return;
  if (roster[i] > 0) atomicAdd(&stats->roster_live_count, 1u);
}

__global__ void count_hash_local_pairs_kernel(GPUHashSet* set, RecbVerifyStats* stats) {
  if (!set || !stats) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= set->capacity) return;
  const long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  // 与花名册同口径：统计全部 key（含 ghost 位置）。花名册由 rebuild 从 MoMo/MoRe
  // 全量 compact（不过滤 ghost），且 walker 出 ghost 后槽位保留 ghost id，
  // 因此 roster_live 应与全量哈希 key 数相等；口径不齐会产生不变量误报。
  atomicAdd(&stats->hash_pair_count, 1u);
}

__global__ void build_nn_info_from_vac_list(const long int* __restrict__ vac_ids,
                                           int vac_count,
                                           dev_Vacancy* __restrict__ out_vac,
                                           dev_nnLattice* __restrict__ out_nn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vac_count) return;
  long int vid = vac_ids[i];
  _type_lattice_id x = vid % g_meta->size_x;
  _type_lattice_id y = (vid / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = vid / (g_meta->size_x * g_meta->size_y);
  out_vac[i].id = static_cast<int>(vid);

  dev_Lattice nn1[8];
  dev_Lattice nn2[6];
  get1nn(x, y, z, nn1);
  get2nn(x, y, z, nn2);

  int base = i * NN_TOTAL;
  for (int j = 0; j < 8; j++) {
    out_nn[base + j].id = static_cast<int>(nn1[j].id);
    out_nn[base + j].type._type = nn1[j].type;
  }
  for (int j = 0; j < 6; j++) {
    out_nn[base + 8 + j].id = static_cast<int>(nn2[j].id);
    out_nn[base + 8 + j].type._type = nn2[j].type;
  }

  for (int b = 0; b < 8; b++) {
    long int nid = nn1[b].id;
    _type_lattice_id nx = nid % g_meta->size_x;
    _type_lattice_id ny = (nid / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id nz = nid / (g_meta->size_x * g_meta->size_y);

    dev_Lattice t1[8];
    dev_Lattice t2[6];
    get1nn(nx, ny, nz, t1);
    get2nn(nx, ny, nz, t2);

    int off1 = base + 14 + b * 8;
    for (int j = 0; j < 8; j++) {
      out_nn[off1 + j].id = static_cast<int>(t1[j].id);
      out_nn[off1 + j].type._type = t1[j].type;
    }

    int off2 = base + 78 + b * 6;
    for (int j = 0; j < 6; j++) {
      out_nn[off2 + j].id = static_cast<int>(t2[j].id);
      out_nn[off2 + j].type._type = t2[j].type;
    }
  }
}
__global__ void compute_types_for_list(const long int* __restrict__ list,
                                      long int* __restrict__ out_types,
                                      int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    out_types[tid] = getType(list[tid]);
}
__global__ void get_surface_and_ghost_type1(long int *surface_type, long int *ghost_type) {
    _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 处理surface数据
    if (tid < g_surface_Hash->capacity) {
        if (g_surface_Hash->keys[tid] != -1 && g_surface_Hash->keys[tid] != -2) {
            surface_type[tid] = getType(g_surface_Hash->keys[tid]);
        }
    }
    
    // 处理ghost数据（需要调整索引计算）
    _type_lattice_count ghost_tid = tid - g_surface_Hash->capacity;
    if (tid >= g_surface_Hash->capacity && ghost_tid < g_ghost_Hash->capacity) {
        if (g_ghost_Hash->keys[ghost_tid] != -1 && g_ghost_Hash->keys[ghost_tid] != -2) {
            ghost_type[ghost_tid] = getType(g_ghost_Hash->keys[ghost_tid]);
        }
    }
}
void recb_solver_GPU(
                    const unsigned int& sector_id,long int arr_size,
                    long int *&h_ghost_Hash,long int *&h_surface_Hash,
                    long int *&h_ghost_type, long int *&h_surface_type,
                    int *sizes
                  ){
    using namespace gpuSimulate;
    //在这里调用一个单核核函数对hash_set进行更新
    // printf("current arr size is %ld",arr_size);
    // HANDLE_HIP(hipMalloc((void**)&Pair_Atoms,arr_size*sizeof(long int)*2));
    // HANDLE_HIP(hipMemset(Pair_Atoms, -1,arr_size * sizeof(long int) * 2));
   
    // printf("starting recb solver with GPU sector_id:%d,arr_size:%d\n",sector_id,arr_size);
    const bool hash_stats = recbHashStatsEnabled();
    const bool do_profile = recbPhaseTimingEnabled();
    const bool verify_detail = recbCorrectnessDebugEnabled();
    const bool event_local = recbEventLocalEnabled();
    if (do_profile) initRecbProfileEvents();

    static int recb_launch_by_sector[8] = {};
    static unsigned long long recb_capped_total = 0;
    const unsigned int sid = sector_id < 8 ? sector_id : 0;
    const int launch_n = ++recb_launch_by_sector[sid];
    int recb_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &recb_rank);
    static bool recb_schedule_config_printed = false;
    if (!recb_schedule_config_printed) {
      if (recb_rank == 0) {
        kiwi::logs::v(" ", " [recb-config] walker=tile8 schedule={} local_audit={} phase_timing={} roster_scan={} roster_capacity={} (KMC_RECB_EVENT_LOCAL={}).\n",
                      event_local ? "event-local" : "full",
                      verify_detail && event_local ? "on" : "off",
                      do_profile ? "on" : "off",
                      pair_roster_scan_size, pair_atoms_size,
                      event_local ? 1 : 0);
      }
      recb_schedule_config_printed = true;
    }

    blockSize = 128;
    // print_Hash_and_Pair
    // printf("+++++++++++++++++++++++++++++++++++++++++++++++++\n");
    // printf("arr size is %ld\n",pair_atoms_size);
    // hipLaunchKernelGGL(print_Hash_and_Pair, 1, 1, 0, 0,arr_size,0);

    if (verify_detail && !d_recb_verify_stats) {
      HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&d_recb_verify_stats), sizeof(RecbVerifyStats)));
    }
    if (hash_stats && !d_recb_walker_lengths && pair_atoms_size > 0) {
      HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&d_recb_walker_lengths),
                           static_cast<size_t>(pair_atoms_size) * sizeof(unsigned long long)));
    }
    // 验证统计的清零/回读/全表计数只在正确性 debug 模式启用；
    // 关闭时 kernel 收到 nullptr，内部所有统计 atomic 自动跳过。
    if (verify_detail) {
      HANDLE_HIP(hipMemset(d_recb_verify_stats, 0, sizeof(RecbVerifyStats)));
    }

    //=======recb 前筛选：把本扇区盒子内的 walker 槽位号压缩出来=======
    // 花名册 g_Pair_Atoms 混装全部 8 个扇区，这里先筛出本扇区，kernel 只启动所需线程。
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_BEGIN], 0));
    }
    hipLaunchKernelGGL(begin_recb_pass, 1, 1, 0, 0,
                       static_cast<int>(sector_id), event_local);
    HANDLE_HIP(hipMemset(d_sector_slot_count, 0, sizeof(int)));
    if (pair_roster_scan_size > 0) {
      const long int filter_grid = (pair_roster_scan_size + blockSize - 1) / blockSize;
      hipLaunchKernelGGL(filter_pair_slots_for_sector, filter_grid, blockSize, 0, 0,
                         (int)sector_id, pair_roster_scan_size, d_sector_slots,
                         d_sector_slot_count, event_local,
                         verify_detail && event_local);
      CHECK_FOR_ERROR();
    }
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_FILTER], 0));
    }
    int h_sector_count = 0;
    HANDLE_HIP(hipMemcpy(&h_sector_count, d_sector_slot_count, sizeof(int), hipMemcpyDeviceToHost));
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_COUNT_D2H], 0));
    }
    //=============================================================

    // query/update/probe 三类原子统计只在哈希统计模式开启。
    if (hash_stats) {
      hipLaunchKernelGGL(reset_hash_stats, 1, 1, 0, 0);
      CHECK_FOR_ERROR();
      const bool stats_on = true;
      HANDLE_HIP(hipMemcpyToSymbol(HIP_SYMBOL(g_hash_stats_on), &stats_on, sizeof(bool)));
    }
    if (h_sector_count > 0) {
      constexpr int walkers_per_block = 128 / 8;
      numBlocks = (h_sector_count + walkers_per_block - 1) / walkers_per_block;
      HANDLE_HIP(hipMemset(d_recb_work_head, 0, sizeof(unsigned int)));
      hipLaunchKernelGGL(calcExchangePairsTile8, numBlocks, blockSize, 0, 0,
                        sector_id, h_sector_count,
                        verify_detail ? d_recb_verify_stats : nullptr, d_sector_slots,
                        verify_detail, hash_stats,
                        hash_stats ? d_recb_walker_lengths : nullptr,
                        d_recb_work_head
                        );
      CHECK_FOR_ERROR();
      // 正确性清理不依赖详细日志开关；同一默认 stream 保证它在游走结束后执行。
      hipLaunchKernelGGL(validate_pair_slots, numBlocks, blockSize, 0, 0,
                         d_sector_slots, h_sector_count,
                         verify_detail ? d_recb_verify_stats : nullptr);
      CHECK_FOR_ERROR();
      if (hash_stats) {
        hipLaunchKernelGGL(collect_recb_walker_stats, numBlocks, blockSize, 0, 0,
                           d_recb_walker_lengths, h_sector_count);
        CHECK_FOR_ERROR();
      }
    }
    if (hash_stats) {
      const bool stats_off = false;
      HANDLE_HIP(hipMemcpyToSymbol(HIP_SYMBOL(g_hash_stats_on), &stats_off, sizeof(bool)));
    }
    if (verify_detail) {
      const int count_block = 256;
      if (pair_atoms_size > 0 && d_Pair_Atoms) {
        const int roster_grid = static_cast<int>((pair_atoms_size + count_block - 1) / count_block);
        hipLaunchKernelGGL(count_roster_live_kernel, roster_grid, count_block, 0, 0,
                           d_Pair_Atoms, pair_atoms_size, d_recb_verify_stats);
      }
      if (MoMo_Hash && MoMo_Hash->device_ptr()) {
        const int cap = MoMo_Hash->get_capacity();
        const int grid = (cap + count_block - 1) / count_block;
        if (grid > 0) {
          hipLaunchKernelGGL(count_hash_local_pairs_kernel, grid, count_block, 0, 0,
                             MoMo_Hash->device_ptr(), d_recb_verify_stats);
        }
      }
      if (MoRe_Hash && MoRe_Hash->device_ptr()) {
        const int cap = MoRe_Hash->get_capacity();
        const int grid = (cap + count_block - 1) / count_block;
        if (grid > 0) {
          hipLaunchKernelGGL(count_hash_local_pairs_kernel, grid, count_block, 0, 0,
                             MoRe_Hash->device_ptr(), d_recb_verify_stats);
        }
      }
      hipDeviceSynchronize();
    }
    unsigned int h_probe_max[RECB_HASH_TABLE_COUNT] = {};
    if (hash_stats) {
      HANDLE_HIP(hipMemcpyFromSymbol(h_probe_max, HIP_SYMBOL(g_probe_max),
                                     sizeof(h_probe_max)));
    }
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_WALKER], 0));
    }

    if (verify_detail) {
      // 每次启动都取回统计（保险丝计数不能漏），但常规日志只打前 5 次
      RecbVerifyStats h_stats{};
      HANDLE_HIP(hipMemcpy(&h_stats, d_recb_verify_stats, sizeof(RecbVerifyStats), hipMemcpyDeviceToHost));
      accumulateRecbRunSummary(h_stats);
      recb_capped_total += h_stats.capped_walks;
      if (recb_rank == 0 && launch_n <= 5) {
        kiwi::logs::v(" ", " [recb-verify] sect={} launch={} sector_pairs={} attempts={} successful_hops={} max_successful_hops_one_pair={} pairs_multi_hop={} captured_walks={} captured_no_move={} exited_walks={} capped_walks={} stale_source_walkers={} unclassified_walkers={} tx_fail={} roster_live={} hash_pairs={}.\n",
                      sid, launch_n, h_stats.sector_pairs, h_stats.attempts,
                      h_stats.successful_hops, h_stats.max_successful_hops,
                      h_stats.multi_hop_pairs, h_stats.captured_walks,
                      h_stats.captured_no_move, h_stats.exited_walks, h_stats.capped_walks,
                      h_stats.stale_source_walkers, h_stats.unclassified_walkers,
                      h_stats.transaction_failures, h_stats.roster_live_count,
                      h_stats.hash_pair_count);
        if (verify_detail) {
          kiwi::logs::v(" ", " [recb-detail] sect={} launch={} targets(mo/re/other)={}/{}/{} failures(lock/src/dst_mo/dst_re/dst_other/tx)={}/{}/{}/{}/{}/{} roster_type_mismatches={}.\n",
                        sid, launch_n, h_stats.target_mo, h_stats.target_re,
                        h_stats.target_other, h_stats.lock_failures,
                        h_stats.src_check_failures, h_stats.dst_mo_check_failures,
                        h_stats.dst_re_check_failures, h_stats.dst_other_check_failures,
                        h_stats.transaction_failures, h_stats.roster_type_mismatches);
          kiwi::logs::v(" ", " [recb-ops] sect={} launch={} attempts=[{},{},{},{}] successes=[{},{},{},{}] failures=[{},{},{},{}].\n",
                        sid, launch_n,
                        h_stats.op_attempts[0], h_stats.op_attempts[1],
                        h_stats.op_attempts[2], h_stats.op_attempts[3],
                        h_stats.op_successes[0], h_stats.op_successes[1],
                        h_stats.op_successes[2], h_stats.op_successes[3],
                        h_stats.op_failures[0], h_stats.op_failures[1],
                        h_stats.op_failures[2], h_stats.op_failures[3]);
          kiwi::logs::v(" ", " [recb-directions] sect={} launch={} counts=[{},{},{},{},{},{},{},{},{},{},{},{},{},{}].\n",
                        sid, launch_n,
                        h_stats.direction_counts[0], h_stats.direction_counts[1],
                        h_stats.direction_counts[2], h_stats.direction_counts[3],
                        h_stats.direction_counts[4], h_stats.direction_counts[5],
                        h_stats.direction_counts[6], h_stats.direction_counts[7],
                        h_stats.direction_counts[8], h_stats.direction_counts[9],
                        h_stats.direction_counts[10], h_stats.direction_counts[11],
                        h_stats.direction_counts[12], h_stats.direction_counts[13]);
        }
      }
      if (h_stats.capped_walks > 0) {
        kiwi::logs::v(" ", " [recb-cap] rank={} sect={} launch={} 触发跳跃上限的 walker 数={} 全程累计={}（疑似病态轨迹，请检查！）.\n",
                      recb_rank, sid, launch_n, h_stats.capped_walks, recb_capped_total);
      }
      // roster/hash 不守恒排查：出现 stale 时打印触发场景分类，直接定位根因。
      // at_start 占满 → 花名册 slot 在 recb 前已陈旧（跨扇区通信/rebuild 嫌疑）；
      // src_changed>0 → 并发游走中源被抢（锁/DCL 机制嫌疑）。
      if (recb_rank == 0 && h_stats.stale_source_walkers > 0) {
        kiwi::logs::v(" ", " [recb-stale] sect={} launch={} total={} at_start={} src_changed={} start_ghost={} start_is_re={} start_is_v={}.\n",
                      sid, launch_n, h_stats.stale_source_walkers,
                      h_stats.stale_at_start, h_stats.stale_src_changed,
                      h_stats.stale_start_ghost, h_stats.stale_start_is_re,
                      h_stats.stale_start_is_v);
      }
      const unsigned int classified_walkers =
          h_stats.captured_walks + h_stats.exited_walks + h_stats.capped_walks +
          h_stats.stale_source_walkers;
      if (recb_rank == 0 &&
          (h_stats.unclassified_walkers != 0 ||
           classified_walkers + h_stats.unclassified_walkers != h_stats.sector_pairs ||
           h_stats.roster_live_count != h_stats.hash_pair_count)) {
        kiwi::logs::v(" ", " [recb-invariant] sect={} launch={} sector_pairs={} classified={} unclassified={} roster_live={} hash_pairs={}（终止原因或花名册/哈希不守恒）.\n",
                      sid, launch_n, h_stats.sector_pairs, classified_walkers,
                      h_stats.unclassified_walkers, h_stats.roster_live_count,
                      h_stats.hash_pair_count);
      }
    }

    hipLaunchKernelGGL(finish_recb_pass, 1, 1, 0, 0, static_cast<int>(sector_id));
    if (verify_detail && event_local) {
      unsigned int h_local_audit[2] = {};
      unsigned int h_recb_was_full = 0;
      HANDLE_HIP(hipMemcpyFromSymbol(h_local_audit, HIP_SYMBOL(g_recb_local_audit),
                                     sizeof(h_local_audit)));
      HANDLE_HIP(hipMemcpyFromSymbol(&h_recb_was_full,
                                     HIP_SYMBOL(g_recb_was_full_pass),
                                     sizeof(h_recb_was_full)));
      if (recb_rank == 0 && h_recb_was_full == 0u) {
        static unsigned long long local_launches = 0;
        static unsigned long long positive_launches = 0;
        static unsigned long long local_candidates = 0;
        static unsigned long long audited_excluded = 0;
        static unsigned long long missed_active = 0;
        const unsigned int selected = h_sector_count >= static_cast<int>(h_local_audit[1])
            ? static_cast<unsigned int>(h_sector_count) - h_local_audit[1] : 0u;
        ++local_launches;
        if (selected > 0) ++positive_launches;
        local_candidates += selected;
        audited_excluded += h_local_audit[0];
        missed_active += h_local_audit[1];
        if (selected > 0 || h_local_audit[1] > 0 || local_launches <= 5 ||
            (local_launches % 100u) == 0u) {
          kiwi::logs::v(" ", " [recb-local-audit] sect={} launch={} selected={} audited_excluded={} missed_active={} totals(local_launches/positive_launches/selected/audited/missed)={}/{}/{}/{}/{}.\n",
                        sid, launch_n, selected, h_local_audit[0],
                        h_local_audit[1], local_launches, positive_launches,
                        local_candidates, audited_excluded, missed_active);
        }
        if (h_local_audit[1] > 0) {
          kiwi::logs::v(" ", " [recb-local-miss] sect={} launch={} missed_active={} recovered_this_launch=1 next_pass_forced_full=1.\n",
                        sid, launch_n, h_local_audit[1]);
        }
      }
    }
    if (hash_stats) {
      unsigned long long h_query[RECB_HASH_TABLE_COUNT] = {};
      unsigned long long h_update[RECB_HASH_TABLE_COUNT] = {};
      unsigned long long h_re_probe_reach[7] = {};
      unsigned long long h_walker_attempt_hist[RECB_WALKER_HIST_BINS] = {};
      unsigned long long h_wave_tail[4] = {};
      unsigned int h_walker_attempt_max = 0;
      unsigned int h_recb_wave_size = 0;
      unsigned long long h_extra_rng_draws = 0;
      unsigned int h_recb_was_full = 0;
      unsigned int h_recb_changed = 0;
      hipMemcpyFromSymbol(h_query, HIP_SYMBOL(g_hash_query), sizeof(h_query));
      hipMemcpyFromSymbol(h_update, HIP_SYMBOL(g_hash_update), sizeof(h_update));
      hipMemcpyFromSymbol(h_re_probe_reach, HIP_SYMBOL(g_re_probe_reach),
                          sizeof(h_re_probe_reach));
      hipMemcpyFromSymbol(h_walker_attempt_hist,
                          HIP_SYMBOL(g_recb_walker_attempt_hist),
                          sizeof(h_walker_attempt_hist));
      hipMemcpyFromSymbol(h_wave_tail, HIP_SYMBOL(g_recb_wave_tail),
                          sizeof(h_wave_tail));
      hipMemcpyFromSymbol(&h_walker_attempt_max,
                          HIP_SYMBOL(g_recb_walker_attempt_max),
                          sizeof(h_walker_attempt_max));
      hipMemcpyFromSymbol(&h_recb_wave_size, HIP_SYMBOL(g_recb_wave_size),
                          sizeof(h_recb_wave_size));
      hipMemcpyFromSymbol(&h_extra_rng_draws,
                          HIP_SYMBOL(g_recb_extra_rng_draws),
                          sizeof(h_extra_rng_draws));
      hipMemcpyFromSymbol(&h_recb_was_full, HIP_SYMBOL(g_recb_was_full_pass),
                          sizeof(h_recb_was_full));
      hipMemcpyFromSymbol(&h_recb_changed, HIP_SYMBOL(g_recb_pairs_changed),
                          sizeof(h_recb_changed));
      if (recb_rank == 0) {
        kiwi::logs::v(" ", " [recb-schedule] sect={} launch={} mode={} candidates={} group_size=8 changed={}\n",
                      sid, launch_n, h_recb_was_full ? "full" : "event-local",
                      h_sector_count, h_recb_changed);
        const double active_lane_eff = h_wave_tail[1] > 0
            ? static_cast<double>(h_wave_tail[0]) / h_wave_tail[1] : 1.0;
        const double full_wave_eff = h_wave_tail[2] > 0
            ? static_cast<double>(h_wave_tail[0]) / h_wave_tail[2] : 1.0;
        kiwi::logs::v(" ", " [recb-walker-attempt-hist] sect={} launch={} bins=0/1-7/8-31/32-63/64-127/128-255/256-511/512-1023/1024-2047/2048-4095/>=4096 counts={}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{} max={}.\n",
                      sid, launch_n,
                      h_walker_attempt_hist[0], h_walker_attempt_hist[1],
                      h_walker_attempt_hist[2], h_walker_attempt_hist[3],
                      h_walker_attempt_hist[4], h_walker_attempt_hist[5],
                      h_walker_attempt_hist[6], h_walker_attempt_hist[7],
                      h_walker_attempt_hist[8], h_walker_attempt_hist[9],
                      h_walker_attempt_hist[10], h_walker_attempt_max);
        kiwi::logs::v(" ", " [recb-wave-tail] sect={} launch={} wave_size={} waves={} attempts={} active_lane_capacity={} full_wave_capacity={} active_lane_eff={:.4f} full_wave_eff={:.4f}.\n",
                      sid, launch_n, h_recb_wave_size, h_wave_tail[3], h_wave_tail[0],
                      h_wave_tail[1], h_wave_tail[2], active_lane_eff,
                      full_wave_eff);
        kiwi::logs::v(" ", " [recb-rng-draws] sect={} launch={} direction={} conditional={} total={}.\n",
                      sid, launch_n, h_wave_tail[0], h_extra_rng_draws,
                      h_wave_tail[0] + h_extra_rng_draws);
        kiwi::logs::v(" ", " [recb-hash] sect={} launch={} tables=MoRe/MoMo/Re/V/Busy/ghost/surface query={}/{}/{}/{}/{}/{}/{} update={}/{}/{}/{}/{}/{}/{} max_probe={}/{}/{}/{}/{}/{}/{}.\n",
                      sid, launch_n,
                      h_query[0], h_query[1], h_query[2], h_query[3], h_query[4], h_query[5], h_query[6],
                      h_update[0], h_update[1], h_update[2], h_update[3], h_update[4], h_update[5], h_update[6],
                      h_probe_max[0], h_probe_max[1], h_probe_max[2], h_probe_max[3],
                      h_probe_max[4], h_probe_max[5], h_probe_max[6]);
        kiwi::logs::v(" ", " [recb-re-probe-reach] sect={} launch={} thresholds=1/2/5/9/17/33/65 counts={}/{}/{}/{}/{}/{}/{}.\n",
                      sid, launch_n,
                      h_re_probe_reach[0], h_re_probe_reach[1],
                      h_re_probe_reach[2], h_re_probe_reach[3],
                      h_re_probe_reach[4], h_re_probe_reach[5],
                      h_re_probe_reach[6]);
      }
    }

    // print_momo_and_more
    //  hipLaunchKernelGGL(print_momo_and_more, 1, 1, 0, 0
                      
    //                   );
    // hipLaunchKernelGGL(print_surface_and_ghost, 1, 1, 0, 0
    //                   );
    // print_surface_and_ghost

    // printf("---------------------------------------------------\n"); 
    // hipLaunchKernelGGL(print_Hash_and_Pair, 1, 1, 0, 0,arr_size,1);
    // printf("+++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    // hipHostFree(Pair_Atoms);
    

    //这是计算后传回的部分
    //ghost和surface区需要传回

    

    HANDLE_HIP(hipMemset(d_surface_count, 0, sizeof(int)));
    HANDLE_HIP(hipMemset(d_ghost_count, 0, sizeof(int)));
    const int compact_block = 256;
    const int surface_grid = (surface_capacity + compact_block - 1) / compact_block;
    const int ghost_grid = (ghost_capacity + compact_block - 1) / compact_block;
    hipLaunchKernelGGL(compact_valid_keys, surface_grid, compact_block, 0, 0,
                       surface_Hash->device_ptr()->keys, surface_capacity, d_surface_list, d_surface_count);
    CHECK_FOR_ERROR();
    hipLaunchKernelGGL(compact_valid_keys, ghost_grid, compact_block, 0, 0,
                       ghost_Hash->device_ptr()->keys, ghost_capacity, d_ghost_list, d_ghost_count);
    CHECK_FOR_ERROR();
    hipDeviceSynchronize();
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_BOUNDARY_COMPACT], 0));
    }

    // 复用上面的既有同步点后再读取 dirty，干净 launch 不再额外等待 GPU。
    // 该刷新只清理 MoRe 墓碑，与通信后的 roster rebuild 无关。
    refreshMoReAfterRecb(sid, launch_n, verify_detail, hash_stats);
    if (hash_stats) logReHashOccupancy(sid, launch_n, recb_rank);
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_REFRESH], 0));
    }

    int h_surface_count = 0;
    int h_ghost_count = 0;
    HANDLE_HIP(hipMemcpy(&h_surface_count, d_surface_count, sizeof(int), hipMemcpyDeviceToHost));
    HANDLE_HIP(hipMemcpy(&h_ghost_count, d_ghost_count, sizeof(int), hipMemcpyDeviceToHost));

    if (gpu_comm_enabled()) {
      commClearGhostBuckets();
      if (h_ghost_count > 0) {
        commBuildGhostBucketsFromList(sector_id, d_ghost_list, h_ghost_count);
      }
      commClearSurfaceBuckets();
      if (h_surface_count > 0) {
        commBuildSurfaceBucketsFromList(d_surface_list, h_surface_count);
      }
    }

    const int type_block = 256;
    if (h_surface_count > 0) {
      const int grid = (h_surface_count + type_block - 1) / type_block;
      hipLaunchKernelGGL(compute_types_for_list, grid, type_block, 0, 0,
                         d_surface_list, d_surface_list_type, h_surface_count);
      CHECK_FOR_ERROR();
    }
    if (h_ghost_count > 0) {
      const int grid = (h_ghost_count + type_block - 1) / type_block;
      hipLaunchKernelGGL(compute_types_for_list, grid, type_block, 0, 0,
                         d_ghost_list, d_ghost_list_type, h_ghost_count);
      CHECK_FOR_ERROR();
    }
    hipDeviceSynchronize();
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_BOUNDARY_TYPE], 0));
    }

    if (h_ghost_count > 0 && h_ghost_Hash && h_ghost_type) {
      HANDLE_HIP(hipMemcpy(h_ghost_Hash, d_ghost_list, h_ghost_count * sizeof(long int), hipMemcpyDeviceToHost));
      HANDLE_HIP(hipMemcpy(h_ghost_type, d_ghost_list_type, h_ghost_count * sizeof(long int), hipMemcpyDeviceToHost));
    }
    if (h_surface_count > 0 && h_surface_Hash && h_surface_type) {
      HANDLE_HIP(hipMemcpy(h_surface_Hash, d_surface_list, h_surface_count * sizeof(long int), hipMemcpyDeviceToHost));
      HANDLE_HIP(hipMemcpy(h_surface_type, d_surface_list_type, h_surface_count * sizeof(long int), hipMemcpyDeviceToHost));
    }
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_COPY], 0));
    }

    ghost_Hash->clear();
    surface_Hash->clear();
    if (do_profile) {
      HANDLE_HIP(hipEventRecord(recb_phase_events[RECB_PHASE_AFTER_CLEAR], 0));
    }
    // for(int i=0;i<surface_Hash->device_ptr()->capacity;i++){
    //   printf("%ld-%ld\t",h_surface_Hash[i],h_surface_type[i]);
    //   if(i%20 == 0)
    //   printf("\n");
    // }
    if (sizes) {
      sizes[0]= h_ghost_count;
      sizes[1]= h_surface_count;
    }

    if (do_profile) accumulateRecbPhaseTimings(h_sector_count);
    // printf("cal exchange finished \n");
}

double calculate_GPU(_type_lattice_id *vac_idArray,  const _type_lattice_count& vac_count) {
  using namespace gpuSimulate;
  
  *sum_rates = 0;
  currentVacCount = vac_count;
  // 如果超出预分配数组大小则重新分配锁页内存和显存
  if(vac_count > vacHostArrayCapacity) {
    vacHostArrayCapacity = vac_count * RE_SCALE_SIZE;
    hipHostFree(hostVacancyInfo);
    hipHostFree(hostNeighbourInfo);
    HANDLE_HIP(hipFree(deviceVacancyInfo));
    HANDLE_HIP(hipFree(deviceNeighbourInfo));

    hipHostMalloc((void**)&hostVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy));
    hipHostMalloc((void**)&hostNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice));
    HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
    HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));
  }

  lattice_list->initGpuInfo(vac_count, vac_idArray, hostVacancyInfo, hostNeighbourInfo);//从哈希表找数据传输到gpu
  HANDLE_HIP(hipMemcpy(deviceVacancyInfo, hostVacancyInfo, vac_count * sizeof(dev_Vacancy), hipMemcpyHostToDevice));
  HANDLE_HIP(hipMemcpy(deviceNeighbourInfo, hostNeighbourInfo, vac_count * NN_TOTAL * sizeof(dev_nnLattice), hipMemcpyHostToDevice));
  // for (_type_lattice_count j = 0; j < vac_count; j++) {
  //   kiwi::logs::v(" ", " latti_id is : {}. \n", hostVacancyInfo[j].id);
  //   for (int i = 0; i < 8; i++) {
  //     kiwi::logs::v(" ", " nn_index is : {} nn_id is : {} nn_type is : {} . \n", i, deviceNeighbourInfo[j * NN_TOTAL + i].id, deviceNeighbourInfo[j * NN_TOTAL + i].type._type);
  //   }
  // }
  blockSize = 128;
  numBlocks = (vac_count + blockSize - 1) / blockSize;

  // 调用GPU内核函数
  hipLaunchKernelGGL(calcRatesGPU, numBlocks, blockSize, 0, 0, 
                    deviceVacancyInfo,
                    deviceNeighbourInfo,
                    d_sum_rates,
                    h_config[0],
                    h_config[1], 
                    vac_count);

  free(vac_idArray);

  CHECK_FOR_ERROR();
  HANDLE_HIP(hipMemcpy(sum_rates, d_sum_rates, sizeof(double), hipMemcpyDeviceToHost));
  return *sum_rates;
}

double calculate_GPU_region(int x_low, int x_high, int y_low, int y_high, int z_low, int z_high) {
  using namespace gpuSimulate;
  HANDLE_HIP(hipMemset(d_sum_rates, 0, sizeof(double)));
  *sum_rates = 0;

  HANDLE_HIP(hipMemset(d_vac_count_all, 0, sizeof(int)));
  const int compact_block = 256;
  const int vac_grid = (vac_capacity_all + compact_block - 1) / compact_block;
  hipLaunchKernelGGL(compact_valid_keys, vac_grid, compact_block, 0, 0,
                     V_Hash->device_ptr()->keys, vac_capacity_all, d_vac_all, d_vac_count_all);
  CHECK_FOR_ERROR();
  hipDeviceSynchronize();

  int h_vac_all = 0;
  HANDLE_HIP(hipMemcpy(&h_vac_all, d_vac_count_all, sizeof(int), hipMemcpyDeviceToHost));
  if (h_vac_all <= 0) return 0.0;

  HANDLE_HIP(hipMemset(d_vac_count_sector, 0, sizeof(int)));
  const int filter_block = 256;
  const int filter_grid = (h_vac_all + filter_block - 1) / filter_block;
  hipLaunchKernelGGL(filter_ids_in_region, filter_grid, filter_block, 0, 0,
                     d_vac_all, h_vac_all,
                     x_low, x_high, y_low, y_high, z_low, z_high,
                     d_vac_sector, d_vac_count_sector);
  CHECK_FOR_ERROR();
  hipDeviceSynchronize();

  int h_vac_sector = 0;
  HANDLE_HIP(hipMemcpy(&h_vac_sector, d_vac_count_sector, sizeof(int), hipMemcpyDeviceToHost));
  if (h_vac_sector <= 0) return 0.0;

  currentVacCount = h_vac_sector;
  if (static_cast<_type_lattice_count>(h_vac_sector) > vacHostArrayCapacity) {
    vacHostArrayCapacity = static_cast<_type_lattice_count>(h_vac_sector);
    HANDLE_HIP(hipFree(deviceVacancyInfo));
    HANDLE_HIP(hipFree(deviceNeighbourInfo));
    HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
    HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));
  }

  blockSize = 128;
  numBlocks = (h_vac_sector + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(build_nn_info_from_vac_list, numBlocks, blockSize, 0, 0,
                     d_vac_sector, h_vac_sector, deviceVacancyInfo, deviceNeighbourInfo);
  CHECK_FOR_ERROR();

  hipLaunchKernelGGL(calcRatesGPU, numBlocks, blockSize, 0, 0,
                     deviceVacancyInfo,
                     deviceNeighbourInfo,
                     d_sum_rates,
                     h_config[0],
                     h_config[1],
                     h_vac_sector);
  CHECK_FOR_ERROR();
  HANDLE_HIP(hipMemcpy(sum_rates, d_sum_rates, sizeof(double), hipMemcpyDeviceToHost));
  return *sum_rates;
}
// void initialize_gpu(int process_rank){
//   int gpu_count;
//   HANDLE_HIP(hipGetDeviceCount(&gpu_count));

//   int dev = process_rank % gpu_count;
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " gpu_count=" << gpu_count
//             << " before hipSetDevice dev=" << dev << std::endl;

//   HANDLE_HIP(hipSetDevice(dev));

//   int current_dev = -1;
//   HANDLE_HIP(hipGetDevice(&current_dev));
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after hipSetDevice current_dev=" << current_dev << std::endl;

//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " before hipDeviceReset" << std::endl;
//   HANDLE_HIP(hipDeviceReset());

//   HANDLE_HIP(hipGetDevice(&current_dev));
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after hipDeviceReset current_dev=" << current_dev << std::endl;

// #ifdef KMC_RCCL_ENABLE
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " before rccl_smoke_test" << std::endl;
//   rccl_smoke_test();
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after rccl_smoke_test" << std::endl;
// #endif
// }


void initialize_gpu(int process_rank){
  int gpu_count;
  HANDLE_HIP(hipGetDeviceCount(&gpu_count));
  const int local_rank = getLocalProcessRank(process_rank);
  const int device_id = local_rank % gpu_count;
  HANDLE_HIP(hipSetDevice(device_id));
  if (shouldPrintGpuBindDebug(process_rank, local_rank)) {
    std::cerr << "[KMC_GPU_BIND] process_rank=" << process_rank
              << " local_rank=" << local_rank
              << " gpu_count=" << gpu_count
              << " device_id=" << device_id << std::endl;
  }
  HANDLE_HIP(hipDeviceReset());
#ifdef KMC_RCCL_ENABLE
  rccl_smoke_test();
#endif
}

void gpu_prepare(double& v, double& T, LatticesList *p_list, comm::ColoredDomain *_p_domain,
                std::vector<long int> pair_atoms, unsigned long long rng_seed
                ){
  using namespace gpuSimulate;

  // 哈希配置打印：跑一次日志即可确认当前二进制实际用的桶大小 / tile / 探测上限 / 路径。
  // BUCKET_SIZE=1 即"经典单线程线性探测开放寻址"；=64 为桶式。哈希操作全部为单线程版
  // （recb: hash_set_contains/insert/remove_atom_op；通信: 同上）。
  {
    int hash_cfg_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &hash_cfg_rank);
    if (hash_cfg_rank == 0 && recbHashStatsEnabled()) {
      kiwi::logs::v(" ", " [hash-config] BUCKET_SIZE={} TILE_SIZE={} PROBE_LIMIT=table_size(no cap) capacity=2x_keys | hash ops=single-threaded (recb & comm) | bucket=1 => classic linear-probing open addressing.\n",
                    (long long)BUCKET_SIZE, (long long)KMC_HASH_TILE_SIZE);
    }
  }

  int device_id;
  hipGetDevice(&device_id);  // 获取当前设备ID
// size_t free_mem = 0;
// size_t total_mem = 0;
// HANDLE_HIP(hipMemGetInfo(&free_mem, &total_mem));

// std::cerr << "[gpu_prepare] current device=" << device_id
//           << " free_mem=" << free_mem
//           << " total_mem=" << total_mem
//           << std::endl;

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device_id);  // 获取设备属性
  std::cout << "GPU名称: " << prop.name << std::endl;
  std::cout << "每个线程块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "每个多处理器最大线程块数: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "线程块最大尺寸（x,y,z）: (" 
            << prop.maxThreadsDim[0] << ", " 
            << prop.maxThreadsDim[1] << ", " 
            << prop.maxThreadsDim[2] << ")" << std::endl;
  std::cout << "网格最大尺寸（x,y,z）: (" 
            << prop.maxGridSize[0] << ", " 
            << prop.maxGridSize[1] << ", " 
            << prop.maxGridSize[2] << ")" << std::endl;

  lattice_list = p_list;
  p_domain = _p_domain;
  h_config[0] = v;   // v
  h_config[1] = T;   // T
  std::vector<long int> region_msg(8 * 6);

  
  int index =0;
  for(auto region_sector:p_domain->local_sector_region){
    region_msg[index * 6 + 0] = region_sector.x_low;
    region_msg[index * 6 + 1] = region_sector.x_high;
    region_msg[index * 6 + 2] = region_sector.y_low;
    region_msg[index * 6 + 3] = region_sector.y_high;
    region_msg[index * 6 + 4] = region_sector.z_low;
    region_msg[index * 6 + 5] = region_sector.z_high;
    index++;
  }
  index = 0;
  printf("init region msg \n");
  HANDLE_HIP(hipMalloc((void**)&d_sum_rates, sizeof(double)));
  HANDLE_HIP(hipMalloc((void**)&d_event, sizeof(dev_event)));
  
  vacHostArrayCapacity = lattice_list->vac_hash.size() + 1;
  HostAtomCapacity = lattice_list->momo_hash.size()
                      + lattice_list->more_hash.size()
                      +lattice_list->rere_hash.size();

  HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
  HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));

  hipHostMalloc((void**)&sum_rates, sizeof(double));
  hipHostMalloc((void**)&h_event, sizeof(dev_event));

  hipHostMalloc((void**)&hostVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy));
  hipHostMalloc((void**)&hostNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice));


  hipHostMalloc((void**)&atoms, HostAtomCapacity * sizeof(dev_atom));
  hipHostMalloc((void**)&ret_atoms, HostAtomCapacity * sizeof(dev_atom));
  HANDLE_HIP(hipMalloc((void**)&d_atoms, HostAtomCapacity * sizeof(dev_atom)));

  //以下是对recb_solver_GPU的空间分配
  //在这里直接把GPU中的哈希表分配掉了，然后在空位交换的时候更新哈希表
  /*
    变量名           位置         解释
    Pair_Atoms      host      间隙对的列表
    d_Pair_Atoms    device    间隙对的列表（GPU）
    meta            host      box中环境变量的集合
    d_meta          device    box中环境变量的集合
    ghost_size      host      ghost区的大小（现在按最大的情况来）
    surface_size    host      surface区的大小（现在按最大的情况来）
    MoRe_Hash       device    GPU中哈希表的位置
    h_MoRe_Hash     host      计算完毕后传回给CPU的空间
    MoMo_Hash等等同理
    busy_set        host      忙碌队列
  */
  pair_roster_scan_size = static_cast<long int>(pair_atoms.size());
  pair_atoms_size = pair_roster_scan_size * 2;
  hipHostMalloc((void**)&Pair_Atoms,pair_atoms.size()*sizeof(long int));
  HANDLE_HIP(hipMalloc((void**)&d_Pair_Atoms,pair_atoms.size()*sizeof(long int)*2));
  HANDLE_HIP(hipMemset(d_Pair_Atoms, -1, pair_atoms.size() * sizeof(long int) * 2));
  //对环境变量以及哈希表进行初始化
  //每组的meta需要重新赋值
  // printf("init gpu_info msg \n");
  lattice_list->initGpuInfo_exchange(pair_atoms,region_msg,
                                    0,meta,Pair_Atoms,
                                    MoRe_Hash,MoMo_Hash,Re_Hash,V_Hash,Busy_Set
                                    );
  {
    int hash_cfg_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &hash_cfg_rank);
    if (hash_cfg_rank == 0 && recbHashStatsEnabled()) {
      kiwi::logs::v(" ", " [more-hash-config] multiplier={} capacity={} scratch_capacity={} scratch_bytes={}.\n",
                    KMC_MORE_HASH_CAPACITY_MULTIPLIER,
                    MoRe_Hash ? MoRe_Hash->get_capacity() : 0,
                    pair_atoms_size,
                    pair_atoms_size * static_cast<long int>(sizeof(long int)));
    }
  }
  // printf("init gpu_info msg end\n");
  HANDLE_HIP(hipMemcpy(d_Pair_Atoms, Pair_Atoms,pair_atoms.size()*sizeof(long int), hipMemcpyHostToDevice));

  // vac_capacity_all = V_Hash->capacity();
  vac_capacity_all = V_Hash->get_capacity();
  vac_capacity_sector = vac_capacity_all;
  HANDLE_HIP(hipMalloc((void**)&d_vac_all, vac_capacity_all * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_sector, vac_capacity_sector * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_count_all, sizeof(int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_count_sector, sizeof(int)));

  int ghost_size = (meta.regions[1] + meta.ghost_x - meta.regions[0])*(meta.regions[3] + meta.ghost_y - meta.regions[2]) * (meta.regions[5] + meta.ghost_z - meta.regions[4])
                    - (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                    + (meta.regions[1]-meta.regions[0])*meta.ghost_y*meta.ghost_z
                    + meta.ghost_x*(meta.regions[3]-meta.regions[2])*meta.ghost_z
                    + meta.ghost_x*(meta.regions[5]-meta.regions[4])*meta.ghost_y;
  int surface_size = (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                      - (meta.regions[1] - meta.ghost_x - meta.regions[0])*(meta.regions[3] - meta.ghost_y - meta.regions[2]) * (meta.regions[5] - meta.ghost_z - meta.regions[4]);

  surface_Hash = new HIPHashSet(surface_size);
  ghost_Hash = new HIPHashSet(ghost_size);
  // surface_capacity = surface_size * 2;
  // ghost_capacity = ghost_size * 2;
  surface_capacity = surface_Hash->get_capacity(); 
  ghost_capacity = ghost_Hash->get_capacity();
  commInitGhostBuckets(ghost_size);
  commInitSurfaceBuckets(surface_size);
  {
    const int dims[comm::DIMENSION_SIZE] = {comm::DIM_X, comm::DIM_Y, comm::DIM_Z};
    std::vector<CommBox> boxes;
    std::array<int, 25> offsets{};
    offsets[0] = 0;
    int bucket = 0;
    for (int sect = 0; sect < 8; sect++) {
      for (int di = 0; di < comm::DIMENSION_SIZE; di++) {
        auto regs = comm::fwCommSectorSendRegion(sect, dims[di], p_domain->lattice_size_ghost,
                                                 p_domain->local_split_coord, p_domain->local_sub_box_lattice_region);
        for (auto &r : regs) {
          CommBox b;
          b.x_low = 2 * r.x_low;
          b.x_high = 2 * r.x_high;
          b.y_low = r.y_low;
          b.y_high = r.y_high;
          b.z_low = r.z_low;
          b.z_high = r.z_high;
          boxes.push_back(b);
        }
        offsets[bucket + 1] = static_cast<int>(boxes.size());
        bucket++;
      }
    }
    commSetSurfaceBoxes(boxes.data(), static_cast<int>(boxes.size()), offsets.data(), static_cast<int>(offsets.size()));
  }
  HANDLE_HIP(hipMalloc((void**)&d_surface_list, surface_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_list, ghost_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_surface_list_type, surface_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_list_type, ghost_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_surface_count, sizeof(int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_count, sizeof(int)));
  //这是对meta的首次更新，以后每次都得更新
  HANDLE_HIP(hipMalloc((void**)&d_meta,sizeof(dev_meta)));
  HANDLE_HIP(hipMemcpy(d_meta, &meta,sizeof(dev_meta), hipMemcpyHostToDevice));
  // printf("init global pointer\n");
  //set hash pointer to gpu global pointer
  // printDevMeta(meta);
//=======分配全局物理位图锁=======
  long int total_lattice_sites = meta.size_x * meta.size_y * meta.size_z; 
  long int num_ull = (total_lattice_sites + 63) / 64; 
  bitmap_lock_bytes = num_ull * sizeof(unsigned long long int);

  HANDLE_HIP(hipMalloc((void**)&d_Global_Bitmap_Lock, bitmap_lock_bytes));
  HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
  // HANDLE_HIP(hipMalloc((void**)&d_Global_Bitmap_Lock, num_ull * sizeof(unsigned long long int)));
  // HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, num_ull * sizeof(unsigned long long int)));
  //===============================
  arr_size = pair_atoms.size();
  // printf("arr_size is %ld",arr_size);
  //=======每个 walker 线程一条独立持久的 RNG 流（Philox）=======
  // 线程总数与 recb kernel 启动口径一致：numBlocks = pair_atoms_size / 128 + 1
  {
    const long int rng_block = 128;
    rng_states_count = (pair_atoms_size / rng_block + 1) * rng_block;
    HANDLE_HIP(hipMalloc((void**)&d_rng_states, rng_states_count * sizeof(hiprandStatePhilox4_32_10_t)));
  }
  //=======recb 前扇区筛选的工作缓冲区（槽位号列表 + 计数）=======
  HANDLE_HIP(hipMalloc((void**)&d_sector_slots, pair_atoms_size * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_sector_slot_count, sizeof(int)));
  HANDLE_HIP(hipMalloc((void**)&d_recb_work_head, sizeof(unsigned int)));
  hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
  //=======MoRe recb 边界墓碑刷新：单表 + LIVE key scratch=======
  HANDLE_HIP(hipMalloc((void**)&d_more_refresh_keys, pair_atoms_size * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_more_refresh_counts, 4 * sizeof(unsigned int)));
  //=============================================================
  hipLaunchKernelGGL(setGlobalPointer, 1, 1, 0, 0,
      d_meta,
      MoRe_Hash->device_ptr(),MoMo_Hash->device_ptr(),Re_Hash->device_ptr(),V_Hash->device_ptr(),Busy_Set->device_ptr(),
      surface_Hash->device_ptr(),ghost_Hash->device_ptr(),d_Pair_Atoms,d_Global_Bitmap_Lock,d_rng_states
  );

  hipDeviceSynchronize();
  initializeMoreOccupancyCounter();
  // 各 rank 错开种子，保证进程间流也互相独立；固定 seed 可整跑复现
  int rng_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rng_rank);
  {
    const long int rng_block = 128;
    const long int rng_grid = (rng_states_count + rng_block - 1) / rng_block;
    hipLaunchKernelGGL(initRngStates, rng_grid, rng_block, 0, 0,
        rng_states_count, rng_seed + 1000003ULL * (unsigned long long)rng_rank);
    CHECK_FOR_ERROR();
  }
  hipDeviceSynchronize();
  // std::cin.get();
  // GPUHashSet* hostPtr;
  // hipMemcpyFromSymbol(&hostPtr, g_MoMo_Hash, sizeof(GPUHashSet*));
  // printf("Global pointer value after synchronization: %p\n", hostPtr);
  // hipLaunchKernelGGL(printpointer, 1, 1, 0, 0,   );
  // system("read -p 'Press Enter to continue...' var");


  //这些是对返回CPU的空间
    // printf("malloc CPU hashset\n");
    // hipHostMalloc((void**)&lattice_list->h_MoRe_Hash,MoRe_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_MoMo_Hash,MoMo_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_Re_Hash,Re_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_V_Hash,V_Hash->device_ptr()->capacity*sizeof(long int));

    hipHostMalloc((void**)&lattice_list->h_ghost_Hash, ghost_capacity * sizeof(long int));
    hipHostMalloc((void**)&lattice_list->h_surface_Hash, surface_capacity * sizeof(long int));

    hipHostMalloc((void**)&lattice_list->h_ghost_Hash_type, ghost_capacity * sizeof(long int));
    hipHostMalloc((void**)&lattice_list->h_surface_Hash_type, surface_capacity * sizeof(long int));

  // printf("finish gpu prepare\n");
  hipHostFree( Pair_Atoms);
//   HANDLE_HIP(hipDeviceSynchronize());

// size_t free_mem_end = 0;
// size_t total_mem_end = 0;
// HANDLE_HIP(hipMemGetInfo(&free_mem_end, &total_mem_end));

// std::cerr << "[gpu_prepare] end current device=" << device_id
//           << " free_mem=" << free_mem_end
//           << " total_mem=" << total_mem_end
//           << std::endl;
}

void gpu_final(){
  using namespace gpuSimulate;

  logRecbRunSummary();
  logRecbPhaseTimings();
  destroyRecbProfileEvents();

  // 释放GPU内存
  HANDLE_HIP(hipFree(d_sum_rates));
  HANDLE_HIP(hipFree(d_event));

  HANDLE_HIP(hipFree(deviceVacancyInfo));
  HANDLE_HIP(hipFree(deviceNeighbourInfo));
  HANDLE_HIP(hipFree(d_vac_all));
  HANDLE_HIP(hipFree(d_vac_sector));
  HANDLE_HIP(hipFree(d_vac_count_all));
  HANDLE_HIP(hipFree(d_vac_count_sector));

  hipHostFree(h_event);
  hipHostFree(sum_rates);

  hipHostFree(hostVacancyInfo);
  hipHostFree(hostNeighbourInfo);

if (d_Global_Bitmap_Lock != nullptr) {
    HANDLE_HIP(hipFree(d_Global_Bitmap_Lock));
  }
  //z这里是recb_solver_GPU的部分
  delete MoRe_Hash;
  delete MoMo_Hash;
  delete Re_Hash;
  delete V_Hash;

  HANDLE_HIP(hipFree(d_surface_list));
  HANDLE_HIP(hipFree(d_ghost_list));
  HANDLE_HIP(hipFree(d_surface_list_type));
  HANDLE_HIP(hipFree(d_ghost_list_type));
  HANDLE_HIP(hipFree(d_surface_count));
  HANDLE_HIP(hipFree(d_ghost_count));

  hipFree(d_Pair_Atoms);
  if (d_rng_states) {
    HANDLE_HIP(hipFree(d_rng_states));
    d_rng_states = nullptr;
    rng_states_count = 0;
  }
  if (d_sector_slots) {
    HANDLE_HIP(hipFree(d_sector_slots));
    d_sector_slots = nullptr;
  }
  if (d_sector_slot_count) {
    HANDLE_HIP(hipFree(d_sector_slot_count));
    d_sector_slot_count = nullptr;
  }
  if (d_recb_work_head) {
    HANDLE_HIP(hipFree(d_recb_work_head));
    d_recb_work_head = nullptr;
  }
  if (d_more_refresh_keys) {
    HANDLE_HIP(hipFree(d_more_refresh_keys));
    d_more_refresh_keys = nullptr;
  }
  if (d_more_refresh_counts) {
    HANDLE_HIP(hipFree(d_more_refresh_counts));
    d_more_refresh_counts = nullptr;
  }
  if (d_recb_verify_stats) {
    HANDLE_HIP(hipFree(d_recb_verify_stats));
    d_recb_verify_stats = nullptr;
  }
  if (d_recb_walker_lengths) {
    HANDLE_HIP(hipFree(d_recb_walker_lengths));
    d_recb_walker_lengths = nullptr;
  }


  // hipHostFree(Pair_Atoms);

  delete ghost_Hash;
  delete surface_Hash;
  delete comm_ghost_bucket0;
  delete comm_ghost_bucket1;
  delete comm_ghost_bucket2;
  delete comm_ghost_bucket3;
  delete comm_ghost_bucket4;
  delete comm_ghost_bucket5;
  delete comm_ghost_bucket6;
  if (d_comm_ghost_regions) HANDLE_HIP(hipFree(d_comm_ghost_regions));
  for (int i = 0; i < 24; i++) {
    delete comm_surface_buckets[i];
  }
  if (d_comm_surface_bucket_ptrs) HANDLE_HIP(hipFree(d_comm_surface_bucket_ptrs));
  if (d_comm_surface_boxes) HANDLE_HIP(hipFree(d_comm_surface_boxes));
  if (d_comm_surface_offsets) HANDLE_HIP(hipFree(d_comm_surface_offsets));

  delete g_Busy_Set;
  hipFree(d_meta);
}

void exchange_lattices(dev_event* h_event){
  using namespace gpuSimulate;

  if (lattice_list->vac_hash.count(h_event->from_id)) {
    lattice_list->vac_hash.erase(h_event->from_id);
    lattice_list->vac_hash.emplace(std::make_pair(h_event->to_id, VacancyHash{}));
  } else assert(0);

  switch (h_event->to_type._type)
  {
    case LatticeTypes::Re:
      lattice_list->re_hash.erase(h_event->to_id);
      lattice_list->re_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Mn:
      lattice_list->mn_hash.erase(h_event->to_id);
      lattice_list->mn_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Ni:
      lattice_list->ni_hash.erase(h_event->to_id);
      lattice_list->ni_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Si:
      lattice_list->si_hash.erase(h_event->to_id);
      lattice_list->si_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Mo:
      break;
    default:
      assert(0);
      break;
  }
}

void add_surface(_type_lattice_id& surface_id,
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z) {
  using namespace gpuSimulate;

  _type_lattice_id surface_x = surface_id % lattice_list->meta.size_x;
  _type_lattice_id surface_y = (surface_id / lattice_list->meta.size_x) % lattice_list->meta.size_y;
  _type_lattice_id surface_z = surface_id / (lattice_list->meta.size_x * lattice_list->meta.size_y);
  const int dims[comm::DIMENSION_SIZE] = {comm::DIM_X, comm::DIM_Y, comm::DIM_Z};
  // todo the regions can be static.
  std::array<std::vector<comm::Region<comm::_type_lattice_coord>>, comm::DIMENSION_SIZE> send_regions; // send regions in each dimension
  // 应该可以不循环 8 次的 ？ 如果发送区没有重叠的话
  for (int sect = 0; sect < 8; sect++) {
    for (int d = 0; d < comm::DIMENSION_SIZE; d++) {
      send_regions[d] = comm::fwCommSectorSendRegion(sect, dims[d], p_domain->lattice_size_ghost,
                                                    p_domain->local_split_coord, p_domain->local_sub_box_lattice_region);
    }
    for(int d = 0; d < comm::DIMENSION_SIZE; d++) {
      for (auto &r : send_regions[d]) {
        _type_lattice_size x_low = 2 * r.x_low;
        _type_lattice_size y_low = r.y_low;
        _type_lattice_size z_low = r.z_low;
        _type_lattice_size x_high = 2 * r.x_high;
        _type_lattice_size y_high = r.y_high;
        _type_lattice_size z_high = r.z_high;
    
        if(x_low <= surface_x && surface_x < x_high && y_low <= surface_y && surface_y < y_high && z_low <= surface_z && surface_z < z_high) {
          if(d == 0){
            if(!exchange_surface_x[sect].count(surface_id)) exchange_surface_x[sect].emplace(surface_id);
          } else if(d == 1) {
            if(!exchange_surface_y[sect].count(surface_id)) exchange_surface_y[sect].emplace(surface_id);
          } else {
            if(!exchange_surface_z[sect].count(surface_id)) exchange_surface_z[sect].emplace(surface_id);
          }
        }
      }
    }
  }
}

void output_time() {
  using namespace gpuSimulate;
  // kiwi::logs::v(" ", " time_event is : {} s. \n", time_event);
  // kiwi::logs::v(" ", " time_event_trans is : {} s. \n", time_event_trans);
  // kiwi::logs::v(" ", " time_event_other : {} s. \n", time_event_other);
  // kiwi::logs::v(" ", " time_update_construct : {} s. \n", time_update_construct);
  // kiwi::logs::v(" ", " time_update_trans : {} s. \n", time_update_trans);
}
__global__ void update_hash_set(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int sector_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int  neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){
    // 单线程版：每个线程独立处理一个 receive buffer 元素，靠位图锁保证跨线程互斥，
    // 不再用 tile 协作（tiled_*_op 的 leader 串行结构在桶=1 时无意义且退化）。
    long int BCC_DBX = 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= receive_len) return;

    long int x=0,y=0,z=0;
    long int o_x=0,o_y=0,o_z=0;
    long int id=0,o_id=0;
   int type = 0;

    o_x= d_buffer[index].x;
    o_y= d_buffer[index].y;
    o_z= d_buffer[index].z;
    o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);

    type = d_buffer[index].type;
    if(dimension == 0){
      if(sector_id % 2 == 0){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    }else if(dimension == 1){

      x = d_buffer[index].x;
      if(sector_id == 0 || sector_id == 1 || sector_id == 4 || sector_id == 5) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    
    }else if(dimension == 2){
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(sector_id < 4) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    }

    // 跨 Block 并发互斥：单线程位图锁（重复 id 由抢锁拦截，一个成功一个放弃）
    if (!try_acquire_bitmap_lock_single(id)) return;

    // 级联删除，物理优化：按 Mo-Re 合金的浓度与反应产物留存概率，从高到低排序。
    // 一个 id 只属于一张表，删到第一张命中的就停。
    if (hash_set_remove_atom_op(g_Re_Hash, id) == 1) {
        // 删掉了 Re
    } else if (hash_set_remove_atom_op(g_MoRe_Hash, id) == 1) {
        // 删掉了 MoRe
    } else if (hash_set_remove_atom_op(g_MoMo_Hash, id) == 1) {
        // 删掉了 MoMo
    } else {
        hash_set_remove_atom_op(g_V_Hash, id);
    }

    // 按新类型插入对应表
    if (type == LatticeTypes::V) {
      hash_set_insert_atom_op(g_V_Hash, id);
    } else if (type == LatticeTypes::Re) {
      hash_set_insert_atom_op(g_Re_Hash, id);
    } else if (type == LatticeTypes::MoMo) {
      hash_set_insert_atom_op(g_MoMo_Hash, id);
    } else if (type == LatticeTypes::MoRe) {
      hash_set_insert_atom_op(g_MoRe_Hash, id);
    }

    release_bitmap_lock_single(id);
    return;
}


__global__ void update_hash_set_tiled(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int sector_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int  neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){
  // printf("env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
  //       neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2);

  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len - 1)
    return;

  long int  x,y,z=0;
  long int o_x=0,o_y=0,o_z=0;
  long int id=0,o_id=0;
  o_x= d_buffer[index].x;
  o_y= d_buffer[index].y;
  o_z= d_buffer[index].y;
  o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
  if(dimension == 0){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      if(sector_id % 2 == 0){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);
      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
           hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      if(sector_id == 0 || sector_id == 1 || sector_id == 4 || sector_id == 5) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      // printf("change sur pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(sector_id < 4) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
      //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      // printf("change sur pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }

  
}
__global__ void update_hash_set2(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int next_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){
  // 单线程版（与 update_hash_set 一致）：每线程独立处理，位图锁保证跨线程互斥。
  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= receive_len) return;

  long int x=0,y=0,z=0;
  long int id=0,o_id=0;
  long int o_x=0,o_y=0,o_z=0;
  int type = 0;

  o_x= d_buffer[index].x;
  o_y= d_buffer[index].y;
  o_z= d_buffer[index].y;
  o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
  type = d_buffer[index].type;

  if(dimension == 0){
    if(next_id % 2 == 1){
      d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
    }else{
      d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
    }
    x = d_buffer[index].x;
    y = d_buffer[index].y;
    z = d_buffer[index].z;
    id = getId(x,y,z);
  }
  else if(dimension == 1){
    x = d_buffer[index].x;
    if(next_id == 2 || next_id == 3 || next_id == 6 || next_id == 7) {
      d_buffer[index].y += sub_box_lattice_size_1;
    } else {
      d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
    }
    y = d_buffer[index].y;
    z = d_buffer[index].z;
    id = getId(x,y,z);
  }
  else if(dimension == 2){
    x = d_buffer[index].x;
    y = d_buffer[index].y;
    if(next_id > 3 ) {
      d_buffer[index].z += sub_box_lattice_size_2;
    }else{
      d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
    }
    z = d_buffer[index].z;
    id = getId(x,y,z);
  }

  // 跨 Block 并发互斥：单线程位图锁
  if (!try_acquire_bitmap_lock_single(id)) return;

  // 级联删除（删到第一张命中的表就停）
  if (hash_set_remove_atom_op(g_Re_Hash, id) == 1) {
  } else if (hash_set_remove_atom_op(g_MoRe_Hash, id) == 1) {
  } else if (hash_set_remove_atom_op(g_MoMo_Hash, id) == 1) {
  } else {
    hash_set_remove_atom_op(g_V_Hash, id);
  }

  // 按新类型插入对应表
  if (type == LatticeTypes::V) {
    hash_set_insert_atom_op(g_V_Hash, id);
  } else if (type == LatticeTypes::Re) {
    hash_set_insert_atom_op(g_Re_Hash, id);
  } else if (type == LatticeTypes::MoMo) {
    hash_set_insert_atom_op(g_MoMo_Hash, id);
  } else if (type == LatticeTypes::MoRe) {
    hash_set_insert_atom_op(g_MoRe_Hash, id);
  }

  release_bitmap_lock_single(id);
  return;
}

__global__ void update_hash_set2_tiled(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int next_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){

  // printf("env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2);
  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len)
    return;

  long int x,y,z;
  long int id=0,o_id=0;
  long int o_x=0,o_y=0,o_z=0;
  // long int id=0;
  o_x= d_buffer[index].x;
  o_y= d_buffer[index].y;
  o_z= d_buffer[index].y;
  o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
  if(dimension == 0){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      if(next_id % 2 == 1){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);
      // printf("after change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld-%ld\n",x,y,z,id,d_buffer[index].type);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        // printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      if(next_id == 2 || next_id == 3 || next_id == 6 || next_id == 7) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(next_id > 3 ) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  
}
void transferBufferToGPU(ChangeLattice *buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id){
using namespace gpuSimulate;
    ChangeLattice_GPU *h_buffer;

      hipHostMalloc((void**)&h_buffer,receive_len*sizeof(ChangeLattice_GPU));
      init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

      ChangeLattice_GPU *d_buffer;
      HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
      const double h2d_start = MPI_Wtime();
      HANDLE_HIP(hipMemcpy(d_buffer,h_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
      gpuSimulate::comm_gpu_maintenance_profiler.h2d_time += MPI_Wtime() - h2d_start;
      const double hash_update_start = MPI_Wtime();
      int blockSize = 128;
      int blockNum = (receive_len)/blockSize + 1;
      if (d_Global_Bitmap_Lock != nullptr && bitmap_lock_bytes > 0) {
        HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
      }
      //printf("start kernel update\n");
      hipLaunchKernelGGL(update_hash_set, blockNum, blockSize, 0, 0, 
        d_buffer,
        receive_len,
        dimension,
        id,
        sub_box_lattice_size[0],
        sub_box_lattice_size[1],
        sub_box_lattice_size[2],
        neighbour_local_sub_box[0],
        neighbour_local_sub_box[1],
        neighbour_local_sub_box[2]
      );
      hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
      hipDeviceSynchronize();
      gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
    hipFree(d_buffer);
    hipHostFree(h_buffer);
}


void transferBufferToGPU2(ChangeLattice *buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id){
using namespace gpuSimulate;
    ChangeLattice_GPU *h_buffer;
    hipHostMalloc((void**)&h_buffer,receive_len*sizeof(ChangeLattice_GPU));
    init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

    ChangeLattice_GPU *d_buffer;
    HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
    const double h2d_start = MPI_Wtime();
    HANDLE_HIP(hipMemcpy(d_buffer,h_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
    gpuSimulate::comm_gpu_maintenance_profiler.h2d_time += MPI_Wtime() - h2d_start;
    const double hash_update_start = MPI_Wtime();

    if (d_Global_Bitmap_Lock != nullptr && bitmap_lock_bytes > 0) {
        HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
    }
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    // hipLaunchKernelGGL(print_g_Pair_Atoms,1,1,0,0,arr_size);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(update_hash_set2,blockNum, blockSize, 0, 0, 
      d_buffer,
      receive_len,
      dimension,
      next_id,
      sub_box_lattice_size[0],
      sub_box_lattice_size[1],
      sub_box_lattice_size[2],
      neighbour_local_sub_box[0],
      neighbour_local_sub_box[1],
      neighbour_local_sub_box[2]
    );
    hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
    hipDeviceSynchronize();
    gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
    hipFree(d_buffer);
    hipHostFree(h_buffer);
}

void transferBufferToGPU_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id){
      const double hash_update_start = MPI_Wtime();
      int blockSize = 128;
      int blockNum = (receive_len)/blockSize + 1;
      hipLaunchKernelGGL(update_hash_set, blockNum, blockSize, 0, 0, 
        d_buffer,
        receive_len,
        dimension,
        id,
        sub_box_lattice_size[0],
        sub_box_lattice_size[1],
        sub_box_lattice_size[2],
        neighbour_local_sub_box[0],
        neighbour_local_sub_box[1],
        neighbour_local_sub_box[2]
      );
      hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
      hipDeviceSynchronize();
      gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
}

void transferBufferToGPU2_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id){
    const double hash_update_start = MPI_Wtime();
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    hipDeviceSynchronize();
    hipLaunchKernelGGL(update_hash_set2,blockNum, blockSize, 0, 0, 
      d_buffer,
      receive_len,
      dimension,
      next_id,
      sub_box_lattice_size[0],
      sub_box_lattice_size[1],
      sub_box_lattice_size[2],
      neighbour_local_sub_box[0],
      neighbour_local_sub_box[1],
      neighbour_local_sub_box[2]
    );
    hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
    hipDeviceSynchronize();
    gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
}

void rebuildPairAtomsAfterComm() {
  using namespace gpuSimulate;
  if (pair_atoms_size <= 0) return;
  if (!d_Pair_Atoms) return;
  if (!MoMo_Hash && !MoRe_Hash) return;

  const double rebuild_cleanup_start = MPI_Wtime();

  static int *d_pair_compact_count = nullptr;
  if (!d_pair_compact_count) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&d_pair_compact_count), sizeof(int)));
  }

  const int max_out = static_cast<int>(pair_atoms_size);

  // 排查统计（对照 key 数、validate 计数/打印）只在正确性 debug 模式启用；
  // clear+reinsert 的修复逻辑不依赖它，始终执行。
  const bool verify_detail = recbCorrectnessDebugEnabled();

  // 对照统计①：compact 之前统计 g_MoMo_Hash + g_MoRe_Hash 的有效 key 数（应 == 花名册应收录数）
  unsigned int h_hash_keys_before = 0;
  if (verify_detail) {
    hipLaunchKernelGGL(reset_validate_counters, 1, 1, 0, 0);
    const int cb = 256;
    if (MoMo_Hash && MoMo_Hash->device_ptr()) {
      const int cap = MoMo_Hash->get_capacity();
      const int grid = (cap + cb - 1) / cb;
      if (grid > 0) hipLaunchKernelGGL(count_hash_keys_all, grid, cb, 0, 0, MoMo_Hash->device_ptr());
    }
    if (MoRe_Hash && MoRe_Hash->device_ptr()) {
      const int cap = MoRe_Hash->get_capacity();
      const int grid = (cap + cb - 1) / cb;
      if (grid > 0) hipLaunchKernelGGL(count_hash_keys_all, grid, cb, 0, 0, MoRe_Hash->device_ptr());
    }
    hipDeviceSynchronize();
    unsigned int h_all[4] = {0, 0, 0, 0};
    hipMemcpyFromSymbol(h_all, HIP_SYMBOL(g_validate_counters), 4 * sizeof(unsigned int));
    h_hash_keys_before = h_all[2];
  }

  // 占用率统计：clear 之前统计 MoMo/MoRe 的 empty/tombstone/live，
  // 有效占用 = (live+tombstone)/capacity 直接决定探测链长度，作为"容量是否足够"的预警。
  if (verify_detail) {
    hipLaunchKernelGGL(reset_validate_counters, 1, 1, 0, 0);
    const int cb = 256;
    if (MoMo_Hash && MoMo_Hash->device_ptr()) {
      const int cap = MoMo_Hash->get_capacity();
      const int grid = (cap + cb - 1) / cb;
      if (grid > 0) hipLaunchKernelGGL(count_hash_occupancy, grid, cb, 0, 0, MoMo_Hash->device_ptr(), 0);
    }
    if (MoRe_Hash && MoRe_Hash->device_ptr()) {
      const int cap = MoRe_Hash->get_capacity();
      const int grid = (cap + cb - 1) / cb;
      if (grid > 0) hipLaunchKernelGGL(count_hash_occupancy, grid, cb, 0, 0, MoRe_Hash->device_ptr(), 1);
    }
    hipDeviceSynchronize();
    unsigned int h_occ[6] = {0, 0, 0, 0, 0, 0};
    hipMemcpyFromSymbol(h_occ, HIP_SYMBOL(g_hash_occ), 6 * sizeof(unsigned int));
    const long long momo_cap = MoMo_Hash ? MoMo_Hash->get_capacity() : 1;
    const long long more_cap = MoRe_Hash ? MoRe_Hash->get_capacity() : 1;
    kiwi::logs::v(" ", " [hash-occupancy] MoMo cap={} empty={} tombstone={} live={} eff_load={:.3f} | MoRe cap={} empty={} tombstone={} live={} eff_load={:.3f}.\n",
                  momo_cap, h_occ[0], h_occ[1], h_occ[2],
                  (h_occ[2] + h_occ[1]) / (double)momo_cap,
                  more_cap, h_occ[3], h_occ[4], h_occ[5],
                  (h_occ[5] + h_occ[4]) / (double)more_cap);
  }

  HANDLE_HIP(hipMemset(d_Pair_Atoms, 0xFF, static_cast<size_t>(pair_atoms_size) * sizeof(long int)));
  HANDLE_HIP(hipMemset(d_pair_compact_count, 0, sizeof(int)));

  const int blockSize = 256;
  long int n_momo = 0;
  long int n_total = 0;

  // compact MoMo（先记录 MoMo 数量，供 reinsert 分桶）
  if (MoMo_Hash && MoMo_Hash->device_ptr()) {
    const int cap = MoMo_Hash->get_capacity();
    const int grid = (cap + blockSize - 1) / blockSize;
    if (grid > 0) {
      hipLaunchKernelGGL(compactHashSetBoundedKernel, grid, blockSize, 0, 0,
                         MoMo_Hash->device_ptr(), d_Pair_Atoms, d_pair_compact_count, max_out);
    }
    hipDeviceSynchronize();
    int n_momo_int = 0;
    HANDLE_HIP(hipMemcpy(&n_momo_int, d_pair_compact_count, sizeof(int), hipMemcpyDeviceToHost));
    n_momo = n_momo_int;
    n_total = n_momo;
  }

  // compact MoRe（接在 MoMo 后面，共享 d_pair_compact_count）
  if (MoRe_Hash && MoRe_Hash->device_ptr()) {
    const int cap = MoRe_Hash->get_capacity();
    const int grid = (cap + blockSize - 1) / blockSize;
    if (grid > 0) {
      hipLaunchKernelGGL(compactHashSetBoundedKernel, grid, blockSize, 0, 0,
                         MoRe_Hash->device_ptr(), d_Pair_Atoms, d_pair_compact_count, max_out);
    }
    hipDeviceSynchronize();
    int n_total_int = 0;
    HANDLE_HIP(hipMemcpy(&n_total_int, d_pair_compact_count, sizeof(int), hipMemcpyDeviceToHost));
    n_total = n_total_int;
  }

  if (n_total > pair_atoms_size) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    kiwi::logs::v(" ", " [pair-roster-capacity-fatal] rank={} live={} capacity={}.\n",
                  rank, n_total, pair_atoms_size);
    std::abort();
  }

  // 关键修复：clear 两个哈希表（清空有效 key 与长期累积的 tombstone），再按 compact 结果重新插入。
  // 否则 tombstone 会拉长开放寻址探测链，占用度逼近 1 时 hash_set_contains 误判为
  // “不存在”，validate_roster_all 因此误清 slot，造成 roster/hash 差值累积（间隙对漏游走）。
  if (MoMo_Hash) MoMo_Hash->clear();
  if (MoRe_Hash) {
    MoRe_Hash->clear();
    hipLaunchKernelGGL(reset_more_occupancy, 1, 1, 0, 0,
                       static_cast<unsigned int>(MoRe_Hash->get_capacity()));
    CHECK_FOR_ERROR();
  }

  const int r_block = 256;
  if (MoMo_Hash && MoMo_Hash->device_ptr() && n_momo > 0) {
    const int grid = static_cast<int>((n_momo + r_block - 1) / r_block);
    hipLaunchKernelGGL(reinsert_from_roster, grid, r_block, 0, 0,
                       MoMo_Hash->device_ptr(), d_Pair_Atoms, 0L, n_momo);
  }
  if (MoRe_Hash && MoRe_Hash->device_ptr() && n_total > n_momo) {
    const int grid = static_cast<int>((n_total - n_momo + r_block - 1) / r_block);
    hipLaunchKernelGGL(reinsert_from_roster, grid, r_block, 0, 0,
                       MoRe_Hash->device_ptr(), d_Pair_Atoms, n_momo, n_total);
  }
  // rebuild 已经得到无墓碑的新表；抵消 reinsert 路径设置的 dirty。
  if (MoRe_Hash) {
    const unsigned int live = static_cast<unsigned int>(n_total - n_momo);
    h_more_cached_counts[0] =
        static_cast<unsigned int>(MoRe_Hash->get_capacity()) - live;
    h_more_cached_counts[1] = 0;
    h_more_cached_counts[2] = live;
    h_more_cache_initialized = true;
  }
  hipLaunchKernelGGL(reset_more_occupancy, 1, 1, 0, 0,
                     MoRe_Hash ? static_cast<unsigned int>(MoRe_Hash->get_capacity()) : 0u);
  hipDeviceSynchronize();

  // 对照统计②：compact 之后统计花名册有效 slot 数（应 == h_hash_keys_before）
  if (verify_detail) {
    unsigned int h_roster_after = 0;
    hipLaunchKernelGGL(reset_validate_counters, 1, 1, 0, 0);
    const int cb = 256;
    const int grid = static_cast<int>((pair_atoms_size + cb - 1) / cb);
    if (grid > 0) {
      hipLaunchKernelGGL(count_roster_slots, grid, cb, 0, 0, pair_atoms_size);
      hipDeviceSynchronize();
      unsigned int h_all2[4] = {0, 0, 0, 0};
      hipMemcpyFromSymbol(h_all2, HIP_SYMBOL(g_validate_counters), 4 * sizeof(unsigned int));
      h_roster_after = h_all2[3];
    }
    if (h_hash_keys_before != h_roster_after) {
      kiwi::logs::v(" ", " [compact-check] hash_keys_before={} roster_after={} diff={}（compact 漏收录的 key 数，应恒为 0）.\n",
                    h_hash_keys_before, h_roster_after,
                    (h_hash_keys_before > h_roster_after) ? (int)(h_hash_keys_before - h_roster_after) : 0);
    }
  }

  // 兜底清理：rebuild 后全量校验花名册，清除仍指向"已不在哈希表的 id"（如游走到 y 通信边界、
  // 被 update_hash_set 移交给邻居而变 Mo 的陈旧 slot），保证 roster 与 g_MoMo_Hash/g_MoRe_Hash 一致。
  // 清理逻辑始终执行（正确性修复）；计数与打印收进正确性 debug 开关。
  {
    const int v_block = 256;
    const int v_grid = static_cast<int>((pair_atoms_size + v_block - 1) / v_block);
    if (v_grid > 0) {
      if (verify_detail) hipLaunchKernelGGL(reset_validate_counters, 1, 1, 0, 0);
      hipLaunchKernelGGL(validate_roster_all, v_grid, v_block, 0, 0, pair_atoms_size, verify_detail);
      hipDeviceSynchronize();
      if (verify_detail) {
        unsigned int h_validate_counters[2] = {0, 0};
        hipMemcpyFromSymbol(h_validate_counters, HIP_SYMBOL(g_validate_counters), 2 * sizeof(unsigned int));
        if (h_validate_counters[0] > 0) {
          kiwi::logs::v(" ", " [validate-count] cleared={} ghost={}（本次 rebuild 清除的陈旧 slot 数，用于对照 roster/hash 差值）.\n",
                        h_validate_counters[0], h_validate_counters[1]);
        }
      }
    }
  }

  hipLaunchKernelGGL(invalidate_recb_stability, 1, 8, 0, 0);
  hipDeviceSynchronize();
  // compactHashSetBoundedKernel writes every surviving pair into the prefix.
  // Holes may appear later, but no live slot can exist beyond this scan bound
  // until the next rebuild updates it again.
  pair_roster_scan_size = n_total;
  gpuSimulate::comm_gpu_maintenance_profiler.rebuild_cleanup_time += MPI_Wtime() - rebuild_cleanup_start;
}

__global__ void packChangeLatticeFromIdsKernel(const _type_lattice_id *ids, ChangeLattice_GPU *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  _type_lattice_id id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  out[i].x = x;
  out[i].y = y;
  out[i].z = z;
  out[i].type = static_cast<int>(getType(id));
}

void packChangeLatticeFromIds_Device(const _type_lattice_id *d_ids, ChangeLattice_GPU *d_out, int n) {
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(packChangeLatticeFromIdsKernel, blockNum, blockSize, 0, 0, d_ids, d_out, n);
  hipDeviceSynchronize();
}
